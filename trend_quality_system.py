#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货品种趋势质量评估系统（单文件版）
依赖：pandas, numpy, scipy, python-dateutil, pandas_ta
"""

import os
import pandas as pd
import numpy as np
from dateutil import parser
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    df.rename(columns={'product': 'symbol'}, inplace=True)
    df['price'] = df['settle_price']
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['symbol', 'date'])
    df['price'] = df['price'].ffill().bfill()
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    并行计算多尺度指标：
      - 短期(5,10)、中期(10,20)、长期(20,60)
      - 每尺度计算：动量、波动率归一化动量、回归斜率、R²
      - 1%–99% 分位数截断异常值
    假设传入的 df 已按 symbol/date 排好序。
    """
    def rolling_slope(x: pd.Series) -> float:
        y = x.values
        if len(y) < 2 or np.isnan(y).any(): return np.nan
        t = np.arange(len(y))
        return np.polyfit(t, y, 1)[0]

    def rolling_r2(x: pd.Series) -> float:
        y = x.values
        if len(y) < 2 or np.isnan(y).any(): return np.nan
        t = np.arange(len(y))
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = intercept + slope * t
        ss_res = ((y - y_pred)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        return 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

    def proc(sym_df: pd.DataFrame) -> pd.DataFrame:
        # sym_df 已按 date 排序
        sym_df = sym_df.copy()
        sym_df['ret_1'] = sym_df['price'].pct_change(1)
        scales = [(5,10), (10,20), (20,60)]
        for m, r in scales:
            # 1) 动量
            sym_df[f'mom_{m}'] = sym_df['price'].pct_change(m)
            # 2) 波动率
            sym_df[f'vol_{m}'] = sym_df['ret_1'].rolling(m, min_periods=5).std()
            # 3) 归一化动量
            sym_df[f'norm_mom_{m}'] = sym_df[f'mom_{m}'] / sym_df[f'vol_{m}']
            # 4) 回归斜率
            sym_df[f'slope_{m}'] = sym_df['price'].rolling(m, min_periods=5)\
                                           .apply(rolling_slope, raw=False)
            # 5) 滚动 R²
            sym_df[f'r2_{r}'] = sym_df['price'].rolling(r, min_periods=5)\
                                         .apply(rolling_r2, raw=False)
            # 6) 异常值过滤
            for col in [f'mom_{m}', f'norm_mom_{m}', f'slope_{m}', f'r2_{r}']:
                lo, hi = sym_df[col].quantile(0.01), sym_df[col].quantile(0.99)
                sym_df[col] = sym_df[col].clip(lo, hi)
        return sym_df.drop(columns=['ret_1'])

    # 并行执行
    parts = Parallel(n_jobs=-1)(
        delayed(proc)(grp) for _, grp in df.groupby('symbol')
    )
    # 直接 concat，不再排序
    return pd.concat(parts, axis=0).reset_index(drop=True)


def score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    多尺度融合打分：
      1) 每尺度内：norm_mom、slope、r2 三者等权平均
      2) 三尺度等权融合 → composite_score
      3) 按日排名 → composite_rank
    """
    scales = [(5,10), (10,20), (20,60)]
    scores = []
    for m, r in scales:
        cols = [f'norm_mom_{m}', f'slope_{m}', f'r2_{r}']
        # 百分位排名后求平均
        s = sum(df[c].rank(pct=True) for c in cols) / len(cols)
        scores.append(s)
    df['composite_score'] = sum(scores) / len(scores)

    # 按日排名
    df['composite_rank'] = (
        df.groupby('date')['composite_score']
          .rank(ascending=False, method='first')
          .fillna(-1)
          .astype(int)
    )
    return df


def generate_report(df: pd.DataFrame, out_csv: str):
    """输出每日 CSV 报表"""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"已生成：{out_csv}")

def backtest(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    简单等权多头回测：
    - 每日根据 composite_score 排名前 top_n 的品种，
    - 持有 1 天，计算第二天的日内收益（close/open - 1），
    - 结果存入 DataFrame 并计算累计收益。
    """
    # 1. 先计算每个品种的日内收益
    df = df.sort_values(['symbol', 'date'])
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    df['ret'] = df['next_open'] / df['open'] - 1

    # 2. 对每个交易日选 top_n，取 next day 的 ret
    records = []
    for day, group in df.groupby('date'):
        top_syms = group.nlargest(top_n, 'composite_score')['symbol']
        # 取下一交易日的 ret
        next_day = day + pd.Timedelta(days=1)
        mask = (df['date'] == next_day) & (df['symbol'].isin(top_syms))
        rets = df.loc[mask, 'ret']
        records.append({
            'date': day,
            'daily_ret': rets.mean() if not rets.empty else 0.0
        })

    back = pd.DataFrame(records).sort_values('date')
    back['cum_ret'] = (1 + back['daily_ret']).cumprod() - 1
    return back


def main():
    # 1. 读数据
    df = load_data('data/all_futures_20250413.csv')
    
    # —— 数据预览 —— 
    print("===== 数据预览：前 5 行 =====")
    print(df.head(), "\n")
    
    print("===== 数据概况 =====")
    print(df.info(), "\n")
    
    print("===== 缺失值统计 =====")
    print(df.isnull().sum(), "\n")
    # ——————————————————
    
    # 2. 清洗
    df = clean_data(df)
    # 3. 指标
    df = compute_indicators(df)
    # 4. 打分 & 排名
    df = score_and_rank(df)
    # 5. 报告
    generate_report(df, 'output/daily_scores.csv')

    # 保存回测结果
    print("===== 开始回测 top 5 =====")
    bt = backtest(df, top_n=5)
    # 打印完整回测表，或者你可以改成 bt.head() 只看前几行
    print(bt)

    # 清洗
    bt_clean = bt.dropna().copy()

    # 统计指标
    n_days = len(bt_clean)
    total_ret = bt_clean['cum_ret'].iloc[-1]
    annual_ret = (1 + total_ret)**(252/n_days) - 1
    drawdown = bt_clean['cum_ret'] - bt_clean['cum_ret'].cummax()
    max_drawdown = drawdown.min()
    sharpe = bt_clean['daily_ret'].mean() / bt_clean['daily_ret'].std() * (252**0.5)

    print(f"年化收益率: {annual_ret:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"年化夏普比率: {sharpe:.2f}")

    # 绘图
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(bt_clean['date'], bt_clean['cum_ret'])
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Backtest Cumulative Return')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
