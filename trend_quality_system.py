#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货品种趋势质量评估系统（单文件版）
依赖：pandas, numpy, scipy, python-dateutil, pandas_ta
"""

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from dateutil import parser

def load_data(path: str) -> pd.DataFrame:
    """读取 CSV，并解析日期列"""
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """1. 缺失值前后填充  2. 合约换月展期（back‑adjusted）"""
    df = df.sort_values(['symbol', 'date'])
    df['price'] = df['price'].fillna(method='ffill').fillna(method='bfill')
    # TODO: 动态展期逻辑
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算趋势强度 & 流畅性指标"""
    # 示例：计算 10 日动量
    df['momentum_10'] = df.groupby('symbol')['price'].pct_change(10)
    # TODO: 加入 R² 拟合优度等流畅性指标
    return df

def score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """合成评分并按日生成排名"""
    # 示例：简易加权
    df['composite_score'] = (
        0.6 * df['momentum_10'].rank(pct=True) +
        0.4 * df['momentum_10'].rank(pct=True)  # 占位，后续改为流畅性指标
    )
    df['momentum_rank'] = df.groupby('date')['momentum_10'] \
                             .rank(ascending=False).astype(int)
    df['persistence_rank'] = df.groupby('date')['momentum_10'] \
                                .rank(ascending=False).astype(int)
    return df

def generate_report(df: pd.DataFrame, out_csv: str):
    """输出每日 CSV 报表"""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"已生成：{out_csv}")

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

if __name__ == '__main__':
    main()
