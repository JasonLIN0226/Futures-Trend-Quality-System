#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监控与历史回测模块
支持两种模式：
 1. backtest - 历史回测
 2. live     - 每天获取最新行情并生成交易信号
"""
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import akshare as ak

from trend_quality_system import (
    load_data, clean_data, compute_indicators,
    score_and_rank_custom, generate_volatility_adjusted_long_signals,
    PortfolioManager, run_best_weight_strategy_with_portfolio_and_plot
)
def future_history_bar(symbol: str, length: int = 750) -> pd.DataFrame:
    start_dt = datetime.now() - timedelta(days=length * 2)
    start_str = start_dt.strftime("%Y%m%d")
    df = ak.futures_main_sina(symbol=symbol, start_date=start_str)
    if df.empty:
        return df
    rename_map = {
        "日期": "date", "开盘价": "open", "最高价": "high",
        "最低价": "low", "收盘价": "close", "成交量": "volume",
        "持仓量": "open_interest", "动态结算价": "settle_price",
    }
    df = df.rename(columns=rename_map)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open","high","low","close","volume","open_interest","settle_price"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    return df.iloc[-length:].reset_index(drop=True)

futures_list = [
    {"symbol":"V0","exchange":"DCE","name":"PVC连续"},
    {"symbol":"P0","exchange":"DCE","name":"棕榈油连续"},
    {"symbol":"B0","exchange":"DCE","name":"豆二连续"},
    {"symbol":"M0","exchange":"DCE","name":"豆粕连续"},
    {"symbol":"I0","exchange":"DCE","name":"铁矿石连续"},
    {"symbol":"JD0","exchange":"DCE","name":"鸡蛋连续"},
    {"symbol":"L0","exchange":"DCE","name":"塑料连续"},
    {"symbol":"PP0","exchange":"DCE","name":"聚丙烯连续"},
    {"symbol":"FB0","exchange":"DCE","name":"纤维板连续"},
    {"symbol":"BB0","exchange":"DCE","name":"胶合板连续"},
    {"symbol":"Y0","exchange":"DCE","name":"豆油连续"},
    {"symbol":"C0","exchange":"DCE","name":"玉米连续"},
    {"symbol":"A0","exchange":"DCE","name":"豆一连续"},
    {"symbol":"J0","exchange":"DCE","name":"焦炭连续"},
    {"symbol":"JM0","exchange":"DCE","name":"焦煤连续"},
    {"symbol":"CS0","exchange":"DCE","name":"淀粉连续"},
    {"symbol":"EG0","exchange":"DCE","name":"乙二醇连续"},
    {"symbol":"RR0","exchange":"DCE","name":"粳米连续"},
    {"symbol":"EB0","exchange":"DCE","name":"苯乙烯连续"},
    {"symbol":"PG0","exchange":"DCE","name":"液化石油气连续"},
    {"symbol":"LH0","exchange":"DCE","name":"生猪连续"},
    {"symbol":"TA0","exchange":"CZCE","name":"PTA连续"},
    {"symbol":"OI0","exchange":"CZCE","name":"菜油连续"},
    {"symbol":"RS0","exchange":"CZCE","name":"菜籽连续"},
    {"symbol":"RM0","exchange":"CZCE","name":"菜粕连续"},
    {"symbol":"WH0","exchange":"CZCE","name":"强麦连续"},
    {"symbol":"JR0","exchange":"CZCE","name":"粳稻连续"},
    {"symbol":"SR0","exchange":"CZCE","name":"白糖连续"},
    {"symbol":"CF0","exchange":"CZCE","name":"棉花连续"},
    {"symbol":"RI0","exchange":"CZCE","name":"早籼稻连续"},
    {"symbol":"MA0","exchange":"CZCE","name":"甲醇连续"},
    {"symbol":"FG0","exchange":"CZCE","name":"玻璃连续"},
    {"symbol":"LR0","exchange":"CZCE","name":"晚籼稻连续"},
    {"symbol":"SF0","exchange":"CZCE","name":"硅铁连续"},
    {"symbol":"SM0","exchange":"CZCE","name":"锰硅连续"},
    {"symbol":"CY0","exchange":"CZCE","name":"棉纱连续"},
    {"symbol":"AP0","exchange":"CZCE","name":"苹果连续"},
    {"symbol":"CJ0","exchange":"CZCE","name":"红枣连续"},
    {"symbol":"UR0","exchange":"CZCE","name":"尿素连续"},
    {"symbol":"SA0","exchange":"CZCE","name":"纯碱连续"},
    {"symbol":"PF0","exchange":"CZCE","name":"短纤连续"},
    {"symbol":"PK0","exchange":"CZCE","name":"花生连续"},
    {"symbol":"SH0","exchange":"CZCE","name":"烧碱连续"},
    {"symbol":"PX0","exchange":"CZCE","name":"对二甲苯连续"},
    {"symbol":"FU0","exchange":"SHFE","name":"燃料油连续"},
    {"symbol":"SC0","exchange":"INE","name":"上海原油连续"},
    {"symbol":"AL0","exchange":"SHFE","name":"铝连续"},
    {"symbol":"RU0","exchange":"SHFE","name":"天然橡胶连续"},
    {"symbol":"ZN0","exchange":"SHFE","name":"沪锌连续"},
    {"symbol":"CU0","exchange":"SHFE","name":"铜连续"},
    {"symbol":"AU0","exchange":"SHFE","name":"黄金连续"},
    {"symbol":"RB0","exchange":"SHFE","name":"螺纹钢连续"},
    {"symbol":"WR0","exchange":"SHFE","name":"线材连续"},
    {"symbol":"PB0","exchange":"SHFE","name":"铅连续"},
    {"symbol":"AG0","exchange":"SHFE","name":"白银连续"},
    {"symbol":"BU0","exchange":"SHFE","name":"沥青连续"},
    {"symbol":"HC0","exchange":"SHFE","name":"热轧卷板连续"},
    {"symbol":"SN0","exchange":"SHFE","name":"锡连续"},
    {"symbol":"NI0","exchange":"SHFE","name":"镍连续"},
    {"symbol":"SP0","exchange":"SHFE","name":"纸浆连续"},
    {"symbol":"NR0","exchange":"INE","name":"20号胶连续"},
    {"symbol":"SS0","exchange":"SHFE","name":"不锈钢连续"},
    {"symbol":"LU0","exchange":"INE","name":"低硫燃料油连续"},
    {"symbol":"BC0","exchange":"INE","name":"国际铜连续"},
    {"symbol":"AO0","exchange":"SHFE","name":"氧化铝连续"},
    {"symbol":"BR0","exchange":"SHFE","name":"丁二烯橡胶连续"},
    {"symbol":"EC0","exchange":"INE","name":"集运指数欧线期货连续"},
    {"symbol":"IF0","exchange":"CFFEX","name":"沪深300指数期货连续"},
    {"symbol":"TF0","exchange":"CFFEX","name":"5年期国债期货连续"},
    {"symbol":"IH0","exchange":"CFFEX","name":"上证50指数期货连续"},
    {"symbol":"IC0","exchange":"CFFEX","name":"中证500指数期货连续"},
    {"symbol":"TS0","exchange":"CFFEX","name":"2年期国债期货连续"},
    {"symbol":"IM0","exchange":"CFFEX","name":"中证连续指数期货连续"},
    {"symbol":"SI0","exchange":"GFEX","name":"工业硅连续"},
    {"symbol":"LC0","exchange":"GFEX","name":"碳酸锂连续"},
]

class TradingEngine:
    def __init__(self, mode, data_path=None, top_n=5, order_fraction=0.1):
        self.mode = mode
        self.data_path = data_path
        self.top_n = top_n
        self.order_fraction = order_fraction
        self.portfolio = None

    def backtest(self):
        run_best_weight_strategy_with_portfolio_and_plot()

    def live(self):
        all_dfs = []
        for item in futures_list:
            sym = item['symbol']
            df = future_history_bar(sym, length=150)
            if df.empty:
                continue
            df['symbol'] = sym
            all_dfs.append(df)
        if not all_dfs:
            print("No data fetched today.")
            return
        today_df = pd.concat(all_dfs, ignore_index=True)
        today_df["price"] = today_df["settle_price"]
        all_hist_path = self.data_path or './data'
        all_files = [f for f in os.listdir(all_hist_path) if f.endswith('.csv')]
        hist_dfs = [pd.read_csv(os.path.join(all_hist_path, f), parse_dates=['date']) for f in all_files]
        hist_df = pd.concat(hist_dfs, ignore_index=True)
        merged = pd.concat([hist_df, today_df], ignore_index=True)
        merged.drop_duplicates(subset=['symbol', 'date'], inplace=True)
        merged = clean_data(merged)
        merged = compute_indicators(merged)
        best_weight = 0.61
        scored = score_and_rank_custom(merged, weight_ts=best_weight)
        latest_date = today_df['date'].max()
        today_signals = scored[scored['date']==latest_date].copy()
        top_signals = today_signals.copy()

        report = top_signals[['symbol','trend_strength','trend_consistency','composite_score','momentum_rank','persistence_rank']]
        report = report.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
        print(f"=== {latest_date.date()} Signals ===")
        print(report.to_dict(orient='records'))

        today_str = latest_date.strftime("%Y%m%d")
        report.to_csv(f"daily_report_{today_str}.csv", index=False)
        print(f"✓ 已导出报告 daily_report_{today_str}.csv")

    def run(self):
        if self.mode == 'backtest':
            self.backtest()
        elif self.mode == 'live':
            self.live()
        else:
            raise ValueError("Unknown mode")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['backtest','live'], required=True)
    parser.add_argument('--data', help='历史数据文件夹路径，例如 ./data')
    args = parser.parse_args()

    engine = TradingEngine(mode=args.mode, data_path=args.data)
    engine.run()
