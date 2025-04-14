#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_20_daily_reports.py

Fetch historical futures data via AKShare and compute trend‐quality reports
for the last N business days.
"""
import os
import sys
import argparse
import datetime
import pandas as pd

# make sure we can import your modules
sys.path.append(os.getcwd())

from realtime_monitor import future_history_bar, futures_list
from trend_quality_system import clean_data, compute_indicators, score_and_rank_custom

def main():
    parser = argparse.ArgumentParser(
        description="生成近 N 个交易日的每日信号报告（无需每日 CSV 快照）"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./daily_reports",
        help="报告输出目录（默认: ./daily_reports）"
    )
    parser.add_argument(
        "--days", "-n",
        type=int,
        default=20,
        help="要回溯的交易日天数（默认: 20）"
    )
    parser.add_argument(
        "--weight", "-w",
        type=float,
        default=0.61,
        help="score_and_rank_custom 中的 weight_ts 参数（默认: 0.61）"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 拉取所有品种的历史数据（默认 750 天）
    print("📥 正在拉取各品种历史数据（750 天）…")
    all_dfs = []
    for item in futures_list:
        sym = item["symbol"]
        print(f"  - {sym}")
        df = future_history_bar(sym)
        if df.empty:
            print(f"    ⚠️ {sym} 无数据，已跳过")
            continue
        df["symbol"] = sym
        # 关键：把 settle_price 复制到 price，供后续 clean_data 使用
        df["price"] = df["settle_price"]
        all_dfs.append(df)

    if not all_dfs:
        print("❌ 未拉取到任何数据，退出")
        return

    # 2) 合并、清洗、计算指标
    hist_df = pd.concat(all_dfs, ignore_index=True)
    hist_df = hist_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    hist_df = clean_data(hist_df)           # 现在 price 已存在
    hist_df = compute_indicators(hist_df)

    # 3) 生成最近 N 个交易日列表
    today = datetime.datetime.today()
    biz_days = pd.bdate_range(end=today, periods=args.days)

    # 4) 对每个交易日做打分并输出 CSV
    for dt in biz_days:
        dt = pd.Timestamp(dt)
        # 只保留至该日的数据
        df_until = hist_df[hist_df["date"] <= dt]
        scored = score_and_rank_custom(df_until, weight_ts=args.weight)

        # 筛出当天信号
        report = scored[scored["date"] == dt].copy()
        if report.empty:
            print(f"⚠️ {dt.date()} 无打分结果，跳过")
            continue

        report = report[[
            "symbol",
            "trend_strength",
            "trend_consistency",
            "composite_score",
            "momentum_rank",
            "persistence_rank"
        ]]
        report = report.sort_values("composite_score", ascending=False).reset_index(drop=True)

        out_path = os.path.join(
            args.output_dir,
            f"daily_report_{dt.strftime('%Y%m%d')}.csv"
        )
        report.to_csv(out_path, index=False)
        print(f"✅ 已保存: {out_path}")

    print("\n🎉 全部完成！")

if __name__ == "__main__":
    main()
