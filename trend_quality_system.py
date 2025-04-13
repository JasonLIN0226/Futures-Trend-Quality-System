#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Futures Trend Quality Evaluation System — Parameter Tuning and Sensitivity Analysis Example

Dependencies: pandas, numpy, scipy, python-dateutil, pandas_ta, joblib, sklearn, matplotlib
"""

import os
import pandas as pd
import numpy as np
from dateutil import parser
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------------------
# Data Loading and Cleaning Functions
# -----------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    df.rename(columns={'product': 'symbol'}, inplace=True)
    df['price'] = df['settle_price']
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['symbol', 'date'])
    df['price'] = df['price'].ffill().bfill()
    return df

# -----------------------------------------
# Indicator Calculation Functions: Including momentum, volatility, normalized momentum,
# rolling slope, conventional R² and robust R² indicators.
# -----------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parallel computation of multi-scale indicators:
      - Short term (5,10), medium term (10,20), long term (20,60)
      - For each scale, compute: momentum, volatility, normalized momentum,
        rolling slope, conventional R² and robust R².
      - Clip outlier values at the 1%-99% quantile.
    Assumes the input df is sorted by symbol/date.
    """
    def rolling_slope(x: pd.Series) -> float:
        y = x.values
        if len(y) < 2 or np.isnan(y).any():
            return np.nan
        t = np.arange(len(y))
        return np.polyfit(t, y, 1)[0]

    def rolling_r2(x: pd.Series) -> float:
        y = x.values
        if len(y) < 2 or np.isnan(y).any():
            return np.nan
        t = np.arange(len(y))
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = intercept + slope * t
        ss_res = ((y - y_pred)**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        return 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

    def robust_r2(x: pd.Series) -> float:
        """
        Robust R² calculation using median instead of mean to reduce noise influence:
        ĤR² = 1 - Σ (y - y_pred)² / Σ (y - median(y))²
        """
        y = x.values
        if len(y) < 2 or np.isnan(y).any():
            return np.nan
        t = np.arange(len(y))
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = intercept + slope * t
        ss_res = ((y - y_pred)**2).sum()
        ss_tot = ((y - np.median(y))**2).sum()
        return 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

    def proc(sym_df: pd.DataFrame) -> pd.DataFrame:
        sym_df = sym_df.copy()
        sym_df['ret_1'] = sym_df['price'].pct_change(1)
        scales = [(5, 10), (10, 20), (20, 60)]
        for m, r in scales:
            # Momentum and volatility
            sym_df[f'mom_{m}'] = sym_df['price'].pct_change(m)
            sym_df[f'vol_{m}'] = sym_df['ret_1'].rolling(m, min_periods=5).std()
            sym_df[f'norm_mom_{m}'] = sym_df[f'mom_{m}'] / sym_df[f'vol_{m}']
            # Rolling slope
            sym_df[f'slope_{m}'] = sym_df['price'].rolling(m, min_periods=5)\
                                               .apply(rolling_slope, raw=False)
            # Conventional R²
            sym_df[f'r2_{r}'] = sym_df['price'].rolling(r, min_periods=5)\
                                         .apply(rolling_r2, raw=False)
            # Robust R² (for trend consistency indicator)
            sym_df[f'robust_r2_{r}'] = sym_df['price'].rolling(r, min_periods=5)\
                                                .apply(robust_r2, raw=False)
            # Clip outliers for each indicator using 1%-99% quantiles
            for col in [f'mom_{m}', f'norm_mom_{m}', f'slope_{m}', f'r2_{r}', f'robust_r2_{r}']:
                lo, hi = sym_df[col].quantile(0.01), sym_df[col].quantile(0.99)
                sym_df[col] = sym_df[col].clip(lo, hi)
        return sym_df.drop(columns=['ret_1'])

    parts = Parallel(n_jobs=-1)(
        delayed(proc)(grp) for _, grp in df.groupby('symbol')
    )
    return pd.concat(parts, axis=0).reset_index(drop=True)

# -----------------------------------------
# Scoring and Ranking Function: Supports custom trend strength weight parameter.
# -----------------------------------------

def score_and_rank_custom(df: pd.DataFrame, weight_ts: float = 0.5) -> pd.DataFrame:
    """
    Computes indicator scores:
      - Trend Strength: Combination of normalized momentum and rolling slope,
        averaged using intraday percentile ranks.
      - Trend Consistency: Based on the intraday percentile rank average of the robust R².
      - Composite Score: composite_score = weight_ts * trend_strength + (1-weight_ts) * trend_consistency
      - Outputs the intraday ranks for each indicator.
    """
    # Trend Strength calculation
    ts_components = (
        df.groupby('date')['norm_mom_5'].rank(pct=True) +
        df.groupby('date')['slope_5'].rank(pct=True) +
        df.groupby('date')['norm_mom_10'].rank(pct=True) +
        df.groupby('date')['slope_10'].rank(pct=True) +
        df.groupby('date')['norm_mom_20'].rank(pct=True) +
        df.groupby('date')['slope_20'].rank(pct=True)
    ) / 6.0
    df['trend_strength'] = ts_components

    # Trend Consistency: Using robust R²
    tc_components = (
        df.groupby('date')['robust_r2_10'].rank(pct=True) +
        df.groupby('date')['robust_r2_20'].rank(pct=True) +
        df.groupby('date')['robust_r2_60'].rank(pct=True)
    ) / 3.0
    df['trend_consistency'] = tc_components

    # Composite Score: Customizable weight for trend strength vs. trend consistency
    df['composite_score'] = weight_ts * df['trend_strength'] + (1 - weight_ts) * df['trend_consistency']

    # Intraday ranking
    df['momentum_rank'] = df.groupby('date')['trend_strength']\
                             .rank(ascending=False, method='first')\
                             .fillna(-1).astype(int)
    df['persistence_rank'] = df.groupby('date')['trend_consistency']\
                                .rank(ascending=False, method='first')\
                                .fillna(-1).astype(int)
    return df

# -----------------------------------------
# Backtest Function (Maintaining Original Logic)
# -----------------------------------------

def backtest(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Simple equal-weight long-only backtest:
      - Each day, select top_n futures based on composite_score.
      - Hold for one day and compute next-day intraday return (next_open / open - 1).
    """
    df = df.sort_values(['symbol', 'date'])
    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    df['ret'] = df['next_open'] / df['open'] - 1

    records = []
    for day, group in df.groupby('date'):
        top_syms = group.nlargest(top_n, 'composite_score')['symbol']
        next_day = day + pd.Timedelta(days=1)
        mask = (df['date'] == next_day) & (df['symbol'].isin(top_syms))
        rets = df.loc[mask, 'ret']
        records.append({
            'date': day,
            'daily_ret': rets.mean() if not rets.empty else 0.0
        })

    bt = pd.DataFrame(records).sort_values('date')
    bt['cum_ret'] = (1 + bt['daily_ret']).cumprod() - 1
    return bt

# -----------------------------------------
# Parameter Tuning and Performance Evaluation
# -----------------------------------------

def evaluate_strategy(df: pd.DataFrame, weight_ts: float, top_n: int = 5) -> dict:
    """
    Computes composite scores and backtesting performance for a given trend strength weight.
    Returns performance metrics:
      - annual_ret: Annualized Return
      - max_drawdown: Maximum Drawdown
      - sharpe: Annualized Sharpe Ratio
    """
    df_scored = score_and_rank_custom(df.copy(), weight_ts)
    bt = backtest(df_scored, top_n=top_n)
    bt_clean = bt.dropna().copy()
    n_days = len(bt_clean)
    if n_days == 0 or bt_clean['daily_ret'].std() == 0:
        return None
    total_ret = bt_clean['cum_ret'].iloc[-1]
    annual_ret = (1 + total_ret)**(252 / n_days) - 1
    drawdown = bt_clean['cum_ret'] - bt_clean['cum_ret'].cummax()
    max_drawdown = drawdown.min()
    sharpe = bt_clean['daily_ret'].mean() / bt_clean['daily_ret'].std() * np.sqrt(252)
    return {
        "weight_ts": weight_ts,
        "annual_ret": annual_ret,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe
    }

def parameter_tuning(df: pd.DataFrame, weight_candidates: list, top_n: int = 5) -> pd.DataFrame:
    """
    Evaluates strategy performance across different candidate values for trend strength weight.
    Returns a DataFrame summarizing the performance metrics.
    """
    results = []
    for weight in weight_candidates:
        metrics = evaluate_strategy(df, weight, top_n=top_n)
        if metrics is not None:
            results.append(metrics)
    return pd.DataFrame(results)

# -----------------------------------------
# Main Function: Load Data, Compute Indicators, and Run Parameter Tuning
# -----------------------------------------

def run_parameter_tuning():
    # Load and preprocess data
    df = load_data('data/all_futures_20250413.csv')
    df = clean_data(df)
    df = compute_indicators(df)
    
    # Candidate values for trend strength weight parameter
    weight_candidates = np.arange(0.4, 0.81, 0.01)
    
    # Execute parameter tuning
    tuning_results = parameter_tuning(df, weight_candidates, top_n=5)
    
    print("Parameter Tuning Results:")
    print(tuning_results)
    
    # Plot Annualized Return vs. Trend Strength Weight
    plt.figure()
    plt.plot(tuning_results['weight_ts'], tuning_results['annual_ret'], marker='o')
    plt.xlabel('Trend Strength Weight (weight_ts)')
    plt.ylabel('Annualized Return')
    plt.title('Parameter Tuning: Annualized Return vs. Trend Strength Weight')
    plt.grid(True)
    plt.show()
    
    # Plot Sharpe Ratio vs. Trend Strength Weight
    plt.figure()
    plt.plot(tuning_results['weight_ts'], tuning_results['sharpe'], marker='o', linestyle='--')
    plt.xlabel('Trend Strength Weight (weight_ts)')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.title('Parameter Tuning: Sharpe Ratio vs. Trend Strength Weight')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_parameter_tuning()
