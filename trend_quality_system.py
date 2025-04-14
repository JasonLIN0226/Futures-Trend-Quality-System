#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

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
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    def robust_r2(x: pd.Series) -> float:
        y = x.values
        if len(y) < 2 or np.isnan(y).any():
            return np.nan
        t = np.arange(len(y))
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = intercept + slope * t
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - np.median(y)) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    def proc(sym_df: pd.DataFrame) -> pd.DataFrame:
        sym_df = sym_df.copy()
        sym_df['ret_1'] = sym_df['price'].pct_change(1)
        scales = [(5, 10), (10, 20), (20, 60)]
        for m, r in scales:
            sym_df[f'mom_{m}'] = sym_df['price'].pct_change(m)
            sym_df[f'vol_{m}'] = sym_df['ret_1'].rolling(m, min_periods=5).std()
            sym_df[f'norm_mom_{m}'] = sym_df[f'mom_{m}'] / sym_df[f'vol_{m}']
            sym_df[f'slope_{m}'] = sym_df['price'].rolling(m, min_periods=5).apply(rolling_slope, raw=False)
            sym_df[f'r2_{r}'] = sym_df['price'].rolling(r, min_periods=5).apply(rolling_r2, raw=False)
            sym_df[f'robust_r2_{r}'] = sym_df['price'].rolling(r, min_periods=5).apply(robust_r2, raw=False)
            for col in [f'mom_{m}', f'norm_mom_{m}', f'slope_{m}', f'r2_{r}', f'robust_r2_{r}']:
                lo, hi = sym_df[col].quantile(0.01), sym_df[col].quantile(0.99)
                sym_df[col] = sym_df[col].clip(lo, hi)
        return sym_df.drop(columns=['ret_1'])

    parts = Parallel(n_jobs=-1)(delayed(proc)(grp) for _, grp in df.groupby('symbol'))
    return pd.concat(parts, axis=0).reset_index(drop=True)

def score_and_rank_custom(df: pd.DataFrame, weight_ts: float = 0.5) -> pd.DataFrame:
    ts_components = (
        df.groupby('date')['norm_mom_5'].rank(pct=True) +
        df.groupby('date')['slope_5'].rank(pct=True) +
        df.groupby('date')['norm_mom_10'].rank(pct=True) +
        df.groupby('date')['slope_10'].rank(pct=True) +
        df.groupby('date')['norm_mom_20'].rank(pct=True) +
        df.groupby('date')['slope_20'].rank(pct=True)
    ) / 6.0
    df['trend_strength'] = ts_components

    tc_components = (
        df.groupby('date')['robust_r2_10'].rank(pct=True) +
        df.groupby('date')['robust_r2_20'].rank(pct=True) +
        df.groupby('date')['robust_r2_60'].rank(pct=True)
    ) / 3.0
    df['trend_consistency'] = tc_components

    df['composite_score'] = weight_ts * df['trend_strength'] + (1 - weight_ts) * df['trend_consistency']
    df['momentum_rank'] = df.groupby('date')['trend_strength'].rank(ascending=False, method='first').fillna(-1).astype(int)
    df['persistence_rank'] = df.groupby('date')['trend_consistency'].rank(ascending=False, method='first').fillna(-1).astype(int)
    return df


def backtest(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
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

def evaluate_strategy(df: pd.DataFrame, weight_ts: float, top_n: int = 5) -> dict:
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
    return {"weight_ts": weight_ts, "annual_ret": annual_ret, "max_drawdown": max_drawdown, "sharpe": sharpe}

def parameter_tuning(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    weight_candidates = np.arange(0.4, 0.81, 0.01)
    results = []
    for weight in weight_candidates:
        metrics = evaluate_strategy(df, weight, top_n=top_n)
        if metrics is not None:
            results.append(metrics)
    return pd.DataFrame(results)

def get_best_weight(tuning_df: pd.DataFrame) -> float:
    tuning_df["composite_metric"] = tuning_df["annual_ret"] + tuning_df["sharpe"]
    best_row = tuning_df.loc[tuning_df["composite_metric"].idxmax()]
    print("最佳调参结果：")
    print(best_row)
    return best_row["weight_ts"]


def generate_volatility_adjusted_long_signals(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:

    signals = []
    for day, group in df.groupby('date'):
        selected = group.nlargest(top_n, 'composite_score')
        for _, row in selected.iterrows():
            vol = row.get('vol_5', 1e-6)
            position_size = 1 / vol  
            signals.append({
                "date": day,
                "symbol": row["symbol"],
                "composite_score": row["composite_score"],
                "volatility": vol,
                "position_size": position_size,
                "signal": "long",
                "entry_score": row["composite_score"]
            })
    return pd.DataFrame(signals)

def dynamic_exit_rule(current_score, entry_score, trailing_stop=0.2, min_exit_threshold=0.4):

    if current_score < min_exit_threshold:
        return True
    if (entry_score - current_score) / entry_score > trailing_stop:
        return True
    return False

def run_dynamic_volatility_strategy(df_scored: pd.DataFrame, top_n: int = 5,
                                    trailing_stop: float = 0.2, min_exit_threshold: float = 0.4):

    entry_signals = generate_volatility_adjusted_long_signals(df_scored, top_n=top_n)
    trades = []
    for _, signal in entry_signals.iterrows():
        entry_date = signal["date"]
        symbol = signal["symbol"]
        entry_score = signal["entry_score"]
        position_size = signal["position_size"]
        
        # 获取该品种入场后所有数据
        trade_data = df_scored[(df_scored['symbol'] == symbol) & (df_scored['date'] > entry_date)].copy()
        if trade_data.empty:
            continue
        
        trade_exit_date = None
        for _, row in trade_data.iterrows():
            current_score = row["composite_score"]
            if dynamic_exit_rule(current_score, entry_score, trailing_stop, min_exit_threshold):
                trade_exit_date = row["date"]
                break
        if trade_exit_date is None:
            trade_exit_date = trade_data['date'].max()
        
        trades.append({
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": trade_exit_date,
            "entry_score": entry_score,
            "exit_score": trade_data.loc[trade_data['date'] == trade_exit_date, "composite_score"].values[0],
            "position_size": position_size
        })
    return pd.DataFrame(trades)


class PortfolioManager:
    def __init__(self, initial_capital, contract_multiplier=1):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {} 
        self.contract_multiplier = contract_multiplier
        self.trade_log = [] 

    def buy(self, symbol, price, order_value, date):
        quantity = int(order_value / (price * self.contract_multiplier))
        if quantity <= 0:
            print(f"{date}: {symbol} 买入数量计算为 0 手。")
            return False

        cost = quantity * price * self.contract_multiplier
        if self.cash >= cost:
            self.cash -= cost
            if symbol in self.positions:
                prev_qty, prev_cost = self.positions[symbol]
                new_qty = prev_qty + quantity
                new_cost = prev_cost + cost
                self.positions[symbol] = (new_qty, new_cost)
            else:
                self.positions[symbol] = (quantity, cost)
            self.trade_log.append({
                "date": date,
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "price": price,
                "cost": cost
            })
            print(f"{date}: 买入 {symbol} {quantity} 手，单价 {price}，花费 {cost:.2f}。")
            return True
        else:
            print(f"{date}: 资金不足，当前现金 {self.cash:.2f}，需要 {cost:.2f} 买入 {symbol}。")
            return False

    def sell(self, symbol, price, quantity, date):
        if symbol not in self.positions:
            print(f"{date}: 无持仓 {symbol}，不能卖出。")
            return False

        held_qty, held_cost = self.positions[symbol]
        if held_qty < quantity:
            print(f"{date}: {symbol} 持仓不足，当前持有 {held_qty} 手，要求卖出 {quantity} 手。")
            return False

        proceeds = quantity * price * self.contract_multiplier
        self.cash += proceeds
        new_qty = held_qty - quantity
        if new_qty == 0:
            del self.positions[symbol]
        else:
            new_cost = held_cost * (new_qty / held_qty)
            self.positions[symbol] = (new_qty, new_cost)
        self.trade_log.append({
            "date": date,
            "symbol": symbol,
            "action": "sell",
            "quantity": quantity,
            "price": price,
            "proceeds": proceeds
        })
        print(f"{date}: 卖出 {symbol} {quantity} 手，单价 {price}，回收 {proceeds:.2f}。")
        return True

    def get_portfolio_value(self, current_prices):
        total_value = self.cash
        for symbol, (qty, _) in self.positions.items():
            price = current_prices.get(symbol, 0)
            total_value += qty * price * self.contract_multiplier
        return total_value


def simulate_trading(trades_df, df_scored, initial_capital=1_000_000, contract_multiplier=10, order_fraction=0.1):

    portfolio = PortfolioManager(initial_capital, contract_multiplier=contract_multiplier)
    
    for idx, trade in trades_df.iterrows():
        entry_date = trade["entry_date"]
        exit_date = trade["exit_date"]
        symbol = trade["symbol"]

        try:
            entry_price = df_scored[(df_scored['date'] == entry_date) & (df_scored['symbol'] == symbol)].iloc[0]["open"]
        except Exception as e:
            print(f"{entry_date} {symbol} 入场价格获取失败：{e}")
            continue
        
        order_value = initial_capital * order_fraction
        success = portfolio.buy(symbol, entry_price, order_value, entry_date)
        if not success:
            continue
        
        try:
            exit_price = df_scored[(df_scored['date'] == exit_date) & (df_scored['symbol'] == symbol)].iloc[0]["open"]
        except Exception as e:
            print(f"{exit_date} {symbol} 出场价格获取失败：{e}")
            continue
        
        held_qty, _ = portfolio.positions.get(symbol, (0, 0))
        if held_qty > 0:
            portfolio.sell(symbol, exit_price, held_qty, exit_date)
    
    print(f"最终现金：{portfolio.cash:.2f}")
    print("持仓情况：", portfolio.positions)
    return portfolio


def simulate_daily_equity(trades_df: pd.DataFrame, df_scored: pd.DataFrame, initial_capital: float = 1_000_000) -> pd.DataFrame:

    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    df_scored['date'] = pd.to_datetime(df_scored['date'])
    
    start_date = (trades_df['entry_date'].min() + pd.Timedelta(days=1)).normalize()
    end_date = trades_df['exit_date'].max().normalize()
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  
    

    df_scored_indexed = df_scored.set_index(['date', 'symbol'])
    
    equity = initial_capital
    equity_curve = []
    
    for current_date in date_range:
        active_trades = trades_df[(trades_df['entry_date'] < current_date) & (trades_df['exit_date'] >= current_date)]
        daily_returns = []
        for _, trade in active_trades.iterrows():
            symbol = trade['symbol']
            try:
                open_today = df_scored_indexed.loc[(current_date, symbol)]['open']
                prev_date = current_date - pd.Timedelta(days=1)
                while (prev_date, symbol) not in df_scored_indexed.index:
                    prev_date = prev_date - pd.Timedelta(days=1)
                open_prev = df_scored_indexed.loc[(prev_date, symbol)]['open']
                if open_prev != 0:
                    r = open_today / open_prev - 1
                    daily_returns.append(r)
            except Exception as e:
                continue
        if daily_returns:
            portfolio_daily_return = np.mean(daily_returns)
        else:
            portfolio_daily_return = 0
        equity = equity * (1 + portfolio_daily_return)
        equity_curve.append({'date': current_date, 'equity': equity, 'daily_return': portfolio_daily_return})
    return pd.DataFrame(equity_curve)


def run_best_weight_strategy_with_portfolio_and_plot():

    df = load_data('data/all_futures_20250413.csv')
    df = clean_data(df)
    df = compute_indicators(df)
    
    tuning_results = parameter_tuning(df, top_n=5)
    print("参数调优结果：")
    print(tuning_results)
    best_weight = get_best_weight(tuning_results)
    print(f"最佳趋势强度权重：{best_weight:.2f}")
    
    df_scored = score_and_rank_custom(df, weight_ts=best_weight)
    
    trades_df = run_dynamic_volatility_strategy(df_scored, top_n=5, trailing_stop=0.2, min_exit_threshold=0.4)
    print("部分交易记录：")
    print(trades_df.head(10))
    
    portfolio = simulate_trading(trades_df, df_scored, initial_capital=1_000_000, contract_multiplier=10, order_fraction=0.1)
    trade_log_df = pd.DataFrame(portfolio.trade_log)
    print("交易日志：")
    print(trade_log_df.head(10))
    
    equity_df = simulate_daily_equity(trades_df, df_scored, initial_capital=1_000_000)
    
    if not equity_df.empty:
        final_equity = equity_df['equity'].iloc[-1]
        total_return = final_equity / 1_000_000 - 1
        num_days = len(equity_df)
        annual_return = (1 + total_return) ** (252 / num_days) - 1
        daily_ret = equity_df['daily_return']
        sharpe_ratio = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() != 0 else np.nan
        cummax = equity_df['equity'].cummax()
        drawdown_pct = equity_df['equity'] / cummax - 1
        max_drawdown = drawdown_pct.min()
    else:
        total_return = annual_return = sharpe_ratio = max_drawdown = np.nan
    
    plt.figure(figsize=(12, 7))
    plt.plot(equity_df['date'], equity_df['equity'], marker='o', markersize=2, label="Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.title("Portfolio Equity Curve (Initial Capital = $1,000,000)")
    plt.grid(True)
    
    metric_text = (f"Total Return: {total_return:.2%}\n"
                   f"Annualized Return: {annual_return:.2%}\n"
                   f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                   f"Max Drawdown: {max_drawdown:.2%}")
    plt.text(0.02, 0.65, metric_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
    
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_best_weight_strategy_with_portfolio_and_plot()
