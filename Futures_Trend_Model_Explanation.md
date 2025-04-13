# Futures Trend Evaluation Model – Technical Explanation

This document introduces the design logic, mathematical basis, and justification for the futures trend quality evaluation system, as well as the backtest framework and trading strategy used in this project.

---

## 📐 Model Components

The model is built around **three core components**:

1. **Trend Strength Indicator**
2. **Trend Consistency Indicator**
3. **Composite Scoring System**

Each is explained in detail below.

---

## 📊 1. Trend Strength Indicator (趋势强度指标)

### ✅ Goal:
To measure how strongly the price is trending in a given direction.

### ✅ Components:
We combine **momentum** and **regression slope** over multiple windows.

### ✅ Mathematical Definitions:

- **Momentum** over window \( m \):
\[
\text{momentum}_t^{(m)} = \frac{P_t - P_{t-m}}{P_{t-m}} = \frac{P_t}{P_{t-m}} - 1
\]

- **Volatility-Adjusted Momentum**:
\[
\widetilde{\text{momentum}}_t^{(m)} = \frac{\text{momentum}_t^{(m)}}{\text{volatility}_t^{(m)}}
\]
where volatility is standard deviation of returns in the same window.

- **Slope** from linear regression:
Given \( P_{t-m+1}, ..., P_t \), fit:
\[
P_i = \beta i + \alpha + \varepsilon_i,\quad i = 0,1,...,m-1
\]
Then: \( \text{slope} = \beta \)

We calculate these over multiple time scales (e.g. 5, 10, 20 days) and normalize.

---

## 🔄 2. Trend Consistency Indicator (趋势流畅性指标)

### ✅ Goal:
To assess how well price follows a consistent, smooth trend (i.e., robustness and noise resistance).

### ✅ Definition:
We use **coefficient of determination (R²)** from a linear fit as a proxy for trend “smoothness”.

### ✅ Formula:
For a given rolling window \( m \):

- Fit \( y_i = \beta x_i + \alpha \) where \( x_i = i \)
- Compute:
\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

Where:
- \( \hat{y}_i \) is the fitted value
- \( \bar{y} \) is the average of \( y_i \)

This value lies in \( [0, 1] \), higher means more consistent.

---

## 🧠 3. Composite Scoring Model (综合评分模型)

### ✅ Goal:
Fuse strength and consistency into a single interpretable score.

### ✅ Method:
At each scale, calculate:
\[
\text{score}_t = \frac{\text{norm\_momentum\_rank} + \text{slope\_rank} + \text{R²\_rank}}{3}
\]

Then aggregate multi-scale scores equally:
\[
\text{composite\_score}_t = \frac{1}{3} \sum_{i=1}^{3} \text{score}^{(i)}_t
\]

Ranks are percentile ranks (0 to 1) among all instruments on the same day.

### ✅ Weighting Justification:
Weights are currently uniform, but can be optimized through:
- Sharpe ratio maximization
- Information ratio on out-of-sample data
- Domain-specific tuning (e.g. emphasize trend or robustness)

---

## 🧪 Backtesting Framework

The backtest is implemented with the following steps:

1. Load and clean historical futures data (from CSV files).
2. Compute trend indicators and scores for each symbol per day.
3. Rank instruments by their composite score on each day.
4. Select top N instruments and simulate buying them the next day using open price.
5. Evaluate strategy performance using:
   - Total return
   - Annualized return
   - Maximum drawdown
   - Annualized Sharpe ratio

The system uses a fixed capital base (e.g., 1 million CNY), and position sizing is based on equal capital allocation across selected instruments.

---

## 📈 Trading Strategy Logic

In real-time mode, the strategy:

1. Fetches latest bar for each futures contract daily.
2. Merges new data with historical records.
3. Recalculates all trend and score indicators.
4. Ranks all available instruments by composite score.
5. Outputs a full daily signal report.

This can be connected to a position management module to:
- Maintain open positions
- Execute entry/exit logic
- Manage portfolio value over time

Currently, only the signal generation logic is active.

---

## 📎 Summary

| Module         | Metric                | Math Basis                     |
|----------------|-----------------------|--------------------------------|
| Strength       | Momentum, Slope       | Pct-change, Linear Regression |
| Consistency    | Smoothness (R²)       | Regression Goodness-of-Fit    |
| Composite      | Percentile Averaging  | Weighted Multi-scale Fusion   |

This framework allows for robust, interpretable, and extendable trend signal generation.
