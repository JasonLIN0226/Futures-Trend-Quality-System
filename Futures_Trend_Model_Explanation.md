# Futures Trend Evaluation Model – Technical Explanation

This document introduces the design logic, mathematical basis, and justification for the futures trend quality evaluation system, as well as the backtest framework and trading strategy used in this project.

---

## Model Components

The model is built around **three core components**:

1. **Trend Strength Indicator**
2. **Trend Consistency Indicator**
3. **Composite Scoring System**

Each is explained in detail below.

---

## 1. Trend Strength Indicator (趋势强度指标)

### Goal:
To measure how strongly the price is trending in a given direction.

### Components:
We combine **momentum** and **regression slope** over multiple windows.

### Mathematical Definitions:

- **Momentum** over window \( m \):

![Momentum 公式](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%5Ctext%7Bmomentum%7D_t%5E%7B(m)%7D%20%3D%20%5Cfrac%7BP_t%20-%20P_%7Bt-m%7D%7D%7BP_%7Bt-m%7D%7D%20%3D%20%5Cfrac%7BP_t%7D%7BP_%7Bt-m%7D%7D%20-%201)


- **Volatility-Adjusted Momentum**:

![标准化动量公式](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%5Cwidetilde%7B%5Ctext%7Bmomentum%7D%7D_t%5E%7B(m)%7D%20%3D%20%5Cfrac%7B%5Ctext%7Bmomentum%7D_t%5E%7B(m)%7D%7D%7B%5Ctext%7Bvolatility%7D_t%5E%7B(m)%7D%7D)

where volatility is standard deviation of returns in the same window.

- **Slope** from linear regression:

Given    

![表达式](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7DP_%7Bt-m%2B1%7D%2C%20%5Cdots%2C%20P_t)

, fit:

![线性回归模型](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%20P_i%20%3D%20%5Cbeta%20i%20%2B%20%5Calpha%20%2B%20%5Cvarepsilon_i%2C%5Cquad%20i%20%3D%200%2C1%2C...%2Cm-1)

Then: 

![斜率公式](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%5Ctext%7Bslope%7D%20%3D%20%5Cbeta)


We calculate these over multiple time scales (e.g. 5, 10, 20 days) and normalize.

---

## 2. Trend Consistency Indicator (趋势流畅性指标)

### Goal:
To assess how well price follows a consistent, smooth trend (i.e., robustness and noise resistance).

### Definition:
We use **coefficient of determination (R²)** from a linear fit as a proxy for trend “smoothness”.

### Formula:
For a given rolling window \( m \):

- Fit     :

![线性回归定义](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%20y_i%20%3D%20%5Cbeta%20x_i%20%2B%20%5Calpha%20%5Cquad%5Ctext%7Bwhere%7D%5Cquad%20x_i%20%3D%20i)

- Compute   :

![R²公式](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%20R%5E2%20%3D%201%20-%20%5Cfrac%7B%5Csum%20%28y_i%20-%20%5Chat%7By%7D_i%29%5E2%7D%7B%5Csum%20%28y_i%20-%20%5Cbar%7By%7D%29%5E2%7D)


Where:
- ![预测值](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%5Chat%7By%7D_i)
 is the fitted value
- ![平均值](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%5Cbar%7By%7D)
 is the average of ![观测值](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7Dy_i)


This value lies in \( [0, 1] \), higher means more consistent.

---

## 3. Composite Scoring Model (综合评分模型)

### Goal:
Fuse strength and consistency into a single interpretable score.

### Method:
At each scale, calculate:

![Score 公式](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%5Ctext%7Bscore%5C_t%7D%20%3D%20%5Cfrac%7B%5Ctext%7Bnorm%5C_momentum%5C_rank%7D%20%2B%20%5Ctext%7Bslope%5C_rank%7D%20%2B%20%5Ctext%7BR%C2%B2%5C_rank%7D%7D%7B3%7D)

Then aggregate multi-scale scores equally:

![Composite Score 公式](https://latex.codecogs.com/svg.image?%5Cdpi%7B120%7D%5Ctext%7Bcomposite%5C_score%7D_t%20%3D%20%5Cfrac%7B1%7D%7B3%7D%20%5Csum_%7Bi%3D1%7D%5E%7B3%7D%20%5Ctext%7Bscore%7D%5E%7B(i)%7D_t)


Ranks are percentile ranks (0 to 1) among all instruments on the same day.

## Trend Score Weight Optimization

To optimize the trading strategy, we adjust the relative weight between **Trend Strength** and **Trend Consistency** using a grid search approach. The goal is to find the best balance that maximizes overall performance (Sharpe ratio and annual return).

### Step 1: Define Evaluation Metrics

For each candidate weight, we evaluate the strategy by:

- Calculating **composite score**:
  ![Composite Score 公式](https://latex.codecogs.com/svg.image?%5Ctext%7Bcomposite%5C_score%7D%20%3D%20%5Ctext%7Bweight%5C_ts%7D%20%5Ccdot%20%5Ctext%7Btrend%5C_strength%7D%20%2B%20%281%20-%20%5Ctext%7Bweight%5C_ts%7D%29%20%5Ccdot%20%5Ctext%7Btrend%5C_consistency%7D)

- Selecting top `N` symbols by score on each day
- Simulating daily returns and computing:
  - Total return
  - Annualized return
  - Sharpe ratio
  - Maximum drawdown

This is implemented in the function `evaluate_strategy()`.

---

### Step 2: Grid Search over Weights

We search for the best `weight_ts` using:

```python
weight_candidates = np.arange(0.4, 0.81, 0.01)
```

For each value, we run `evaluate_strategy()` and log the resulting performance metrics. This is done in the function `parameter_tuning()`.

---

### Step 3: Select the Best Weight

We compute a **composite objective** to evaluate each weight:

\[
\text{composite\_metric} = \text{Annual Return} + \text{Sharpe Ratio}
\]

Then we select the weight with the **highest composite_metric**:

```python
best_row = tuning_df.loc[tuning_df["composite_metric"].idxmax()]
```

This best weight is used for scoring and signal generation in later parts of the strategy.

---

---


## Mock Trading Strategy with Portfolio Management

This system simulates real-world trading behavior by executing buy/sell orders based on trend scores and managing a virtual portfolio. It combines **position sizing**, **dynamic exits**, and **account equity simulation** to evaluate the quality of trading signals.

---

### Components

#### 1. **Signal Generation**

Top `N` futures contracts are selected each day using a **composite score** based on:
- `trend_strength` (momentum & slope-based)
- `trend_consistency` (robust R²-based)

Each selected contract receives a **position size inversely proportional to its volatility**:
\[
\text{position\_size} = \frac{1}{\text{volatility}_{5d}}
\]

Implemented in: `generate_volatility_adjusted_long_signals()`

---

#### 2. **Dynamic Exit Rule**

Each long position is monitored daily. Exit is triggered if:

- Composite score falls below a `min_exit_threshold`, or
- A **trailing stop** condition is met:
  \[
  \frac{\text{entry\_score} - \text{current\_score}}{\text{entry\_score}} > \text{stop\_threshold}
  \]

Implemented in: `dynamic_exit_rule()` and `run_dynamic_volatility_strategy()`

---

#### 3. **Order Execution and Portfolio Simulation**

- Trades are executed through a mock `PortfolioManager` class.
- For each entry/exit:
  - Determine quantity to buy/sell based on `order_fraction` of capital.
  - Cash balance and holdings are updated.
  - All trades are logged (`trade_log`).

Implemented in: `simulate_trading()`

---

#### 4. **Equity Curve and Performance Metrics**

Simulate **daily equity** by:
- Tracking which contracts are held on each day
- Calculating average return from active positions
- Compounding daily returns to form an **equity curve**

Metrics calculated:
- **Total return**
- **Annualized return**
- **Sharpe ratio**
- **Maximum drawdown**

Visualization is done with matplotlib, showing the portfolio value over time and annotated metrics.

Implemented in: `simulate_daily_equity()` and `run_best_weight_strategy_with_portfolio_and_plot()`

---

