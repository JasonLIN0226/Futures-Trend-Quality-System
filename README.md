# futures_trend_qc

**compute_indicators 函数说明**

**1. 动量 (Momentum)**
- **公式**  
  \[
    \mathrm{mom}_t^{(m)} = \frac{P_t - P_{t-m}}{P_{t-m}}
  \]  
- **实现**  
  ```python
  df[f'mom_{m}'] = df['price'].pct_change(m)
  ```
- **设计思路 & 好处**  
  捕捉价格相对于过去 \(m\) 日的涨跌幅，反映趋势方向和初步强度。

---

**2. 波动率 (Volatility)**
- **公式**  
  \[
    \mathrm{vol}_t^{(m)} = \sigma\bigl(r_{t-m+1}, \dots, r_t\bigr)
  \]  
- **实现**  
  ```python
  df[f'vol_{m}'] = df['ret_1'].rolling(window=m, min_periods=5).std()
  ```
- **设计思路 & 好处**  
  衡量历史收益标准差，为后续归一化动量提供基准。

---

**3. 波动率归一化动量 (Normalized Momentum)**
- **公式**  
![标准化动量公式](https://latex.codecogs.com/svg.image?\dpi{120}\widetilde{\mathrm{mom}}_t^{(m)}%20=%20\frac{\mathrm{mom}_t^{(m)}}{\mathrm{vol}_t^{(m)}})
- **实现**  
  ```python
  df[f'norm_mom_{m}'] = df[f'mom_{m}'] / df[f'vol_{m}']
  ```
- **设计思路 & 好处**  
  消除不同品种或不同周期的波动率差异，提升可比性。

---

**4. 回归斜率 (Slope)**
- **公式**  
  \[
    P_i = \beta\,i + \alpha + \varepsilon_i,\quad i=0,\dots,m-1
  \]  
- **实现**  
  ```python
  def rolling_slope(x):
      t = np.arange(len(x))
      return np.polyfit(t, x.values, 1)[0]
  df[f'slope_{m}'] = df['price'].rolling(window=m, min_periods=5)\
                                 .apply(rolling_slope, raw=False)
  ```
- **设计思路 & 好处**  
  反映价格变化的平均速率，对噪声有平滑效果。

---

**5. 拟合优度 R² (R‑squared)**
- **公式**  
  \[
    R^2 = 1 - \frac{\sum (P_i - \hat P_i)^2}{\sum (P_i - \bar P)^2}
  \]
- **实现**  
  ```python
  def rolling_r2(x):
      y = x.values
      t = np.arange(len(y))
      β, α = np.polyfit(t, y, 1)
      y_pred = α + β*t
      ss_res = ((y - y_pred)**2).sum()
      ss_tot = ((y - y.mean())**2).sum()
      return 1 - ss_res/ss_tot
  df[f'r2_{r}'] = df['price'].rolling(window=r, min_periods=5)\
                             .apply(rolling_r2, raw=False)
  ```
- **设计思路 & 好处**  
  衡量价格序列与理想趋势线的拟合程度，高 R² 表示更平滑的趋势。

---

**6. 异常值过滤 (Outlier Clipping)**
- **方法**  
  ```python
  lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
  df[col] = df[col].clip(lo, hi)
  ```
- **设计思路 & 好处**  
  去除离群点，防止极端值在后续排名或回测中造成过度影响。

---

**7. 多尺度融合 (Multi‑scale)**
- **配置**  
  - 短期：动量 5 日 + R² 10 日  
  - 中期：动量 10 日 + R² 20 日  
  - 长期：动量 20 日 + R² 60 日
- **设计思路 & 好处**  
  不同尺度兼顾灵敏与稳健，多尺度等权融合后信号更可靠。

---

**8. 并行计算 (Parallelization)**
- **实现**  
  ```python
  from joblib import Parallel, delayed
  processed = Parallel(n_jobs=-1)(
      delayed(proc_symbol)(grp) for _, grp in df.groupby('symbol')
  )
  ```
- **设计思路 & 好处**  
  利用多核并行加速，显著缩短大品种池下的计算时间。
