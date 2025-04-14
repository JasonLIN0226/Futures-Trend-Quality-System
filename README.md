# Futures Trend Quality System | 期货趋势质量评估系统

This project is a futures trend evaluation and signal generation system.  
本项目是一个期货趋势质量评估与信号生成系统，支持以下两种模式：

- 📈 **Historical Backtest** – Using existing CSV data to simulate past performance  
  📈 **历史回测**：使用历史 CSV 数据回测交易表现  
- 🔔 **Real-Time Monitoring** – Fetching today's futures data, scoring trend quality, and generating daily trading signals  
  🔔 **实时监控**：获取当日主力合约数据，评估趋势质量，生成每日交易信号

---

## ⚙️ Setup | 安装依赖

Install dependencies using pip:  
使用 pip 安装依赖：

```bash
pip install -r requirements.txt
```

You also need [AKShare](https://akshare.xyz/) for fetching futures data:  
你还需要安装 [AKShare](https://akshare.xyz/) 来获取期货数据：

```bash
pip install akshare
```

---

## 🚀 Usage | 使用方法

### 🔹 Run Real-Time Monitoring (Generate Daily Report)  
### 🔹 运行实时监控（生成每日报告）

```bash
python realtime_monitor.py --mode live --data ./data
```

This will:  
程序将会执行以下操作：

- Fetch today's bar data for all futures  
  获取今日所有期货主力合约的K线数据  
- Merge it with existing historical data from `/data`  
  与 `/data` 中已有的历史数据合并  
- Compute trend strength, consistency, composite score  
  计算趋势强度、趋势流畅性、综合评分  
- Output a CSV report: `daily_report_YYYYMMDD.csv`  
  输出报告：`daily_report_YYYYMMDD.csv`

📄 Example Report Format | 示例输出格式:
```csv
symbol,trend_strength,trend_consistency,composite_score,momentum_rank,persistence_rank
RB0,0.85,0.79,0.82,3,5
M0,0.81,0.76,0.79,4,6
...
```

---

### 🔹 Run Historical Backtest  
### 🔹 运行历史回测

```bash
python realtime_monitor.py --mode backtest --data data/all_futures_20250413.csv
```

This will:  
程序将会执行以下操作：

- Load all data in `/data`  
  加载 `/data` 中的所有历史数据  
- Compute signals based on composite score  
  根据综合评分生成交易信号  
- Simulate trades using a simple equal-weight strategy  
  使用等权重策略模拟交易  
- Print final portfolio cash  
  输出最终账户资金结果

---

## 📊 Scoring Logic | 评分机制说明

- **Trend Strength 趋势强度**：基于归一化多窗口动量与回归斜率  
- **Trend Consistency 趋势流畅性**：基于滚动回归 R² 的稳定性指标  
- **Composite Score 综合评分**：趋势强度和流畅性加权平均（默认权重 = 0.61）

---

## 🧑‍💻 Author | 作者信息

Quantitative research prototype by Linxiangyu  
林祥宇开发的量化研究原型项目  
用于中国期货品种的趋势质量评估与信号生成。
