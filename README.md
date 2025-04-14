# Futures Trend Quality System

This project is a futures trend evaluation and signal generation system. It supports both:

- 📈 **Historical Backtest** – Using existing CSV data to simulate past performance  
- 🔔 **Real-Time Monitoring** – Fetching today's futures data, scoring trend quality, and generating daily trading signals

---

## ⚙️ Setup

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

You also need [AKShare](https://akshare.xyz/) for fetching futures data:

```bash
pip install akshare
```

---

## 🚀 Usage

### 🔹 Run Real-Time Monitoring (Generate Daily Report)

```bash
python realtime_monitor.py --mode live --data ./data
```

This will:
- Fetch today's bar data for all futures
- Merge it with existing historical data from `/data`
- Compute trend strength, consistency, composite score
- Output a CSV report: `daily_report_YYYYMMDD.csv`

📄 Example Report Format:
```csv
symbol,trend_strength,trend_consistency,composite_score,momentum_rank,persistence_rank
RB0,0.85,0.79,0.82,3,5
M0,0.81,0.76,0.79,4,6
...
```

---

### 🔹 Run Historical Backtest

```bash
python realtime_monitor.py --mode backtest --data data/all_futures_20250413.csv
```

This will:
- Load all data in `/data`
- Compute signals based on composite score
- Simulate trades using a simple equal-weight strategy
- Print final portfolio cash

---

## 📊 Scoring Logic

- **Trend Strength**: Based on normalized momentum and slope (multi-scale)
- **Trend Consistency**: Based on robust R² over rolling windows
- **Composite Score**: Weighted average of strength and consistency (default weight = 0.61)

---

## 🧑‍💻 Author

Quantitative research prototype by Linxiangyu  
Built for trend quality evaluation of Chinese futures instruments.
