# 📈 Quantitative Pair Trading with Cointegration & ML Filtering

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end algorithmic quantitative pair trading framework built on **Engle-Granger Cointegration**, **Z-score Mean Reversion Signals**, and **Random Forest Machine Learning Filters**. Includes interactive Streamlit analytics, walk-forward optimization, risk controls, transaction cost simulation, and paper trading capabilities.

---

## 📑 Table of Contents
- [📌 System Architecture & Pipeline](#-system-architecture--pipeline)
- [⚡ Quick Start & Setup](#-quick-start--setup)
- [🔁 Pipeline Execution Order](#-pipeline-execution-order)
- [📊 Interactive Dashboard](#-interactive-dashboard)
- [📁 Key Outputs & Verification](#-key-outputs--verification)
- [🛠️ Troubleshooting & Common Fixes](#️-troubleshooting--common-fixes)

---

## 📌 System Architecture & Pipeline

```mermaid
flowchart TD
    A["📈 01. Data Collection<br/>Yahoo Finance / Nifty 50"] --> B["🔍 02. EDA & Stationarity<br/>ADF Test & Price Ratio Analysis"]
    B --> C["🎯 03. Pair Selection<br/>Engle-Granger & Johansen Cointegration"]
    C --> D["⚡ 04. Signal Backtest<br/>Z-score Mean Reversion"]
    D --> E["🤖 05. ML Filtering<br/>Random Forest Regime & Quality Filter"]
    E --> F["🖥️ 06. Streamlit Dashboard<br/>Interactive Analytics & Live Monitoring"]
    E --> G["🔄 Walk-Forward & Paper Trading<br/>Out-of-Sample Validation"]
```

### 🧠 Core Algorithmic Components

```mermaid
graph LR
    subgraph Signal Generation
        A1["Price Ratio / Spread"] --> A2["Z-Score Calculation"]
        A2 --> A3["Entry: |Z| > Threshold<br/>Exit: |Z| < Exit Level"]
    end

    subgraph Machine Learning Quality Filter
        B1["Market Regime Features<br/>Volatility, ADX, Spread Dist"] --> B2["Random Forest Classifier"]
        B2 --> B3{"Quality Score > Threshold?"}
    end

    A3 --> C{"Filter Passed?"}
    B3 --> C
    C -- Yes --> D["Execute Trade with Risk & Cost Controls"]
    C -- No --> E["Suppress False Signal"]
```

---

## ⚡ Quick Start & Setup

### 1. Prerequisites
* **Python 3.10+**
* **VS Code** with Python & Jupyter Extensions installed

### 2. Environment Setup

#### **Windows (PowerShell)**
```powershell
# 1. Create Virtual Environment
python -m venv .venv

# 2. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade Pip & Install Dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### **macOS / Linux**
```bash
# 1. Create Virtual Environment
python -m venv .venv

# 2. Activate Virtual Environment
source .venv/bin/activate

# 3. Upgrade Pip & Install Dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 🔁 Pipeline Execution Order

Execute the Jupyter Notebooks sequentially in `notebooks/` from top to bottom. Each stage relies on artifacts produced by the previous step:

| Step | Notebook | Description | Output Artifact |
| :--- | :--- | :--- | :--- |
| 1️⃣ | [`notebooks/01_data_collection.ipynb`](notebooks/01_data_collection.ipynb) | Fetches clean stock price history from `yfinance`. | `data/nifty_prices_clean.csv` |
| 2️⃣ | [`notebooks/02_eda_pairs.ipynb`](notebooks/02_eda_pairs.ipynb) | Analyzes spread stationarity and correlation. | EDA plots & stats |
| 3️⃣ | [`notebooks/03_pair_selection.ipynb`](notebooks/03_pair_selection.ipynb) | Runs Engle-Granger tests to filter cointegrated pairs. | `data/selected_pairs.csv` |
| 4️⃣ | [`notebooks/04_signal_backtest.ipynb`](notebooks/04_signal_backtest.ipynb) | Backtests baseline Z-score mean-reversion strategy. | `data/baseline_backtest_stats.csv` |
| 5️⃣ | [`notebooks/05_ml_filter_random_forest.ipynb`](notebooks/05_ml_filter_random_forest.ipynb) | Trains ML classifier to reduce drawdown & false signals. | `data/ml_vs_baseline_stats.csv` |

> [!NOTE]  
> If using the alternate Phase 3 methodology, run [`notebooks/03_pair_selection1.ipynb`](notebooks/03_pair_selection1.ipynb) instead of notebook 3.

---

## 📊 Interactive Dashboard

Launch the Streamlit Web Application to visualize pair performance, equity curves, drawdown curves, and live trade signals:

```powershell
python -m streamlit run dashboard.py --server.port 8501
```

Access the UI at: **`http://localhost:8501`**

### Dashboard Features
- 📈 **Interactive Pair Explorer:** View dynamic price ratios, Z-scores, and cointegration statistics.
- ⚖️ **Baseline vs ML Strategy Comparison:** Equity curves, Sharpe ratios, and max drawdown comparison tables.
- 🛡️ **Risk & Cost Analytics:** Slippage, brokerages, and stop-loss impact simulations.
- 📡 **Live Signal Monitor:** Real-time z-score thresholds and entry/exit triggers.

---

## 📁 Key Outputs & Verification

Upon completing all notebook runs, verify that the following core files exist:

- `data/nifty_prices_clean.csv` — Clean historical price dataset
- `data/selected_pairs.csv` — Statistically verified cointegrated pairs
- `data/baseline_backtest_stats.csv` — Unfiltered Z-score backtest metrics
- `data/ml_vs_baseline_stats.csv` — Machine Learning enhanced strategy metrics
- `outputs/06_ml_equity_and_importance.png` — ML feature importance & equity comparison plots

---

## 🛠️ Troubleshooting & Common Fixes

> [!TIP]
> **Missing Dependencies:**  
> Run `python -m pip install <package_name>` inside your active virtual environment.

> [!WARNING]
> **Data Download Errors:**  
> If a stock ticker fails in `yfinance`, update the ticker symbol list in `notebooks/01_data_collection.ipynb` and re-run the pipeline.

> [!IMPORTANT]
> **Missing Downstream Data Files:**  
> Notebooks must be run sequentially in order. If a file is missing, re-run from Phase 1.

---

## 🧭 Nifty 200 Regime-Aware Relative-Strength Pairs (research pipeline)

A second, script-based pipeline extends the project from the 22-stock Nifty 50 sector set to the full
**Nifty 200**, with market regimes (trend, breadth, volatility), cointegration pairing, and a
relative-strength + volatility signal (no z-score, no regression hedge ratio).
Results and methodology: [research/REPORT.md](research/REPORT.md).

Run from the project root:

```powershell
.\.venv\Scripts\python.exe universe.py          # Nifty 200 prices, Nifty 200 index, India VIX
.\.venv\Scripts\python.exe regime.py            # daily regime labels
.\.venv\Scripts\python.exe pit_pair_system.py   # final strategy: train 2015-2024, test 2025+ vs Nifty 200 TRI
.\.venv\Scripts\python.exe pit_pair_robustness.py
.\.venv\Scripts\python.exe report_figures.py
.\.venv\Scripts\python.exe -m pytest tests -q   # 58 tests
```

The dashboard picks up the new results automatically as the **Nifty200 Regime Research** page in the sidebar.

| File | Purpose |
|---|---|
| universe.py | Downloads Nifty 200 closes, Nifty 200 index, India VIX |
| regime.py | Bullish / Bearish / Sideways / Low Vol / High Vol regime per day |
| rs_pairs.py | Cointegration pair formation, RS signal, portfolio backtester, costs |
| perf.py | Sharpe, Sortino, max drawdown, beta, win rate, yearly / monthly returns |
| research.py | Static dev (2014-21) / holdout (2022+) study, parameter grid, variants, per-regime runs |
| walk_forward_rs.py | Walk-forward model selection (final model) |
| report_figures.py | Charts in research/figures |
| tune_h15.py | Walk-forward lookback tuning with a fixed 15-calendar-day hold |
| kaufman_factors.py | Kaufman-style regime factors (Efficiency Ratio, net new highs, ...), percentile normalization, correlation filter |
| kaufman_system.py | Regime-timed index + pair system, trained on 2015-2024 and tested on 2025+ (research/kaufman/) |
| rs_lowvol_system.py | RS + low-vol stock selection + regime hedge + pairs, judged against EW buy & hold and the index (research/rs_lowvol/) |
| tri.py | Nifty 200 TRI loader (official file if saved as data/nifty200_tri.csv) |
| pit_universe.py | Point-in-time Nifty 200 membership 2013-2026 from NSE constituent archives + review press releases |
| pit_pair_system.py | **Final strategy**: survivorship-free selection on 2015-2024, single out-of-sample test vs Nifty 200 TRI (research/pit_pairs/) |
| pit_pair_robustness.py | Cost, benchmark and neighbour robustness checks for the final strategy |
| final_oos.py | Out-of-sample scorecard of all frozen strategies vs the TRI |
| tools/parse_nse_*.py | Parsers used to build the point-in-time membership files (need `pip install pdfplumber xlrd`) |

**Final result (research/REPORT.md section 10):** out of sample (2025 to Oct 2026), the frozen point-in-time pair strategy
returned +6.8% (Sharpe 0.53, vol 7.4%, max DD -7.2%) vs the Nifty 200 TRI's -1.7% (Sharpe 0.00, vol 14.2%, max DD -14.6%).
Read the caveats there: thin edge, needs futures-level execution costs, weak 2015-2024 training record.
