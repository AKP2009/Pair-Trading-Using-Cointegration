import numpy as np
import pandas as pd
import streamlit as st

from core import data
from core.style import page_header, pct

page_header(
    "Reference",
    "Limitations, glossary and reproducibility",
    "What the result does and does not show, the terms used across the dashboard, and how to regenerate every "
    "number from the code in this repository.",
)

card = data.scorecard().set_index("period")
oos, train = card.loc["OUT-OF-SAMPLE 2025..latest"], card.loc["train 2015-2024"]
m = data.meta()
years = len(data.equity_oos()) / 252
se = np.sqrt((1 + 0.5 * oos["sharpe"] ** 2) / years)

st.header("Limitations")
st.markdown(
    f"1. **Short test window.** {years:.1f} years out of sample. The standard error of a Sharpe ratio over that span is "
    f"≈ {se:.2f} (Lo, 2002: √((1 + SR²/2) / T)), so a Sharpe of {oos['sharpe']:.2f} is consistent with a modest real edge "
    "but does not prove one.\n"
    f"2. **Weak training record.** Training Sharpe {train['sharpe']:.2f}; {pct(train['total_return'])} over 2015–24 while the "
    f"TRI returned {pct(train['tri_total_return'])}. The goal was met in the test because the index went sideways while "
    "the pair book stayed positive — a market-neutral strategy will lag in strong bull markets.\n"
    "3. **Execution costs.** Profitable at 5 bps slippage per side, barely at 10 bps, clearly negative at 20 bps or with "
    "cash-market delivery and stock borrowing. Real fills, impact on smaller names and futures roll timing matter.\n"
    "4. **Futures tradability.** The NSE F&O stock list could not be downloaded, so every pair is assumed tradable through "
    "stock futures. Most Nifty 200 names are, but not all; lot sizes are also ignored.\n"
    f"5. **Residual survivorship.** {m['n_missing_history']} historical members have no price data (delisted or merged), "
    "all before the test window. Their absence can only flatter the training years.\n"
    "6. **Approximate benchmark.** The TRI is the price index plus 1.5% a year of dividends; the official TRI download "
    "was blocked. The result also holds at 2.5% (breakeven yield ≈ 6%).\n"
    "7. **A seen test window.** Two earlier designs were evaluated on the same 2025–26 window before the point-in-time "
    "protocol was written. The final selection was mechanical and fixed in advance, but the decision to rebuild the "
    "universe followed those failures.\n"
    "8. **Daily closes only.** No intraday data, no corporate-action handling beyond Yahoo's adjusted closes, and "
    "fills at the close are assumed achievable."
)

st.header("What would make the evidence stronger")
st.markdown(
    "- **Forward paper trading** of the frozen configuration — the only fully clean test left, since all history has been seen.\n"
    "- **The official Nifty 200 TRI** in `data/nifty200_tri.csv` (every script picks it up automatically).\n"
    "- **The NSE F&O list and lot sizes**, to drop untradable names and size trades in whole lots.\n"
    "- **Stock-vs-basket relative strength** (each stock against its whole cointegrated cluster) for more independent bets.\n"
    "- **Execution research** — limit orders around the close, or trading the most liquid futures only."
)

st.header("Glossary")
st.table(pd.DataFrame([
    ["Pair trading", "Buying one stock and shorting a related one in equal ₹ amounts, betting on their relative price, not the market."],
    ["Cointegration", "Two prices that each wander but whose combination (here the log ratio) keeps returning to a stable level."],
    ["ADF test", "Augmented Dickey–Fuller unit-root test. A low p-value (≤ 0.05) is evidence a series is stationary (mean-reverting)."],
    ["Engle–Granger test", "Cointegration test used in Phase 1: regress A on B, then test the residual spread with ADF."],
    ["Relative strength (RS)", "Stock A's return minus stock B's over the lookback (20 days): ln(A/B) today minus 20 days ago."],
    ["Percentile rank", "Where today's value sits among the last N values, from 0 (lowest) to 1 (highest). Used instead of z-scores."],
    ["Z-score", "(value − rolling mean) ÷ rolling standard deviation. Used in Phase 1 only."],
    ["Regime", "A label for the market state each day: Bullish, Bearish, Sideways, Low-vol or High-vol."],
    ["Rank-IC", "Spearman correlation between a factor and what it should predict; used to rank regime factors."],
    ["Efficiency Ratio", "Kaufman's trend-quality measure: net move ÷ total path length over 20 days."],
    ["Total return", "Percentage gain over the whole period, with returns compounded."],
    ["Sharpe ratio", "Annualised mean daily return ÷ annualised volatility (0% risk-free rate)."],
    ["Sortino ratio", "Like Sharpe, but divides by downside volatility only."],
    ["Volatility", "Annualised standard deviation of daily returns."],
    ["Max drawdown", "The largest peak-to-trough fall of the equity curve."],
    ["Beta", "Sensitivity of the strategy's daily returns to the index's. ≈ 0 means market-neutral."],
    ["TRI", "Total Return Index: the index with dividends reinvested — the fair benchmark for dividend-adjusted prices."],
    ["In-sample / out-of-sample", "Data used to choose the model vs data it never saw before the final test."],
    ["Walk-forward", "Re-choosing the model each year from past data only, then trading the next year; repeated through time."],
    ["Plateau score", "A candidate's score averaged with its grid neighbours, to prefer stable settings over lucky spikes."],
    ["Survivorship bias", "Flattering a backtest by using only stocks that survived to today."],
    ["Point-in-time universe", "The index membership as it was on each historical date."],
    ["Slippage", "The gap between the modelled price and the price actually achieved, in basis points (1 bp = 0.01%)."],
    ["SLB", "Securities Lending & Borrowing — how stock is borrowed in India to hold a short in the cash market."],
], columns=["Term", "Meaning"]).set_index("Term"))

st.header("Reproduce every number")
st.markdown("From the project root, with the full research environment (`pip install -r requirements.txt`):")
st.code("""\
.venv\\Scripts\\python.exe universe.py              # Nifty 200 prices, Nifty 200 index, India VIX (Yahoo Finance)
.venv\\Scripts\\python.exe regime.py                # first regime model -> data/regimes.csv
.venv\\Scripts\\python.exe research.py              # Phase 2A: static dev / holdout study, grid, variants
.venv\\Scripts\\python.exe walk_forward_rs.py       # Phase 2B: walk-forward v3
.venv\\Scripts\\python.exe tune_h15.py              # Phase 2C: 15-day hold, walk-forward lookback tuning
.venv\\Scripts\\python.exe kaufman_system.py        # Phase 2D: regime-timed index + pairs
.venv\\Scripts\\python.exe rs_lowvol_system.py      # Phase 2E: RS + low-vol stocks
.venv\\Scripts\\python.exe pit_pair_system.py       # FINAL: point-in-time selection on 2015-24, single test vs TRI
.venv\\Scripts\\python.exe pit_pair_robustness.py   # cost / benchmark / neighbour robustness
.venv\\Scripts\\python.exe final_oos.py             # all frozen strategies vs the TRI
.venv\\Scripts\\python.exe app\\build_data.py        # export this dashboard's bundle (checks it reproduces the final equity)
.venv\\Scripts\\python.exe -m pytest tests -q       # unit and no-lookahead tests
.venv\\Scripts\\python.exe -m streamlit run app\\streamlit_app.py""", language="bash")
st.caption(f"Bundle built {m['built']}; its re-run of the frozen strategy matches research/pit_pairs/full_equity.csv to "
           f"{m['equity_check_max_abs_diff']:.0e}.")

st.header("Code map")
st.table(pd.DataFrame([
    ["universe.py", "Downloads and caches Nifty 200 closes, the Nifty 200 index and India VIX"],
    ["pit_universe.py", "Point-in-time Nifty 200 membership 2013–2026 from NSE constituent files and review press releases"],
    ["tools/parse_nse_*.py", "Parsers that built the membership files from NSE PDFs / spreadsheets"],
    ["regime.py", "First regime model: trend / breadth / volatility vote"],
    ["kaufman_factors.py", "13 regime factors, percentile normalisation, correlation filter, composite regime"],
    ["rs_pairs.py", "Pair formation (ADF on the log ratio), RS signals, the portfolio backtester and the futures/cash cost model"],
    ["perf.py", "Sharpe, Sortino, drawdown, beta, win rate, yearly and monthly tables"],
    ["research.py · walk_forward_rs.py · tune_h15.py", "Phase 2A–2C studies"],
    ["kaufman_system.py · rs_lowvol_system.py", "Phase 2D–2E systems on a fixed 2015–24 / 2025+ split"],
    ["pit_pair_system.py", "Final strategy: survivorship-free selection and the single out-of-sample test"],
    ["pit_pair_robustness.py · final_oos.py", "Robustness checks and the scorecard of every frozen strategy"],
    ["tri.py", "Nifty 200 TRI loader (official file if present)"],
    ["notebooks/01–05 · model_pipeline.py", "Phase 1: data, EDA, Engle–Granger pairs, z-score backtest, Random Forest filter"],
    ["costs.py · risk.py · simulate.py", "Phase 1 realistic cost model, position sizing and stops, day-by-day engine"],
    ["walk_forward.py · paper_trading.py", "Phase 1 walk-forward validation and the daily paper-trading ledger"],
    ["app/", "This dashboard: build_data.py exports the bundle, streamlit_app.py and views/ render it"],
], columns=["File", "Purpose"]).set_index("File"))
