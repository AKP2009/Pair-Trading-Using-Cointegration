import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import charts, data
from core.style import AXIS, BENCHMARK, CRITICAL, GOOD, MUTED, SERIES, STRATEGY, THIRD, chip, page_header, pct, show, step_cards

sys.path.append(str(data.ROOT))
import costs  # noqa: E402  (the project's own cost model: pure Python, no heavy dependencies)

page_header(
    "Research journey · phase 1",
    "Phase 1: classic cointegration, z-scores and a Random Forest filter",
    "The project started with the textbook approach on 22 Nifty 50 stocks in four sectors: find cointegrated pairs "
    "with the Engle–Granger test, trade the z-score of the regression spread, and use a Random Forest to filter out "
    "bad entries. It was then given a realistic cost engine, walk-forward validation and a paper-trading ledger. "
    "It did not survive those tests, and why it failed shaped everything that came after.",
)

step_cards([
    ("Data (nb 01)", "Adjusted closes for 22 stocks in Banking, IT, Auto and Energy, 2015–2025, from Yahoo Finance."),
    ("Stationarity (nb 02)", "ADF tests on price levels; train 2015–21 / validation 2022–23 / test 2024–25 split."),
    ("Pairs (nb 03)", "Same-sector pairs with price correlation ≥ 0.80, Engle–Granger cointegration test, OLS hedge ratio."),
    ("Z-score signal (nb 04)", "Enter at |z| > 2, exit at |z| < 0.5, on a rolling-β spread."),
    ("ML filter (nb 05)", "A Random Forest predicts whether the spread will revert within 10 days; trade only if p ≥ 0.55."),
])

# ---------------------------------------------------------------- stationarity + pairs
st.header("1 · Stationarity and pair selection")
c1, c2 = st.columns([1, 1.3], gap="large")
adf = data.data_csv("adf_levels_train.csv")
with c1:
    st.markdown(
        f"**Every one of the {len(adf)} price series is non-stationary** in levels (ADF p > 0.05 on 2015–21) — prices "
        "wander, so they cannot be traded on their own level. The question is whether a *combination* of two prices is "
        "stationary. That is cointegration."
    )
    st.latex(r"P_{A,t} = \alpha + \beta P_{B,t} + \varepsilon_t \qquad \text{cointegrated} \iff \varepsilon_t \text{ is stationary}")
    st.markdown("The Engle–Granger test estimates β by OLS on the training period and runs a unit-root test on the "
                "residual spread εₜ.")
cand = data.data_csv("all_candidate_pairs.csv")
cand["pair"] = cand["stock_a"].str.replace(".NS", "") + " / " + cand["stock_b"].str.replace(".NS", "")
with c2:
    cs = cand.sort_values("coint_pvalue")
    fig = charts.hbar(cs["pair"], cs["coint_pvalue"], "Engle–Granger p-value of the 23 candidate pairs (train 2015–21)",
                      fmt=".3f", colors=[STRATEGY if ok else "#c3c2b7" for ok in cs["cointegrated_5pct"]], text=False,
                      log=True)
    fig.add_vline(x=0.05, line=dict(color=CRITICAL, width=1.5))
    fig.add_annotation(x=np.log10(0.05), y=1, yref="paper", text="5% threshold", showarrow=False, xanchor="left",
                       xshift=4, yanchor="bottom", font=dict(size=11, color=MUTED))
    fig.update_xaxes(title_text="p-value (log scale; left of the line = cointegrated)")
    show(fig, height=580)
st.markdown("Three pairs pass at 5% (blue): " + "".join(chip(p, "good") for p in cand.loc[cand["cointegrated_5pct"], "pair"])
            , unsafe_allow_html=True)

# ---------------------------------------------------------------- z-score
st.header("2 · The z-score strategy")
c1, c2 = st.columns([1, 1.6], gap="large")
with c1:
    st.latex(r"S_t = P_{A,t} - \beta_t P_{B,t} \qquad z_t = \frac{S_t - \overline{S}_{30}}{\sigma_{30}(S)}")
    st.markdown(
        "- β is a rolling 252-day regression slope (the static training β until enough data exists).\n"
        "- **Short the spread** when z > 2 (A rich vs B), **long** when z < −2, **exit** when |z| < 0.5.\n"
        "- 10 bps cost per unit of position change."
    )
    base = data.data_csv("baseline_backtest_stats.csv")
    st.dataframe(base.assign(CAGR=base["CAGR"] * 100, Vol=base["Vol"] * 100, MaxDD=base["MaxDD"] * 100,
                             strategy=base["strategy"].map({"zscore_strategy": "Z-score strategy",
                                                            "raw_spread_buyhold": "Hold the spread"})),
                 hide_index=True, column_config={
                     "strategy": "HEROMOTOCO / EICHERMOT, 2015–24",
                     "CAGR": st.column_config.NumberColumn(format="%.1f%%"),
                     "Vol": st.column_config.NumberColumn(format="%.1f%%"),
                     "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                     "MaxDD": st.column_config.NumberColumn("Max DD", format="%.1f%%")})
    st.caption("Barely positive (CAGR 0.7%) with a 43% drawdown on the best-ranked pair.")
with c2:
    bt = data.data_csv("baseline_trade_series.csv", index_col=0, parse_dates=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.07,
                        subplot_titles=("Spread z-score with entry (±2) and exit (±0.5) bands", "Equity, growth of ₹1"))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["zscore"], name="z-score", line=dict(color=MUTED, width=1),
                             hovertemplate="%{y:.2f}<extra>z</extra>"), row=1, col=1)
    for y in (2, -2):
        fig.add_hline(y=y, line=dict(color=CRITICAL, width=1), row=1, col=1)
    for y in (0.5, -0.5):
        fig.add_hline(y=y, line=dict(color=AXIS, width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name="Z-score strategy", line=dict(color=STRATEGY, width=2),
                             hovertemplate="%{y:.2f}<extra>strategy</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=bt.index, y=bt["buyhold_spread"], name="Hold the spread", line=dict(color=BENCHMARK, width=1.6),
                             hovertemplate="%{y:.2f}<extra>hold spread</extra>"), row=2, col=1)
    fig.update_layout(hovermode="x unified")
    fig.update_annotations(font=dict(size=13))
    show(fig, height=480)

# ---------------------------------------------------------------- RF
st.header("3 · The Random Forest trade filter")
FEAT = {"z": "Current z-score", "z_chg_5": "Change in z over 5 days", "z_chg_20": "Change in z over 20 days",
        "corr_30": "30-day price correlation", "spread_vol_20": "20-day spread-return volatility",
        "spread_vol_60": "60-day spread-return volatility", "half_life": "Mean-reversion half-life (rolling AR(1))",
        "hurst": "Hurst exponent of the spread (< 0.5 hints at reversion)",
        "beta_dev": "Kalman-filter β minus the static β (hedge drift)"}
c1, c2 = st.columns([1, 1.2], gap="large")
with c1:
    st.markdown(
        "**Label:** 1 if the spread is closer to its rolling mean 10 days later, else 0.\n\n"
        "**Model:** Random Forest (400 trees, depth 7, min leaf 15, balanced classes), trained on 2015–21, threshold "
        "tuned on 2022–23, tested on 2024–25. A baseline z-score entry is taken only if the model's probability ≥ 0.55.\n\n"
        "`model_pipeline.py` trains one model per pair with the nine features below (notebook 05 used the first six)."
    )
    st.dataframe(pd.DataFrame(FEAT.items(), columns=["Feature", "Meaning"]), hide_index=True)
with c2:
    imp = data.bundle_csv("rf_feature_importance.csv", index_col=0)
    imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index]
    fig = go.Figure()
    for i, col in enumerate(imp.columns):
        fig.add_trace(go.Bar(y=imp.index, x=imp[col], name=col.replace("__", " / "), orientation="h",
                             marker_color=SERIES[i], marker_line_width=0,
                             hovertemplate="%{y}: %{x:.3f}<extra>" + col.replace("__", " / ") + "</extra>"))
    fig.update_layout(barmode="group", title="Feature importance (mean decrease in impurity)", bargroupgap=0.1)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#e1e0d9")
    show(fig, height=460)
    st.caption("Importance is spread almost evenly: no single feature carries much information about whether a spread "
               "will revert — an early sign the filter had little to work with.")

reg = data.data_csv("../models/pair_model_registry.csv")
mets = reg[["model_name", "validation_f1", "validation_roc_auc", "test_f1", "test_roc_auc"]].copy()
mets["model_name"] = mets["model_name"].str.replace("__", " / ")
c1, c2 = st.columns([1.3, 1], gap="large")
with c1:
    st.markdown("**Classification quality per pair**")
    st.dataframe(mets, hide_index=True, column_config={
        "model_name": "Pair", "validation_f1": st.column_config.NumberColumn("Valid F1", format="%.2f"),
        "validation_roc_auc": st.column_config.NumberColumn("Valid AUC", format="%.2f"),
        "test_f1": st.column_config.NumberColumn("Test F1", format="%.2f"),
        "test_roc_auc": st.column_config.NumberColumn("Test AUC", format="%.2f")})
    st.caption("AUC 0.5 is a coin flip. Validation AUCs of 0.50–0.57 mean the model could barely rank good entries above bad ones.")
with c2:
    ml = data.data_csv("ml_vs_baseline_stats.csv")
    st.markdown("**Did the filter help the backtest?**")
    st.dataframe(ml.assign(CAGR=ml["CAGR"] * 100, Vol=ml["Vol"] * 100, MaxDD=ml["MaxDD"] * 100,
                           strategy=ml["strategy"].map({"baseline": "Z-score baseline", "rf_filtered": "With RF filter"})),
                 hide_index=True, column_config={
                     "strategy": "Strategy", "CAGR": st.column_config.NumberColumn(format="%.1f%%"),
                     "Vol": st.column_config.NumberColumn(format="%.1f%%"), "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                     "MaxDD": st.column_config.NumberColumn("Max DD", format="%.1f%%")})
    st.caption("No: the filter cut the good trades as often as the bad ones, and the filtered strategy lost money "
               "(Sharpe −1.03).")

mlt = data.data_csv("ml_trade_series.csv", index_col=0, parse_dates=True)
show(charts.lines(mlt[["equity_base", "equity_ml"]].rename(columns={"equity_base": "Z-score baseline",
                                                                     "equity_ml": "With RF filter"}),
                  "HEROMOTOCO / EICHERMOT: baseline vs Random-Forest-filtered equity (notebook 05)",
                  colors={"Z-score baseline": STRATEGY, "With RF filter": BENCHMARK}, ytitle="Growth of ₹1",
                  label_fmt=lambda v: f"{v:.2f}"), height=340)

# ---------------------------------------------------------------- costs & risk
st.header("4 · A realistic cost and risk engine")
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown(
        "The notebooks charged a flat 10 bps. `costs.py`, `risk.py` and `simulate.py` replaced that with:\n\n"
        "- **Indian cash-market charges:** STT 0.1% on both sides, exchange fee 0.00297% + 18% GST, SEBI fee, stamp duty "
        "0.015% on buys, a ₹15 DP charge per delivery sale.\n"
        "- **Short selling through SLB:** the short leg is borrowed at 8% a year, because NSE cash equity cannot be held "
        "short overnight otherwise.\n"
        "- **Volatility-scaled slippage** (half the recent spread volatility) instead of a fixed guess.\n"
        "- **Whole shares:** positions are rounded to integer quantities, and the hedge error from rounding is tracked.\n"
        "- **Risk controls:** volatility-targeted size (1% of capital per day of spread vol), a z-score stop at |z| ≥ 4, "
        "a 40-day time stop, at most 3 concurrent pairs and no stock in two pairs."
    )
with c2:
    st.markdown("**Try the cost model** (`costs.round_trip_cost_pct`, the project's own code)")
    k1, k2, k3 = st.columns(3)
    notional = k1.number_input("₹ per leg", 10_000, 10_000_000, 100_000, step=10_000)
    days = k2.number_input("Holding days", 1, 120, 10)
    vol = k3.number_input("Daily spread vol (%)", 0.0, 5.0, 1.0, step=0.1) / 100
    rt = costs.round_trip_cost_pct(notional, notional, vol, days)
    fee = (costs.leg_transaction_cost(notional, True) * 2 + costs.leg_transaction_cost(notional, False, True)
           + costs.leg_transaction_cost(notional, False)) / (2 * notional)
    borrow = costs.short_borrow_cost(notional, days) / (2 * notional)
    st.metric("Round-trip cost, % of both legs", f"{rt:.3%}", border=True)
    st.dataframe(pd.DataFrame([["Regulatory + exchange charges", fee], ["SLB borrow on the short leg", borrow],
                               ["Slippage (vol-scaled)", rt - fee - borrow]], columns=["Component", "Share"]).assign(
        Share=lambda x: x["Share"] * 100), hide_index=True,
        column_config={"Share": st.column_config.NumberColumn("% of both legs", format="%.3f%%")})
    st.caption("With a 1% spread vol, slippage dominates. Reverting by more than this inside the holding period is the "
               "bar every trade must clear — the z-score strategy's average edge did not.")

# ---------------------------------------------------------------- walk-forward
st.header("5 · Walk-forward validation and paper trading")
wf = data.data_csv("walk_forward_stats.csv")
wfe = data.data_csv("walk_forward_equity.csv", parse_dates=["date"])
agg = wf[wf["fold"] == "ALL"].copy()
c1, c2 = st.columns([1.1, 1.3], gap="large")
with c1:
    st.markdown(
        "`walk_forward.py` retrains the Random Forest on an expanding window (4 years to start), trades the next 6 "
        "months strictly out of sample with the realistic engine, then rolls forward: 12 folds from mid-2019 to the "
        "end of 2024, ₹2 lakh per pair."
    )
    agg["pair"] = agg["pair"].str.replace("__", " / ").replace({"PORTFOLIO": "Portfolio"})
    agg["n_trades"] = pd.to_numeric(agg["n_trades"], errors="coerce")
    st.dataframe(agg.assign(MaxDD=agg["MaxDD"] * 100, win_rate=agg["win_rate"] * 100)[
        ["pair", "n_trades", "win_rate", "Sharpe", "MaxDD"]],
                 hide_index=True, column_config={
                     "pair": "Pair", "n_trades": st.column_config.NumberColumn("Trades", format="%d"),
                     "win_rate": st.column_config.NumberColumn("Win rate", format="%.0f%%"),
                     "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                     "MaxDD": st.column_config.NumberColumn("Max DD", format="%.1f%%")})
    st.caption("Every pair and the portfolio lost money out of sample (portfolio Sharpe −1.0). Few trades, low win rates.")
with c2:
    piv = wfe.pivot_table(index="date", columns="pair", values="equity", aggfunc="last")
    norm = piv.div(piv.bfill().iloc[0])
    norm.columns = [c.replace("__", " / ") if c != "PORTFOLIO" else "Portfolio (3 pairs)" for c in norm.columns]
    cols = {"Portfolio (3 pairs)": "#0b0b0b"}
    show(charts.lines(norm, "Walk-forward out-of-sample equity (growth of ₹1)", colors=cols, ytitle="Growth of ₹1",
                      label_fmt=lambda v: f"{v:.2f}", widths={"Portfolio (3 pairs)": 2.5}), height=380)

live = data.data_csv("live_prediction.csv")
pos = data.data_csv("paper_trading_positions.csv")
with st.expander("Paper trading ledger (paper_trading.py)"):
    st.markdown(
        "`paper_trading.py` is meant to run once per trading day after the NSE close: it pulls the latest prices, scores "
        "every pair's model and advances the same simulation engine, appending to `data/paper_trading_*.csv`. The "
        f"ledger's last run was **{pos['last_date'].max()}**, with every pair flat. The model's last live call:"
    )
    st.dataframe(live.assign(stock_a=live["stock_a"].str.replace(".NS", ""), stock_b=live["stock_b"].str.replace(".NS", ""))[
        ["date", "stock_a", "stock_b", "zscore", "rf_probability", "baseline_action", "final_action"]], hide_index=True,
        column_config={"zscore": st.column_config.NumberColumn("z-score", format="%.2f"),
                       "rf_probability": st.column_config.NumberColumn("RF probability", format="%.2f")})

st.header("What Phase 1 taught us")
st.markdown(
    "1. **Three pairs are not a strategy.** With so few pairs, results hinge on a handful of trades; breadth (many "
    "independent pairs) matters more than a clever filter.\n"
    "2. **The regression hedge ratio drifts.** β estimated in one period stops holding in the next, so the 'stationary' "
    "spread is not. → Phase 2 uses a **fixed 1 : 1 log-ratio** and re-tests pairs every six months.\n"
    "3. **Costs are first-order.** Cash-market STT and stock borrowing eat a thin reversion edge. → Phase 2 is costed "
    "as **stock futures**.\n"
    "4. **ML cannot rescue a weak signal.** A classifier with AUC ≈ 0.55 filters noise with noise.\n"
    "5. **Only out-of-sample evidence counts.** Walk-forward testing and a strict train/test protocol became the rule."
)
st.page_link("views/research.py", label="Phase 2: what was built next on the Nifty 200", icon=":material/science:")
