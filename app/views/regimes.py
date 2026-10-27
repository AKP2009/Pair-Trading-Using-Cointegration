import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import charts, data
from core.style import (AXIS, BENCHMARK, MUTED, REGIME_COLORS, REGIME_LABEL, REGIME_ORDER, STRATEGY, chip,
                        page_header, show)

page_header(
    "The model · regime engine",
    "Market regime engine",
    "Pair reversion does not work equally well in every market. The regime engine labels each trading day from "
    "trend, breadth and volatility factors, and the strategy uses that label as an on/off switch for new trades.",
)

m = data.meta()
cfg = m["config"]
rg = cfg["regime"]
reg = data.regime_daily()
regime = reg["regime"].dropna()

# ---------------------------------------------------------------- today
today = regime.iloc[-1]
run = (regime != regime.shift()).cumsum()
days_in = int((run == run.iloc[-1]).sum())
c = st.columns(4)
c[0].metric(f"Regime on {regime.index[-1]:%d %b %Y}", REGIME_LABEL[today], border=True)
c[1].metric("Direction score", f"{reg['direction'].iloc[-1]:.2f}", help=f"0–1. Bullish ≥ {rg['bull']}, bearish ≤ {rg['bear']}",
            border=True)
c[2].metric("Volatility score", f"{reg['volatility'].iloc[-1]:.2f}",
            help=f"0–1. High-vol ≥ {rg['high_vol']}, low-vol ≤ {rg['low_vol']}", border=True)
c[3].metric("Pair strategy", "Entries allowed" if today != "LOW_VOL" else "No new entries",
            f"{days_in} trading days in this regime", delta_color="off", border=True)

# ---------------------------------------------------------------- timeline
st.header("Regime timeline")
span = st.segmented_control("Window", ["All", "5 years", "2 years", "1 year"], default="All", key="reg_span")
start = {"All": regime.index[0], "5 years": regime.index[-1] - pd.DateOffset(years=5),
         "2 years": regime.index[-1] - pd.DateOffset(years=2), "1 year": regime.index[-1] - pd.DateOffset(years=1)}[span or "All"]
fig = charts.regime_ribbon(reg.loc[start:, "benchmark"], regime.loc[start:],
                           "Nifty 200 index with the daily regime underneath")
show(fig, height=460)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(
    f"Direction score (trend + breadth): bullish ≥ {rg['bull']}, bearish ≤ {rg['bear']}",
    f"Volatility score: high-vol ≥ {rg['high_vol']}, low-vol ≤ {rg['low_vol']}"))
d = reg.loc[start:]
fig.add_trace(go.Scatter(x=d.index, y=d["direction"], name="Direction score", line=dict(color=STRATEGY, width=1.3),
                         hovertemplate="%{y:.2f}<extra>direction</extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=d.index, y=d["volatility"], name="Volatility score", line=dict(color=BENCHMARK, width=1.3),
                         hovertemplate="%{y:.2f}<extra>volatility</extra>"), row=2, col=1)
for y, r in ((rg["bull"], 1), (rg["bear"], 1), (rg["high_vol"], 2), (rg["low_vol"], 2)):
    fig.add_hline(y=y, line=dict(color=AXIS, width=1), row=r, col=1)
fig.update_yaxes(range=[0, 1], tickvals=[0, rg["bear"], rg["bull"], 1], tickformat=".2f", row=1, col=1)
fig.update_yaxes(range=[0, 1], tickvals=[0, rg["low_vol"], 0.5, rg["high_vol"]], tickformat=".2f", row=2, col=1)
fig.update_layout(hovermode="x unified", showlegend=False)
fig.update_annotations(font=dict(size=13, color="#0b0b0b"), x=0, xanchor="left")
show(fig, height=440)
st.caption("Both scores are averages of percentile ranks, so they live between 0 and 1. The rule checks volatility first "
           "(High-vol overrides everything), then direction, then Low-vol; whatever is left is Sideways.")

# ---------------------------------------------------------------- shares
st.header("How often each regime occurs")
c1, c2 = st.columns([1.25, 1.35], gap="large")
share = regime.value_counts(normalize=True).reindex(REGIME_ORDER)
spells = regime.groupby(run.reindex(regime.index)).agg(["first", "size"])
mean_len = spells.groupby("first")["size"].mean().reindex(REGIME_ORDER)
with c1:
    tbl = pd.DataFrame({"Regime": [REGIME_LABEL[r] for r in REGIME_ORDER], "Share of days": share.values * 100,
                        "Average spell (days)": mean_len.values,
                        "Strategy trades?": ["✘ no new entries" if r == "LOW_VOL" else "✔ yes" for r in REGIME_ORDER]})
    st.dataframe(tbl, hide_index=True, column_config={
        "Share of days": st.column_config.NumberColumn(format="%.1f%%"),
        "Average spell (days)": st.column_config.NumberColumn(format="%.1f")})
    st.caption(f"Labelled days from {regime.index[0]:%b %Y} (the first date with {rg['pct_window']} days of factor history).")
with c2:
    by_year = pd.crosstab(regime.index.year, regime, normalize="index").reindex(columns=REGIME_ORDER).fillna(0)
    fig = go.Figure()
    for r in REGIME_ORDER:
        fig.add_trace(go.Bar(x=by_year.index.astype(str), y=by_year[r], name=REGIME_LABEL[r], marker_color=REGIME_COLORS[r],
                             marker_line=dict(color="white", width=1), hovertemplate="%{x}: %{y:.0%}<extra>" + REGIME_LABEL[r] + "</extra>"))
    fig.update_layout(barmode="stack", title="Share of trading days per regime, by year", bargap=0.2)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    show(fig, height=360)

# ---------------------------------------------------------------- per regime performance
st.header("Does the regime gate make sense?")
pr = data.research_csv("pit_pairs/per_regime.csv")
pr["Regime"] = pr["regime"].map(REGIME_LABEL)
period = st.segmented_control("Period", ["train", "test"], default="train",
                              format_func=lambda x: "Training 2015–24" if x == "train" else "Out-of-sample 2025–26",
                              key="reg_period") or "train"
p = pr[pr["period"] == period].set_index("Regime").reindex([REGIME_LABEL[r] for r in REGIME_ORDER])
fig = charts.grouped_bars(p[["strategy_ann_return", "tri_ann_return"]].rename(
    columns={"strategy_ann_return": "Pair strategy", "tri_ann_return": "Nifty 200 TRI"}),
    "Annualised return earned on the days in each regime", yfmt=".0%", text=True, text_fmt=".1%")
show(fig, height=380)
st.dataframe(p[["days", "strategy_ann_return", "strategy_sharpe", "tri_ann_return", "tri_sharpe"]].assign(
    strategy_ann_return=lambda x: x["strategy_ann_return"] * 100, tri_ann_return=lambda x: x["tri_ann_return"] * 100),
    column_config={"days": "Days", "strategy_ann_return": st.column_config.NumberColumn("Strategy ann. return", format="%.1f%%"),
                   "strategy_sharpe": st.column_config.NumberColumn("Strategy Sharpe", format="%.2f"),
                   "tri_ann_return": st.column_config.NumberColumn("TRI ann. return", format="%.1f%%"),
                   "tri_sharpe": st.column_config.NumberColumn("TRI Sharpe", format="%.2f")})
st.markdown(
    "Days are grouped by the regime that **set the position** (the label two days earlier, because a decision at "
    "close *t* fills at *t + 1* and earns from *t + 2*). In training, **Low-vol days lost money** at an annualised "
    "−16% (Sharpe −2.25), while High-vol and Sideways days carried the profit — exactly why new entries are blocked in "
    "Low-vol. In the test, open trades are still managed on Low-vol days, so a few test days carry that label."
)

# ---------------------------------------------------------------- factors
st.header("Which factors feed the scores")
use = data.bundle_csv("factor_usefulness.csv").set_index("factor")
corr = data.bundle_csv("factor_correlation.csv", index_col=0)
kept_all = cfg["factors"]["direction"] + cfg["factors"]["volatility"]
rows = []
for g, members in (("direction", ["dist_ema21", "dist_ema50", "dist_ema200", "er_signed_20", "roc_63",
                                  "pct_above_200ema", "pct_above_50ema", "ad_ratio_10", "net_highs_252"]),
                   ("volatility", ["vix", "rv_20", "avgabs_14", "dispersion_20"])):
    chosen = []
    for f in use.loc[members].sort_values("rank_ic", ascending=False).index:
        ic = use.at[f, "rank_ic"]
        if not ic > 0:
            why = "IC ≤ 0 (no predictive value)"
        else:
            clash = [(k, corr.at[f, k]) for k in chosen if abs(corr.at[f, k]) > 0.70]
            why = f"ρ = {clash[0][1]:.2f} with {clash[0][0]} (already kept)" if clash else "kept"
            if not clash:
                chosen.append(f)
        rows.append({"Score": g, "Factor": f, "Group": use.at[f, "group"], "Rank-IC": ic, "Decision": why})
ftab = pd.DataFrame(rows)
assert set(ftab.loc[ftab["Decision"] == "kept", "Factor"]) == set(kept_all)
DESC = {"dist_ema21": "Index distance from its 21-day EMA", "dist_ema50": "Index distance from its 50-day EMA",
        "dist_ema200": "Index distance from its 200-day EMA", "er_signed_20": "Kaufman Efficiency Ratio (20d), signed",
        "roc_63": "3-month rate of change", "pct_above_200ema": "% of members above their 200-day EMA",
        "pct_above_50ema": "% of members above their 50-day EMA", "ad_ratio_10": "Advancers ÷ decliners, 10-day sums",
        "net_highs_252": "(52-week highs − lows) ÷ members", "vix": "India VIX level",
        "rv_20": "Index 20-day realised volatility", "avgabs_14": "14-day mean |daily return| (ATR proxy)",
        "dispersion_20": "Cross-sectional std of member returns, 20-day mean"}
ftab.insert(2, "Meaning", ftab["Factor"].map(DESC))
st.markdown(
        "**Selection (training 2015–24 only).** Usefulness is the Spearman rank-IC of the factor's percentile with the "
        "index's next-20-day return (direction factors) or next-20-day realised volatility (volatility factors). Within "
        "each score, factors are taken in IC order and kept only if |ρ| ≤ 0.70 with every factor already kept — so "
        "near-duplicates do not get double weight."
)
st.dataframe(ftab, hide_index=True, height=500, column_config={
    "Rank-IC": st.column_config.NumberColumn(format="%.3f")})
show(charts.corr_heatmap(corr, "Spearman correlation of factor percentiles (training)", highlight=kept_all), height=620)
st.caption("Bold labels are the factors the final model uses. Kaufman's Efficiency Ratio replaced the original "
           "21-day-EMA trend factor (ρ = 0.84 between them, higher IC).")

with st.expander("Kaufman's Efficiency Ratio, the trend factor with the best IC"):
    st.latex(r"ER_{t} = \operatorname{sign}(P_t - P_{t-20})\cdot\frac{|P_t - P_{t-20}|}{\sum_{i=t-19}^{t}|P_i - P_{i-1}|}")
    st.markdown("It is the net move over 20 days divided by the total distance travelled: near ±1 for a clean trend, "
                "near 0 for a choppy market (P. Kaufman, *Trading Systems and Methods*).")

with st.expander("The first regime model (regime.py), used in the earlier walk-forward studies"):
    st.markdown(
        "Before the factor-selection approach, a simple three-pillar vote was used:\n\n"
        "| Pillar | Indicator | Vote |\n|---|---|---|\n"
        "| Trend | Index distance from its 21-day EMA | +1 above +1%, −1 below −1% |\n"
        "| Breadth | % of stocks above their 200-day EMA | +1 above 60%, −1 below 40% |\n"
        "| Breadth | 10-day advance/decline ratio | +1 above 1.2, −1 below 0.8 |\n"
        "| Volatility | Mean of India VIX and 20-day realised vol, each as a 1-year percentile | — |\n\n"
        "High-vol if the volatility percentile ≥ 80%; else Bullish if the vote ≥ +2; Bearish if ≤ −2; Low-vol if the "
        "volatility percentile ≤ 25%; otherwise Sideways. It used today's index members, not the point-in-time list."
    )
    old = data.data_csv("regimes.csv", index_col=0, parse_dates=True)["regime"].dropna()
    old_share = old.value_counts(normalize=True).reindex(REGIME_ORDER)
    st.html("".join(chip(f"{REGIME_LABEL[r]} {old_share[r]:.0%}") for r in REGIME_ORDER))
