import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import data
from core.style import AXIS, BENCHMARK, GOOD, CRITICAL, MUTED, STRATEGY, page_header, pct, show

page_header(
    "Methodology",
    "How the model works",
    "The strategy is a pipeline of six steps. Each step uses only information available at the close of the day "
    "it is computed, and every parameter below was fixed on 2015–2024 data before the out-of-sample test was run.",
)

m = data.meta()
cfg = m["config"]
pp = cfg["pair_params"]
rg = cfg["regime"]

flow = [("1 · Universe", "Point-in-time Nifty 200 members"), ("2 · Regime", "Trend, breadth, volatility → 5 states; gates entries"),
        ("3 · Pairs", "Cointegrated, same industry, re-formed every 6 months"),
        ("4 · Signal", "Percentile of 20-day relative strength"), ("5 · Portfolio", "10 slots, equal ₹ legs, futures costs"),
        ("6 · Evaluation", "One out-of-sample test vs the Nifty 200 TRI")]
st.html('<div class="flow">' + '<div class="arrow">→</div>'.join(
    f'<div class="box{" end" if i == len(flow) - 1 else ""}"><b>{t}</b><span>{d}</span></div>' for i, (t, d) in enumerate(flow))
    + "</div>")

# ---------------------------------------------------------------- 1 universe
st.header("1 · Universe: who could be traded on each day")
st.markdown(
    "The backtest only lets the strategy pick from stocks that were **members of the Nifty 200 on that date**. "
    "Membership is rebuilt from NSE's own month-end constituent files (Dec 2013 – Mar 2022) and from the index-review "
    "press releases since then. Former members that later dropped out are included while they were members, which "
    f"removes most **survivorship bias**. {m['n_missing_history']} historical members (delisted or merged, e.g. DHFL, "
    "HDFC Ltd) have no price history left and could not be included."
)
st.page_link("views/data_universe.py", label="Details: data sources and membership through time", icon=":material/database:")

# ---------------------------------------------------------------- 2 regime
st.header("2 · Market regime: when the strategy may trade")
c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    st.markdown(
        "Thirteen candidate indicators are computed every day. Each one is converted to a **percentile rank** within its "
        f"own trailing {rg['pct_window']}-day history (0 = lowest reading in 3 years, 1 = highest). No z-scores are used "
        "anywhere in the final model.\n\n"
        "On the training data only, each indicator is scored by how well it predicts the next 20 days (rank-IC), and a "
        "greedy filter keeps an indicator only if its correlation with every one already kept is ≤ 0.70. The survivors "
        "are averaged into two scores between 0 and 1:"
    )
    st.markdown(
        f"- **Direction score** = mean of {', '.join(f'`{f}`' for f in cfg['factors']['direction'])}\n"
        f"- **Volatility score** = mean of {', '.join(f'`{f}`' for f in cfg['factors']['volatility'])}"
    )
with c2:
    st.markdown("**Classification (first rule that matches wins)**")
    st.table(pd.DataFrame([
        ["1", "High vol", f"volatility ≥ {rg['high_vol']:.2f}", "✔ trades"],
        ["2", "Bullish", f"direction ≥ {rg['bull']:.2f}", "✔ trades"],
        ["3", "Bearish", f"direction ≤ {rg['bear']:.2f}", "✔ trades"],
        ["4", "Low vol", f"volatility ≤ {rg['low_vol']:.2f}", "✘ no new entries"],
        ["5", "Sideways", "everything else", "✔ trades"],
    ], columns=["#", "Regime", "Rule", "Pair strategy"]).set_index("#"))
    st.caption("Why block Low-vol? Short-term reversal profits are a reward for supplying liquidity, and that reward is "
               "smallest in calm markets (Nagel, 2012). Low-vol was also the worst regime in every earlier study here.")
st.page_link("views/regimes.py", label="Details: regime timeline, factor selection and per-regime performance",
             icon=":material/partly_cloudy_day:")

# ---------------------------------------------------------------- 3 pairs
st.header("3 · Pair formation: cointegration within an industry")
c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    st.markdown(
        "Every **1 January and 1 July**, all pairs of stocks from the **same NSE industry** that were index members the "
        "day before are tested on the trailing "
        f"**{pp['formation_days']} trading days (≈3 years)**. A pair qualifies if:"
    )
    st.latex(r"\text{ADF p-value of } \ln\!\left(\tfrac{P_A}{P_B}\right) \le 0.05"
             r"\qquad \rho(\text{daily returns}) \ge 0.5")
    st.markdown(
        "The **Augmented Dickey–Fuller test** checks whether the log price ratio keeps returning to a stable level "
        "(is *stationary*). If it does, the two stocks are cointegrated with a fixed 1 : 1 rupee hedge, so **no "
        "regression hedge ratio has to be estimated**, which removes a parameter that drifts. Stocks with a daily move "
        "above 40% inside the window are dropped as bad data. The pair list is then traded for the next six months "
        "only, so pair selection never sees the period it trades."
    )
with c2:
    st.markdown("**Why these choices**")
    st.markdown(
        "- Same industry: pairs share the same economic drivers, so a gap is more likely to be noise than news.\n"
        "- 3-year window: in the walk-forward studies it was the only formation window that worked in both 2014–21 and "
        "2022+ (§7 of the report), and every 2-year variant failed the final test.\n"
        "- Six-month refresh: relationships decay; the list is rebuilt from scratch twice a year."
    )

sched = data.pairs_by_period()
trd = data.trades()
ex_pair = trd["pair"].value_counts().index[0]
a, b = ex_pair.split(" / ")
ex_periods = sched[sched["pair"] == ex_pair].sort_values("period_start")
p0 = ex_periods.iloc[-1]
sig = data.pair_signal(a, b, int(pp["lookback"]), int(pp["pct_window"]))
form = sig.loc[:p0["period_start"] - pd.Timedelta(days=1), "log_ratio"].dropna().iloc[-int(pp["formation_days"]):]
trade_win = sig.loc[p0["period_start"]:p0["period_end"] - pd.Timedelta(days=1), "log_ratio"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=form.index, y=form, name="Formation window (tested)", line=dict(color=MUTED, width=1.5),
                         hovertemplate="%{y:.3f}<extra>ln(A/B)</extra>"))
fig.add_trace(go.Scatter(x=trade_win.index, y=trade_win, name="Following 6 months (traded)",
                         line=dict(color=STRATEGY, width=2), hovertemplate="%{y:.3f}<extra>ln(A/B)</extra>"))
fig.add_hline(y=form.mean(), line=dict(color=AXIS, width=1), annotation_text="formation mean",
              annotation_font=dict(size=11, color=MUTED))
fig.update_layout(title=f"Example: {a} / {b}, log price ratio before and after the {p0['period_start']:%b %Y} test",
                  yaxis_title="ln(P_A / P_B)", hovermode="x unified")
show(fig, height=360)
st.caption(f"ADF p = {p0['adf_p']:.3f}, daily-return correlation {p0['corr']:.2f}. "
           f"{a} / {b} is the pair the strategy traded most often ({(trd['pair'] == ex_pair).sum()} trades across "
           f"{len(ex_periods)} half-year periods). The grey part is what the ADF test sees; the blue part is the period "
           "the pair is then allowed to trade.")

# ---------------------------------------------------------------- 4 signal
st.header("4 · Signal: relative strength, ranked as a percentile")
c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    L, W = int(pp["lookback"]), int(pp["pct_window"])
    st.markdown(f"**Relative strength** is how much stock A has out-performed stock B over the last {L} trading days:")
    st.latex(rf"RS_t = \ln\frac{{P_{{A,t}}}}{{P_{{B,t}}}} - \ln\frac{{P_{{A,t-{L}}}}}{{P_{{B,t-{L}}}}}"
             rf"\;=\; r^{{({L}d)}}_A - r^{{({L}d)}}_B")
    st.markdown(f"It is then ranked against its own last {W} days, giving a number between 0 and 1:")
    st.latex(rf"p_t = \text{{rank of }} RS_t \text{{ among }} \{{RS_{{t-{W - 1}}},\dots,RS_t\}} \;/\; {W}")
    st.markdown(
        f"- **Enter** when $p_t \\ge {pp['entry_pct']:.2f}$: A has run unusually far ahead → **short A, buy B**.\n"
        f"- **Enter** when $p_t \\le {1 - pp['entry_pct']:.2f}$: A has fallen unusually far behind → **buy A, short B**.\n"
        "- **Exit** when $p_t$ crosses 0.5 (relative strength back to its median), or after **15 calendar days**, "
        "whichever comes first."
    )
with c2:
    st.markdown("**Why a percentile and not a z-score?**")
    st.markdown(
        "- A percentile does not assume returns are normally distributed; one extreme day cannot distort it.\n"
        "- It is scale-free: a quiet utility pair and a volatile PSU-bank pair are judged on the same 0–1 scale.\n"
        "- It measures *recent divergence*, not the distance from a long-run mean of the spread level, so it reacts to "
        "the relationship as it is now.\n\n"
        "In the earlier studies, trading *with* relative strength (momentum) lost heavily (Sharpe −1.31), while "
        "trading *against* it (reversion) worked, which confirms the direction of the bet."
    )

recent = sig.loc[sig.index >= sig.index[-1] - pd.Timedelta(days=550)]
pt = trd[(trd["pair"] == ex_pair) & (trd["entry_date"] >= recent.index[0])]
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.06)
fig.add_trace(go.Scatter(x=recent.index, y=recent["rs"], name=f"{L}-day relative strength (A − B)",
                         line=dict(color=MUTED, width=1.5), hovertemplate="%{y:.1%}<extra>RS</extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=recent.index, y=recent["rs_pct"], name="Percentile of RS (signal)",
                         line=dict(color=STRATEGY, width=2), hovertemplate="%{y:.2f}<extra>percentile</extra>"), row=2, col=1)
for y in (pp["entry_pct"], 1 - pp["entry_pct"], 0.5):
    fig.add_hline(y=y, line=dict(color=AXIS, width=1), row=2, col=1)
for _, t in pt.iterrows():
    win = t["ret"] > 0
    fig.add_vrect(x0=t["entry_date"], x1=t["exit_date"], fillcolor=GOOD if win else CRITICAL, opacity=0.10,
                  line_width=0, row=2, col=1)
fig.update_yaxes(tickformat=".0%", title_text="RS", row=1, col=1)
fig.update_yaxes(range=[-0.02, 1.02], title_text="Percentile", tickvals=[0, 1 - pp["entry_pct"], 0.5, pp["entry_pct"], 1],
                 row=2, col=1)
fig.update_layout(title=f"{a} / {b}: the signal over the last 18 months", hovermode="x unified")
show(fig, height=460)
st.caption(f"Bottom panel: lines at {pp['entry_pct']:.0%} and {1 - pp['entry_pct']:.0%} are the entry triggers, 50% is the "
           "exit. Shaded spans are actual trades — green ones made money, red ones lost.")
st.page_link("views/pair_explorer.py", label="Explore any pair the same way", icon=":material/query_stats:")

# ---------------------------------------------------------------- 5 portfolio
st.header("5 · Portfolio, timing and costs")
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown(
        "- **10 slots.** Each leg is sized at equity ÷ 10, so a full book is 100% long and 100% short — market-neutral by "
        "construction.\n"
        "- **Ranking.** When more signals fire than there are free slots, the most extreme percentiles go first.\n"
        "- **Concentration limit.** A stock can be in at most 2 open pairs.\n"
        "- **Regime gate.** New entries are allowed in every regime except Low-vol; open trades are always managed to exit.\n"
        "- **No look-ahead.** Signals use closes up to day *t*; orders fill at the close of *t + 1*. Time exits fill on the "
        "close the holding period ends.\n"
        "- **Start capital** ₹1 crore (the scale does not affect percentage returns)."
    )
with c2:
    r = m["cost_rates_ex_slippage"]
    slip = 0.0005
    fut_rt = r["futures"]["buy"] + r["futures"]["sell"] + 2 * slip
    cash_rt = r["cash"]["buy"] + r["cash"]["sell"] + 2 * slip
    st.markdown("**Costs charged in the backtest** (per ₹100 of one leg, round trip)")
    st.table(pd.DataFrame([
        ["Stock futures (used)", f"₹{r['futures']['buy'] * 100:.4f}", f"₹{r['futures']['sell'] * 100:.4f}", "₹0.10",
         f"₹{fut_rt * 100:.3f}"],
        ["Cash + stock borrowing", f"₹{r['cash']['buy'] * 100:.4f}", f"₹{r['cash']['sell'] * 100:.4f}", "₹0.10",
         f"₹{cash_rt * 100:.3f} + borrow"],
    ], columns=["Route", "Buy", "Sell", "Slippage", "Round trip"]).set_index("Route"))
    st.caption("Slippage is 5 bps on each side. Futures also pay a roll cost when a position crosses a monthly expiry; "
               "the cash route pays 8% a year to borrow the short leg. Futures: STT 0.02% on the sell side, exchange fee + GST, SEBI fee, 0.002% stamp duty on the buy side. "
               "Cash delivery: STT 0.1% on both sides and 0.015% stamp duty, plus stock-borrow cost for the short. "
               f"Cash + SLB costs about {cash_rt / fut_rt:.1f}× as much per round trip before borrowing, and a 15-day "
               f"borrow adds another ≈{0.08 * 15 / 365 * 100:.2f}% on the short leg — enough to turn the strategy negative.")

# ---------------------------------------------------------------- 6 selection
st.header("6 · How the parameters were chosen")
st.markdown(
    "Selection used **2015–2024 only**, in two stages, and the winner was run **once** on 2025-01 → 2026-10.\n\n"
    "1. **Regime definition** — 1,080 candidates (3 percentile windows × 4 threshold pairs × 3 high-vol × 2 low-vol "
    "cut-offs × 3 confirmation lags × 5 index profiles).\n"
    "2. **Pair strategy** — 216 candidates: signal type (RS percentile or Kaufman's stress indicator) × RS lookback "
    "{5, 10, 20} × percentile window {126, 252} × entry {90th, 95th} × median exit {on, off} × formation window "
    "{2, 3 years} × regime mode {Sideways + High-vol, all but Low-vol, all}.\n\n"
    "Each candidate is scored by the **average percentile rank** of three consistency measures — the worse half's "
    "Sharpe (2015–19 vs 2020–24), the worse half's return-to-drawdown ratio, and the full-period Sharpe — and needs at "
    "least 100 training trades. The score is then **averaged with its grid neighbours** (a plateau), so a broad, stable "
    "region beats a lucky isolated setting."
)
g = data.research_csv("pit_pairs/stage2_pair_grid.csv")
g["chosen"] = np.arange(len(g)) == 0
fig = go.Figure()
for chosen, color, name, size in ((False, "#9ec5f4", "Other candidates", 8), (True, STRATEGY, "Selected configuration", 14)):
    d = g[g["chosen"] == chosen]
    fig.add_trace(go.Scatter(x=d["train_sharpe"], y=d["plateau_score"], mode="markers", name=name,
                             marker=dict(color=color, size=size, line=dict(color="white", width=2)),
                             customdata=np.c_[d["mode"], d["label"], d["n_trades_train"]],
                             hovertemplate="%{customdata[1]}<br>%{customdata[0]}<br>train Sharpe %{x:.2f} · "
                                           "plateau score %{y:.2f} · %{customdata[2]} trades<extra></extra>"))
fig.update_layout(title="Stage 2: the 216 pair-strategy candidates on training data",
                  xaxis_title="Training Sharpe 2015–2024", yaxis_title="Plateau score (selection criterion)")
show(fig, height=380)
st.caption("The pick is not the highest training Sharpe: it sits in the most consistent neighbourhood. Settings with a "
           "higher training Sharpe but a weaker plateau were passed over on purpose.")

st.subheader("The frozen configuration")
st.table(pd.DataFrame([
    ["Universe", "Point-in-time Nifty 200 members"],
    ["Pair test", f"ADF p ≤ 0.05 on ln(A/B), return correlation ≥ 0.5, same industry, {pp['formation_days']}-day window, every 6 months"],
    ["Signal", f"Percentile of {pp['lookback']}-day relative strength within {pp['pct_window']} days"],
    ["Entry / exit", f"{float(pp['entry_pct']):.0%} / {1 - float(pp['entry_pct']):.0%} percentile; exit at median or 15 calendar days"],
    ["Regimes traded", cfg["pair_mode"].replace("LOW_VOL", "Low-vol")],
    ["Regime model", f"{rg['pct_window']}-day percentiles; bull ≥ {rg['bull']}, bear ≤ {rg['bear']}, "
                     f"high-vol ≥ {rg['high_vol']}, low-vol ≤ {rg['low_vol']}"],
    ["Portfolio", "10 slots, equal ₹ per leg, ≤ 2 pairs per stock"],
    ["Costs", "NSE stock futures + 5 bps slippage per side per leg"],
], columns=["Component", "Setting"]).set_index("Component"))
