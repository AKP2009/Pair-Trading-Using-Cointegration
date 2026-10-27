import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import data
from core.style import AXIS, BENCHMARK, CRITICAL, GOOD, MUTED, REGIME_LABEL, STRATEGY, chip, page_header, pct, show

page_header(
    "Results · pair explorer",
    "Pair explorer",
    "Pick any pair the final strategy traded and see the whole mechanism at work: the two prices, their log ratio, "
    "the relative-strength percentile that drives entries and exits, and every trade it took.",
)

m = data.meta()
pp = m["config"]["pair_params"]
L, W, E = int(pp["lookback"]), int(pp["pct_window"]), float(pp["entry_pct"])
sched = data.pairs_by_period()
trd = data.trades()

n_tr = trd["pair"].value_counts()
pairs = sorted(sched["pair"].unique(), key=lambda p: (-n_tr.get(p, 0), p))
industry = sched.groupby("pair")["industry"].first()
recent = trd.loc[trd["period"] == "Out-of-sample", "pair"].value_counts()
default = st.session_state.pop("explorer_pair", None) or (recent.index[0] if len(recent) else pairs[0])
c1, c2 = st.columns([2, 1])
pair = c1.selectbox("Pair", pairs, index=pairs.index(default) if default in pairs else 0,
                    format_func=lambda p: f"{p}  ·  {industry[p]}  ·  {n_tr.get(p, 0)} trades")
span = c2.segmented_control("Window", ["Full history", "Last 3 years", "Last year"], default="Last 3 years",
                            key="px_span") or "Full history"

a, b = pair.split(" / ")
sig = data.pair_signal(a, b, L, W)
periods = sched[sched["pair"] == pair].sort_values("period_start")
t = trd[trd["pair"] == pair].sort_values("entry_date")

c = st.columns(4)
c[0].metric("Half-years listed", f"{len(periods)}", help="Formation periods in which the pair passed the cointegration test",
            border=True)
c[1].metric("Trades", f"{len(t)}", border=True)
c[2].metric("Win rate", f"{(t['net_pnl'] > 0).mean():.0%}" if len(t) else "–", border=True)
c[3].metric("Contribution", pct(t["net_pnl"].sum() / 10_000_000, 2) if len(t) else "–",
            help="Net P&L as a share of the ₹1 crore starting capital", border=True)
st.html(f'<span class="muted"><b>{a}</b>: {data.company(a)} &nbsp;·&nbsp; <b>{b}</b>: {data.company(b)} '
        f'&nbsp;·&nbsp; industry: {industry[pair]}</span>')
st.html("Listed in: " + "".join(chip(f"{p:%b %Y}") for p in periods["period_start"]))

end = sig.index[-1]
start = {"Full history": sig["log_ratio"].first_valid_index(), "Last 3 years": end - pd.DateOffset(years=3),
         "Last year": end - pd.DateOffset(years=1)}[span]
s = sig.loc[start:]
tw = t[t["exit_date"] >= start]

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.34, 0.28, 0.38], vertical_spacing=0.06,
                    subplot_titles=(f"Prices, rebased to 100 at {s.index[0]:%b %Y}", "Log price ratio ln(A / B)",
                                    f"Signal: percentile of {L}-day relative strength (window {W} days)"))
base = s[["price_a", "price_b"]].dropna().iloc[0]
fig.add_trace(go.Scatter(x=s.index, y=s["price_a"] / base["price_a"] * 100, name=a, line=dict(color=STRATEGY, width=1.6),
                         hovertemplate="%{y:.1f}<extra>" + a + "</extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=s.index, y=s["price_b"] / base["price_b"] * 100, name=b, line=dict(color=BENCHMARK, width=1.6),
                         hovertemplate="%{y:.1f}<extra>" + b + "</extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=s.index, y=s["log_ratio"], name="ln(A/B)", line=dict(color="#3a3936", width=1.4),
                         showlegend=False, hovertemplate="%{y:.3f}<extra>ln(A/B)</extra>"), row=2, col=1)
fig.add_trace(go.Scatter(x=s.index, y=s["rs_pct"], name="RS percentile", line=dict(color=STRATEGY, width=1.4),
                         showlegend=False, hovertemplate="%{y:.2f}<extra>RS percentile</extra>"), row=3, col=1)
for _, p in periods.iterrows():
    x0, x1 = max(p["period_start"], s.index[0]), min(p["period_end"], end)
    if x1 > x0:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=STRATEGY, opacity=0.07, line_width=0, row=2, col=1)
for y in (E, 1 - E):
    fig.add_hline(y=y, line=dict(color=AXIS, width=1), row=3, col=1)
fig.add_hline(y=0.5, line=dict(color=AXIS, width=1, dash="dot"), row=3, col=1)
for won in (True, False):
    x = tw[(tw["ret"] > 0) == won]
    if x.empty:
        continue
    color = GOOD if won else CRITICAL
    label = "Winning trade" if won else "Losing trade"
    entry_y = sig["rs_pct"].reindex(x["entry_date"]).to_numpy()
    exit_y = sig["rs_pct"].reindex(x["exit_date"]).to_numpy()
    fig.add_trace(go.Scatter(
        x=x["entry_date"], y=entry_y, mode="markers", name=f"{label} — entry", legendgroup=label,
        marker=dict(symbol="circle", size=10, color=color, line=dict(color="white", width=2)),
        customdata=np.c_[x["long_leg"], x["short_leg"], x["ret"] * 100, x["exit_date"].dt.strftime("%d %b %Y")],
        hovertemplate="entry %{x|%d %b %Y}<br>long %{customdata[0]} / short %{customdata[1]}<br>"
                      "net %{customdata[2]:+.2f}% · exit %{customdata[3]}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x["exit_date"], y=exit_y, mode="markers", name=f"{label} — exit", legendgroup=label, showlegend=False,
        marker=dict(symbol="x-thin", size=10, color=color, line=dict(color=color, width=2)),
        hovertemplate="exit %{x|%d %b %Y}<extra></extra>"), row=3, col=1)
fig.update_yaxes(title_text="Rebased", row=1, col=1)
fig.update_yaxes(title_text="ln(A/B)", row=2, col=1)
fig.update_yaxes(title_text="Percentile", range=[-0.03, 1.03], tickvals=[0, 1 - E, 0.5, E, 1], row=3, col=1)
fig.update_layout(hovermode="x unified", margin=dict(t=60))
fig.update_annotations(font=dict(size=13, color="#0b0b0b"))
show(fig, height=760)
if t.empty is False and tw.empty:
    st.info(f"This pair's trades are all before {start:%b %Y}. Switch the window to **Full history** to see them.",
            icon=":material/info:")
st.caption(f"Shaded bands in the middle panel are the half-years when the pair was on the cointegrated list (only then "
           f"can it trade). In the bottom panel, entries happen above {E:.0%} (short {a}, buy {b}) or below "
           f"{1 - E:.0%} (buy {a}, short {b}); dots are entries, crosses exits. A pair only trades when it is listed "
           "and the regime is not Low-vol, so not every crossing becomes a trade.")

st.subheader(f"Trades in {pair}")
if t.empty:
    st.info("This pair was listed but never triggered a trade: its signal never reached an extreme while a slot was free "
            "and the regime allowed entries.")
else:
    st.dataframe(t.sort_values("entry_date", ascending=False).assign(
        entry_regime=t["entry_regime"].map(REGIME_LABEL), gross_ret=t["gross_ret"] * 100, ret=t["ret"] * 100,
        entry_rs_pct=t["entry_rs_pct"] * 100)[
        ["entry_date", "exit_date", "long_leg", "short_leg", "entry_regime", "entry_rs_pct", "days", "gross_ret", "ret",
         "exit_reason", "period"]],
        hide_index=True, column_config={
            "entry_date": st.column_config.DateColumn("Entry", format="D MMM YYYY"),
            "exit_date": st.column_config.DateColumn("Exit", format="D MMM YYYY"),
            "long_leg": "Long", "short_leg": "Short", "entry_regime": "Regime",
            "entry_rs_pct": st.column_config.NumberColumn("RS percentile at signal", format="%.0f"),
            "days": st.column_config.NumberColumn("Days", format="%d"),
            "gross_ret": st.column_config.NumberColumn("Gross", format="%+.2f%%"),
            "ret": st.column_config.NumberColumn("Net", format="%+.2f%%"),
            "exit_reason": "Exit", "period": "Sample"})

with st.expander("Cointegration test results for each half-year"):
    st.dataframe(periods[["period_start", "period_end", "adf_p", "corr", "ratio_vol"]].assign(ratio_vol=lambda x: x["ratio_vol"] * 100),
                 hide_index=True, column_config={
                     "period_start": st.column_config.DateColumn("Trades from", format="D MMM YYYY"),
                     "period_end": st.column_config.DateColumn("Until", format="D MMM YYYY"),
                     "adf_p": st.column_config.NumberColumn("ADF p-value", format="%.4f"),
                     "corr": st.column_config.NumberColumn("Return correlation", format="%.2f"),
                     "ratio_vol": st.column_config.NumberColumn("Daily ratio vol", format="%.2f%%")})
    st.caption(f"Each test uses the {pp['formation_days']} trading days before the period starts. p ≤ 0.05 is required.")
