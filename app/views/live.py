import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data
from core.style import AXIS, CRITICAL, GOOD, MUTED, REGIME_COLORS, REGIME_LABEL, STRATEGY, page_header, show

m = data.meta()
pp = m["config"]["pair_params"]
E = float(pp["entry_pct"])
asof = pd.Timestamp(m["data_end"])

page_header(
    "Results · live",
    "Live signal monitor",
    f"What the frozen strategy sees at the close of <b>{asof:%A %d %B %Y}</b>, the last day in the data: the market "
    "regime, the signal on every pair in the current list, and the positions it is holding.",
)

reg = data.regime_daily()
today = reg["regime"].dropna().iloc[-1]
live = data.bundle_csv("live_signals.csv")
sched = data.pairs_by_period()
cur = sched[sched["period_start"] == sched["period_start"].max()].iloc[0]
trd = data.trades()
open_pos = trd[trd["exit_reason"] == "end_of_test"].copy()

live["signal"] = np.select([live["rs_pct"] >= E, live["rs_pct"] <= 1 - E],
                           [live["stock_a"].radd("Short ") + " / buy " + live["stock_b"],
                            live["stock_a"].radd("Buy ") + " / short " + live["stock_b"]], "No signal")
live["pair"] = live["stock_a"] + " / " + live["stock_b"]
live["to_trigger"] = np.where(live["signal"] != "No signal", 0.0,
                              np.minimum(E - live["rs_pct"], live["rs_pct"] - (1 - E)))
n_sig = int((live["signal"] != "No signal").sum())

c = st.columns(4)
c[0].metric("Regime", REGIME_LABEL[today], border=True)
c[1].metric("New entries", "Allowed" if today != "LOW_VOL" else "Blocked (Low-vol)", border=True)
c[2].metric("Pairs on the current list", f"{len(live)}", f"formed {cur['period_start']:%d %b %Y}", delta_color="off",
            border=True)
c[3].metric("Pairs at an extreme today", f"{n_sig}", help=f"Percentile ≥ {E:.0%} or ≤ {1 - E:.0%}", border=True)

st.header("Signal on every live pair")
lv = live.sort_values("rs_pct")
colors = np.where(lv["signal"] != "No signal", STRATEGY, "#9ec5f4")
fig = go.Figure()
fig.add_vrect(x0=E, x1=1.02, fillcolor=STRATEGY, opacity=0.07, line_width=0)
fig.add_vrect(x0=-0.02, x1=1 - E, fillcolor=STRATEGY, opacity=0.07, line_width=0)
fig.add_vline(x=0.5, line=dict(color=AXIS, width=1))
fig.add_trace(go.Scatter(x=lv["rs_pct_prev"], y=lv["pair"], mode="markers", name="Previous day",
                         marker=dict(color="#c3c2b7", size=8), hovertemplate="previous day %{x:.2f}<extra>%{y}</extra>"))
fig.add_trace(go.Scatter(x=lv["rs_pct"], y=lv["pair"], mode="markers", name=f"{asof:%d %b}",
                         marker=dict(color=colors, size=12, line=dict(color="white", width=2)),
                         customdata=lv["signal"], hovertemplate="%{x:.2f} · %{customdata}<extra>%{y}</extra>"))
fig.add_annotation(x=(E + 1) / 2, y=1.0, yref="paper", text="short A / buy B", showarrow=False, yanchor="bottom",
                   font=dict(size=11, color=MUTED))
fig.add_annotation(x=(1 - E) / 2, y=1.0, yref="paper", text="buy A / short B", showarrow=False, yanchor="bottom",
                   font=dict(size=11, color=MUTED))
fig.update_layout(title="", xaxis_title=f"Percentile of 20-day relative strength (A vs B)", margin=dict(t=50))
fig.update_xaxes(range=[-0.02, 1.02], tickvals=[0, 1 - E, 0.5, E, 1], tickformat=".0%", showgrid=True, gridcolor="#e1e0d9")
fig.update_yaxes(showgrid=False)
show(fig, height=120 + 24 * len(lv))
st.caption("Each row is a pair on the current list (stock A / stock B). Dark dots are in an entry zone today; grey dots "
           "show the previous day, so you can see which way each pair is moving. A signal becomes an order for the next "
           "close only if a slot is free (10 at most) and neither stock is already in two open pairs.")

st.dataframe(live.sort_values("to_trigger")[["pair", "industry", "rs_pct", "rs_pct_prev", "signal", "to_trigger"]].assign(
    rs_pct=lambda x: x["rs_pct"] * 100, rs_pct_prev=lambda x: x["rs_pct_prev"] * 100, to_trigger=lambda x: x["to_trigger"] * 100),
    hide_index=True, column_config={
        "pair": "Pair", "industry": "Industry",
        "rs_pct": st.column_config.ProgressColumn("Percentile today", min_value=0, max_value=100, format="%.0f"),
        "rs_pct_prev": st.column_config.NumberColumn("Previous day", format="%.0f"),
        "signal": "Signal", "to_trigger": st.column_config.NumberColumn("Points to a trigger", format="%.0f")})

st.header("Positions open at the data cut-off")
if open_pos.empty:
    st.info("The book was flat on the last day.")
else:
    st.markdown(f"**{len(open_pos)} pairs** were open at the close of {asof:%d %b %Y}. The P&L below marks them at that "
                "close, including the cost of closing them.")
    st.dataframe(open_pos.sort_values("entry_date")[["pair", "long_leg", "short_leg", "entry_date", "entry_regime", "days",
                                                      "entry_rs_pct", "ret"]].assign(
        entry_regime=lambda x: x["entry_regime"].map(REGIME_LABEL), entry_rs_pct=lambda x: x["entry_rs_pct"] * 100,
        ret=lambda x: x["ret"] * 100),
        hide_index=True, column_config={
            "pair": "Pair", "long_leg": "Long", "short_leg": "Short",
            "entry_date": st.column_config.DateColumn("Entered", format="D MMM YYYY"),
            "entry_regime": "Regime at entry", "days": st.column_config.NumberColumn("Days held", format="%d"),
            "entry_rs_pct": st.column_config.NumberColumn("Percentile at signal", format="%.0f"),
            "ret": st.column_config.NumberColumn("Net P&L so far", format="%+.2f%%")})

st.header("Most recent closed trades")
done = trd[trd["exit_reason"] != "end_of_test"].sort_values("exit_date", ascending=False).head(15)
st.dataframe(done[["exit_date", "pair", "long_leg", "short_leg", "days", "ret", "exit_reason"]].assign(
    ret=lambda x: x["ret"] * 100,
    exit_reason=lambda x: x["exit_reason"].map({"time": "15-day limit", "revert": "back to median"})),
    hide_index=True, column_config={
        "exit_date": st.column_config.DateColumn("Closed", format="D MMM YYYY"), "pair": "Pair", "long_leg": "Long",
        "short_leg": "Short", "days": st.column_config.NumberColumn("Days", format="%d"),
        "ret": st.column_config.NumberColumn("Net return", format="%+.2f%%"), "exit_reason": "Exit"})

with st.expander("How to refresh this page with new data"):
    st.markdown(
        "The dashboard reads a frozen snapshot. To roll it forward, re-download prices and rebuild the bundle — the "
        "strategy settings do not change:\n"
        "```\n"
        ".venv\\Scripts\\python.exe universe.py          # latest Nifty 200 prices, index and VIX\n"
        ".venv\\Scripts\\python.exe pit_pair_system.py   # re-runs selection on 2015-24 (same pick) and the test\n"
        ".venv\\Scripts\\python.exe app\\build_data.py    # re-exports this dashboard's data\n"
        "```\n"
        "This is a research tool, not investment advice: the strategy has a short and statistically thin "
        "out-of-sample record."
    )
