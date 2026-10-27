import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import charts, data
from core.style import (BENCHMARK, CRITICAL, GOOD, MUTED, REGIME_COLORS, REGIME_LABEL, STRATEGY, chip, page_header,
                        pct, show)

page_header(
    "Results",
    "Final strategy results",
    "The frozen point-in-time pair strategy, judged against the Nifty 200 Total Return Index. The goal, set before "
    "the test: over 2025-01-01 → 2026-10-23, beat the TRI on all four of total return, Sharpe ratio, volatility and "
    "maximum drawdown.",
)

m = data.meta()
pp = m["config"]["pair_params"]
card = data.scorecard()
st.html(chip(f"RS percentile · L{pp['lookback']} · window {pp['pct_window']}") + chip(f"entry {float(pp['entry_pct']):.0%} / exit at median")
        + chip("15-day max hold") + chip(f"{int(pp['formation_days']) // 252}-year cointegration window")
        + chip("all regimes except Low-vol") + chip("NSE futures costs + 5 bps"))

# ---------------------------------------------------------------- scorecard
st.header("Scorecard vs the Nifty 200 TRI")
names = {"OUT-OF-SAMPLE 2025..latest": "Out-of-sample 2025 → Oct 2026", "train 2015-2024": "Training 2015–2024",
         "train half 2015-19": "Training half 2015–19", "train half 2020-24": "Training half 2020–24"}
rows = []
for _, r in card.iterrows():
    rows.append({
        "Period": names.get(r["period"], r["period"]),
        "Return": f"{pct(r['total_return'])} vs {pct(r['tri_total_return'])} {'✔' if r['1_higher_return'] else '✘'}",
        "Sharpe": f"{r['sharpe']:.2f} vs {r['tri_sharpe']:.2f} {'✔' if r['2_higher_sharpe'] else '✘'}",
        "Volatility": f"{pct(r['vol'], sign=False)} vs {pct(r['tri_vol'], sign=False)} {'✔' if r['3_lower_vol'] else '✘'}",
        "Max drawdown": f"{pct(r['max_drawdown'])} vs {pct(r['tri_maxdd'])} {'✔' if r['4_lower_maxdd'] else '✘'}",
        "Beta": f"{r['beta']:.2f}",
        "Criteria met": f"{int(r['criteria_met'])} / 4",
    })
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption("Each cell reads strategy vs TRI. ✔ = goal criterion met (higher return, higher Sharpe, lower volatility, "
           "smaller drawdown). The out-of-sample row is the result; the training rows show how the strategy behaved on "
           "the data it was chosen on — it always wins on risk, never on return in a rising market.")

# ---------------------------------------------------------------- equity
st.header("Equity curves")
tab1, tab2 = st.tabs(["Out-of-sample (the test)", "Full history 2015 → 2026"])
with tab1:
    show(charts.equity_and_drawdown(data.equity_oos(), "Growth of ₹1 from 1 Jan 2025"), height=470)
with tab2:
    full = data.equity_full()
    show(charts.equity_and_drawdown(full, "Growth of ₹1 from 1 Jan 2015 (shaded: out-of-sample)",
                                    shade_from="2025-01-01"), height=500)
    st.caption("Over the full history the index compounds far faster: the strategy is market-neutral (beta ≈ 0.03) and "
               "earns a small spread return. Its value is in the risk column — about half the index's drawdown.")

yr = data.research_csv("pit_pairs/yearly_returns.csv", index_col=0)
yr.index = yr.index.astype(str)
show(charts.grouped_bars(yr, "Calendar-year returns", text=True, text_fmt=".1%"), height=400)
beat = yr.index[yr["PIT pair strategy"] > yr["Nifty 200 TRI"]].tolist()
st.caption(f"{m['data_end'][:4]} runs to {pd.Timestamp(m['data_end']):%d %b}. The strategy beat the TRI in {len(beat)} of "
           f"{len(yr)} calendar years ({', '.join(beat)}) — including both out-of-sample years; in 2026 it lost less than "
           "the index. The strategy starts trading in July 2015, once three years of history exist for the first "
           "cointegration test.")
mt = data.research_csv("pit_pairs/monthly_returns.csv", index_col=0)
show(charts.monthly_heatmap(mt, "Monthly returns of the strategy (blank = no position that month)"), height=480)

# ---------------------------------------------------------------- book usage
st.header("How much of the book is used")
d = data.daily().loc["2015-07-01":]
c1, c2, c3 = st.columns(3)
c1.metric("Average open pairs", f"{d['n_open'].mean():.1f} of 10", border=True)
c2.metric("Average gross exposure", f"{d['gross_exposure'].mean():.0%}", help="Long + short notional ÷ equity", border=True)
c3.metric("Days with no position", f"{(d['n_open'] == 0).mean():.0%}", border=True)
monthly_open = d["n_open"].resample("ME").mean()
fig = go.Figure(go.Bar(x=monthly_open.index, y=monthly_open, marker_color=STRATEGY, marker_line_width=0,
                       hovertemplate="%{x|%b %Y}: %{y:.1f} pairs<extra></extra>"))
fig.update_layout(title="Average number of open pairs per month (maximum 10)", yaxis_title="Open pairs", bargap=0.15)
fig.update_yaxes(range=[0, 10])
show(fig, height=300)
st.caption("Signals are scarce, and Low-vol spells switch new entries off, so the book is rarely full. That is why "
           "returns are small even when the risk-adjusted numbers are good.")

# ---------------------------------------------------------------- trade stats
st.header("Trade statistics")
trd = data.trades()
which = st.segmented_control("Trades", ["All", "Training", "Out-of-sample"], default="Out-of-sample", key="res_trades") or "All"
t = trd if which == "All" else trd[trd["period"] == which]
wins, losses = t.loc[t["net_pnl"] > 0, "net_pnl"].sum(), -t.loc[t["net_pnl"] < 0, "net_pnl"].sum()
c = st.columns(6)
c[0].metric("Trades", f"{len(t):,}", border=True)
c[1].metric("Win rate", f"{(t['net_pnl'] > 0).mean():.1%}", border=True)
c[2].metric("Avg net return", pct(t["ret"].mean(), 2), help="Net P&L ÷ the notional of one leg", border=True)
c[3].metric("Profit factor", f"{wins / losses:.2f}" if losses > 0 else "–", help="Gross winnings ÷ gross losses", border=True)
c[4].metric("Avg holding", f"{t['days'].mean():.1f} days", help="Trading days", border=True)
c[5].metric("Avg cost / trade", pct(t["costs"].div(t["leg_notional"]).mean(), 2, sign=False),
            help="All charges + slippage, as a share of one leg's notional", border=True)

c1, c2 = st.columns([1.3, 1], gap="large")
with c1:
    fig = go.Figure()
    for won, color, name in ((True, GOOD, "Winning trades"), (False, CRITICAL, "Losing trades")):
        x = t.loc[(t["ret"] > 0) == won, "ret"]
        fig.add_trace(go.Histogram(x=x, name=name, marker_color=color, opacity=0.85, xbins=dict(size=0.01),
                                   marker_line=dict(color="white", width=1), hovertemplate="%{x}: %{y} trades<extra></extra>"))
    fig.add_vline(x=t["ret"].mean(), line=dict(color=MUTED, width=1), annotation_text=f"mean {pct(t['ret'].mean(), 2)}",
                  annotation_font=dict(size=11, color=MUTED))
    fig.update_layout(barmode="overlay", title="Distribution of net trade returns", xaxis_title="Net return per leg notional",
                      yaxis_title="Trades")
    fig.update_xaxes(tickformat=".0%")
    show(fig, height=360)
with c2:
    ex = t["exit_reason"].map({"time": "15-day time limit", "revert": "RS back to median", "end_of_test": "Open at data end",
                               "missing_price": "Missing price"}).fillna(t["exit_reason"])
    tcfg = {"trades": "Trades", "win": st.column_config.NumberColumn("Win rate", format="%.0f%%"),
            "avg": st.column_config.NumberColumn("Avg net return", format="%+.2f%%")}
    g = t.assign(exit=ex).groupby("exit").agg(trades=("ret", "size"), win=("net_pnl", lambda s: (s > 0).mean()),
                                              avg=("ret", "mean")).sort_values("trades", ascending=False)
    st.markdown("**How trades end**")
    st.dataframe(g.assign(win=g["win"] * 100, avg=g["avg"] * 100).reset_index(), hide_index=True,
                 column_config={"exit": "Exit", **tcfg})
    r = t.groupby("entry_regime").agg(trades=("ret", "size"), win=("net_pnl", lambda s: (s > 0).mean()), avg=("ret", "mean"))
    r.index = r.index.map(REGIME_LABEL)
    st.markdown("**By regime at entry**")
    st.dataframe(r.assign(win=r["win"] * 100, avg=r["avg"] * 100).reset_index(), hide_index=True,
                 column_config={"entry_regime": "Regime", **tcfg})
    st.caption("Exits at the median are the winners; trades that hit the 15-day limit had not reverted and mostly lose.")

# ---------------------------------------------------------------- robustness
st.header("Robustness: does the result survive small changes?")
rob = data.research_csv("pit_pairs/robustness.csv")
labels = {"frozen strategy": "As selected (futures, 5 bps slippage)", "slippage 10 bps/side": "Slippage 10 bps per side",
          "slippage 20 bps/side": "Slippage 20 bps per side", "cash delivery + 8% SLB borrow": "Cash delivery + 8% stock borrow",
          "benchmark TRI at 2.5%/yr dividends": "Harder benchmark: TRI with 2.5% dividends"}
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    rb = pd.DataFrame({
        "Check": rob["check"].map(labels).fillna(rob["check"]),
        "Return": rob["total_return"] * 100, "Sharpe": rob["sharpe"], "Max DD": rob["max_drawdown"] * 100,
        "TRI return": rob["tri_total_return"] * 100, "Criteria met": rob["criteria_met"].astype(int).astype(str) + " / 4"})
    st.dataframe(rb, hide_index=True, column_config={
        "Return": st.column_config.NumberColumn(format="%+.1f%%"), "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Max DD": st.column_config.NumberColumn(format="%.1f%%"), "TRI return": st.column_config.NumberColumn(format="%+.1f%%")})
    st.markdown(
        "- **Costs are the weak point.** Doubling slippage to 10 bps leaves only +0.4%; at 20 bps the strategy loses "
        "11%. Cash-market delivery with stock borrowing is worse still. It is only viable through stock futures with "
        "good execution.\n"
        "- **The benchmark is not the issue.** Even crediting the TRI with a 2.5% dividend yield, all four criteria hold."
    )
with c2:
    nb = data.research_csv("pit_pairs/top20_oos_diagnostic.csv")
    nb["window"] = np.where(nb["label"].str.endswith("f756"), "3-year cointegration window", "2-year cointegration window")
    fig = go.Figure()
    for w, color in (("3-year cointegration window", STRATEGY), ("2-year cointegration window", BENCHMARK)):
        s = nb[nb["window"] == w]
        fig.add_trace(go.Scatter(x=s["train_sharpe"], y=s["oos_sharpe"], mode="markers", name=w,
                                 marker=dict(color=color, size=11, line=dict(color="white", width=2)),
                                 customdata=np.c_[s["train_rank"], s["label"], s["oos_criteria_met"]],
                                 hovertemplate="#%{customdata[0]} %{customdata[1]}<br>train Sharpe %{x:.2f} → "
                                               "OOS Sharpe %{y:.2f} · %{customdata[2]}/4 criteria<extra></extra>"))
    sel = nb.iloc[0]
    fig.add_annotation(x=sel["train_sharpe"], y=sel["oos_sharpe"], text="selected", showarrow=True, arrowhead=0,
                       ax=30, ay=-25, font=dict(size=11, color=MUTED), arrowcolor=MUTED)
    fig.add_hline(y=0, line=dict(color="#c3c2b7", width=1))
    fig.update_layout(title="The 20 best training settings, and how each did out of sample",
                      xaxis_title="Training Sharpe", yaxis_title="Out-of-sample Sharpe")
    show(fig, height=400)
    n_ok = int((nb["oos_criteria_met"] == 4).sum())
    st.caption(f"Diagnostic only (not used for selection). {n_ok} of the top 20 meet all four criteria out of sample, "
               "and they are exactly the ones with the 3-year cointegration window; every 2-year variant fails. The "
               "3-year window was also the most robust choice in the earlier walk-forward study.")

st.header("Read before relying on it")
train = card.set_index("period").loc["train 2015-2024"]
st.markdown(
    f"1. **Weak training record.** Sharpe {train['sharpe']:.2f} over 2015–24 ({pct(train['total_return'])} in total). "
    "It passes the goal in 2025–26 because the index went nowhere while the pair book stayed positive.\n"
    "2. **Statistically thin.** A 22-month Sharpe of 0.53 has a standard error of about 0.75 — consistent with a real "
    "but modest edge, not proof of one.\n"
    "3. **Execution cost is critical** (see above).\n"
    "4. **The test window had been seen before.** Two earlier designs were evaluated on the same 2025–26 window. The "
    "point-in-time selection itself was mechanical and fixed in advance, but the decision to rebuild the universe came "
    "after those failures.\n"
    "5. **Residual survivorship** (46 delisted members without prices) and an **approximate TRI**.\n"
    "6. The next honest step is **forward paper trading** of this frozen configuration."
)
st.page_link("views/limitations.py", label="All limitations, and how to reproduce every number", icon=":material/fact_check:")
