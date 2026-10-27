import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import charts, data
from core.style import CRITICAL, GOOD, SERIES, STRATEGY, page_header, show

CAPITAL = 10_000_000  # the backtest starts with ₹1 crore

page_header(
    "Results · pairs",
    "All pairs & trades",
    "Every pair the final strategy was allowed to trade, every half-year period it qualified in, and every trade it "
    "made. Pairs are re-tested from scratch each January and July, so the list changes over time.",
)

sched = data.pairs_by_period()
trd = data.trades()
stocks = data.stocks().set_index("symbol")

# trades -> the formation period they belong to
trd["period_start"] = pd.NaT
for s, e in sched[["period_start", "period_end"]].drop_duplicates().itertuples(index=False):
    trd.loc[(trd["entry_date"] >= s) & (trd["entry_date"] < e), "period_start"] = s
pp_stats = trd.groupby(["period_start", "pair"]).agg(trades=("ret", "size"), net_pnl=("net_pnl", "sum"),
                                                     wins=("net_pnl", lambda x: int((x > 0).sum())))
sched = sched.merge(pp_stats, left_on=["period_start", "pair"], right_index=True, how="left")
sched[["trades", "net_pnl", "wins"]] = sched[["trades", "net_pnl", "wins"]].fillna(0)

c = st.columns(5)
c[0].metric("Pair-periods", f"{len(sched):,}", help="A pair listed in one half-year period", border=True)
c[1].metric("Distinct pairs", f"{sched['pair'].nunique():,}", border=True)
c[2].metric("Stocks involved", f"{len(set(sched['stock_a']) | set(sched['stock_b']))}", border=True)
c[3].metric("Industries", f"{sched['industry'].nunique()}", border=True)
c[4].metric("Trades", f"{len(trd):,}", f"{(trd['period'] == 'Out-of-sample').sum()} out of sample", delta_color="off",
            border=True)

# ---------------------------------------------------------------- pairs per period
top = sched["industry"].value_counts().index[:7].tolist()
sched["industry_grp"] = np.where(sched["industry"].isin(top), sched["industry"], "Other industries")
order = top + ["Other industries"]
cnt = sched.groupby(["period_start", "industry_grp"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
fig = go.Figure()
for i, ind in enumerate(order):
    fig.add_trace(go.Bar(x=cnt.index, y=cnt[ind], name=ind, marker_color=SERIES[i] if ind != "Other industries" else "#c3c2b7",
                         marker_line=dict(color="white", width=1), width=86_400_000 * 150,
                         hovertemplate="%{x|%b %Y}: %{y} pairs<extra>" + ind + "</extra>"))
fig.add_vrect(x0="2024-12-15", x1=cnt.index[-1] + pd.Timedelta(days=170), fillcolor=STRATEGY, opacity=0.06, line_width=0)
fig.update_layout(barmode="stack", title="Cointegrated pairs found at each formation date, by industry",
                  yaxis_title="Pairs", legend=dict(font=dict(size=11)))
show(fig, height=420)
st.caption("Each bar is one formation date (1 Jan or 1 Jul); the pairs then trade for six months. Shaded: out-of-sample. "
           "Pair counts swell after volatile spells (2021, 2023) when many stocks in a sector move together, and fall "
           "when a sector's stocks diverge for fundamental reasons.")

# ---------------------------------------------------------------- filters
st.header("Browse the pairs")
f1, f2, f3 = st.columns([1, 1.4, 1])
periods = sorted(sched["period_start"].dt.date.unique(), reverse=True)
period = f1.selectbox("Formation period", ["All periods"] + periods,
                      format_func=lambda p: p if isinstance(p, str) else f"{p:%b %Y} → +6 months")
inds = f2.multiselect("Industry", sorted(sched["industry"].unique()), placeholder="All industries")
query = f3.text_input("Search a stock", placeholder="e.g. SBIN").strip().upper()


def filt(df: pd.DataFrame) -> pd.DataFrame:
    if period != "All periods":
        df = df[df["period_start"].dt.date == period]
    if inds:
        df = df[df["industry"].isin(inds)]
    if query:
        df = df[df["stock_a"].str.contains(query, regex=False) | df["stock_b"].str.contains(query, regex=False)]
    return df


tab1, tab2, tab3 = st.tabs(["Pair leaderboard", "Pairs by period", "Trade log"])

with tab1:
    view = filt(sched)
    tsub = trd.merge(view[["period_start", "pair"]], on=["period_start", "pair"])
    lb = view.groupby("pair").agg(industry=("industry", "first"), periods=("period_start", "size"),
                                  first=("period_start", "min"), last=("period_start", "max"),
                                  adf_p=("adf_p", "median"))
    ts = tsub.groupby("pair").agg(trades=("ret", "size"), win=("net_pnl", lambda x: (x > 0).mean()),
                                  avg=("ret", "mean"), pnl=("net_pnl", "sum"))
    lb = lb.join(ts).fillna({"trades": 0, "pnl": 0})
    lb["pnl_pct"] = lb["pnl"] / CAPITAL * 100
    lb["companies"] = [f"{stocks['company'].get(a, a)} / {stocks['company'].get(b, b)}"
                       for a, b in (p.split(" / ") for p in lb.index)]
    lb = lb.sort_values("pnl", ascending=False).reset_index()
    st.markdown(f"**{len(lb)} pairs** match the filters. Contribution = net P&L as a share of the ₹1 crore starting "
                "capital. Select a row to open it in the pair explorer.")
    ev = st.dataframe(
        lb[["pair", "pnl_pct", "trades", "win", "avg", "industry", "periods", "first", "last", "adf_p", "companies"]].assign(
            win=lambda x: x["win"] * 100, avg=lambda x: x["avg"] * 100),
        hide_index=True, on_select="rerun", selection_mode="single-row", key="lb_table", height=420,
        column_config={
            "pair": "Pair", "companies": "Companies", "industry": "Industry", "periods": "Periods listed",
            "first": st.column_config.DateColumn("First listed", format="MMM YYYY"),
            "last": st.column_config.DateColumn("Last listed", format="MMM YYYY"),
            "trades": st.column_config.NumberColumn("Trades", format="%d"),
            "win": st.column_config.NumberColumn("Win rate", format="%.0f%%"),
            "avg": st.column_config.NumberColumn("Avg net return", format="%.2f%%"),
            "pnl_pct": st.column_config.NumberColumn("Contribution", format="%+.2f%%"),
            "adf_p": st.column_config.NumberColumn("Median ADF p", format="%.3f")})
    sel = ev.selection.rows if ev and ev.selection else []
    if sel:
        chosen = lb.iloc[sel[0]]["pair"]
        if st.button(f"Open {chosen} in the pair explorer", icon=":material/query_stats:", type="primary"):
            st.session_state["explorer_pair"] = chosen
            st.switch_page("views/pair_explorer.py")
    traded = lb[lb["trades"] > 0]
    if len(traded):
        n = min(12, len(traded) // 2 or 1)
        best, worst = traded.head(n), traded.tail(n).iloc[::-1]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            show(charts.hbar(best["pair"], best["pnl_pct"] / 100, f"Top {n} pairs by contribution", color=GOOD,
                             fmt="+.2%"), height=90 + 28 * n, key="best_pairs")
        with c2:
            show(charts.hbar(worst["pair"], worst["pnl_pct"] / 100, f"Bottom {n} pairs by contribution",
                             color=CRITICAL, fmt="+.2%"), height=90 + 28 * n, key="worst_pairs")
        by_ind = tsub.groupby("industry")["net_pnl"].sum().sort_values(ascending=False) / CAPITAL
        show(charts.hbar(by_ind.index, by_ind.values, "Net contribution by industry", fmt="+.1%",
                         colors=[GOOD if v > 0 else CRITICAL for v in by_ind.values]),
             height=100 + 30 * len(by_ind), key="by_ind")

with tab2:
    view = filt(sched).sort_values(["period_start", "industry", "adf_p"], ascending=[False, True, True])
    st.markdown(f"**{len(view)} pair-periods.** ADF p is the cointegration test's p-value on the formation window "
                "(lower = stronger evidence the ratio mean-reverts); ratio vol is the daily std of ln(A/B).")
    st.dataframe(
        view[["period_start", "period_end", "pair", "industry", "adf_p", "corr", "ratio_vol", "trades", "wins", "net_pnl"]].assign(
            ratio_vol=lambda x: x["ratio_vol"] * 100, net_pnl=lambda x: x["net_pnl"] / CAPITAL * 100),
        hide_index=True, height=520,
        column_config={
            "period_start": st.column_config.DateColumn("Trades from", format="D MMM YYYY"),
            "period_end": st.column_config.DateColumn("Until", format="D MMM YYYY"),
            "pair": "Pair", "industry": "Industry",
            "adf_p": st.column_config.NumberColumn("ADF p", format="%.4f"),
            "corr": st.column_config.NumberColumn("Return corr.", format="%.2f"),
            "ratio_vol": st.column_config.NumberColumn("Ratio vol (daily)", format="%.2f%%"),
            "trades": st.column_config.NumberColumn("Trades", format="%d"),
            "wins": st.column_config.NumberColumn("Wins", format="%d"),
            "net_pnl": st.column_config.NumberColumn("Contribution", format="%+.2f%%")})
    st.download_button("Download the pair list (CSV)", view.drop(columns=["industry_grp"]).to_csv(index=False),
                       "final_strategy_pairs.csv", "text/csv", icon=":material/download:")

with tab3:
    tv = filt(trd.dropna(subset=["period_start"]))
    side = st.segmented_control("Show", ["All", "Training", "Out-of-sample"], default="All", key="tl_side") or "All"
    if side != "All":
        tv = tv[tv["period"] == side]
    tv = tv.sort_values("entry_date", ascending=False)
    st.markdown(f"**{len(tv):,} trades.** Each trade is long one stock and short the other, with equal ₹ on both legs. "
                "Returns are net of costs, per ₹ of one leg.")
    st.dataframe(
        tv[["entry_date", "exit_date", "pair", "long_leg", "short_leg", "entry_regime", "entry_rs_pct", "days", "gross_ret",
            "ret", "exit_reason", "industry"]].assign(gross_ret=lambda x: x["gross_ret"] * 100, ret=lambda x: x["ret"] * 100,
                                                       entry_rs_pct=lambda x: x["entry_rs_pct"] * 100),
        hide_index=True, height=520,
        column_config={
            "entry_date": st.column_config.DateColumn("Entry", format="D MMM YYYY"),
            "exit_date": st.column_config.DateColumn("Exit", format="D MMM YYYY"),
            "pair": "Pair", "long_leg": "Long", "short_leg": "Short", "entry_regime": "Regime at entry",
            "entry_rs_pct": st.column_config.NumberColumn("RS percentile", format="%.0f"),
            "days": st.column_config.NumberColumn("Days held", format="%d"),
            "gross_ret": st.column_config.NumberColumn("Gross", format="%+.2f%%"),
            "ret": st.column_config.NumberColumn("Net", format="%+.2f%%"),
            "exit_reason": "Exit reason", "industry": "Industry"})
    st.download_button("Download the trade log (CSV)", tv.to_csv(index=False), "final_strategy_trades.csv", "text/csv",
                       icon=":material/download:")

with st.expander("Phase 1 pairs (Nifty 50 sectors, Engle–Granger test)"):
    cand = data.data_csv("all_candidate_pairs.csv")
    st.markdown(f"The first version of the project tested **{len(cand)} same-sector pairs** (train-period price "
                "correlation ≥ 0.80) with the Engle–Granger cointegration test; **3 passed at 5%** and were traded with a "
                "z-score rule.")
    st.dataframe(cand.assign(stock_a=cand["stock_a"].str.replace(".NS", ""), stock_b=cand["stock_b"].str.replace(".NS", "")),
                 hide_index=True, column_config={
                     "corr_train": st.column_config.NumberColumn("Price corr.", format="%.3f"),
                     "coint_stat": st.column_config.NumberColumn("EG statistic", format="%.2f"),
                     "coint_pvalue": st.column_config.NumberColumn("EG p-value", format="%.4f"),
                     "spread_adf_pvalue": st.column_config.NumberColumn("Spread ADF p", format="%.4f"),
                     "beta_a_on_b": st.column_config.NumberColumn("Hedge ratio β", format="%.3f"),
                     "cointegrated_5pct": "Cointegrated at 5%"})
    st.page_link("views/phase1.py", label="Phase 1 in full", icon=":material/history_edu:")
