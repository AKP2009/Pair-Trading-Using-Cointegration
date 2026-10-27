import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import charts, data
from core.style import AXIS, BENCHMARK, CRITICAL, MUTED, STRATEGY, chip, page_header, show

page_header(
    "The model · data",
    "Data & the point-in-time universe",
    "Daily adjusted closing prices for every stock that has been in the Nifty 200 since December 2013, the Nifty 200 "
    "index and India VIX, from January 2012 to October 2026. The key design decision is to trade only stocks that "
    "were actually index members on each date.",
)

m = data.meta()
reg = data.regime_daily()
counts = data.bundle_csv("pit_membership_counts.csv", parse_dates=["date"])
stocks = data.stocks()

c = st.columns(4)
c[0].metric("Trading days", f"{len(reg):,}", help=f"{m['data_start']} to {m['data_end']}", border=True)
c[1].metric("Ever in the index", "420", help="Stocks that were Nifty 200 members at some point since Dec 2013", border=True)
c[2].metric("Stocks with price history", f"{m['n_stocks_with_prices']}", border=True)
c[3].metric("Members with no data left", f"{m['n_missing_history']}", help="Delisted or merged; Yahoo no longer carries them",
            border=True)

st.header("Why a point-in-time universe matters")
c1, c2 = st.columns([1, 1.1], gap="large")
with c1:
    st.markdown(
        "Backtesting on **today's** index members is a classic trap. Companies that collapsed were removed from the index, "
        "so a backtest that only uses today's list never holds them: it quietly avoids the losers.\n\n"
        "This was measured directly. In the 2025–26 test, an earlier frozen pair strategy made **+0.9%** on today's list "
        "but **−2.8%** on the members actually in the index at the time. The 14 trades in stocks that were members then "
        "and were *dropped later* averaged **−2.7%** each, against +0.15% for the other 94 trades."
    )
    st.info("Every result on the Results pages uses the point-in-time universe. The earlier experiments (Phase 2 "
            "research pages) used today's list and are labelled accordingly.", icon=":material/info:")
with c2:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=counts["date"], y=counts["with_prices"], name="Members with price data",
                         marker_color=STRATEGY, marker_line_width=0, hovertemplate="%{x|%b %Y}: %{y}<extra>with data</extra>"))
    fig.add_trace(go.Bar(x=counts["date"], y=counts["without_prices"], name="Members with no data (delisted / merged)",
                         marker_color=CRITICAL, marker_line_width=0,
                         hovertemplate="%{x|%b %Y}: %{y}<extra>no data</extra>"))
    fig.update_layout(barmode="stack", title="Nifty 200 members at each review date", yaxis_title="Stocks", bargap=0.3)
    show(fig, height=330)
    st.caption("Before 2019 up to a sixth of the members no longer have price data — that residual survivorship sits in "
               "the training years, not in the 2025–26 test, where coverage is complete.")

st.subheader("Where membership comes from")
st.table(pd.DataFrame([
    ["Dec 2013 – Mar 2022", "Official month-end Nifty 200 constituent files from NSE Indices (17 PDFs, 200–201 names, weights sum to 100%)", "Exact"],
    ["Apr 2022 – Mar 2023", "Mar 2022 list rolled forward through the Sep 2022 semi-annual review", "Exact for reviews"],
    ["Apr 2023 – Mar 2024", "Today's list rolled back through the reviews", "Approximate (202 names; one ad-hoc change missing)"],
    ["Mar 2024 – today", "Today's list rolled back through NSE 'Replacements in indices' press releases (Mar 2024 – Sep 2026)", "Exact (200 names at every review)"],
], columns=["Period", "Source", "Accuracy"]).set_index("Period"))
st.caption("Old symbols are mapped to current ones (e.g. ZOMATO → ETERNAL, LTI → LTM, TATAMOTORS → TMPV), "
           "because Yahoo Finance only keeps the current ticker. Parsers: tools/parse_nse_constituents.py and "
           "tools/parse_nse_index_changes.py; logic: pit_universe.py.")

tab1, tab2, tab3 = st.tabs(["Index review changes", "Members without price data", "Industries"])
with tab1:
    ch = data.data_csv("nifty200_index_changes.csv", parse_dates=["effective_date"])
    ch = ch[ch["action"].isin(["include", "exclude"])]
    per = ch.groupby([ch["effective_date"].dt.date, "action"]).size().unstack(fill_value=0)
    st.markdown(f"**{len(ch)} inclusions and exclusions** across {ch['effective_date'].nunique()} review dates since Sep 2022.")
    eff = st.selectbox("Review effective date", sorted(ch["effective_date"].dt.date.unique(), reverse=True))
    sub = ch[ch["effective_date"].dt.date == eff]
    a, b = st.columns(2)
    a.markdown("**Added to the index**")
    a.dataframe(sub[sub["action"] == "include"][["symbol", "company"]], hide_index=True)
    b.markdown("**Removed from the index**")
    b.dataframe(sub[sub["action"] == "exclude"][["symbol", "company"]], hide_index=True)
with tab2:
    st.markdown("These stocks were Nifty 200 members at some point but have **no price history left** on Yahoo Finance "
                "(delisted, merged or suspended), so the backtest cannot trade them while they were members.")
    st.html("".join(chip(s) for s in m["missing_history"]))
with tab3:
    cur = data.data_csv("nifty200_constituents.csv")
    ind = cur["Industry"].value_counts().sort_values()
    used = stocks[stocks["in_final_pairs"]]["industry"].value_counts()
    fig = go.Figure()
    fig.add_trace(go.Bar(y=ind.index, x=ind.values, orientation="h", name="Nifty 200 members today",
                         marker_color="#c3c2b7", marker_line_width=0, hovertemplate="%{y}: %{x}<extra>members today</extra>"))
    fig.add_trace(go.Bar(y=ind.index, x=used.reindex(ind.index).fillna(0).values, orientation="h",
                         name="Stocks ever in a final-strategy pair", marker_color=STRATEGY, marker_line_width=0,
                         hovertemplate="%{y}: %{x}<extra>in a traded pair</extra>"))
    fig.update_layout(barmode="group", title="Industries (NSE classification), used to group pair candidates",
                      xaxis_title="Stocks", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True)
    show(fig, height=560)
    st.caption("Pairs are only formed within an industry. Financial Services dominates: it is the largest industry and "
               "banks and NBFCs share the same rate and credit drivers, so they cointegrate most often.")

st.header("Market data")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.07)
fig.add_trace(go.Scatter(x=reg.index, y=reg["benchmark"], name="Nifty 200 index", line=dict(color=BENCHMARK, width=1.6),
                         hovertemplate="%{y:,.0f}<extra>Nifty 200</extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=reg.index, y=reg["vix"], name="India VIX", line=dict(color=MUTED, width=1.3),
                         hovertemplate="%{y:.1f}<extra>India VIX</extra>"), row=2, col=1)
fig.update_yaxes(type="log", title_text="Nifty 200 (log)", row=1, col=1)
fig.update_yaxes(title_text="India VIX", row=2, col=1)
fig.update_layout(title="Nifty 200 index (^CNX200) and India VIX (^INDIAVIX), daily", hovermode="x unified")
show(fig, height=430)
st.markdown(
    "**Benchmark.** The goal is set against the **Nifty 200 Total Return Index (TRI)**, which reinvests dividends — the "
    "fair comparison, because the stock prices used are dividend-adjusted. The official TRI download is blocked by a bot "
    "check, so the TRI is approximated as the **price index plus 1.5% a year of dividends**, reinvested daily. That is "
    "at or above the index's real yield (≈1.1–1.4%), so the benchmark is if anything harder to beat; the result also "
    "holds at 2.5%. Dropping the official file into `data/nifty200_tri.csv` makes every script use it automatically."
)

with st.expander("Phase 1 dataset: 22 Nifty 50 stocks in four sectors (2015–2025)"):
    sectors = json.loads((data.DATA / "sectors.json").read_text())
    st.dataframe(pd.DataFrame([(k, ", ".join(t.replace(".NS", "") for t in v), len(v)) for k, v in sectors.items()],
                              columns=["Sector", "Stocks", "Count"]), hide_index=True)
    st.caption("Used by notebooks 01–05 and the classic z-score / Random Forest pipeline. Train 2015–21, validation "
               "2022–23, test 2024–25.")
