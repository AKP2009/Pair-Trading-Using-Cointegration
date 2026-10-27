import streamlit as st

from core import charts, data
from core.style import chip, page_header, pct, show, step_cards

page_header(
    "Quantitative research project · NSE India",
    "Regime-aware pair trading on the Nifty 200",
    "A market-neutral strategy that finds pairs of stocks whose prices move together (cointegration), waits "
    "until one of them runs unusually far ahead of the other, then bets on the gap closing. A market-regime "
    "model decides when the strategy is allowed to trade. Everything was chosen on 2015–2024 data and then "
    "tested once, untouched, on January 2025 – October 2026.",
)

card = data.scorecard().set_index("period")
oos = card.loc["OUT-OF-SAMPLE 2025..latest"]
m = data.meta()

st.subheader("Final result: out of sample, Jan 2025 → Oct 2026")
st.html(
    chip(f"{int(oos['criteria_met'])} of 4 goal criteria met", "good")
    + chip("Benchmark: Nifty 200 Total Return Index")
    + chip(f"Beta to the index {oos['beta']:.2f}")
    + chip("Net of futures costs + 5 bps slippage")
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total return", pct(oos["total_return"]), f"{(oos['total_return'] - oos['tri_total_return']) * 100:+.1f} pts vs TRI "
          f"({pct(oos['tri_total_return'])})", border=True, help="Goal 1: higher total return than the TRI")
c2.metric("Sharpe ratio", f"{oos['sharpe']:.2f}", f"{oos['sharpe'] - oos['tri_sharpe']:+.2f} vs TRI ({oos['tri_sharpe']:.2f})",
          border=True, help="Goal 2: higher risk-adjusted return (annualised mean / volatility of daily returns)")
c3.metric("Volatility", pct(oos["vol"], sign=False), f"{(oos['vol'] - oos['tri_vol']) * 100:+.1f} pts vs TRI "
          f"({pct(oos['tri_vol'], sign=False)})", delta_color="inverse", border=True,
          help="Goal 3: lower annualised volatility")
c4.metric("Max drawdown", pct(oos["max_drawdown"]), f"{(oos['max_drawdown'] - oos['tri_maxdd']) * 100:+.1f} pts vs TRI "
          f"({pct(oos['tri_maxdd'])})", border=True, help="Goal 4: smaller worst peak-to-trough loss")

eq = data.equity_oos()
show(charts.equity_and_drawdown(eq, "Growth of ₹1 since 1 Jan 2025: strategy vs Nifty 200 TRI"), height=470)
st.caption("Top: value of ₹1 invested on 1 Jan 2025. Bottom: drawdown, the loss from the previous peak. "
           "The strategy made money while the index went nowhere, and its worst fall was half the index's.")

counts = data.pairs_by_period().groupby("period_start").size()
n_lo, n_hi, n_med = counts.min(), counts.max(), int(counts.median())

st.divider()
st.subheader("The model in five steps")
step_cards([
    ("Point-in-time universe", "Only stocks that were <b>actually in the Nifty 200 on each date</b> (420 names over "
     "2013–2026), so the backtest cannot quietly skip companies that later fell out of the index."),
    ("Market regime", "Every day is labelled Bullish, Bearish, Sideways, Low-vol or High-vol from trend, breadth "
     "and volatility factors. <b>No new trades in Low-vol</b> markets, where reversion pays least."),
    ("Find pairs", "Every January and July, test same-industry stocks for <b>cointegration</b> (ADF test on the log "
     f"price ratio, 3 years of history). {n_lo}–{n_hi} pairs qualify per half-year (median {n_med})."),
    ("Signal", "Measure each pair's 20-day <b>relative strength</b> and rank it against its last 126 days. Above the "
     "90th percentile: short the leader, buy the laggard. Below the 10th: the reverse."),
    ("Exit & risk", "Close when relative strength crosses its median or after <b>15 calendar days</b>. Up to 10 pairs, "
     "equal ₹ per leg, a stock in at most 2 pairs, traded through stock futures."),
])

st.divider()
left, right = st.columns([1.15, 1], gap="large")
with left:
    st.subheader("How we got here")
    st.caption("Seven designs were built and tested. The failures shaped the final model, so they are shown, not hidden.")
    items = [
        ("PHASE 1 · NIFTY 50", "Classic cointegration + z-score + Random Forest filter",
         "3 pairs from 22 large caps. Baseline Sharpe 0.14 with a −43% drawdown; the ML filter made it worse (Sharpe −1.03); "
         "walk-forward Sharpe −1.0. Too few pairs, fragile regression hedge.", "bad", "did not work"),
        ("PHASE 2A · NIFTY 200", "Static relative-strength model, tuned on 2014–21",
         "Sharpe 1.11 in-sample, −0.56 on the 2022+ holdout. In-sample tuning had fitted the 2020–21 COVID dispersion.",
         "bad", "failed holdout"),
        ("PHASE 2B", "Walk-forward model selection (v1 → v3)",
         "Out of sample 2017–2026: +7.5%, Sharpe 0.19, beta 0.02. Positive but not significant. Key finding: reversion works in "
         "sideways / high-vol markets and fails in low-vol ones.", "warn", "weak positive"),
        ("PHASE 2C", "15-day hold, every lookback tuned walk-forward",
         "Tuning everything gave Sharpe −0.22. What stayed robust: short RS lookbacks and a 3-year cointegration window.",
         "bad", "negative"),
        ("PHASE 2D", "Kaufman-style regime-timed index + pairs",
         "Beat buy & hold in training (Sharpe 1.00 vs 0.79) but lost in the 2025–26 test (−0.60 vs −0.09): index timing whipsawed. "
         "The pair sleeve stayed positive.", "bad", "failed test"),
        ("PHASE 2E", "Relative-strength + low-vol stock picking with a regime hedge",
         "Met every goal in training and over 2013–2026, failed all four in the test. Training was flattered by survivorship.",
         "bad", "failed test"),
        ("FINAL", "Point-in-time universe + re-selected pair strategy",
         f"Out of sample: {pct(oos['total_return'])}, Sharpe {oos['sharpe']:.2f}, vol {pct(oos['vol'], sign=False)}, "
         f"max DD {pct(oos['max_drawdown'])} vs the TRI's {pct(oos['tri_total_return'])} / {oos['tri_sharpe']:.2f} / "
         f"{pct(oos['tri_vol'], sign=False)} / {pct(oos['tri_maxdd'])}.", "good", "4 / 4 goals met"),
    ]
    html = '<div class="timeline">'
    for when, what, why, kind, verdict in items:
        cls = "item final" if when == "FINAL" else "item"
        html += (f'<div class="{cls}"><div class="when">{when}</div><div class="what">{what} {chip(verdict, kind)}</div>'
                 f'<div class="why">{why}</div></div>')
    st.html(html + "</div>")

with right:
    st.subheader("The honest bottom line")
    st.success(f"**It met the goal.** Out of sample the strategy beat the Nifty 200 TRI on return, Sharpe, volatility "
               f"and drawdown, with almost no market exposure (beta {oos['beta']:.2f}).", icon=":material/check_circle:")
    train = card.loc["train 2015-2024"]
    st.warning(f"**The edge is thin.** Training Sharpe was only {train['sharpe']:.2f} ({pct(train['total_return'])} over "
               "10 years), and it lagged the TRI in 9 of the 10 training years. It wins on *risk*, not on return in rising markets.",
               icon=":material/warning:")
    st.warning("**Execution decides it.** It survives 10 bps slippage per side but not 20 bps, and not cash-market "
               "delivery with stock borrowing. It has to be traded through stock futures.", icon=":material/warning:")
    st.info("**22 months is short.** A 22-month Sharpe of 0.53 has a standard error of roughly 0.75. The next honest "
            "test is forward paper trading of this frozen configuration.", icon=":material/info:")

    st.subheader("Explore")
    st.page_link("views/how_it_works.py", label="How the model works, step by step", icon=":material/account_tree:")
    st.page_link("views/results.py", label="Full results, robustness and trade statistics", icon=":material/trophy:")
    st.page_link("views/all_pairs.py", label="Every pair the strategy traded", icon=":material/join_inner:")
    st.page_link("views/live.py", label="Today's signals and open positions", icon=":material/sensors:")
