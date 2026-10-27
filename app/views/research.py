import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import charts, data
from core.style import (BENCHMARK, CRITICAL, GOOD, MUTED, REGIME_LABEL, SEQUENTIAL, STRATEGY, THIRD, page_header, pct,
                        show)

page_header(
    "Research journey · phase 2",
    "Phase 2: from the Nifty 200 research model to the final strategy",
    "Phase 2 moved to the whole Nifty 200, replaced the z-score with relative strength, priced trades as stock "
    "futures and added market regimes. Six designs were tested in turn. Each tab below shows what was built, what "
    "it returned on data it had not seen, and what was learned. These studies used <b>today's</b> Nifty 200 list; "
    "only the final strategy uses the point-in-time universe.",
)

SUMMARY_COLS = {"total_return": "Return", "CAGR": "CAGR", "sharpe": "Sharpe", "sortino": "Sortino",
                "vol": "Volatility", "max_drawdown": "Max DD", "beta": "Beta"}


def metrics_table(df: pd.DataFrame, label_cols: list[str]) -> None:
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    out = df[label_cols + cols].copy()
    for c in ("total_return", "CAGR", "vol", "max_drawdown"):
        if c in out:
            out[c] = out[c] * 100
    cfg = {c: st.column_config.NumberColumn(SUMMARY_COLS[c], format="%+.1f%%" if c in ("total_return", "CAGR") else
                                            "%.1f%%" if c in ("vol", "max_drawdown") else "%.2f") for c in cols}
    st.dataframe(out, hide_index=True, column_config=cfg)


tabs = st.tabs(["A · Static model", "B · Walk-forward", "C · 15-day hold", "D · Kaufman regime system",
                "E · RS + low-vol stocks", "F · All frozen strategies vs TRI"])

# ---------------------------------------------------------------- A static
with tabs[0]:
    st.subheader("A · Static relative-strength model, tuned on 2014–21, held out on 2022+")
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.markdown(
            "**Signal.** Relative strength scaled by the ratio's own volatility, then a fixed hold:")
        st.latex(r"\text{score}_t = \frac{\ln(A/B)_t - \ln(A/B)_{t-L}}{\sigma_{60}\big(\Delta\ln(A/B)\big)\sqrt{L}}")
        st.latex(r"|\text{score}_t| \ge k \;\Rightarrow\; \text{fade the move, hold } H \text{ days}")
        st.markdown(
            "**Pairs:** ADF on the log ratio over 504 days (2 years), same industry, re-formed every 6 months.\n\n"
            "**Tuning:** a 210-cell grid of lookback *L* × hold *H* × entry *k* on 2014-07 → 2021 only, picking the cell "
            "with the best *neighbourhood-averaged* Sharpe. The pick: **L = 40, H = 10, k = 1.0**, a stock in up to 2 "
            "pairs, and per-regime slot limits."
        )
    with c2:
        g = data.research_csv("grid_dev.csv")
        k = st.segmented_control("Entry threshold k", sorted(g["entry_k"].unique()), default=1.0, key="grid_k",
                                 format_func=lambda v: f"k = {v:.1f}") or 1.0
        h = g[g["entry_k"] == k].pivot(index="lookback", columns="hold", values="sharpe")
        fig = go.Figure(go.Heatmap(z=h.to_numpy(), x=[f"H={c}" for c in h.columns], y=[f"L={i}" for i in h.index],
                                   colorscale=[[0, "#b8322f"], [0.5, "#f0efec"], [1, "#1c5cab"]], zmid=0, zmin=-1.5, zmax=1.5,
                                   text=np.round(h.to_numpy(), 2), texttemplate="%{text}", xgap=2, ygap=2,
                                   colorbar=dict(title="Sharpe", thickness=10, outlinewidth=0),
                                   hovertemplate="%{y}, %{x}: Sharpe %{z:.2f}<extra></extra>"))
        fig.update_layout(title=f"Development-period Sharpe by lookback L and hold H (k = {k:.1f})")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(fig, height=360)
        st.caption("1–2-day holds lose everywhere (costs swamp a one-day reversal); the good zone is H = 10–20 days.")

    fs = data.research_csv("final_summary.csv")
    sc = fs.pivot_table(index="label", columns="period", values="sharpe")[["dev", "holdout"]].reset_index()
    c1, c2 = st.columns([1.2, 1.1], gap="large")
    with c1:
        st.markdown("**Dev vs holdout Sharpe, same model, different assumptions**")
        st.dataframe(sc, hide_index=True, column_config={
            "label": "Scenario", "dev": st.column_config.NumberColumn("Dev 2014–21", format="%.2f"),
            "holdout": st.column_config.NumberColumn("Holdout 2022+", format="%.2f")})
        var = data.research_csv("variants_dev.csv")[["label", "sharpe"]]
        st.markdown("**Model variants tested on dev**")
        st.dataframe(var, hide_index=True, column_config={"label": "Variant",
                                                         "sharpe": st.column_config.NumberColumn("Dev Sharpe", format="%.2f")})
    with c2:
        fe = data.research_csv("final_equity.csv", index_col=0, parse_dates=True)["equity"]
        fe = fe / fe.iloc[0]
        show(charts.equity_and_drawdown(fe.to_frame("Static model"), "Static model: tuned on 2014–21, then the holdout",
                                        colors={"Static model": STRATEGY}, shade_from="2022-01-01",
                                        shade_label="Holdout"), height=440)
    st.error("**Verdict: failed its holdout.** Sharpe 1.11 in development, −0.56 on 2022+ (−33%). The development years were "
             "driven by the 2020–21 COVID dispersion. After 2022 reversions got faster: signals still reverted over 1–5 days "
             "but gave the move back by 10–20 days, so a 10-day hold fixed on old data sat on the losing side.",
             icon=":material/cancel:")
    st.markdown("**Kept:** mean reversion is the right direction (momentum: Sharpe −1.31), and cointegration beats plain "
                "correlation (non-cointegrated control pairs: 0.49 vs 0.72).")

# ---------------------------------------------------------------- B walk-forward
with tabs[1]:
    st.subheader("B · Walk-forward model selection, out of sample 2017 → 2026")
    st.markdown(
        "Instead of tuning once, every year is traded with the **top-5 grid settings ranked on the trailing 3 years**, run "
        "as an equal-weight ensemble, plus a per-regime gate that blocks regimes whose trailing trades lost money. Each "
        "year is therefore out of sample.\n\n"
        "- **v1** — base grid. **v2** — adds breadth options (ADF p ≤ 0.10, a 25-pair book). "
        "**v3 (final, declared before it was run)** — ensemble members must differ, and no new entries in Low-vol."
    )
    rows = []
    for v in ("v1", "v2", "v3"):
        s = data.research_csv(f"{v}_wf_summary.csv")
        r = s[(s["model"] == "WF ensemble + regime gate") & (s["period"] == "OOS 2017-latest")].iloc[0]
        rows.append({"Version": v, **{k: r[k] for k in ("total_return", "CAGR", "sharpe", "sortino", "max_drawdown", "beta")}})
    nb = s[(s["model"] == "Nifty 200 (buy & hold)") & (s["period"] == "OOS 2017-latest")].iloc[0]
    rows.append({"Version": "Nifty 200 buy & hold", **{k: nb[k] for k in ("total_return", "CAGR", "sharpe", "sortino",
                                                                          "max_drawdown", "beta")}})
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        metrics_table(pd.DataFrame(rows), ["Version"])
        st.caption("v3 came out below v2 and is still the reported model: switching back after seeing the scores would be "
                   "selection on out-of-sample data.")
        ra = data.research_csv("v3_wf_regime_attribution.csv")
        ra["regime"] = ra["regime"].map(REGIME_LABEL)
        st.markdown("**v3 by regime of the day**")
        st.dataframe(ra[["regime", "days", "sharpe", "n_trades", "win_rate", "avg_trade_ret"]].assign(
            win_rate=lambda x: x["win_rate"] * 100, avg_trade_ret=lambda x: x["avg_trade_ret"] * 100),
            hide_index=True, column_config={
                "regime": "Regime", "days": "Days", "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "n_trades": st.column_config.NumberColumn("Trades", format="%d"),
                "win_rate": st.column_config.NumberColumn("Win rate", format="%.0f%%"),
                "avg_trade_ret": st.column_config.NumberColumn("Avg trade", format="%+.2f%%")})
    with c2:
        eq = data.research_csv("v3_wf_equity.csv", index_col=0, parse_dates=True)
        eq = eq[["WF ensemble + regime gate", "WF ensemble", "nifty200"]].rename(columns={
            "WF ensemble + regime gate": "v3 final (with regime gate)", "WF ensemble": "v3 without the gate",
            "nifty200": "Nifty 200"})
        show(charts.lines(eq, "Walk-forward v3, growth of ₹1 (out of sample)",
                          colors={"v3 final (with regime gate)": STRATEGY, "v3 without the gate": THIRD, "Nifty 200": BENCHMARK},
                          ytitle="Growth of ₹1", label_fmt=lambda v: f"{v:.2f}"), height=380)
        yr = data.research_csv("v3_wf_yearly_returns.csv", index_col=0)[["WF ensemble + regime gate", "Nifty 200"]]
        yr.columns = ["v3 final", "Nifty 200"]
        show(charts.grouped_bars(yr, "v3 calendar-year returns", colors={"v3 final": STRATEGY}), height=320)
    st.warning("**Verdict: positive but not significant.** +7.5% over ~10 years, Sharpe 0.19 (t-stat ≈ 0.6), beta 0.02. "
               "Positive in 7 of 10 years; 2023's momentum rally (−10%) was the worst.", icon=":material/warning:")
    st.markdown("**Kept — the most robust finding of the project:** relative-strength pair reversion works in **Sideways "
                "and High-vol** markets and fails in **Low-vol** markets. The raw-signal event study agrees in both the dev "
                "and holdout periods (Low-vol forward returns negative at 5–20 days; Sideways positive at every horizon).")

# ---------------------------------------------------------------- C h15
with tabs[2]:
    st.subheader("C · Holding period fixed at 15 calendar days, every lookback tuned walk-forward")
    st.markdown(
        "Every trade exits on the first close at least 15 calendar days after entry. Each year, a 576-setting model grid "
        "(RS lookback, ratio-vol window, cointegration window 1/2/3 years, entry k, vol filter) and a 243-setting regime "
        "grid were re-chosen from the trailing 3 years."
    )
    s = data.research_csv("h15/wf_summary.csv")
    s = s[s["period"] == "OOS 2017-latest"]
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        metrics_table(s, ["model"])
        sens = data.research_csv("h15/sensitivity_model.csv")
        sens = sens[sens["dimension"].isin(["lookback", "formation_days"])]
        sp = sens.pivot_table(index=["dimension", "value"], columns="period", values="median_sharpe").reset_index()
        sp["dimension"] = sp["dimension"].map({"lookback": "RS lookback", "formation_days": "Cointegration window"})
        st.markdown("**Which settings were robust in both periods** (median Sharpe across all other settings)")
        st.dataframe(sp, hide_index=True, column_config={
            "dimension": "Setting", "value": st.column_config.NumberColumn("Value", format="%d"),
            "dev 2014-21": st.column_config.NumberColumn("Dev 2014–21", format="%.2f"),
            "holdout 2022+": st.column_config.NumberColumn("Holdout 2022+", format="%.2f")})
    with c2:
        eq = data.research_csv("h15/wf_equity.csv", index_col=0, parse_dates=True)
        show(charts.lines(eq, "15-day hold, walk-forward out of sample (growth of ₹1)", ytitle="Growth of ₹1",
                          label_fmt=lambda v: f"{v:.2f}"), height=380)
    st.error("**Verdict: tuning everything did not produce an edge** (Sharpe −0.22). Re-picking the regime lookbacks every "
             "year *hurt*: their differences are smaller than the noise.", icon=":material/cancel:")
    st.markdown("**Kept:** short RS lookbacks (5–10 days) and the **3-year cointegration window** were positive in both "
                "periods (the 3-year window: holdout median Sharpe +0.35, 86% of settings positive). Having a regime gate "
                "helps; its exact windows barely matter. The final strategy uses the 3-year window.")

# ---------------------------------------------------------------- D kaufman
with tabs[3]:
    st.subheader("D · Regime-timed index + pair system (Kaufman-style), train 2015–24, test 2025–26")
    st.markdown(
        "Concepts from P. Kaufman's *Trading Systems and Methods*: the Efficiency Ratio, percentile normalisation (no "
        "z-scores) and robustness testing. The pair *stress indicator* is from Kaufman's *Alpha Trading*.\n\n"
        "- **Index sleeve:** Nifty 200 exposure set by regime — 100% in Bullish/Sideways/Low-vol, 50% in Bearish, 0% in High-vol.\n"
        "- **Pair sleeve:** percentile-RS pairs, 15-day hold, only in Sideways + High-vol.\n"
        "- **Selection:** 1,080 regime/index candidates and 145 pair candidates; each had to beat buy & hold on both Sharpe "
        "and drawdown in training, then a percentile-rank consistency score with plateau smoothing."
    )
    ks = data.research_csv("kaufman/summary.csv")
    ks = ks[ks["period"].isin(["train 2015-2024", "test 2025-latest"])]
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        metrics_table(ks, ["strategy", "period"])
    with c2:
        ke = data.research_csv("kaufman/equity.csv", index_col=0, parse_dates=True)
        show(charts.lines(ke, "Growth of ₹1 (shaded: test)", colors={"Final system": STRATEGY, "Index sleeve only": THIRD,
                                                                   "Pair sleeve only": "#6250d6",
                                                                   "Nifty 200 buy & hold": BENCHMARK},
                          ytitle="Growth of ₹1", label_fmt=lambda v: f"{v:.2f}").add_vrect(
            x0="2025-01-01", x1=ke.index[-1], fillcolor=STRATEGY, opacity=0.06, line_width=0), height=400)
    dist = data.research_csv("kaufman/stage1_test_distribution.csv")
    passed = dist[dist["beats_bh_train"]]
    fig = go.Figure(go.Histogram(x=passed["test_sharpe"], xbins=dict(size=0.1), marker_color=STRATEGY,
                                 marker_line=dict(color="white", width=1), hovertemplate="%{x}: %{y} settings<extra></extra>"))
    bh = ks[(ks["strategy"] == "Nifty 200 buy & hold") & (ks["period"] == "test 2025-latest")]["sharpe"].iloc[0]
    fig.add_vline(x=bh, line=dict(color=BENCHMARK, width=2), annotation_text=f"buy & hold {bh:.2f}",
                  annotation_font=dict(size=11, color=MUTED))
    fig.update_layout(title=f"Test Sharpe of the {len(passed)} regime settings that beat buy & hold in training",
                      xaxis_title="Test Sharpe 2025–26", yaxis_title="Settings")
    show(fig, height=320)
    share = (passed["test_sharpe"] > bh).mean()
    st.error(f"**Verdict: failed the test.** Training Sharpe 1.00 vs 0.79, but test −0.60 vs −0.09. Only {share:.0%} of the "
             "settings that beat buy & hold in training also beat it in the test: the failure is general. Thresholds of "
             "0.55/0.45 flipped exposure ~26 times a year, buying near tops and cutting near bottoms in a choppy market.",
             icon=":material/cancel:")
    st.markdown("**Kept:** the pair sleeve with the percentile-RS signal and a regime gate was the best pair result so far "
                "(training Sharpe 0.57, positive in both halves, still positive in the test) and nearly uncorrelated with "
                "the index sleeve (ρ = 0.06). Index timing was dropped.")

# ---------------------------------------------------------------- E rs lowvol
with tabs[4]:
    st.subheader("E · Relative-strength + low-volatility stock selection with a regime hedge")
    st.markdown(
        "Goal at the time: beat (i) an equal-weight buy & hold of the universe and (ii) the Nifty 200 index on return, "
        "Sharpe, volatility and drawdown.\n\n"
        "- **Equity sleeve:** each month hold the top 40 stocks by a composite of 12-1-month relative-strength percentile "
        "(70%) and low-volatility percentile (30%), with a buffer to the top 60.\n"
        "- **Hedge:** short Nifty 200 futures — 50% in Bearish, 100% in High-vol.  **Pairs:** the sleeve from tab D.\n"
        "- **Selection:** 864 candidates on 2015–24; 349 beat both benchmarks on all four criteria in training."
    )
    aw = data.research_csv("rs_lowvol/all_windows_summary.csv")
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        metrics_table(aw, ["window", "strategy"])
    with c2:
        re_ = data.research_csv("rs_lowvol/equity_all_windows.csv", index_col=0, parse_dates=True)
        fig = charts.lines(re_, "Growth of ₹1, log scale (shaded: test)", log=True,
                           colors={"Final system": STRATEGY, "(i) EW buy & hold": THIRD, "(ii) Nifty 200 index": BENCHMARK},
                           ytitle="Growth of ₹1 (log)", label_fmt=lambda v: f"{v:.1f}")
        fig.add_vrect(x0="2025-01-01", x1=re_.index[-1], fillcolor=STRATEGY, opacity=0.06, line_width=0)
        show(fig, height=420)
    st.error("**Verdict: met the goal in training and over 2013–26, failed all four criteria in the 2025–26 test.** None of "
             "the 864 candidates passed the test. The equal-weight benchmark and the momentum tilt were both inflated by "
             "survivorship (today's winners held from the start).", icon=":material/cancel:")
    st.markdown("**Kept:** the lesson that **survivorship bias had to be removed** before any further test — which led to the "
                "point-in-time universe and the final strategy. Two designs had now used the same 22-month test window, so "
                "no more designs were generated against it.")

# ---------------------------------------------------------------- F scorecard
with tabs[5]:
    st.subheader("F · Every frozen strategy, out of sample 2025 → Oct 2026, vs the Nifty 200 TRI")
    so = data.research_csv("final_oos/scorecard_vs_tri.csv")
    so = so[so["period"].str.startswith("OUT")][["strategy", "total_return", "sharpe", "vol", "max_drawdown", "criteria_met"]]
    fin = data.scorecard().iloc[0]
    so = pd.concat([pd.DataFrame([{"strategy": "FINAL · point-in-time pair strategy (re-selected)", **{
        k: fin[k] for k in ("total_return", "sharpe", "vol", "max_drawdown", "criteria_met")}}]), so], ignore_index=True)
    so["strategy"] = so["strategy"].replace({
        "Regime-aware cointegrated pair strategy (percentile RS)": "Pair sleeve from D (today's universe)"})
    tri_row = {"strategy": "Nifty 200 TRI (benchmark)", "total_return": fin["tri_total_return"], "sharpe": fin["tri_sharpe"],
               "vol": fin["tri_vol"], "max_drawdown": fin["tri_maxdd"], "criteria_met": np.nan}
    so = pd.concat([so, pd.DataFrame([tri_row])], ignore_index=True)
    st.dataframe(so.assign(total_return=so["total_return"] * 100, vol=so["vol"] * 100, max_drawdown=so["max_drawdown"] * 100),
                 hide_index=True, column_config={
                     "strategy": "Strategy", "total_return": st.column_config.NumberColumn("Return", format="%+.1f%%"),
                     "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                     "vol": st.column_config.NumberColumn("Volatility", format="%.1f%%"),
                     "max_drawdown": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
                     "criteria_met": st.column_config.NumberColumn("Criteria met (of 4)", format="%d")})
    st.markdown(
        "The pair sleeve from D also met 4/4 — but only on **today's** universe (+0.9%). On the members actually in the "
        "index at the time it made −2.8%: survivorship had flattered it. The final strategy re-ran the same, pre-fixed "
        "selection procedure on the point-in-time universe, on 2015–24 only, and was then tested once."
    )
    st.page_link("views/results.py", label="Final strategy results", icon=":material/trophy:")
