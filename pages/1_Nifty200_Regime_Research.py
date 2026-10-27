"""Streamlit page: Nifty 200 regime-aware relative-strength pair research (reads research/ outputs)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "research"
DATA = ROOT / "data"

st.title("Nifty 200 - Regime-aware Relative-Strength Pairs")

if not (RES / "v3_wf_summary.csv").exists():
    st.warning("Run `python research.py` and `python walk_forward_rs.py` first to generate research outputs.")
    st.stop()

regimes = pd.read_csv(DATA / "regimes.csv", index_col=0, parse_dates=True)
latest = regimes.dropna(subset=["regime"]).iloc[-1]
summary = pd.read_csv(RES / "v3_wf_summary.csv")
final = summary[(summary["model"] == "WF ensemble + regime gate") & (summary["period"] == "OOS 2017-latest")].iloc[0]

st.subheader(f"Current regime ({latest.name.date()}): {latest['regime'].replace('_', ' ').title()}")
c = st.columns(4)
c[0].metric("Distance from 21 EMA", f"{latest['dist_ema21']:+.2%}")
c[1].metric("% stocks above 200 EMA", f"{latest['pct_above_200ema']:.0%}")
c[2].metric("A/D ratio (10d)", f"{latest['ad_ratio']:.2f}")
c[3].metric("India VIX", f"{latest['vix']:.1f}")

PIT = RES / "pit_pairs"
if (PIT / "scorecard_vs_tri.csv").exists():
    st.subheader("Final strategy vs Nifty 200 TRI - out of sample, 2025 to latest")
    card = pd.read_csv(PIT / "scorecard_vs_tri.csv").set_index("period").loc["OUT-OF-SAMPLE 2025..latest"]
    g = st.columns(4)
    g[0].metric("Total return", f"{card['total_return']:+.1%}", f"TRI {card['tri_total_return']:+.1%}", delta_color="off")
    g[1].metric("Sharpe", f"{card['sharpe']:.2f}", f"TRI {card['tri_sharpe']:.2f}", delta_color="off")
    g[2].metric("Volatility", f"{card['vol']:.1%}", f"TRI {card['tri_vol']:.1%}", delta_color="off")
    g[3].metric("Max drawdown", f"{card['max_drawdown']:.1%}", f"TRI {card['tri_maxdd']:.1%}", delta_color="off")
    st.caption("Cointegrated pairs from the point-in-time Nifty 200 (survivorship-corrected), percentile relative-strength signal, "
               "regime-gated, frozen on 2015-2024. TRI approximated as price index + 1.5%/yr dividends unless "
               "data/nifty200_tri.csv is provided. Thin edge: needs futures-level costs (fails at 20 bps slippage). "
               "See research/REPORT.md section 10.")
    oos = pd.read_csv(PIT / "oos_equity.csv", index_col=0, parse_dates=True)
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=oos.index, y=oos["PIT pair strategy"], name="Pair strategy (PIT universe)", line=dict(color="#2a78d6", width=2.4)))
    f2.add_trace(go.Scatter(x=oos.index, y=oos["Nifty 200 TRI"], name="Nifty 200 TRI", line=dict(color="#52514e", width=1.6)))
    f2.add_hline(y=1.0, line_color="#c3c2b7")
    f2.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Growth of 1", hovermode="x unified")
    st.plotly_chart(f2, width="stretch")

st.subheader("Earlier research model - walk-forward out-of-sample, 2017 to latest")
k = st.columns(6)
k[0].metric("Total return", f"{final['total_return']:+.1%}")
k[1].metric("Sharpe", f"{final['sharpe']:.2f}")
k[2].metric("Sortino", f"{final['sortino']:.2f}")
k[3].metric("Max drawdown", f"{final['max_drawdown']:.1%}")
k[4].metric("Beta vs Nifty 200", f"{final['beta']:.3f}")
k[5].metric("Win rate", f"{final['win_rate']:.1%}")
st.caption("After NSE stock-futures costs + 5 bps slippage per side. Edge is small and not statistically significant; "
           "2022 onward is flat to negative. See research/REPORT.md.")

eq = pd.read_csv(RES / "v3_wf_equity.csv", index_col=0, parse_dates=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=eq.index, y=eq["WF ensemble + regime gate"], name="Final model (v3)", line=dict(color="#2a78d6", width=2)))
fig.add_trace(go.Scatter(x=eq.index, y=eq["WF ensemble"], name="Same, no regime gate", line=dict(color="#eb6834", width=1.5)))
fig.add_hline(y=1.0, line_color="#c3c2b7")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Growth of 1", hovermode="x unified")
st.plotly_chart(fig, width="stretch")

tab_reg, tab_year, tab_month, tab_pairs, tab_static = st.tabs(
    ["Per regime", "Year by year", "Month by month", "Pairs per regime", "Static dev vs holdout"])
with tab_reg:
    st.markdown("**Backtest run separately in each regime state** (new entries only in that state)")
    st.dataframe(pd.read_csv(RES / "v3_wf_per_regime_backtests.csv", index_col=0)
                 [["total_return", "CAGR", "sharpe", "sortino", "max_drawdown", "beta"]].style.format("{:.3f}"))
    st.markdown("**Attribution of the final model by the regime of each day / entry**")
    st.dataframe(pd.read_csv(RES / "v3_wf_regime_attribution.csv", index_col=0).style.format("{:.3f}", na_rep="-"))
    st.image(str(RES / "figures" / "regime_sharpe.png"))
with tab_year:
    st.dataframe(pd.read_csv(RES / "v3_wf_yearly_returns.csv", index_col=0).style.format("{:+.1%}"))
with tab_month:
    st.image(str(RES / "figures" / "monthly_returns_v3.png"))
with tab_pairs:
    st.dataframe(pd.read_csv(RES / "pairs_by_regime.csv", index_col=0).style.format("{:.3f}", na_rep="-"))
    st.dataframe(pd.read_csv(RES / "pair_counts_by_period.csv"))
with tab_static:
    st.image(str(RES / "figures" / "static_dev_vs_holdout.png"))
    st.dataframe(pd.read_csv(RES / "final_summary.csv")[["label", "period", "CAGR", "sharpe", "sortino", "max_drawdown", "beta", "win_rate", "n_trades"]])
