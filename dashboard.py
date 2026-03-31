from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"


st.set_page_config(
    page_title="Pair Trading Intelligence Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f4f7fb;
        --card: #ffffff;
        --ink: #0f172a;
        --muted: #475569;
        --brand: #0f766e;
        --accent: #ea580c;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(15, 118, 110, 0.10), transparent 30%),
            radial-gradient(circle at 85% 20%, rgba(234, 88, 12, 0.12), transparent 35%),
            linear-gradient(120deg, #f8fafc 0%, #f1f5f9 100%);
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0.2px;
    }

    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #0ea5a3 60%, #14b8a6 100%);
        border-radius: 20px;
        padding: 24px;
        color: white;
        box-shadow: 0 10px 30px rgba(2, 8, 23, 0.15);
        margin-bottom: 16px;
    }

    .kpi {
        background: var(--card);
        border-radius: 16px;
        padding: 14px 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    .kpi-title {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 4px;
    }

    .kpi-value {
        color: var(--ink);
        font-size: 24px;
        font-weight: 700;
        line-height: 1.1;
    }

    .mono {
        font-family: 'IBM Plex Mono', monospace;
        color: #334155;
        font-size: 12px;
    }

    div[data-baseweb="tab-list"] button {
        color: #000000 !important;
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #000000 !important;
    }

    div[data-baseweb="tab-list"] button[aria-selected="false"] {
        color: #000000 !important;
    }

    div[data-baseweb="tab-list"] button p {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.DataFrame()


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def as_float(df: pd.DataFrame, strategy: str, col: str) -> float | None:
    if df.empty or "strategy" not in df.columns or col not in df.columns:
        return None
    rows = df[df["strategy"] == strategy]
    if rows.empty:
        return None
    val = rows.iloc[0][col]
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    return None


def explain_block(what: str, why: str, how: str) -> None:
    st.markdown(
        f"""
        <div style=\"background:#ffffffcc;border:1px solid rgba(148,163,184,0.25);border-radius:14px;padding:12px 14px;margin:6px 0 12px 0;\">
            <div style=\"font-size:13px;color:#0f172a;\"><b>What:</b> {what}</div>
            <div style=\"font-size:13px;color:#0f172a;\"><b>Why:</b> {why}</div>
            <div style=\"font-size:13px;color:#0f172a;\"><b>How to read:</b> {how}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


prices = load_csv("nifty_prices_clean.csv")
prices_train = load_csv("prices_train.csv")
sector_auto = load_csv("sector_auto.csv")
sector_banking = load_csv("sector_banking.csv")
sector_energy = load_csv("sector_energy.csv")
sector_it = load_csv("sector_it.csv")

selected_pairs = load_table("selected_pairs.csv")
candidate_pairs = load_table("all_candidate_pairs.csv")
baseline_stats = load_table("baseline_backtest_stats.csv")
ml_stats = load_table("ml_vs_baseline_stats.csv")
ml_trade = load_csv("ml_trade_series.csv")


st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">Pair Trading Intelligence Dashboard</h1>
      <p style="margin:8px 0 0 0; opacity:0.95;">
        Cointegration-based statistical arbitrage on Indian equities with baseline strategy and ML trade filter.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if prices.empty:
    st.error("Missing data/nifty_prices_clean.csv. Run the notebooks first.")
    st.stop()

st.sidebar.header("Controls")
sector_option = st.sidebar.selectbox(
    "Sector view",
    ["Banking", "IT", "Auto", "Energy"],
    index=0,
)

show_last_n_days = st.sidebar.slider(
    "Focus window (days)",
    min_value=180,
    max_value=min(2500, len(prices)),
    value=min(900, len(prices)),
    step=30,
)

price_view_mode = st.sidebar.radio(
    "Price visualization mode",
    ["Normalized (base=100)", "Log returns"],
    index=0,
)

z_entry = st.sidebar.slider("Z-score entry threshold", min_value=1.0, max_value=3.5, value=2.0, step=0.1)
z_exit = st.sidebar.slider("Z-score exit threshold", min_value=0.1, max_value=1.5, value=0.5, step=0.1)

pair_labels = []
if not selected_pairs.empty:
    pair_labels = [f"{row.stock_a} vs {row.stock_b}" for row in selected_pairs.itertuples()]

selected_label = st.sidebar.selectbox(
    "Selected pair",
    pair_labels if pair_labels else ["No pair file found"],
    index=0,
)


def metric_value(df: pd.DataFrame, strategy: str, col: str) -> str:
    if df.empty or "strategy" not in df.columns or col not in df.columns:
        return "NA"
    rows = df[df["strategy"] == strategy]
    if rows.empty:
        return "NA"
    val = rows.iloc[0][col]
    if isinstance(val, (int, float, np.floating)):
        if col.lower().startswith("maxdd"):
            return f"{val:.2%}"
        if col in {"CAGR", "Vol"}:
            return f"{val:.2%}"
        return f"{val:.2f}"
    return str(val)


def strategy_insight(base_sharpe: float | None, ml_sharpe: float | None) -> str:
    if base_sharpe is None or ml_sharpe is None:
        return "Strategy comparison is unavailable until both baseline and ML stats are generated."
    delta = ml_sharpe - base_sharpe
    if delta > 0.2:
        return f"RF filtering materially improves risk-adjusted return (Sharpe delta {delta:.2f})."
    if delta > 0:
        return f"RF filtering gives a mild Sharpe improvement (delta {delta:.2f})."
    if delta == 0:
        return "RF and baseline have identical Sharpe in current run."
    return f"Baseline currently beats RF filter on Sharpe (delta {delta:.2f}); retune threshold/probability cutoff."


k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-title">Stocks</div>
          <div class="kpi-value">{prices.shape[1]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-title">Trading Days</div>
          <div class="kpi-value">{prices.shape[0]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-title">Baseline Sharpe</div>
          <div class="kpi-value">{metric_value(baseline_stats, 'zscore_strategy', 'Sharpe')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-title">RF Sharpe</div>
          <div class="kpi-value">{metric_value(ml_stats, 'rf_filtered', 'Sharpe')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"<div class='mono'>Date range: {prices.index.min().date()} to {prices.index.max().date()}</div>",
    unsafe_allow_html=True,
)

base_sharpe = as_float(baseline_stats, "zscore_strategy", "Sharpe")
ml_sharpe = as_float(ml_stats, "rf_filtered", "Sharpe")

st.info(strategy_insight(base_sharpe, ml_sharpe))


sector_map = {
    "Banking": sector_banking,
    "IT": sector_it,
    "Auto": sector_auto,
    "Energy": sector_energy,
}
sector_df = sector_map.get(sector_option, pd.DataFrame())

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Market Structure",
        "Pairs and Cointegration",
        "Strategy Performance",
        "Presentation Assets",
    ]
)

with tab1:
    st.subheader("Normalized price action by sector")
    if not sector_df.empty:
        sector_focus = sector_df.tail(show_last_n_days)
        if price_view_mode == "Normalized (base=100)":
            chart_df = sector_focus / sector_focus.iloc[0] * 100
            y_label = "Index (Base = 100)"
            explain_block(
                "Relative performance of each stock from a common baseline.",
                "Pairs trading needs co-movement. This shows who diverges and by how much.",
                "Lines moving together imply stable relationship; persistent separation suggests weak pair quality.",
            )
        else:
            chart_df = np.log(sector_focus / sector_focus.shift(1)).dropna()
            y_label = "Log Return"
            explain_block(
                "Day-to-day return behavior of each stock.",
                "Returns expose volatility regimes and short-term shocks that can break pair stability.",
                "Frequent sharp spikes mean higher risk; smoother co-moves support robust pair candidates.",
            )

        fig_norm = px.line(chart_df, x=chart_df.index, y=chart_df.columns)
        fig_norm.update_layout(
            height=500,
            template="plotly_white",
            legend_title="Ticker",
            xaxis_title="Date",
            yaxis_title=y_label,
        )
        st.plotly_chart(fig_norm, width="stretch")

        if price_view_mode == "Normalized (base=100)":
            perf = (chart_df.iloc[-1] - 100).sort_values(ascending=False)
            perf_df = perf.rename("return_pct").reset_index()
            perf_df.columns = ["ticker", "return_pct"]
            fig_perf = px.bar(
                perf_df,
                x="ticker",
                y="return_pct",
                color="return_pct",
                color_continuous_scale="Tealgrn",
                title=f"{sector_option}: Relative return over selected window",
            )
            fig_perf.update_layout(template="plotly_white", height=420, xaxis_title="Ticker", yaxis_title="Return %")
            st.plotly_chart(fig_perf, width="stretch")
            st.caption("Context: taller bars indicate stronger outperformance within the sector in the selected period.")

        corr = sector_focus.corr()
        explain_block(
            "Correlation matrix between stocks in the chosen sector.",
            "Cointegration search is more efficient and reliable among highly correlated names.",
            "Cells closer to +1 are stronger co-movers; look for blocks of consistently high values.",
        )
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdBu_r",
                zmid=0,
                colorbar=dict(title="Corr"),
            )
        )
        fig_corr.update_layout(
            title=f"{sector_option} Correlation Heatmap",
            height=550,
            template="plotly_white",
        )
        st.plotly_chart(fig_corr, width="stretch")

        roll_vol = sector_focus.pct_change().rolling(30).std().mean(axis=1) * np.sqrt(252)
        explain_block(
            "Rolling annualized volatility regime of the sector basket.",
            "High volatility often reduces mean-reversion reliability and increases false entries.",
            "Rising regimes call for tighter risk controls; calmer regimes usually improve spread behavior.",
        )
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, mode="lines", name="Annualized Volatility"))
        fig_vol.update_layout(
            title=f"{sector_option}: Regime risk (30D rolling annualized volatility)",
            template="plotly_white",
            height=380,
            xaxis_title="Date",
            yaxis_title="Volatility",
        )
        st.plotly_chart(fig_vol, width="stretch")
        st.caption("Context: volatility spikes often coincide with unstable pair relationships and noisier spread behavior.")
    else:
        st.warning("Sector CSV is missing. Re-run Notebook 1.")

with tab2:
    st.subheader("Candidate and selected pairs")
    explain_block(
        "Statistical screening output for all tested stock pairs.",
        "This stage controls model quality before backtesting. Better screening means fewer bad trades later.",
        "Prefer high train correlation, low cointegration p-value, and stationary spread residuals.",
    )
    if not selected_pairs.empty:
        st.dataframe(selected_pairs, width="stretch")
    else:
        st.warning("selected_pairs.csv not found.")

    if not candidate_pairs.empty:
        st.write("Top candidate pairs by cointegration p-value")
        show_cols = [
            "sector",
            "stock_a",
            "stock_b",
            "corr_train",
            "coint_pvalue",
            "spread_adf_pvalue",
            "cointegrated_5pct",
        ]
        show_cols = [c for c in show_cols if c in candidate_pairs.columns]
        st.dataframe(candidate_pairs[show_cols].head(30), width="stretch")

        if {"corr_train", "coint_pvalue", "sector"}.issubset(candidate_pairs.columns):
            fig_sc = px.scatter(
                candidate_pairs,
                x="corr_train",
                y="coint_pvalue",
                color="sector",
                hover_data=[c for c in ["stock_a", "stock_b", "spread_adf_pvalue"] if c in candidate_pairs.columns],
                title="Pair quality map: high correlation + low p-value preferred",
            )
            fig_sc.add_hline(y=0.05, line_dash="dash", line_color="#dc2626", annotation_text="5% cutoff")
            fig_sc.update_layout(template="plotly_white", height=460)
            st.plotly_chart(fig_sc, width="stretch")
            st.caption("Context: ideal candidates cluster in the lower-right zone (high corr, low cointegration p-value).")

    if not selected_pairs.empty and not prices_train.empty and selected_label != "No pair file found":
        row = selected_pairs.iloc[pair_labels.index(selected_label)]
        a = row["stock_a"]
        b = row["stock_b"]
        beta = float(row["beta_a_on_b"])

        spread = prices_train[a] - beta * prices_train[b]
        z = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()

        spread_df = pd.DataFrame({"Spread": spread, "Zscore": z}).dropna()

        entry_events = spread_df[(spread_df["Zscore"] > z_entry) | (spread_df["Zscore"] < -z_entry)]
        exit_events = spread_df[spread_df["Zscore"].abs() < z_exit]

        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(x=spread_df.index, y=spread_df["Spread"], name="Spread", line=dict(color="#0f766e")))
        fig_spread.add_trace(go.Scatter(x=spread_df.index, y=spread_df["Zscore"], name="Zscore", yaxis="y2", opacity=0.9, line=dict(color="#ea580c")))
        fig_spread.add_hline(y=z_entry, line_dash="dot", line_color="#dc2626", yref="y2")
        fig_spread.add_hline(y=-z_entry, line_dash="dot", line_color="#16a34a", yref="y2")
        fig_spread.add_hline(y=0, line_dash="dash", line_color="#334155", yref="y2")
        fig_spread.add_hline(y=z_exit, line_dash="dash", line_color="#64748b", yref="y2")
        fig_spread.add_hline(y=-z_exit, line_dash="dash", line_color="#64748b", yref="y2")
        fig_spread.add_trace(
            go.Scatter(
                x=entry_events.index,
                y=entry_events["Zscore"],
                mode="markers",
                marker=dict(size=6, color="#dc2626", symbol="diamond"),
                name="Entry trigger",
                yaxis="y2",
            )
        )
        fig_spread.update_layout(
            title=f"Spread and Zscore: {a} vs {b}",
            template="plotly_white",
            height=520,
            yaxis=dict(title="Spread"),
            yaxis2=dict(title="Zscore", overlaying="y", side="right"),
        )
        st.plotly_chart(fig_spread, width="stretch")

        signal_count = int((spread_df["Zscore"].abs() > 2).sum())
        st.caption(
            f"Context: this pair generated {signal_count} extreme events at |Z| > 2. "
            f"With current controls, entry triggers = {len(entry_events)} and neutral-zone exits = {len(exit_events)}."
        )

        c_meta1, c_meta2, c_meta3 = st.columns(3)
        c_meta1.metric("Hedge ratio beta", f"{beta:.3f}")
        c_meta2.metric("Train spread std", f"{spread_df['Spread'].std():.3f}")
        c_meta3.metric("Latest zscore", f"{spread_df['Zscore'].iloc[-1]:.2f}")

with tab3:
    st.subheader("Baseline vs ML-filtered strategy")
    explain_block(
        "Performance comparison between pure statistical signals and ML-filtered signals.",
        "Shows whether ML is actually improving risk-adjusted behavior, not just returns.",
        "Focus on Sharpe, Max Drawdown, and shape of equity/drawdown curves across the same window.",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.write("Baseline stats")
        st.dataframe(baseline_stats, width="stretch")
    with c2:
        st.write("ML vs Baseline stats")
        st.dataframe(ml_stats, width="stretch")

    if not ml_trade.empty and {"equity_base", "equity_ml"}.issubset(set(ml_trade.columns)):
        ml_focus = ml_trade.tail(show_last_n_days).copy()
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=ml_focus.index, y=ml_focus["equity_base"], name="Baseline", line=dict(color="#334155", width=2)))
        fig_eq.add_trace(go.Scatter(x=ml_focus.index, y=ml_focus["equity_ml"], name="RF Filtered", line=dict(color="#0f766e", width=3)))
        fig_eq.update_layout(
            title="Equity Curve Comparison",
            template="plotly_white",
            height=520,
            xaxis_title="Date",
            yaxis_title="Equity",
        )
        st.plotly_chart(fig_eq, width="stretch")

        dd_base = ml_focus["equity_base"] / ml_focus["equity_base"].cummax() - 1
        dd_ml = ml_focus["equity_ml"] / ml_focus["equity_ml"].cummax() - 1
        dd_df = pd.DataFrame({"Baseline DD": dd_base, "RF DD": dd_ml})
        fig_dd = px.area(dd_df, x=dd_df.index, y=dd_df.columns, title="Drawdown profile")
        fig_dd.update_layout(template="plotly_white", height=380, xaxis_title="Date", yaxis_title="Drawdown")
        st.plotly_chart(fig_dd, width="stretch")

        if {"strategy_ret", "strategy_ret_ml"}.issubset(set(ml_focus.columns)):
            roll = pd.DataFrame(index=ml_focus.index)
            roll["Baseline Sharpe (63D)"] = ml_focus["strategy_ret"].rolling(63).mean() / (ml_focus["strategy_ret"].rolling(63).std() + 1e-9) * np.sqrt(252)
            roll["RF Sharpe (63D)"] = ml_focus["strategy_ret_ml"].rolling(63).mean() / (ml_focus["strategy_ret_ml"].rolling(63).std() + 1e-9) * np.sqrt(252)
            fig_roll = px.line(roll.dropna(), x=roll.dropna().index, y=roll.dropna().columns, title="Rolling Sharpe (63D)")
            fig_roll.update_layout(template="plotly_white", height=380, xaxis_title="Date", yaxis_title="Sharpe")
            st.plotly_chart(fig_roll, width="stretch")
            st.caption("Context: rolling Sharpe reveals whether performance stability improved or only one-time gains were captured.")

        cperf1, cperf2 = st.columns(2)
        with cperf1:
            if base_sharpe is not None:
                st.metric("Baseline Sharpe", f"{base_sharpe:.2f}")
        with cperf2:
            if ml_sharpe is not None:
                delta = ml_sharpe - base_sharpe if base_sharpe is not None else 0.0
                st.metric("RF Sharpe", f"{ml_sharpe:.2f}", delta=f"{delta:.2f}")
    else:
        st.info("ml_trade_series.csv not available or missing required columns.")

with tab4:
    st.subheader("Generated report images")
    explain_block(
        "Export-ready visual assets from your notebook pipeline.",
        "Useful for viva, review slides, and submission appendix.",
        "Use these as static evidence while the interactive tabs support deeper drill-down discussion.",
    )
    image_names = [
        "01_raw_prices.png",
        "02_normalised_prices.png",
        "03_sector_heatmaps_train.png",
        "04_best_pair_spread_zscore.png",
        "05_baseline_equity_curve.png",
        "06_ml_equity_and_importance.png",
    ]

    cols = st.columns(2)
    idx = 0
    for img in image_names:
        path = OUT_DIR / img
        if path.exists():
            with cols[idx % 2]:
                st.image(str(path), caption=img, width="stretch")
            idx += 1

    st.markdown("### Data downloads")
    downloadable = {
        "Selected pairs": DATA_DIR / "selected_pairs.csv",
        "All candidate pairs": DATA_DIR / "all_candidate_pairs.csv",
        "Baseline stats": DATA_DIR / "baseline_backtest_stats.csv",
        "ML vs baseline stats": DATA_DIR / "ml_vs_baseline_stats.csv",
    }
    d1, d2 = st.columns(2)
    for i, (name, path) in enumerate(downloadable.items()):
        if path.exists():
            with (d1 if i % 2 == 0 else d2):
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Download {name}",
                        data=f,
                        file_name=path.name,
                        mime="text/csv",
                        width="stretch",
                    )

st.sidebar.markdown("---")
st.sidebar.success("Dashboard is live.")

with st.sidebar.expander("How to read this dashboard"):
    st.write("1) Start with Market Structure to validate sector co-movement.")
    st.write("2) Move to Pairs and Cointegration to inspect statistical quality.")
    st.write("3) Finish with Strategy Performance to compare baseline vs ML filter.")
    st.write("4) Use Presentation Assets and downloads for report material.")
