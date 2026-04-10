from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
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

    /* Force Streamlit metric text to black (value, label, delta) */
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }

    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }

    [data-testid="stMetricDelta"] * {
        color: #000000 !important;
    }

    /* Make download button text white in presentation section */
    [data-testid="stDownloadButton"] button {
        color: #ffffff !important;
    }

    [data-testid="stDownloadButton"] button * {
        color: #ffffff !important;
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


def load_model_table(name: str) -> pd.DataFrame:
    path = MODEL_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_model_json(name: str) -> dict:
    path = MODEL_DIR / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_model_artifact(path_str: str):
    path = Path(path_str)
    if path.exists():
        return joblib.load(path)
    return None


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


def build_pair_features(
    prices_window: pd.DataFrame,
    stock_a: str,
    stock_b: str,
    beta_static: float,
    beta_window: int = 252,
    z_window: int = 30,
    horizon: int = 10,
) -> pd.DataFrame:
    beta_roll = prices_window[stock_a].rolling(beta_window).cov(prices_window[stock_b]) / prices_window[stock_b].rolling(beta_window).var()
    beta = beta_roll.fillna(beta_static)

    spread = prices_window[stock_a] - beta * prices_window[stock_b]
    spread_mean = spread.rolling(z_window).mean()
    spread_std = spread.rolling(z_window).std()
    z = (spread - spread_mean) / spread_std

    ret_a = prices_window[stock_a].pct_change()
    ret_b = prices_window[stock_b].pct_change()
    spread_ret = ret_a - beta * ret_b

    feat = pd.DataFrame(index=prices_window.index)
    feat["z"] = z
    feat["z_chg_5"] = z - z.shift(5)
    feat["z_chg_20"] = z - z.shift(20)
    feat["corr_30"] = prices_window[stock_a].rolling(30).corr(prices_window[stock_b])
    feat["spread_vol_20"] = spread_ret.rolling(20).std()
    feat["spread_vol_60"] = spread_ret.rolling(60).std()
    feat["beta"] = beta
    feat["spread"] = spread

    future_spread = spread.shift(-horizon)
    dist_now = (spread - spread_mean).abs()
    dist_future = (future_spread - spread_mean).abs()
    feat["y"] = (dist_future < dist_now).astype(int)

    return feat.dropna()


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
live_pred = load_table("live_prediction.csv")
pair_registry = load_model_table("pair_model_registry.csv")


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

available_years = sorted(prices.index.year.unique().tolist())
timeline_years = st.sidebar.select_slider(
    "Timeline window (analysis)",
    options=available_years,
    value=(available_years[0], available_years[-1]),
)

analysis_start_year, analysis_end_year = timeline_years
analysis_start = pd.Timestamp(f"{analysis_start_year}-01-01")
analysis_end = pd.Timestamp(f"{analysis_end_year}-12-31")
if analysis_end < analysis_start:
    analysis_start, analysis_end = analysis_end, analysis_start

analysis_label = f"{analysis_start.year} to {analysis_end.year}"

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

st.markdown("### Live Model Recommendation")
if not live_pred.empty:
    row_live = live_pred.iloc[0]
    live_date = str(row_live.get("date", "NA"))
    live_pair = f"{row_live.get('stock_a', 'NA')} vs {row_live.get('stock_b', 'NA')}"
    live_action = str(row_live.get("final_action", "NO_TRADE"))
    baseline_action = str(row_live.get("baseline_action", "NO_TRADE"))
    live_prob = float(row_live.get("rf_probability", np.nan))
    live_z = float(row_live.get("zscore", np.nan))
    live_thr = float(row_live.get("threshold", np.nan))

    if live_action == "LONG_SPREAD":
        action_color = "#166534"
        action_bg = "#dcfce7"
        action_text = "LONG SPREAD"
    elif live_action == "SHORT_SPREAD":
        action_color = "#9f1239"
        action_bg = "#ffe4e6"
        action_text = "SHORT SPREAD"
    else:
        action_color = "#1e293b"
        action_bg = "#e2e8f0"
        action_text = "NO TRADE"

    st.markdown(
        f"""
        <div style=\"background:#ffffff;border:1px solid rgba(148,163,184,0.25);border-radius:14px;padding:14px 16px;\">
            <div style=\"font-size:13px;color:#334155;\">As of {live_date} | Pair: {live_pair}</div>
            <div style=\"margin-top:8px;display:inline-block;padding:6px 12px;border-radius:999px;background:{action_bg};color:{action_color};font-weight:700;\">{action_text}</div>
            <div style=\"margin-top:10px;font-size:13px;color:#0f172a;\">RF confidence: {live_prob:.3f} (threshold {live_thr:.2f}) | Current Z-score: {live_z:.3f}</div>
            <div style=\"margin-top:6px;font-size:13px;color:#475569;\">Baseline signal: {baseline_action}. Final action is after ML filter.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("Live prediction not found. Run model_pipeline.py to generate data/live_prediction.csv.")


sector_map = {
    "Banking": sector_banking,
    "IT": sector_it,
    "Auto": sector_auto,
    "Energy": sector_energy,
}
sector_df = sector_map.get(sector_option, pd.DataFrame())

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Market Structure",
        "Pairs and Cointegration",
        "Strategy Performance",
        "Timeline Model",
        "Outcome Story",
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
    st.subheader("Timeline Model")
    explain_block(
        "A date-range analysis mode that ranks pairs with the trained models.",
        "This answers: for a chosen year window, which pair looks best and what action should be taken.",
        "Use the timeline selector in the sidebar. The table is ranked by model score, then the top pair is analyzed in detail.",
    )

    if pair_registry.empty:
        st.warning("Pair model registry not found. Run model_pipeline.py first.")
    else:
        feature_cols = ["z", "z_chg_5", "z_chg_20", "corr_30", "spread_vol_20", "spread_vol_60"]
        scored_rows: list[dict[str, object]] = []
        window_start_buffer = analysis_start - pd.Timedelta(days=420)
        past_window = analysis_end <= pd.Timestamp(prices.index.max())

        for _, reg_row in pair_registry.iterrows():
            model = load_model_artifact(str(reg_row["model_path"]))
            if model is None:
                continue

            stock_a = str(reg_row["stock_a"])
            stock_b = str(reg_row["stock_b"])
            beta_static = float(reg_row["beta_static"])
            threshold = float(reg_row.get("threshold", 0.55))

            price_slice = prices.loc[window_start_buffer:analysis_end].copy()
            if stock_a not in price_slice.columns or stock_b not in price_slice.columns:
                continue

            feature_df = build_pair_features(price_slice, stock_a, stock_b, beta_static)
            window_df = feature_df.loc[analysis_start:analysis_end].copy()
            if window_df.empty:
                continue

            probs = pd.Series(model.predict_proba(window_df[feature_cols])[:, 1], index=window_df.index)
            base_actions = np.where(window_df["z"] > z_entry, "SHORT_SPREAD", np.where(window_df["z"] < -z_entry, "LONG_SPREAD", "NO_TRADE"))
            model_actions = np.where((probs >= threshold) & (base_actions != "NO_TRADE"), base_actions, "NO_TRADE")

            trade_rate = float((model_actions != "NO_TRADE").mean())
            mean_prob = float(probs.mean())
            hit_rate = float(window_df["y"].mean())
            latest_action = str(model_actions[-1])
            latest_prob = float(probs.iloc[-1])
            score = 0.45 * float(reg_row["validation_f1"]) + 0.35 * mean_prob + 0.20 * trade_rate

            model_return = np.nan
            base_return = np.nan
            pred_accuracy = np.nan
            if past_window:
                spread_ret = window_df[stock_a].pct_change().fillna(0) - window_df["beta"] * window_df[stock_b].pct_change().fillna(0)
                base_pos = pd.Series(np.where(base_actions == "LONG_SPREAD", 1, np.where(base_actions == "SHORT_SPREAD", -1, 0)), index=window_df.index)
                model_pos = pd.Series(np.where(model_actions == "LONG_SPREAD", 1, np.where(model_actions == "SHORT_SPREAD", -1, 0)), index=window_df.index)
                base_curve = 1 + (base_pos.shift(1).fillna(0) * spread_ret - base_pos.diff().abs().fillna(0) * 0.001)
                model_curve = 1 + (model_pos.shift(1).fillna(0) * spread_ret - model_pos.diff().abs().fillna(0) * 0.001)
                base_return = float(base_curve.prod() - 1)
                model_return = float(model_curve.prod() - 1)
                pred_accuracy = float((pd.Series((model_actions != "NO_TRADE").astype(int), index=window_df.index) == window_df["y"]).mean())

            scored_rows.append(
                {
                    "model_name": reg_row["model_name"],
                    "stock_a": stock_a,
                    "stock_b": stock_b,
                    "validation_f1": float(reg_row["validation_f1"]),
                    "validation_roc_auc": float(reg_row["validation_roc_auc"]),
                    "mean_probability": mean_prob,
                    "trade_rate": trade_rate,
                    "hit_rate": hit_rate,
                    "latest_action": latest_action,
                    "latest_probability": latest_prob,
                    "model_score": score,
                    "threshold": threshold,
                    "window_model_return": model_return,
                    "window_baseline_return": base_return,
                    "prediction_accuracy": pred_accuracy,
                    "model_path": reg_row["model_path"],
                }
            )

        if not scored_rows:
            st.warning("No pair models could be evaluated for the selected window.")
        else:
            scored_df = pd.DataFrame(scored_rows).sort_values(
                ["model_score", "validation_f1", "mean_probability"],
                ascending=[False, False, False],
            )
            best = scored_df.iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best pair", f"{best.stock_a} vs {best.stock_b}")
            c2.metric("Best action", str(best.latest_action))
            c3.metric("Model score", f"{best.model_score:.3f}")
            c4.metric("Avg RF prob", f"{best.mean_probability:.3f}")

            st.dataframe(
                scored_df[[
                    "model_name",
                    "stock_a",
                    "stock_b",
                    "model_score",
                    "validation_f1",
                    "mean_probability",
                    "trade_rate",
                    "latest_action",
                    "latest_probability",
                ]].head(10),
                width="stretch",
            )

            best_model = load_model_artifact(str(best["model_path"]))
            if best_model is not None:
                best_price_slice = prices.loc[window_start_buffer:analysis_end].copy()
                best_feat = build_pair_features(best_price_slice, str(best["stock_a"]), str(best["stock_b"]), float(pair_registry.loc[pair_registry["model_name"] == best["model_name"], "beta_static"].iloc[0]))
                best_window = best_feat.loc[analysis_start:analysis_end].copy()

                if not best_window.empty:
                    best_probs = pd.Series(best_model.predict_proba(best_window[feature_cols])[:, 1], index=best_window.index)
                    best_base_actions = np.where(best_window["z"] > z_entry, "SHORT_SPREAD", np.where(best_window["z"] < -z_entry, "LONG_SPREAD", "NO_TRADE"))
                    best_model_actions = np.where((best_probs >= float(best["threshold"])) & (best_base_actions != "NO_TRADE"), best_base_actions, "NO_TRADE")

                    st.markdown("#### Prediction vs Reality")
                    explain_block(
                        "Compares model signal to realized outcome in the selected past window.",
                        "Shows whether the model is calling good trades or just producing confident predictions.",
                        "If predicted trade = 1 but actual = 0, the model would have taken a bad trade.",
                    )

                    pred_df = pd.DataFrame(
                        {
                            "RF Probability": best_probs,
                            "Actual Outcome": best_window["y"].rolling(10).mean(),
                        }
                    ).dropna()
                    fig_pred = px.line(pred_df, x=pred_df.index, y=pred_df.columns, title="Predicted confidence vs realized outcome")
                    fig_pred.update_layout(template="plotly_white", height=380, xaxis_title="Date", yaxis_title="Value")
                    st.plotly_chart(fig_pred, width="stretch")

                    pred_trade = pd.Series((best_model_actions != "NO_TRADE").astype(int), index=best_window.index)
                    actual_trade = best_window["y"].astype(int)
                    confusion = pd.crosstab(pred_trade, actual_trade, rownames=["Predicted Trade"], colnames=["Actual Outcome"], dropna=False)
                    st.write("Prediction vs actual table")
                    st.dataframe(confusion, width="stretch")

                    st.markdown("#### Profit / Loss")
                    explain_block(
                        "Separate performance view for the same chosen window.",
                        "This answers whether the model made or lost money over the selected years.",
                        "Positive ending equity means profit; negative means loss after simple transaction costs.",
                    )

                    spread_ret = best_window[str(best["stock_a"])].pct_change().fillna(0) - best_window["beta"] * best_window[str(best["stock_b"])].pct_change().fillna(0)
                    base_pos = pd.Series(np.where(best_base_actions == "LONG_SPREAD", 1, np.where(best_base_actions == "SHORT_SPREAD", -1, 0)), index=best_window.index)
                    model_pos = pd.Series(np.where(best_model_actions == "LONG_SPREAD", 1, np.where(best_model_actions == "SHORT_SPREAD", -1, 0)), index=best_window.index)
                    base_curve = (1 + (base_pos.shift(1).fillna(0) * spread_ret - base_pos.diff().abs().fillna(0) * 0.001)).cumprod()
                    model_curve = (1 + (model_pos.shift(1).fillna(0) * spread_ret - model_pos.diff().abs().fillna(0) * 0.001)).cumprod()

                    curve_df = pd.DataFrame({"Baseline": base_curve, "ML Filtered": model_curve})
                    fig_curve = px.line(curve_df, x=curve_df.index, y=curve_df.columns, title="Profit / Loss curve")
                    fig_curve.update_layout(template="plotly_white", height=420, xaxis_title="Date", yaxis_title="Equity")
                    st.plotly_chart(fig_curve, width="stretch")

                    model_return = float(model_curve.iloc[-1] - 1)
                    base_return = float(base_curve.iloc[-1] - 1)
                    pnl_col1, pnl_col2, pnl_col3 = st.columns(3)
                    pnl_col1.metric("Model return", f"{model_return:.2%}")
                    pnl_col2.metric("Baseline return", f"{base_return:.2%}")
                    pnl_col3.metric("Return delta", f"{(model_return - base_return):.2%}")

                    if model_return > 0:
                        st.success("This time window produced profit after filtering by the model.")
                    elif model_return < 0:
                        st.error("This time window produced a loss after filtering by the model.")
                    else:
                        st.info("This time window was approximately breakeven after filtering by the model.")

            st.markdown("#### Why this pair was chosen")
            st.write("- Highest combined model score across the chosen window.")
            st.write("- Strong validation performance from the pair-specific classifier.")
            st.write("- Acceptable signal rate without firing on every noisy move.")

            if not past_window:
                st.warning("The selected end year extends beyond currently available historical data, so prediction-vs-reality and P/L are unavailable.")

with tab5:
    st.subheader("What this model got us")
    explain_block(
        "A plain-language outcome summary from baseline and ML-filtered strategy metrics.",
        "This is the business-style answer to: what did the model improve, what did it hurt, and what should we do next.",
        "Green signals indicate meaningful improvement; yellow means mixed; red means the filter likely needs retuning.",
    )

    base_cagr = as_float(baseline_stats, "zscore_strategy", "CAGR")
    base_mdd = as_float(baseline_stats, "zscore_strategy", "MaxDD")
    rf_cagr = as_float(ml_stats, "rf_filtered", "CAGR")
    rf_mdd = as_float(ml_stats, "rf_filtered", "MaxDD")

    final_base = None
    final_ml = None
    accept_rate = None
    active_base = None
    active_ml = None
    if not ml_trade.empty:
        if "equity_base" in ml_trade.columns:
            final_base = float(ml_trade["equity_base"].iloc[-1])
        if "equity_ml" in ml_trade.columns:
            final_ml = float(ml_trade["equity_ml"].iloc[-1])
        if "rf_accept" in ml_trade.columns:
            accept_rate = float(ml_trade["rf_accept"].mean())
        if "position_lag" in ml_trade.columns:
            active_base = float((ml_trade["position_lag"].abs() > 0).mean())
        if "position_ml" in ml_trade.columns:
            active_ml = float((ml_trade["position_ml"].abs() > 0).mean())

    summary = pd.DataFrame(
        [
            {
                "Metric": "Sharpe",
                "Baseline": base_sharpe,
                "RF Filtered": ml_sharpe,
                "Delta (RF - Base)": None if (base_sharpe is None or ml_sharpe is None) else (ml_sharpe - base_sharpe),
            },
            {
                "Metric": "CAGR",
                "Baseline": base_cagr,
                "RF Filtered": rf_cagr,
                "Delta (RF - Base)": None if (base_cagr is None or rf_cagr is None) else (rf_cagr - base_cagr),
            },
            {
                "Metric": "Max Drawdown",
                "Baseline": base_mdd,
                "RF Filtered": rf_mdd,
                "Delta (RF - Base)": None if (base_mdd is None or rf_mdd is None) else (rf_mdd - base_mdd),
            },
            {
                "Metric": "Final Equity",
                "Baseline": final_base,
                "RF Filtered": final_ml,
                "Delta (RF - Base)": None if (final_base is None or final_ml is None) else (final_ml - final_base),
            },
        ]
    )

    def fmt_num(v: float | None, pct: bool = False) -> str:
        if v is None or pd.isna(v):
            return "NA"
        return f"{v:.2%}" if pct else f"{v:.3f}"

    c_out1, c_out2, c_out3, c_out4 = st.columns(4)
    c_out1.metric("Sharpe delta", fmt_num(None if (base_sharpe is None or ml_sharpe is None) else (ml_sharpe - base_sharpe)))
    c_out2.metric("CAGR delta", fmt_num(None if (base_cagr is None or rf_cagr is None) else (rf_cagr - base_cagr), pct=True))
    c_out3.metric("Drawdown delta", fmt_num(None if (base_mdd is None or rf_mdd is None) else (rf_mdd - base_mdd), pct=True))
    c_out4.metric("RF acceptance", fmt_num(accept_rate, pct=True))

    story_lines = []
    if base_sharpe is not None and ml_sharpe is not None:
        if ml_sharpe > base_sharpe:
            story_lines.append(f"Risk-adjusted performance improved: Sharpe increased from {base_sharpe:.2f} to {ml_sharpe:.2f}.")
        else:
            story_lines.append(f"Risk-adjusted performance weakened: Sharpe moved from {base_sharpe:.2f} to {ml_sharpe:.2f}.")

    if base_mdd is not None and rf_mdd is not None:
        if rf_mdd > base_mdd:
            story_lines.append(f"Capital protection improved: max drawdown reduced from {base_mdd:.2%} to {rf_mdd:.2%}.")
        else:
            story_lines.append(f"Drawdown worsened: max drawdown changed from {base_mdd:.2%} to {rf_mdd:.2%}.")

    if accept_rate is not None:
        story_lines.append(f"Model selectivity: RF accepted {accept_rate:.1%} of candidate trade opportunities.")
    if active_base is not None and active_ml is not None:
        story_lines.append(
            f"Exposure changed from {active_base:.1%} (baseline active time) to {active_ml:.1%} (RF active time)."
        )

    if not summary.empty:
        show_summary = summary.copy()
        for col in ["Baseline", "RF Filtered", "Delta (RF - Base)"]:
            show_summary[col] = show_summary[col].apply(lambda x: np.nan if x is None else x)
        st.dataframe(show_summary, width="stretch")

    if story_lines:
        st.markdown("### What happened")
        for line in story_lines:
            st.write(f"- {line}")

    if base_sharpe is not None and ml_sharpe is not None and base_mdd is not None and rf_mdd is not None:
        improved_sharpe = ml_sharpe > base_sharpe
        improved_drawdown = rf_mdd > base_mdd
        if improved_sharpe and improved_drawdown:
            st.success("Outcome: ML filter added value by improving quality and reducing risk.")
        elif improved_sharpe or improved_drawdown:
            st.warning("Outcome: ML filter gave mixed results. Keep it, but tune threshold/features.")
        else:
            st.error("Outcome: ML filter underperformed baseline. Retune model or revert to baseline rules.")

    st.markdown("### Why this likely happened")
    st.write("- RF model is acting as a gate, rejecting part of the raw Z-score signals.")
    st.write("- If rejected signals were mostly noisy, Sharpe and drawdown improve.")
    st.write("- If rejected signals included many profitable trades, CAGR and final equity can drop.")
    st.write("- Balance this trade-off by tuning probability cutoff and feature set.")

with tab6:
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
