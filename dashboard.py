import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #f4f7f2;
        --card: #ffffff;
        --ink: #0f172a;
        --muted: #55635a;
        --brand: #0f766e;
        --brand-dark: #0b5c56;
        --accent: #b8790f;
        --gain: #157a45;
        --loss: #b7362b;
        --line: rgba(15, 46, 38, 0.10);
    }

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(15, 118, 110, 0.08), transparent 32%),
            radial-gradient(circle at 88% 12%, rgba(184, 121, 15, 0.08), transparent 34%),
            linear-gradient(160deg, #f7faf5 0%, #f0f4ec 100%);
        color: var(--ink);
    }

    h1, h2, h3 { color: var(--ink); letter-spacing: 0.1px; font-weight: 700; }

    /* ---------- hero ---------- */
    .hero {
        background: linear-gradient(135deg, #0b5c56 0%, #0f766e 55%, #14b8a6 100%);
        border-radius: 20px;
        padding: 28px 30px;
        color: white;
        box-shadow: 0 14px 34px rgba(6, 40, 34, 0.22);
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(circle at 85% -10%, rgba(255,255,255,0.18), transparent 55%);
        pointer-events: none;
    }
    .hero .eyebrow {
        font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; letter-spacing: 0.14em;
        text-transform: uppercase; opacity: 0.85; margin-bottom: 8px;
    }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p.sub { margin: 8px 0 0 0; opacity: 0.95; font-size: 1.02rem; max-width: 62ch; }

    /* ---------- kpi cards ---------- */
    .kpi {
        background: var(--card);
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 6px 18px rgba(15, 46, 38, 0.06);
        border: 1px solid var(--line);
        transition: box-shadow .15s ease, transform .15s ease;
    }
    .kpi:hover { box-shadow: 0 10px 26px rgba(15, 46, 38, 0.10); transform: translateY(-1px); }
    .kpi-title {
        color: var(--muted); font-size: 12.5px; margin-bottom: 5px;
        font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 0.04em;
    }
    .kpi-value { color: var(--ink); font-size: 25px; font-weight: 700; line-height: 1.1; }
    .kpi-value.brand { color: var(--brand-dark); }
    .kpi-value.accent { color: var(--accent); }

    .mono { font-family: 'IBM Plex Mono', monospace; color: var(--muted); font-size: 12px; }

    /* ---------- section framing ---------- */
    .section-card {
        background: var(--card); border: 1px solid var(--line); border-radius: 16px;
        padding: 16px 18px; box-shadow: 0 6px 18px rgba(15, 46, 38, 0.05); margin: 4px 0 14px 0;
    }
    .callout {
        background: #ffffffcc; border: 1px solid var(--line); border-left: 3px solid var(--accent);
        border-radius: 12px; padding: 12px 16px; margin: 8px 0 14px 0;
    }
    .callout .what, .callout .why, .callout .how { font-size: 13px; color: var(--ink); margin: 2px 0; }
    .callout b { color: var(--brand-dark); }

    .pill { display:inline-block; padding:6px 14px; border-radius:999px; font-weight:700; font-size: 13px; }
    .pill.long { background:#dcfce7; color:#166534; }
    .pill.short { background:#ffe4e6; color:#9f1239; }
    .pill.flat { background:#e2e8f0; color:#1e293b; }

    /* ---------- tabs ---------- */
    div[data-baseweb="tab-list"] {
        gap: 4px; border-bottom: 1px solid var(--line);
    }
    div[data-baseweb="tab-list"] button {
        color: var(--muted) !important;
        font-weight: 600;
        border-radius: 10px 10px 0 0 !important;
    }
    div[data-baseweb="tab-list"] button p { color: inherit !important; font-size: 13.5px; }
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        color: var(--brand-dark) !important;
        background: rgba(15, 118, 110, 0.08);
    }
    div[data-baseweb="tab-highlight"] { background-color: var(--brand) !important; }

    /* ---------- metrics ---------- */
    [data-testid="stMetric"] {
        background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 10px 14px 6px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: 'IBM Plex Mono', monospace; font-size: 11.5px !important; text-transform: uppercase; letter-spacing: .03em; }
    [data-testid="stMetricValue"] {
        color: var(--ink) !important; font-size: 1.65rem !important;
        white-space: normal !important; overflow-wrap: anywhere; line-height: 1.15;
    }
    [data-testid="stMetricDelta"] svg { display: inline; }

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--brand-dark); }

    /* ---------- download buttons ---------- */
    [data-testid="stDownloadButton"] button {
        background: var(--brand) !important; color: #ffffff !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
    }
    [data-testid="stDownloadButton"] button:hover { background: var(--brand-dark) !important; }

    /* ---------- dataframes ---------- */
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }

    hr { border-color: var(--line); }
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
        <div class="callout">
            <div class="what"><b>What</b> &nbsp;{what}</div>
            <div class="why"><b>Why</b> &nbsp;{why}</div>
            <div class="how"><b>How to read</b> &nbsp;{how}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


from model_pipeline import FEATURE_COLS, build_features as build_pair_features
from paper_trading import CAPITAL_PER_PAIR as CAPITAL_PER_PAIR_DASHBOARD


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
walk_forward_stats = load_table("walk_forward_stats.csv")
walk_forward_equity = load_table("walk_forward_equity.csv")
paper_positions = load_table("paper_trading_positions.csv")
paper_trades = load_table("paper_trading_trades.csv")
paper_equity = load_table("paper_trading_equity.csv")


best_meta = load_model_json("rf_pair_model_meta.json")

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Spread &amp; Signal &middot; NSE Statistical Arbitrage</div>
      <h1>Pair Trading Intelligence Dashboard</h1>
      <p class="sub">
        Cointegration-based statistical arbitrage on Indian equities &mdash; screening, a Random Forest trade filter,
        a realistic NSE cost &amp; risk model, walk-forward validation, and a live paper-trading ledger, in one view.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if prices.empty:
    st.error("Missing data/nifty_prices_clean.csv. Run the notebooks first.")
    st.stop()

if best_meta:
    st.caption(
        f"Top-ranked model (by validation F1): **{best_meta.get('stock_a', '?')} vs {best_meta.get('stock_b', '?')}** "
        f"&middot; validation F1 {float(best_meta.get('validation_f1', 0)):.3f} "
        f"&middot; test ROC-AUC {float(best_meta.get('test_roc_auc', 0)):.3f} "
        f"&middot; as of {best_meta.get('dataset_last_date', 'n/a')}"
    )

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
        pill_class, action_text = "long", "LONG SPREAD"
    elif live_action == "SHORT_SPREAD":
        pill_class, action_text = "short", "SHORT SPREAD"
    else:
        pill_class, action_text = "flat", "NO TRADE"

    st.markdown(
        f"""
        <div class="section-card">
            <div class="mono" style="font-size:13px;">As of {live_date} &nbsp;&middot;&nbsp; Pair: {live_pair}</div>
            <div style="margin-top:10px;"><span class="pill {pill_class}">{action_text}</span></div>
            <div style="margin-top:10px;font-size:13px;color:var(--ink);">RF confidence: <b>{live_prob:.3f}</b> (threshold {live_thr:.2f}) &nbsp;&middot;&nbsp; Current Z-score: <b>{live_z:.3f}</b></div>
            <div style="margin-top:6px;font-size:13px;color:var(--muted);">Baseline signal: {baseline_action}. Final action is after the ML filter.</div>
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "📈 Market Structure",
        "🔗 Pairs & Cointegration",
        "⚖️ Strategy Performance",
        "🕒 Timeline Model",
        "📝 Outcome Story",
        "🖼️ Presentation Assets",
        "🧪 Walk-Forward Validation",
        "📒 Live Paper Trading",
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

    st.markdown("---")
    st.markdown("### Flat-cost backtest vs. realistic-cost walk-forward")
    explain_block(
        "Compares the numbers above (flat 10bps-per-turnover, unlimited position size) against the "
        "walk-forward result under costs.py's realistic Indian cash-equity cost model (STT, exchange/SEBI "
        "charges, stamp duty, GST, SLB short-borrow cost, volatility-scaled slippage) and risk.py's "
        "volatility-targeted position sizing with a z-score blowout / time stop.",
        "A strategy that only looks good under a flat 10bps assumption is not evidence of a real edge.",
        "If Sharpe/CAGR survive under realistic costs, that is much stronger evidence for the paper than the "
        "flat-cost baseline alone.",
    )
    if walk_forward_stats.empty:
        st.warning("walk_forward_stats.csv not found. Run walk_forward.py first.")
    else:
        agg = walk_forward_stats[walk_forward_stats["fold"] == "ALL"]
        st.dataframe(agg, width="stretch")
        portfolio_row = agg[agg["pair"] == "PORTFOLIO"]
        if not portfolio_row.empty and base_sharpe is not None:
            wf_sharpe = float(portfolio_row.iloc[0]["Sharpe"])
            delta = wf_sharpe - base_sharpe
            st.metric("Walk-forward portfolio Sharpe (realistic costs)", f"{wf_sharpe:.2f}", delta=f"{delta:.2f} vs flat-cost baseline")

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
        with st.expander("Full pair model registry (all trained models, raw metrics)"):
            registry_cols = [
                c
                for c in [
                    "model_name", "stock_a", "stock_b", "beta_static", "threshold",
                    "final_action", "live_probability", "live_zscore",
                    "validation_f1", "validation_roc_auc", "test_f1", "test_roc_auc",
                    "dataset_last_date",
                ]
                if c in pair_registry.columns
            ]
            st.dataframe(pair_registry[registry_cols], width="stretch")

        feature_cols = FEATURE_COLS
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
                aligned_prices = price_slice.loc[window_df.index, [stock_a, stock_b]].copy()
                aligned_prices["beta"] = window_df["beta"]
                spread_ret = aligned_prices[stock_a].pct_change().fillna(0) - aligned_prices["beta"] * aligned_prices[stock_b].pct_change().fillna(0)
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

            pill_class = "flat"
            if str(best.latest_action) == "LONG_SPREAD":
                pill_class = "long"
            elif str(best.latest_action) == "SHORT_SPREAD":
                pill_class = "short"

            st.markdown(
                f"""
                <div class="section-card">
                    <div class="mono" style="font-size:13px;">Timeline summary for <b>{analysis_label}</b></div>
                    <div style="margin-top:8px;font-size:19px;color:var(--ink);font-weight:700;">Best Pair: {best.stock_a} vs {best.stock_b}</div>
                    <div style="margin-top:8px;"><span class="pill {pill_class}">Recommended: {best.latest_action}</span></div>
                    <div style="margin-top:10px;font-size:13px;color:var(--muted);">Model score: <b>{best.model_score:.3f}</b> &nbsp;&middot;&nbsp; Validation F1: <b>{best.validation_f1:.3f}</b> &nbsp;&middot;&nbsp; Avg RF prob: <b>{best.mean_probability:.3f}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
            short_a = str(best.stock_a).replace(".NS", "")
            short_b = str(best.stock_b).replace(".NS", "")
            c1.metric("Best pair", f"{short_a}/{short_b}")
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
                    st.write("Prediction vs actual")

                    # force 2x2 layout for consistent interpretation
                    confusion = confusion.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                    fig_cm = go.Figure(
                        data=go.Heatmap(
                            z=confusion.values,
                            x=["Actual 0 (No convergence)", "Actual 1 (Convergence)"],
                            y=["Pred 0 (Skip)", "Pred 1 (Take trade)"],
                            colorscale="Blues",
                            text=confusion.values,
                            texttemplate="%{text}",
                            textfont={"size": 14},
                            colorbar=dict(title="Count"),
                        )
                    )
                    fig_cm.update_layout(template="plotly_white", height=360, title="Confusion Matrix (Predicted vs Reality)")
                    st.plotly_chart(fig_cm, width="stretch")

                    hit_rate = float((pred_trade == actual_trade).mean())
                    if hit_rate >= 0.60:
                        st.success(f"Prediction quality looks strong in this window (accuracy {hit_rate:.1%}).")
                    elif hit_rate >= 0.50:
                        st.warning(f"Prediction quality is moderate in this window (accuracy {hit_rate:.1%}).")
                    else:
                        st.error(f"Prediction quality is weak in this window (accuracy {hit_rate:.1%}).")

                    st.markdown("#### Profit / Loss")
                    explain_block(
                        "Separate performance view for the same chosen window.",
                        "This answers whether the model made or lost money over the selected years.",
                        "Positive ending equity means profit; negative means loss after simple transaction costs.",
                    )

                    aligned_prices = best_price_slice.loc[best_window.index, [str(best["stock_a"]), str(best["stock_b"])]].copy()
                    aligned_prices["beta"] = best_window["beta"]
                    spread_ret = aligned_prices[str(best["stock_a"])].pct_change().fillna(0) - aligned_prices["beta"] * aligned_prices[str(best["stock_b"])].pct_change().fillna(0)
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

                    pnl_class = "flat"
                    pnl_text = "BREAKEVEN"
                    if model_return > 0:
                        pnl_class, pnl_text = "long", "PROFIT"
                    elif model_return < 0:
                        pnl_class, pnl_text = "short", "LOSS"

                    st.markdown(
                        f'<span class="pill {pnl_class}">Window Result: {pnl_text}</span>',
                        unsafe_allow_html=True,
                    )

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

with tab7:
    st.subheader("Walk-forward validation (out-of-sample, realistic costs)")
    explain_block(
        "Every fold retrains the model on an expanding historical window and evaluates strictly on the "
        "next unseen block, then chains all out-of-sample blocks into one continuous equity curve.",
        "A single static train/valid/test split (used elsewhere in this app) is weak evidence of a real "
        "edge -- walk-forward is the standard way to show a strategy isn't just fit to one lucky window.",
        "Compare per-fold Sharpe/CAGR/MaxDD for consistency across time, and look at the chained portfolio "
        "equity curve for the overall out-of-sample track record.",
    )

    if walk_forward_stats.empty or walk_forward_equity.empty:
        st.warning("walk_forward_stats.csv / walk_forward_equity.csv not found. Run walk_forward.py first.")
    else:
        fold_stats = walk_forward_stats[walk_forward_stats["fold"] != "ALL"]
        st.write("Per-fold metrics")
        st.dataframe(fold_stats, width="stretch")

        st.write("Aggregate metrics (all folds chained)")
        st.dataframe(walk_forward_stats[walk_forward_stats["fold"] == "ALL"], width="stretch")

        portfolio_curve = walk_forward_equity[walk_forward_equity["pair"] == "PORTFOLIO"].copy()
        if not portfolio_curve.empty:
            portfolio_curve["date"] = pd.to_datetime(portfolio_curve["date"])
            fig_wf = go.Figure()
            fig_wf.add_trace(
                go.Scatter(x=portfolio_curve["date"], y=portfolio_curve["equity"], name="Walk-forward portfolio equity", line=dict(color="#0f766e", width=3))
            )
            fig_wf.update_layout(title="Chained out-of-sample portfolio equity curve", template="plotly_white", height=480, xaxis_title="Date", yaxis_title="Equity")
            st.plotly_chart(fig_wf, width="stretch")

        pair_options = [p for p in walk_forward_equity["pair"].unique() if p != "PORTFOLIO"]
        if pair_options:
            wf_pair = st.selectbox("Inspect one pair's walk-forward equity", pair_options, key="wf_pair_select")
            pair_curve = walk_forward_equity[walk_forward_equity["pair"] == wf_pair].copy()
            pair_curve["date"] = pd.to_datetime(pair_curve["date"])
            fig_pair_wf = go.Figure()
            fig_pair_wf.add_trace(go.Scatter(x=pair_curve["date"], y=pair_curve["equity"], name=wf_pair, line=dict(color="#334155")))
            for fold_id in pair_curve["fold"].unique():
                fold_start = pair_curve[pair_curve["fold"] == fold_id]["date"].min()
                fig_pair_wf.add_vline(x=fold_start, line_dash="dot", line_color="#94a3b8")
            fig_pair_wf.update_layout(title=f"{wf_pair}: walk-forward equity (dotted lines = fold boundaries)", template="plotly_white", height=420, xaxis_title="Date", yaxis_title="Equity")
            st.plotly_chart(fig_pair_wf, width="stretch")

with tab8:
    st.subheader("Live paper trading (forward test, no broker required)")
    explain_block(
        "An accumulating day-by-day ledger: each day paper_trading.py evaluates every pair's trained model "
        "against that day's real closing prices and logs what the strategy would have done, under the same "
        "realistic cost and risk-sizing model as the walk-forward backtest.",
        "This is genuine out-of-sample forward-test evidence -- generated after the model was trained, not "
        "backtested on history the model could have learned from. Strong evidence for a research paper.",
        "Run `python paper_trading.py` daily (see the scheduling command in its module docstring) to build "
        "up this track record over time.",
    )

    if paper_equity.empty:
        st.warning("No paper-trading history yet. Run `python paper_trading.py` to start the ledger.")
    else:
        equity_wide = paper_equity.copy()
        equity_wide["date"] = pd.to_datetime(equity_wide["date"])
        portfolio_daily = equity_wide.groupby("date")["pnl"].sum().sort_index()
        portfolio_equity_curve = portfolio_daily.cumsum() + CAPITAL_PER_PAIR_DASHBOARD * equity_wide["pair"].nunique()

        c_pt1, c_pt2, c_pt3 = st.columns(3)
        c_pt1.metric("Trading days logged", f"{portfolio_daily.shape[0]}")
        c_pt2.metric("Portfolio equity", f"{portfolio_equity_curve.iloc[-1]:,.0f}")
        c_pt3.metric("Cumulative P&L", f"{portfolio_daily.sum():,.0f}")

        fig_paper = go.Figure()
        fig_paper.add_trace(go.Scatter(x=portfolio_equity_curve.index, y=portfolio_equity_curve.values, name="Paper-trading portfolio equity", line=dict(color="#ea580c", width=3)))
        fig_paper.update_layout(title="Live paper-trading equity curve", template="plotly_white", height=460, xaxis_title="Date", yaxis_title="Equity")
        st.plotly_chart(fig_paper, width="stretch")

        st.write("Current open positions")
        if not paper_positions.empty:
            open_pos = paper_positions[paper_positions["direction"] != 0]
            st.dataframe(open_pos if not open_pos.empty else paper_positions, width="stretch")
        else:
            st.info("No positions file yet.")

        st.write("Closed-trade log")
        if not paper_trades.empty:
            st.dataframe(paper_trades, width="stretch")
        else:
            st.info("No closed trades logged yet.")

st.sidebar.markdown("---")
st.sidebar.success("Dashboard is live.")

with st.sidebar.expander("How to read this dashboard"):
    st.write("1) Start with Market Structure to validate sector co-movement.")
    st.write("2) Move to Pairs and Cointegration to inspect statistical quality.")
    st.write("3) Finish with Strategy Performance to compare baseline vs ML filter.")
    st.write("4) Use Presentation Assets and downloads for report material.")
