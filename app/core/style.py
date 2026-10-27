"""Colours, the Plotly template, page CSS and small layout helpers shared by every page."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---- palette (validated categorical order; text never wears a series colour)
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

STRATEGY = "#2a78d6"   # the strategy line, everywhere
BENCHMARK = "#eb6834"  # Nifty 200 TRI / index, everywhere
THIRD = "#1baf7a"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

REGIME_ORDER = ["BULLISH", "SIDEWAYS", "BEARISH", "HIGH_VOL", "LOW_VOL"]
REGIME_COLORS = {"BULLISH": "#1baf7a", "BEARISH": "#e34948", "SIDEWAYS": "#c3c2b7",
                 "HIGH_VOL": "#eda100", "LOW_VOL": "#2a78d6"}
REGIME_LABEL = {"BULLISH": "Bullish", "BEARISH": "Bearish", "SIDEWAYS": "Sideways",
                "HIGH_VOL": "High vol", "LOW_VOL": "Low vol"}

DIVERGING = [[0.0, "#b8322f"], [0.25, "#e88b85"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#5598e7"], [1.0, "#104281"]]

PLOTLY_CONFIG = {"displaylogo": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"]}

pio.templates["pairs"] = go.layout.Template(layout=go.Layout(
    font=dict(family="system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif", size=13, color=INK_2),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=SERIES,
    xaxis=dict(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, zeroline=False,
               title=dict(font=dict(size=12, color=MUTED))),
    yaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS,
               title=dict(font=dict(size=12, color=MUTED))),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12, color=INK_2),
                bgcolor="rgba(0,0,0,0)", traceorder="normal"),
    hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(color=INK, size=12)),
    margin=dict(l=8, r=8, t=40, b=8),
    bargap=0.25,
))
pio.templates.default = "plotly_white+pairs"


CSS = """
<style>
.block-container {padding-top: 4.2rem; padding-bottom: 4rem; max-width: 1280px;}
h1 {letter-spacing: -0.02em;}
h2, h3 {letter-spacing: -0.01em;}
.eyebrow {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
          color: #2a78d6; margin-bottom: -0.4rem;}
.lede {font-size: 1.08rem; line-height: 1.6; color: #3a3936; max-width: 62rem;}
.muted {color: #6b6a65; font-size: 0.9rem;}
.step-card {border: 1px solid rgba(11,11,11,0.10); border-radius: 0.6rem; padding: 0.9rem 1rem;
            background: #ffffff; height: 100%;}
.step-card .num {font-size: 0.75rem; font-weight: 700; color: #2a78d6; letter-spacing: 0.06em;}
.step-card .ttl {font-weight: 650; font-size: 1.0rem; margin: 0.15rem 0 0.35rem 0; color: #0b0b0b;}
.step-card .txt {font-size: 0.88rem; line-height: 1.45; color: #52514e;}
.flow {display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.3rem; margin: 0.4rem 0 1.2rem 0;}
.flow .box {flex: 1 1 6.6rem; border: 1px solid rgba(11,11,11,0.10); border-radius: 0.55rem; background: #ffffff;
            padding: 0.6rem 0.75rem;}
.flow .box.end {border-color: #2a78d6; background: #eef4fc;}
.flow .box b {display: block; font-size: 0.92rem; color: #0b0b0b; margin-bottom: 0.15rem;}
.flow .box span {font-size: 0.8rem; color: #52514e; line-height: 1.35; display: block;}
.flow .arrow {align-self: center; color: #898781; font-size: 1.1rem;}
.chip {display: inline-block; padding: 0.08rem 0.55rem; border-radius: 999px; font-size: 0.78rem;
       font-weight: 600; border: 1px solid rgba(11,11,11,0.10); background: #f3f2ee; color: #3a3936;
       margin-right: 0.3rem; white-space: nowrap;}
.chip.good {background: #e7f6e7; color: #1d6b1d; border-color: #bfe3bf;}
.chip.bad {background: #fbeaea; color: #9b2626; border-color: #f0c4c4;}
.chip.warn {background: #fff4dc; color: #7a5600; border-color: #f3dca4;}
.timeline {border-left: 2px solid #e1e0d9; margin-left: 0.4rem; padding-left: 1.1rem;}
.timeline .item {position: relative; margin-bottom: 1.05rem;}
.timeline .item:before {content: ""; position: absolute; left: -1.48rem; top: 0.35rem; width: 0.7rem;
                        height: 0.7rem; border-radius: 50%; background: #c3c2b7; border: 2px solid #fcfcfb;}
.timeline .item.final:before {background: #2a78d6;}
.timeline .when {font-size: 0.75rem; color: #898781; font-weight: 600; letter-spacing: 0.04em;}
.timeline .what {font-weight: 650; color: #0b0b0b;}
.timeline .why {font-size: 0.9rem; color: #52514e; line-height: 1.45;}
.regime-key span {display: inline-flex; align-items: center; margin-right: 0.9rem; font-size: 0.85rem; color: #3a3936;}
.regime-key i {display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 3px; margin-right: 0.35rem;}
div[data-testid="stMetric"] {background: #ffffff;}
div[data-testid="stMetricValue"] {font-size: 1.75rem;}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def page_header(eyebrow: str, title: str, lede: str | None = None) -> None:
    st.html(f'<div class="eyebrow">{eyebrow}</div>')
    st.title(title)
    if lede:
        st.html(f'<p class="lede">{lede}</p>')


def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{text}</span>'


def regime_key() -> None:
    items = "".join(f'<span><i style="background:{REGIME_COLORS[r]}"></i>{REGIME_LABEL[r]}</span>' for r in REGIME_ORDER)
    st.html(f'<div class="regime-key">{items}</div>')


def show(fig: go.Figure, height: int | None = None, key: str | None = None) -> None:
    """Render a figure with title, legend and subplot titles stacked in fixed pixel bands so they never collide."""
    h = height or fig.layout.height or 450
    has_title = bool(fig.layout.title.text)
    has_legend = fig.layout.showlegend is not False and sum(
        1 for t in fig.data if t.showlegend is not False and t.name) > 1
    sub_titles = any(a.yref == "paper" and a.yanchor == "bottom" and (a.y or 0) > 0.9 for a in fig.layout.annotations)
    title_px = 30 if has_title else 0
    legend_px = 26 if has_legend else 0
    top = 10 + title_px + legend_px + (28 if sub_titles else 0)
    fig.update_layout(
        height=h, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(t=max(top, 16), l=fig.layout.margin.l if fig.layout.margin.l is not None else 8,
                    r=fig.layout.margin.r if fig.layout.margin.r is not None else 8, b=8),
        title=dict(yref="container", y=1 - 8 / h, yanchor="top"),
        legend=dict(yref="container", y=1 - (8 + title_px) / h, yanchor="top"),
    )
    if not has_legend:
        fig.update_layout(showlegend=False)
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def step_cards(steps: list[tuple[str, str]]) -> None:
    cols = st.columns(len(steps))
    for i, (col, (ttl, txt)) in enumerate(zip(cols, steps), start=1):
        col.html(f'<div class="step-card"><div class="num">STEP {i}</div><div class="ttl">{ttl}</div>'
                 f'<div class="txt">{txt}</div></div>')


# ---- number formatting
def pct(x: float, digits: int = 1, sign: bool = True) -> str:
    if x is None or x != x:
        return "–"
    s = f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"
    return s.replace("-", "−")


def num(x: float, digits: int = 2, sign: bool = False) -> str:
    if x is None or x != x:
        return "–"
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace("-", "−")


def pts(x: float, digits: int = 1) -> str:
    """A difference of two percentages, in percentage points."""
    return f"{x * 100:+.{digits}f} pts".replace("-", "−")


def tick(ok: bool) -> str:
    return "✔" if ok else "✘"
