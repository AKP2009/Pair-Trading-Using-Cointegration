"""Reusable Plotly figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.style import (AXIS, BENCHMARK, DIVERGING, GRID, INK_2, MUTED, REGIME_COLORS, REGIME_LABEL, REGIME_ORDER,
                        SERIES, STRATEGY)

DEFAULT_COLORS = {"PIT pair strategy": STRATEGY, "Pair strategy": STRATEGY, "Strategy": STRATEGY,
                  "Nifty 200 TRI": BENCHMARK, "Nifty 200": BENCHMARK, "Nifty 200 (buy & hold)": BENCHMARK,
                  "Nifty 200 buy & hold": BENCHMARK}


def _color(name: str, i: int, colors: dict | None) -> str:
    if colors and name in colors:
        return colors[name]
    return DEFAULT_COLORS.get(name, SERIES[i % len(SERIES)])


def end_labels(fig: go.Figure, df: pd.DataFrame, colors: dict | None = None, fmt=lambda v: f"{v:.2f}",
               row: int | None = None, col: int | None = None) -> None:
    """Label the last point of each line (selective direct labelling)."""
    for i, c in enumerate(df.columns):
        s = df[c].dropna()
        if s.empty:
            continue
        kw = dict(row=row, col=col) if row else {}
        fig.add_annotation(x=s.index[-1], y=s.iloc[-1], text=f"<b>{fmt(s.iloc[-1])}</b>", showarrow=False,
                           xanchor="left", xshift=6, font=dict(size=12, color=_color(c, i, colors)), **kw)


def lines(df: pd.DataFrame, title: str = "", colors: dict | None = None, ytitle: str = "", yfmt: str = ".2f",
          log: bool = False, label_fmt=None, hover_fmt: str | None = None, widths: dict | None = None) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c, mode="lines",
                                 line=dict(color=_color(c, i, colors), width=(widths or {}).get(c, 2)),
                                 hovertemplate=f"%{{y:{hover_fmt or yfmt}}}<extra>{c}</extra>"))
    fig.update_layout(title=title, hovermode="x unified", yaxis_title=ytitle)
    fig.update_yaxes(tickformat=yfmt, type="log" if log else "linear")
    if label_fmt:
        end_labels(fig, df, colors, label_fmt)
        fig.update_layout(margin=dict(r=70))
    return fig


def equity_and_drawdown(eq: pd.DataFrame, title: str = "", colors: dict | None = None,
                        shade_from: str | None = None, shade_label: str = "Out-of-sample") -> go.Figure:
    """Growth of 1 on top, drawdown below (two panels, one y-scale each)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
    for i, c in enumerate(eq.columns):
        s = eq[c].dropna()
        color = _color(c, i, colors)
        fig.add_trace(go.Scatter(x=s.index, y=s, name=c, mode="lines", line=dict(color=color, width=2),
                                 legendgroup=c, hovertemplate="%{y:.3f}<extra>" + c + "</extra>"), row=1, col=1)
        dd = s / s.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=c, mode="lines", line=dict(color=color, width=1.5),
                                 legendgroup=c, showlegend=False,
                                 hovertemplate="%{y:.1%}<extra>" + c + " drawdown</extra>"), row=2, col=1)
    end_labels(fig, eq, colors, lambda v: f"{v:.2f}", row=1, col=1)
    if shade_from:
        for r in (1, 2):
            fig.add_vrect(x0=shade_from, x1=eq.index[-1], fillcolor="#2a78d6", opacity=0.06, line_width=0, row=r, col=1)
        fig.add_annotation(x=shade_from, y=1, yref="y domain", text=shade_label, showarrow=False, xanchor="left",
                           xshift=6, yshift=-4, yanchor="top", font=dict(size=11, color=MUTED))
    fig.update_yaxes(title_text="Growth of ₹1", tickformat=".2f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(title=title, hovermode="x unified", margin=dict(r=60))
    return fig


def grouped_bars(df: pd.DataFrame, title: str = "", colors: dict | None = None, yfmt: str = ".0%",
                 ytitle: str = "", text: bool = False, text_fmt: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(df.columns):
        fig.add_trace(go.Bar(x=[str(x) for x in df.index], y=df[c], name=c, marker_color=_color(c, i, colors),
                             marker_line_width=0,
                             text=[f"{v:{text_fmt or yfmt}}" if v == v else "" for v in df[c]] if text else None,
                             textposition="outside", textfont=dict(size=11, color=INK_2),
                             hovertemplate=f"%{{x}}: %{{y:{yfmt}}}<extra>{c}</extra>"))
    fig.update_layout(title=title, barmode="group", bargroupgap=0.08, yaxis_title=ytitle)
    fig.update_yaxes(tickformat=yfmt, zeroline=True, zerolinecolor=AXIS, zerolinewidth=1)
    return fig


def monthly_heatmap(tbl: pd.DataFrame, title: str = "") -> go.Figure:
    months = [c for c in tbl.columns if c != "Year"]
    z = tbl[months].to_numpy(dtype=float)
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 0.05
    text = [[("" if not np.isfinite(v) or v == 0 else f"{v * 100:+.1f}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=months, y=[str(y) for y in tbl.index], colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim,
        text=text, texttemplate="%{text}", textfont=dict(size=11), xgap=2, ygap=2,
        colorbar=dict(title="Return", tickformat=".0%", thickness=10, outlinewidth=0),
        hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(showline=False, ticks="")
    return fig


def regime_ribbon(index: pd.Series, regime: pd.Series, title: str = "", log: bool = True,
                  index_name: str = "Nifty 200 index") -> go.Figure:
    """Index on top, one coloured bar per day below showing that day's regime."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=index.index, y=index, name=index_name, mode="lines",
                             line=dict(color="#3a3936", width=1.5), hovertemplate="%{y:,.0f}<extra>Index</extra>"),
                  row=1, col=1)
    reg = regime.dropna()
    for r in REGIME_ORDER:
        d = reg.index[reg == r]
        if len(d) == 0:
            continue
        fig.add_trace(go.Bar(x=d, y=np.ones(len(d)), name=REGIME_LABEL[r], marker_color=REGIME_COLORS[r],
                             marker_line_width=0, width=86_400_000 * 1.6,
                             hovertemplate="%{x|%d %b %Y}<extra>" + REGIME_LABEL[r] + "</extra>"), row=2, col=1)
    fig.update_layout(title=title, barmode="overlay", bargap=0, hovermode="closest",
                      legend=dict(traceorder="normal"))
    fig.update_yaxes(type="log" if log else "linear", title_text="Index level", row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1, range=[0, 1])
    return fig


def corr_heatmap(corr: pd.DataFrame, title: str = "", highlight: list[str] | None = None) -> go.Figure:
    labels = [f"<b>{c}</b>" if highlight and c in highlight else c for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels, colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1, xgap=2, ygap=2,
        text=np.round(corr.to_numpy(), 2), texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="Spearman ρ", thickness=10, outlinewidth=0),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=-40, showline=False, ticks="")
    return fig


def hbar(labels, values, title: str = "", color: str = STRATEGY, fmt: str = ".1%", xtitle: str = "",
         colors: list[str] | None = None, text: bool = True, log: bool = False) -> go.Figure:
    fig = go.Figure(go.Bar(
        y=list(labels), x=list(values), orientation="h", marker_color=colors or color, marker_line_width=0,
        text=[f"{v:{fmt}}" for v in values] if text else None, textposition="outside",
        textfont=dict(size=11, color=INK_2), cliponaxis=False,
        hovertemplate=f"%{{y}}: %{{x:{fmt}}}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=xtitle, showlegend=False)
    fig.update_xaxes(tickformat=fmt, gridcolor=GRID, showgrid=True, zeroline=True, zerolinecolor=AXIS)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    if log:
        fig.update_xaxes(type="log", tickformat=None)
    else:  # leave room for the value labels outside the bar ends
        v = [x for x in values if x == x]
        lo, hi = min(0.0, min(v)), max(0.0, max(v))
        pad = 0.22 * (hi - lo or 1)
        fig.update_xaxes(range=[lo - pad if lo < 0 else lo, hi + pad if hi > 0 else hi])
    return fig
