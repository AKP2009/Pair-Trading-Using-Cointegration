"""Figures for research/REPORT.md, built from the CSVs written by research.py and walk_forward_rs.py."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from research import OUT
from universe import DATA_DIR

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
REGIME_COLORS = {"BULLISH": SERIES[2], "BEARISH": SERIES[1], "SIDEWAYS": "#c3c2b7", "LOW_VOL": SERIES[0], "HIGH_VOL": "#e34948"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def regime_timeline() -> None:
    reg = pd.read_csv(DATA_DIR / "regimes.csv", index_col=0, parse_dates=True)
    market = pd.read_csv(DATA_DIR / "nifty200_market.csv", index_col=0, parse_dates=True)
    reg = reg.loc["2013-01-01":]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw={"height_ratios": [4, 0.6]})
    ax.plot(market.loc[reg.index, "benchmark"], color=INK2, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_title("Nifty 200 and the daily market regime")
    ax.set_ylabel("Nifty 200 (log scale)")
    for r, c in REGIME_COLORS.items():
        mask = (reg["regime"] == r).to_numpy()
        ax2.fill_between(reg.index, 0, 1, where=mask, color=c, step="post", linewidth=0)
    ax2.set_yticks([])
    ax2.grid(False)
    ax2.legend(handles=[Patch(color=c, label=r.replace("_", " ").title()) for r, c in REGIME_COLORS.items()],
               ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.6), fontsize=9)
    _save(fig, "regime_timeline.png")


def static_dev_vs_holdout() -> None:
    eq = pd.read_csv(OUT / "final_equity.csv", index_col=0, parse_dates=True)["equity"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(eq / eq.iloc[0], color=SERIES[0])
    ax.axvline(pd.Timestamp("2022-01-01"), color=MUTED, linestyle="--", linewidth=1)
    ymax = (eq / eq.iloc[0]).max()
    ax.text(pd.Timestamp("2021-12-01"), ymax, "parameters frozen  |  holdout", ha="right", va="top", color=INK2, fontsize=9)
    ax.set_title("Static model: tuned on 2014-2021, run once on 2022+  (growth of 1)")
    _save(fig, "static_dev_vs_holdout.png")


def walk_forward_equity() -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, v in enumerate(["v2", "v3"]):
        eq = pd.read_csv(OUT / f"{v}_wf_equity.csv", index_col=0, parse_dates=True)["WF ensemble + regime gate"]
        lbl = {"v2": "v2: wider breadth + learned regime gate", "v3": "v3 (final): + distinct ensemble, no LOW_VOL entries"}[v]
        ax.plot(eq, color=SERIES[i], label=lbl, linewidth=2.4 if v == "v3" else 1.6)
        ax.annotate(f"{eq.iloc[-1] - 1:+.1%}", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    ax.axhline(1.0, color=AXIS, linewidth=1)
    ax.axvline(pd.Timestamp("2022-01-01"), color=MUTED, linestyle="--", linewidth=1)
    ax.set_title("Walk-forward out-of-sample equity, 2017-2026 (growth of 1, after futures costs)")
    ax.legend(loc="upper left")
    _save(fig, "walk_forward_equity.png")


def regime_sharpe() -> None:
    dev = pd.read_csv(OUT / "per_regime_dev.csv", index_col=0)["sharpe"]
    hold = pd.read_csv(OUT / "per_regime_holdout.csv", index_col=0)["sharpe"]
    wf = pd.read_csv(OUT / "v3_wf_per_regime_backtests.csv", index_col=0)["sharpe"]
    regs = list(dev.index)
    x = np.arange(len(regs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4))
    for j, (s, lbl) in enumerate([(dev, "Static, dev 2014-21"), (hold, "Static, holdout 2022+"), (wf, "Walk-forward OOS 2017+")]):
        ax.bar(x + (j - 1) * w, s.reindex(regs), width=w - 0.03, color=SERIES[j], label=lbl)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, [r.replace("_", " ").title() for r in regs])
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Backtest run separately in each regime state (entries only in that state)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.grid(axis="x", visible=False)
    _save(fig, "regime_sharpe.png")


def dev_grid_heatmap() -> None:
    g = pd.read_csv(OUT / "grid_dev.csv")
    piv = g.pivot_table(index="lookback", columns="hold", values="sharpe", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lim = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xlabel("Holding period (days)")
    ax.set_ylabel("RS lookback (days)")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax.set_title("Dev 2014-21 Sharpe by lookback x holding (mean over entry thresholds)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Sharpe")
    _save(fig, "dev_grid_heatmap.png")


def monthly_heatmap() -> None:
    m = pd.read_csv(OUT / "v3_wf_monthly_returns.csv", index_col=0)
    vals = m.drop(columns="Year")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    lim = np.nanmax(np.abs(vals.values))
    im = ax.imshow(vals.values, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(vals.shape[1]), vals.columns)
    ax.set_yticks(range(vals.shape[0]), vals.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(vals.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.1%}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_title("Final model (v3) month-on-month out-of-sample returns")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Monthly return")
    _save(fig, "monthly_returns_v3.png")


def kaufman_equity() -> None:
    eq = pd.read_csv(OUT / "kaufman" / "equity.csv", index_col=0, parse_dates=True)
    series = [("Final system", SERIES[0], 2.4), ("Index sleeve only", SERIES[1], 1.4),
              ("Pair sleeve only", SERIES[2], 1.4), ("Nifty 200 buy & hold", INK2, 1.6)]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for name, color, lw in series:
        e = eq[name].dropna()
        ax.plot(e, color=color, linewidth=lw, label=name)
        ax.annotate(f"{e.iloc[-1]:.2f}x", (e.index[-1], e.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
        ax2.plot(e / e.cummax() - 1, color=color, linewidth=lw * 0.7)
    for a_ in (ax, ax2):
        a_.axvline(pd.Timestamp("2025-01-01"), color=MUTED, linestyle="--", linewidth=1)
    ax.text(pd.Timestamp("2024-11-15"), 0.04, "train 2015-2024  |  test", ha="right", va="bottom",
            color=INK2, fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_title("Regime-timed system vs Nifty 200 buy & hold (growth of 1, after costs)")
    ax.legend(loc="upper left")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save(fig, "kaufman_equity.png")


def rs_lowvol_all_windows() -> None:
    eq = pd.read_csv(OUT / "rs_lowvol" / "equity_all_windows.csv", index_col=0, parse_dates=True)
    series = [("Final system", SERIES[0], 2.4), ("(i) EW buy & hold", SERIES[1], 1.6), ("(ii) Nifty 200 index", SERIES[2], 1.6)]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    for a_ in (ax, ax2):
        a_.axvspan(eq.index[0], pd.Timestamp("2014-12-31"), color=GRID, alpha=0.6, linewidth=0)
        a_.axvspan(pd.Timestamp("2025-01-01"), eq.index[-1], color=GRID, alpha=0.6, linewidth=0)
    for name, color, lw in series:
        e = eq[name].dropna()
        ax.plot(e, color=color, linewidth=lw, label=name)
        ax.annotate(f"{e.iloc[-1]:.1f}x", (e.index[-1], e.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
        ax2.plot(e / e.cummax() - 1, color=color, linewidth=lw * 0.7)
    ax.set_yscale("log")
    for x, label in [(pd.Timestamp("2014-04-01"), "pre-test"), (pd.Timestamp("2019-12-01"), "train 2015-2024"),
                     (pd.Timestamp("2025-11-01"), "test")]:
        ax.text(x, 0.03, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_title("RS + low-vol system vs both benchmarks (growth of 1, log scale, after costs)")
    ax.legend(loc="center left")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save(fig, "rs_lowvol_all_windows.png")


def pit_pairs_oos() -> None:
    eq = pd.read_csv(OUT / "pit_pairs" / "oos_equity.csv", index_col=0, parse_dates=True)
    series = [("PIT pair strategy", SERIES[0], 2.4, "Pair strategy (point-in-time universe)"),
              ("Nifty 200 TRI", INK2, 1.8, "Nifty 200 TRI")]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]})
    for col, color, lw, label in series:
        e = eq[col].dropna()
        ax.plot(e, color=color, linewidth=lw, label=label)
        ax.annotate(f"{e.iloc[-1] - 1:+.1%}", (e.index[-1], e.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
        ax2.plot(e / e.cummax() - 1, color=color, linewidth=lw * 0.7)
    ax.axhline(1.0, color=AXIS, linewidth=1)
    ax.set_title("Out-of-sample 2025 to Oct 2026: frozen pair strategy vs Nifty 200 TRI (growth of 1)")
    ax.legend(loc="lower center", ncol=2)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save(fig, "pit_pairs_oos.png")


if __name__ == "__main__":
    regime_timeline()
    static_dev_vs_holdout()
    walk_forward_equity()
    regime_sharpe()
    dev_grid_heatmap()
    monthly_heatmap()
    kaufman_equity()
    rs_lowvol_all_windows()
    pit_pairs_oos()
    print(f"Figures written to {FIG}")
