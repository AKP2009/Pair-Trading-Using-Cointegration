"""
End-to-end research run for the regime-aware relative-strength pair strategy
on the Nifty 200.

Protocol (to keep the final numbers honest):
- DEVELOPMENT period 2014-07 .. 2021-12: every parameter (lookback,
  holding period, entry threshold, exit variant, number of pairs per
  regime) is chosen here and only here.
- HOLDOUT period 2022-01 .. latest: run exactly once with the frozen
  configuration. Nothing is re-tuned after looking at it.
- Parameter choice uses a neighbourhood-smoothed Sharpe (average of each
  grid cell and its neighbours) instead of the single best cell, so a
  lucky isolated spike in the grid is not selected.

Outputs go to research/ (CSV tables, figures, REPORT.md).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from perf import monthly_returns_table, perf_metrics, regime_attribution, yearly_returns
from regime import REGIMES, compute_regimes
from rs_pairs import Config, backtest, build_schedule, compute_scores, form_pairs, formation_starts, signal_forward_returns
from universe import DATA_DIR, load_universe

OUT = Path(__file__).resolve().parent / "research"
OUT.mkdir(exist_ok=True)

DEV_START, DEV_END = "2014-07-01", "2021-12-31"
HOLD_START = "2022-01-01"

GRID_L = [2, 3, 5, 10, 20, 40]
GRID_H = [1, 2, 3, 5, 10, 20, 40]
GRID_K = [1.0, 1.5, 2.0, 2.5, 3.0]
GRID_MAX_PAIRS = [3, 5, 8, 10, 15]
MIN_DEV_TRADES = 150
KEY_METRICS = ["total_return", "CAGR", "sharpe", "sortino", "max_drawdown", "calmar", "beta", "corr_bench",
               "vol", "n_trades", "win_rate", "avg_trade_ret", "profit_factor", "avg_hold_days", "pct_positive_months"]


class Lab:
    def __init__(self) -> None:
        self.prices, self.market, self.meta = load_universe()
        reg = compute_regimes(self.prices, self.market)
        reg.to_csv(DATA_DIR / "regimes.csv", index_label="Date")
        self.regime_frame = reg
        self.regimes = reg["regime"]
        sched_file = DATA_DIR / "rs_pair_schedule.csv"
        if sched_file.exists():
            self.schedule = pd.read_csv(sched_file, parse_dates=["period_start", "period_end"])
        else:
            self.schedule = build_schedule(self.prices, self.meta)
            self.schedule.to_csv(sched_file, index=False)
        self.bench = self.market["benchmark"]
        self._sig: dict = {}

    def signals(self, lookback: int, vol_window: int = 60, schedule: pd.DataFrame | None = None, tag: str = "main"):
        key = (lookback, vol_window, tag)
        if key not in self._sig:
            self._sig[key] = compute_scores(self.prices, self.schedule if schedule is None else schedule, lookback, vol_window)
        return self._sig[key]

    def run(self, cfg: Config, schedule: pd.DataFrame | None = None, tag: str = "main") -> dict:
        sched = self.schedule if schedule is None else schedule
        res = backtest(cfg, self.prices, self.regimes, sched, self.signals(cfg.lookback, cfg.vol_window, sched, tag))
        res["metrics"] = perf_metrics(res["equity"]["equity"], self.bench, res["trades"])
        return res


def row(cfg: Config, metrics: dict, **extra) -> dict:
    base = {"label": cfg.label, "lookback": cfg.lookback, "hold": cfg.hold, "entry_k": cfg.entry_k,
            "max_pairs": cfg.max_pairs, "cost_model": cfg.cost_model, "slippage_bps": cfg.slippage_bps,
            "exit_on_revert": cfg.exit_on_revert, "stop_k": cfg.stop_k, "direction": cfg.direction,
            "pairs_by_regime": json.dumps(cfg.pairs_by_regime) if cfg.pairs_by_regime else "ALL"}
    return {**base, **extra, **{k: metrics.get(k, np.nan) for k in KEY_METRICS}}


def smoothed_pick(grid: pd.DataFrame) -> pd.Series:
    """Pick (L, H, k) maximizing the mean Sharpe of the cell and its +/-1 grid neighbours."""
    idx = {"lookback": GRID_L, "hold": GRID_H, "entry_k": GRID_K}
    g = grid.set_index(["lookback", "hold", "entry_k"])
    smooth = []
    for (L, H, k), r in g.iterrows():
        li, hi, ki = GRID_L.index(L), GRID_H.index(H), GRID_K.index(k)
        neigh = []
        for dl, dh, dk in itertools.product((-1, 0, 1), repeat=3):
            a, b, c = li + dl, hi + dh, ki + dk
            if 0 <= a < len(GRID_L) and 0 <= b < len(GRID_H) and 0 <= c < len(GRID_K):
                key = (idx["lookback"][a], idx["hold"][b], idx["entry_k"][c])
                if key in g.index:
                    neigh.append(g.loc[key, "sharpe"])
        smooth.append(np.nanmean(neigh))
    grid = grid.copy()
    grid["sharpe_smoothed"] = smooth
    eligible = grid[grid["n_trades"] >= MIN_DEV_TRADES]
    return grid, eligible.sort_values("sharpe_smoothed", ascending=False).iloc[0]


def random_control_schedule(lab: Lab, like: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Same-industry, correlated but clearly NON-cointegrated pairs (ADF p > 0.3), same count per period."""
    rng = np.random.default_rng(seed)
    out = []
    counts = like.groupby("period_start").size()
    starts = formation_starts(lab.prices.index)
    for i, s in enumerate(starts):
        if s not in counts.index:
            continue
        cand = form_pairs(lab.prices, lab.meta, s, max_adf_p=1.0)
        cand = cand[cand["adf_p"] > 0.3]
        if cand.empty:
            continue
        take = cand.iloc[rng.choice(len(cand), size=min(len(cand), int(counts[s])), replace=False)].copy()
        end = starts[i + 1] if i + 1 < len(starts) else lab.prices.index[-1] + pd.Timedelta(days=1)
        take.insert(0, "period_end", end)
        take.insert(0, "period_start", s)
        out.append(take)
    return pd.concat(out, ignore_index=True)


def forward_return_table(lab: Lab, signals: list[tuple[str, int, float]]) -> pd.DataFrame:
    """Mean and hit rate of the pair's trade-direction forward return after each signal, by period and regime."""
    rows = []
    for name, lookback, k in signals:
        ev = signal_forward_returns(lab.prices, lab.signals(lookback), lab.regimes, k)
        ev["period"] = np.where(ev["date"] <= DEV_END, "dev 2014-21", "holdout 2022+")
        for (period, regime), g in pd.concat([ev.assign(regime="ALL"), ev]).groupby(["period", "regime"]):
            row = {"signal": name, "period": period, "regime": regime, "n_signals": len(g)}
            for h in (1, 5, 10, 20):
                x = g[f"fwd_{h}d"].dropna()
                row[f"fwd_{h}d_mean"] = x.mean()
                row[f"fwd_{h}d_hit"] = (x > 0).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    lab = Lab()
    print(f"Universe: {lab.prices.shape[1]} stocks, {lab.prices.index[0].date()} .. {lab.prices.index[-1].date()}")

    # ---------------------------------------------------------------- 1. regimes
    rf = lab.regime_frame.copy()
    rf["bench_fwd_20d"] = lab.bench.pct_change(20).shift(-20)
    regime_summary = rf.groupby("regime").agg(
        days=("regime", "size"), dist_ema21=("dist_ema21", "mean"), ad_ratio=("ad_ratio", "mean"),
        pct_above_200ema=("pct_above_200ema", "mean"), vix=("vix", "mean"), realized_vol=("realized_vol", "mean"),
        bench_fwd_20d=("bench_fwd_20d", "mean"),
    )
    regime_summary["share"] = regime_summary["days"] / regime_summary["days"].sum()
    regime_summary.to_csv(OUT / "regime_summary.csv")
    regime_by_year = pd.crosstab(rf.index.year, rf["regime"], normalize="index")
    regime_by_year.to_csv(OUT / "regime_by_year.csv")

    # ---------------------------------------------------------------- 2. pair universe per period / regime
    pair_counts = lab.schedule.groupby("period_start").agg(
        n_pairs=("stock_a", "size"), median_adf_p=("adf_p", "median"), n_industries=("industry", "nunique"))
    pair_counts.to_csv(OUT / "pair_counts_by_period.csv")

    # ---------------------------------------------------------------- 3. grid search on DEV
    base = Config(max_pairs=5, cost_model="futures", slippage_bps=5.0, start=DEV_START, end=DEV_END)
    grid_rows = []
    for L, H, k in itertools.product(GRID_L, GRID_H, GRID_K):
        cfg = replace(base, lookback=L, hold=H, entry_k=k, label="grid")
        grid_rows.append(row(cfg, lab.run(cfg)["metrics"]))
    grid = pd.DataFrame(grid_rows)
    grid, best = smoothed_pick(grid)
    grid.to_csv(OUT / "grid_dev.csv", index=False)
    L, H, K = int(best["lookback"]), int(best["hold"]), float(best["entry_k"])
    print(f"[dev grid] picked lookback={L} hold={H} entry_k={K} "
          f"sharpe={best['sharpe']:.2f} smoothed={best['sharpe_smoothed']:.2f} trades={best['n_trades']}")
    core = replace(base, lookback=L, hold=H, entry_k=K)

    # edge cases of the grid itself (very short / very long horizons) for the report
    edge = grid[(grid["lookback"].isin([GRID_L[0], GRID_L[-1]])) | (grid["hold"].isin([GRID_H[0], GRID_H[-1]]))]
    edge.groupby(["lookback", "hold"])["sharpe"].max().unstack().to_csv(OUT / "grid_edge_cases.csv")

    # ---------------------------------------------------------------- 4. model variants on DEV
    control = random_control_schedule(lab, lab.schedule)
    loose = build_schedule(lab.prices, lab.meta, max_adf_p=0.10)
    variants = {
        "core (fixed hold)": (core, None, "main"),
        "+ exit on RS revert": (replace(core, exit_on_revert=True), None, "main"),
        "+ stop 2 vol": (replace(core, stop_k=2.0), None, "main"),
        "+ stop 3 vol": (replace(core, stop_k=3.0), None, "main"),
        "+ revert exit + stop 3": (replace(core, exit_on_revert=True, stop_k=3.0), None, "main"),
        "stock in up to 2 pairs": (replace(core, max_per_stock=2), None, "main"),
        "momentum (trade with RS)": (replace(core, direction="momentum"), None, "main"),
        "looser coint (ADF p<=0.10)": (core, loose, "loose"),
        "control: NON-cointegrated pairs": (core, control, "control"),
    }
    var_rows, var_res = [], {}
    for name, (cfg, sched, tag) in variants.items():
        cfg = replace(cfg, label=name)
        res = lab.run(cfg, sched, tag)
        var_res[name] = res
        var_rows.append(row(cfg, res["metrics"]))
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(OUT / "variants_dev.csv", index=False)

    # adopt a variant only if it beats the core dev Sharpe by a margin (parsimony), never the controls
    core_sharpe = var_df.loc[var_df["label"] == "core (fixed hold)", "sharpe"].iloc[0]
    adoptable = var_df[~var_df["label"].str.startswith(("control", "momentum"))]
    top = adoptable.sort_values("sharpe", ascending=False).iloc[0]
    chosen_name = top["label"] if top["sharpe"] > core_sharpe + 0.10 else "core (fixed hold)"
    chosen_cfg, chosen_sched, chosen_tag = variants[chosen_name]
    print(f"[dev variants] chosen: {chosen_name}")

    # ---------------------------------------------------------------- 5. number of top pairs (portfolio size) on DEV
    mp_rows = []
    for n in GRID_MAX_PAIRS:
        cfg = replace(chosen_cfg, max_pairs=n, label=f"max_pairs={n}")
        mp_rows.append(row(cfg, lab.run(cfg, chosen_sched, chosen_tag)["metrics"]))
    mp_df = pd.DataFrame(mp_rows)
    mp_df.to_csv(OUT / "max_pairs_dev.csv", index=False)
    # Sharpe barely depends on size; pick the most diversified size whose dev Sharpe is within 0.1 of the best
    ok = mp_df[mp_df["sharpe"] >= mp_df["sharpe"].max() - 0.10]
    n_pairs = int(ok["max_pairs"].max())
    chosen_cfg = replace(chosen_cfg, max_pairs=n_pairs)
    print(f"[dev] max concurrent pairs = {n_pairs}")

    # ---------------------------------------------------------------- 6. regime study on DEV
    def per_regime(cfg: Config, start: str, end: str | None) -> pd.DataFrame:
        rows = []
        for R in REGIMES:
            c = replace(cfg, start=start, end=end, pairs_by_regime={R: cfg.max_pairs}, label=f"only {R}")
            m = lab.run(c, chosen_sched, chosen_tag)["metrics"]
            rows.append({"regime": R, **{k: m.get(k, np.nan) for k in KEY_METRICS}})
        return pd.DataFrame(rows).set_index("regime")

    reg_dev = per_regime(chosen_cfg, DEV_START, DEV_END)
    reg_dev.to_csv(OUT / "per_regime_dev.csv")
    # rule: full allocation where the regime-only dev Sharpe > 0.3, half where 0 < Sharpe <= 0.3, none otherwise
    pairs_by_regime = {}
    for R in REGIMES:
        s = reg_dev.loc[R, "sharpe"]
        pairs_by_regime[R] = n_pairs if s > 0.3 else (max(1, n_pairs // 2) if s > 0 else 0)
    regime_cfg = replace(chosen_cfg, pairs_by_regime=pairs_by_regime)
    print(f"[dev] pairs by regime: {pairs_by_regime}")

    forward_return_table(lab, [(f"chosen (L={chosen_cfg.lookback}, k={chosen_cfg.entry_k})", chosen_cfg.lookback, chosen_cfg.entry_k),
                               ("short-term (L=5, k=2.0)", 5, 2.0)]).to_csv(OUT / "forward_returns.csv", index=False)

    # ---------------------------------------------------------------- 7. FROZEN configs -> holdout (run once)
    finals = {
        "Regime-aware RS pairs (final)": regime_cfg,
        "RS pairs, no regime filter": chosen_cfg,
    }
    summary_rows, final_res = [], {}
    for name, cfg in finals.items():
        for period, (s, e) in {"dev": (DEV_START, DEV_END), "holdout": (HOLD_START, None), "full": (DEV_START, None)}.items():
            c = replace(cfg, start=s, end=e, label=name)
            res = lab.run(c, chosen_sched, chosen_tag)
            final_res[(name, period)] = res
            summary_rows.append(row(c, res["metrics"], period=period))
    # robustness of the final config on the holdout: costs and slippage stress
    stress = {
        "futures, 10bps slippage": replace(regime_cfg, slippage_bps=10.0),
        "futures, 20bps slippage": replace(regime_cfg, slippage_bps=20.0),
        "cash delivery + SLB borrow": replace(regime_cfg, cost_model="cash"),
        "zero cost (gross edge)": replace(regime_cfg, cost_model="none"),
    }
    for name, cfg in stress.items():
        for period, (s, e) in {"dev": (DEV_START, DEV_END), "holdout": (HOLD_START, None)}.items():
            c = replace(cfg, start=s, end=e, label=f"stress: {name}")
            summary_rows.append(row(c, lab.run(c, chosen_sched, chosen_tag)["metrics"], period=period))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "final_summary.csv", index=False)

    reg_hold = per_regime(chosen_cfg, HOLD_START, None)
    reg_hold.to_csv(OUT / "per_regime_holdout.csv")

    final_full = final_res[("Regime-aware RS pairs (final)", "full")]
    eq = final_full["equity"]
    eq.to_csv(OUT / "final_equity.csv")
    final_full["trades"].to_csv(OUT / "final_trades.csv", index=False)
    yearly = pd.DataFrame({name: yearly_returns(final_res[(name, "full")]["equity"]["equity"]) for name in finals})
    yearly["Nifty 200"] = yearly_returns(lab.bench.loc[DEV_START:])
    yearly.to_csv(OUT / "yearly_returns.csv")
    monthly_returns_table(eq["equity"]).to_csv(OUT / "monthly_returns.csv")
    regime_attribution(eq, final_full["trades"]).to_csv(OUT / "regime_attribution_full.csv")
    regime_attribution(final_res[("Regime-aware RS pairs (final)", "holdout")]["equity"],
                       final_res[("Regime-aware RS pairs (final)", "holdout")]["trades"]).to_csv(OUT / "regime_attribution_holdout.csv")
    trades = final_full["trades"]
    trades["year"] = pd.to_datetime(trades["exit_date"]).dt.year
    trades.groupby("year").agg(n=("ret", "size"), win_rate=("net_pnl", lambda s: (s > 0).mean()),
                               avg_ret=("ret", "mean")).to_csv(OUT / "trades_by_year.csv")

    frozen = {"core": asdict(core), "chosen_variant": chosen_name, "final": asdict(regime_cfg)}
    (OUT / "final_config.json").write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[["label", "period", "CAGR", "sharpe", "sortino", "max_drawdown", "beta", "n_trades", "win_rate"]].round(3).to_string(index=False))
        print("\nPer-regime (dev):\n", reg_dev[["sharpe", "CAGR", "n_trades", "win_rate"]].round(3))
        print("\nPer-regime (holdout):\n", reg_hold[["sharpe", "CAGR", "n_trades", "win_rate"]].round(3))


if __name__ == "__main__":
    main()
