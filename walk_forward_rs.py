"""
Walk-forward model selection for the regime-aware RS pair strategy (v2).

research.py showed that one parameter set frozen on 2014-2021 did not
survive 2022+: the speed of mean reversion changed. Here nothing is
frozen. Each calendar year Y is traded with models chosen ONLY from the
trailing TRAIN_YEARS years:

1. Every config in the grid (lookback x hold x entry_k x vol filter) is
   simulated once over the whole sample, giving a daily return stream.
2. For year Y, each config is scored by its Sharpe over [Y-3, Y-1],
   smoothed over its grid neighbours; the TOP_K configs (with enough
   trades) form an equal-weight ensemble traded through year Y.
3. Regime layer: for each selected config, regimes whose trailing
   trades (entered in that regime) lost money on average are blocked for
   new entries in year Y, and the config is re-simulated with that gate.

The concatenated year-Y slices form a genuinely out-of-sample return
stream from 2017 onward. Outputs go to research/wf_*.
"""

from __future__ import annotations

import itertools
import json
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd

from perf import monthly_returns_table, perf_metrics, regime_attribution, yearly_returns
from regime import REGIMES
from research import OUT, Lab
from rs_pairs import Config, build_schedule

SIM_START = "2014-07-01"
FIRST_OOS_YEAR = 2017
TRAIN_YEARS = 3
TOP_K = 5
MIN_TRAIN_TRADES = 60
MIN_REGIME_TRADES = 15
# v3 (final, pre-specified): ensemble members must differ in (lookback, hold, entry_k), and no
# new entries in LOW_VOL -- reversal profits are compensation for liquidity provision, which is
# least rewarded in calm markets (Nagel 2012); LOW_VOL was also the weakest regime in the
# 2014-2021 development study (research/per_regime_dev.csv).
DISTINCT_ENSEMBLE = True
ALWAYS_BLOCKED_REGIMES = {"LOW_VOL"}

# Structural dimensions widen breadth: a looser cointegration cut (more pairs) and a
# bigger book of smaller positions (stock may sit in more pairs). (max_per_stock, max_pairs)
GRID = {
    "lookback": [3, 5, 10, 20, 40],
    "hold": [5, 10, 20],
    "entry_k": [1.0, 1.5, 2.0, 2.5],
    "max_vol_ratio": [None, 1.0],
    "adf_p": [0.05, 0.10],
    "book": [(2, 10), (3, 25)],
}
BASE = Config(cost_model="futures", slippage_bps=5.0, start=SIM_START)
CACHE = OUT / "wf_grid_cache.pkl"


def grid_configs() -> list[Config]:
    return [replace(BASE, lookback=L, hold=H, entry_k=k, max_vol_ratio=v, max_per_stock=b[0], max_pairs=b[1],
                    extra={"adf_p": p}, label=f"L{L}_H{H}_k{k}_v{v}_p{p}_b{b[0]}x{b[1]}")
            for L, H, k, v, p, b in itertools.product(*GRID.values())]


def grid_key(c: Config) -> tuple:
    return (c.lookback, c.hold, c.entry_k, c.max_vol_ratio, c.extra["adf_p"], (c.max_per_stock, c.max_pairs))


_SCHEDULES: dict[float, pd.DataFrame] = {}


def run_cfg(lab: Lab, cfg: Config) -> dict:
    p = cfg.extra["adf_p"]
    if p not in _SCHEDULES:
        _SCHEDULES[p] = lab.schedule if p == 0.05 else build_schedule(lab.prices, lab.meta, max_adf_p=p)
    return lab.run(cfg, _SCHEDULES[p], tag=f"p{p}")


def run_grid(lab: Lab) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    rets, trades = {}, {}
    for cfg in grid_configs():
        res = run_cfg(lab, cfg)
        rets[cfg.label] = res["equity"]["equity"].pct_change().fillna(0.0)
        trades[cfg.label] = res["trades"]
    out = (pd.DataFrame(rets), trades)
    CACHE.write_bytes(pickle.dumps(out))
    return out


def _neighbours(label_idx: dict, key: tuple) -> list[tuple]:
    dims = list(GRID.values())
    pos = [dims[d].index(key[d]) for d in range(len(dims))]
    out = []
    for delta in itertools.product((-1, 0, 1), repeat=len(dims)):
        p = [pos[d] + delta[d] for d in range(len(dims))]
        if all(0 <= p[d] < len(dims[d]) for d in range(len(dims))):
            k = tuple(dims[d][p[d]] for d in range(len(dims)))
            if k in label_idx:
                out.append(label_idx[k])
    return out


def select_for_year(year: int, rets: pd.DataFrame, trades: dict[str, pd.DataFrame], configs: list[Config]) -> list[Config]:
    lo, hi = f"{year - TRAIN_YEARS}-01-01", f"{year - 1}-12-31"
    window = rets.loc[lo:hi]
    sharpe = window.mean() / window.std().replace(0, np.nan) * np.sqrt(252)
    n_tr = {c.label: int(((trades[c.label]["exit_date"] >= lo) & (trades[c.label]["exit_date"] <= hi)).sum())
            if len(trades[c.label]) else 0 for c in configs}
    keyed = {grid_key(c): c.label for c in configs}
    smooth = {c.label: np.nanmean([sharpe[l] for l in _neighbours(keyed, grid_key(c))]) for c in configs}
    ranked = sorted((c for c in configs if n_tr[c.label] >= MIN_TRAIN_TRADES), key=lambda c: -smooth[c.label])
    if DISTINCT_ENSEMBLE:
        picked, seen = [], set()
        for c in ranked:
            if (c.lookback, c.hold, c.entry_k) not in seen:
                seen.add((c.lookback, c.hold, c.entry_k))
                picked.append(c)
        ranked = picked
    return ranked[:TOP_K], {c.label: (sharpe[c.label], smooth[c.label], n_tr[c.label]) for c in ranked[:TOP_K]}


def regime_gate(year: int, cfg_trades: pd.DataFrame, max_pairs: int) -> dict[str, int]:
    lo, hi = pd.Timestamp(f"{year - TRAIN_YEARS}-01-01"), pd.Timestamp(f"{year - 1}-12-31")
    t = cfg_trades[(cfg_trades["exit_date"] >= lo) & (cfg_trades["exit_date"] <= hi)]
    gate = {}
    for R in REGIMES:
        tr = t[t["entry_regime"] == R]
        losing = len(tr) >= MIN_REGIME_TRADES and tr["ret"].mean() <= 0
        gate[R] = 0 if losing or R in ALWAYS_BLOCKED_REGIMES else max_pairs
    return gate


def main() -> None:
    lab = Lab()
    configs = grid_configs()
    rets, trades = run_grid(lab)
    print(f"Grid: {len(configs)} configs simulated")

    last_year = lab.prices.index[-1].year
    streams = {"WF ensemble": [], "WF ensemble + regime gate": [], "WF top-1": []}
    regime_only = {R: [] for R in REGIMES}
    wf_trades, selection_log = [], []

    for year in range(FIRST_OOS_YEAR, last_year + 1):
        chosen, info = select_for_year(year, rets, trades, configs)
        ys, ye = f"{year}-01-01", f"{year}-12-31"
        streams["WF ensemble"].append(rets.loc[ys:ye, [c.label for c in chosen]].mean(axis=1))
        streams["WF top-1"].append(rets.loc[ys:ye, chosen[0].label])

        gated = []
        for c in chosen:
            gate = regime_gate(year, trades[c.label], c.max_pairs)
            res = run_cfg(lab, replace(c, pairs_by_regime=gate, end=ye))
            gated.append(res["equity"]["equity"].pct_change().fillna(0.0).loc[ys:ye])
            t = res["trades"]
            t = t[(t["exit_date"] >= ys) & (t["exit_date"] <= ye)].assign(config=c.label, weight=1 / len(chosen))
            wf_trades.append(t)
            sh, sm, n = info[c.label]
            selection_log.append({"year": year, "config": c.label, "lookback": c.lookback, "hold": c.hold,
                                  "entry_k": c.entry_k, "max_vol_ratio": c.max_vol_ratio, "adf_p": c.extra["adf_p"],
                                  "max_per_stock": c.max_per_stock, "max_pairs": c.max_pairs, "train_sharpe": sh,
                                  "train_sharpe_smoothed": sm, "train_trades": n,
                                  "blocked_regimes": ",".join(r for r, v in gate.items() if v == 0) or "-"})
            for R in REGIMES:
                r_only = run_cfg(lab, replace(c, pairs_by_regime={R: c.max_pairs}, end=ye))
                regime_only[R].append(r_only["equity"]["equity"].pct_change().fillna(0.0).loc[ys:ye].rename(c.label))
        streams["WF ensemble + regime gate"].append(pd.concat(gated, axis=1).mean(axis=1))
        print(f"{year}: {[c.label for c in chosen]}")

    bench = lab.bench
    eq = {name: (1 + pd.concat(parts)).cumprod() for name, parts in streams.items()}
    all_trades = pd.concat(wf_trades, ignore_index=True)
    rows = []
    for name, e in eq.items():
        e = pd.concat([pd.Series([1.0], index=[e.index[0] - pd.Timedelta(days=1)]), e])
        for period, (s, en) in {"OOS 2017-2021": ("2016-12-31", "2021-12-31"), "OOS 2022-latest": ("2021-12-31", None),
                                "OOS 2017-latest": (None, None)}.items():
            seg = e.loc[s:en]
            m = perf_metrics(seg, bench, all_trades if name == "WF ensemble + regime gate" and period == "OOS 2017-latest" else None)
            rows.append({"model": name, "period": period, **m})
    nifty = bench.loc["2016-12-30":]
    for period, (s, en) in {"OOS 2017-2021": (None, "2021-12-31"), "OOS 2022-latest": ("2021-12-31", None), "OOS 2017-latest": (None, None)}.items():
        rows.append({"model": "Nifty 200 (buy & hold)", "period": period, **perf_metrics(nifty.loc[s:en], bench)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "wf_summary.csv", index=False)

    final_eq = eq["WF ensemble + regime gate"]
    pd.DataFrame({k: v for k, v in eq.items()}).assign(nifty200=bench.reindex(final_eq.index) / bench.reindex(final_eq.index).iloc[0]) \
        .to_csv(OUT / "wf_equity.csv", index_label="date")
    pd.DataFrame(selection_log).to_csv(OUT / "wf_selection_log.csv", index=False)
    all_trades.to_csv(OUT / "wf_trades.csv", index=False)

    yearly = pd.DataFrame({k: yearly_returns(pd.concat([pd.Series([1.0], index=[v.index[0] - pd.Timedelta(days=1)]), v]))
                           for k, v in eq.items()})
    yearly["Nifty 200"] = yearly_returns(bench.loc["2016-12-30":])
    yearly.to_csv(OUT / "wf_yearly_returns.csv")
    monthly_returns_table(pd.concat([pd.Series([1.0], index=[final_eq.index[0] - pd.Timedelta(days=1)]), final_eq])) \
        .to_csv(OUT / "wf_monthly_returns.csv")

    day_reg = lab.regimes.reindex(final_eq.index)
    attr = regime_attribution(pd.DataFrame({"equity": final_eq, "regime": day_reg}), all_trades)
    attr.to_csv(OUT / "wf_regime_attribution.csv")

    reg_rows = []
    for R, parts in regime_only.items():
        stream = pd.concat(_yearly_mean(parts))
        e = (1 + stream).cumprod()
        e = pd.concat([pd.Series([1.0], index=[e.index[0] - pd.Timedelta(days=1)]), e])
        reg_rows.append({"regime": R, **perf_metrics(e, bench)})
    pd.DataFrame(reg_rows).to_csv(OUT / "wf_per_regime_backtests.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["model", "period", "total_return", "CAGR", "sharpe", "sortino", "max_drawdown", "beta", "pct_positive_months"]
        print(summary[cols].round(3).to_string(index=False))
        print(yearly.round(3).to_string())
        print(pd.DataFrame(reg_rows).set_index("regime")[["CAGR", "sharpe", "sortino", "max_drawdown", "beta"]].round(3))
        print(attr.round(3))


def _yearly_mean(parts: list[pd.Series]) -> list[pd.Series]:
    """Group per-config yearly slices by year and average them into one ensemble slice per year."""
    by_year: dict[int, list[pd.Series]] = {}
    for p in parts:
        if len(p):
            by_year.setdefault(p.index[0].year, []).append(p)
    return [pd.concat(v, axis=1).mean(axis=1) for _, v in sorted(by_year.items())]


if __name__ == "__main__":
    main()
