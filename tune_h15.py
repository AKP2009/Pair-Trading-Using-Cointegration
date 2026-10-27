"""
Fine-tune every lookback around a FIXED 15-calendar-day holding period.

The holding period is no longer a parameter: every trade exits on the first
close at least 15 calendar days after entry. Everything else with a window
is tuned:

Model lookbacks (MODEL_GRID)
- lookback        RS lookback L (days)
- vol_window      window of the pair-ratio volatility that scales RS
- formation_days  cointegration (ADF) formation window
- entry_k         |RS / vol| entry threshold
- max_vol_ratio   volatility-spike filter (10d / vol_window ratio vol), None = off

Regime lookbacks (REGIME_GRID)
- ema_trend    EMA for the benchmark trend distance
- adr_window   advance/decline summation window
- ema_breadth  per-stock EMA for "% of stocks above EMA"
- rv_window    realized-vol window of the benchmark
- pct_window   trailing window that turns VIX / realized vol into percentiles

Protocol (walk-forward, no peeking): for each calendar year Y
1. Model: score every model config by its neighbourhood-smoothed Sharpe
   over the trailing 3 years; trade the top-5 configs that differ in
   (lookback, entry_k) as an equal-weight ensemble.
2. Regime: label the selected configs' trailing trades by each regime
   definition, learn a gate (block regimes whose trailing trades lost
   money, and LOW_VOL as in the v3 model), and keep the definition whose
   surviving trades have the best (smoothed) t-statistic.
3. Re-simulate year Y exactly with that model ensemble and regime gate.

Outputs: research/h15/*.csv, research/h15/figures/*.png
"""

from __future__ import annotations

import itertools
import json
import pickle
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from perf import monthly_returns_table, perf_metrics, regime_attribution, yearly_returns
from regime import REGIMES, RegimeParams, adr_indicator, breadth_indicator, classify, trend_indicator, vol_indicator
from rs_pairs import Config, backtest, build_schedule, compute_scores
from universe import load_universe
from research import OUT as RESEARCH_DIR

OUT = RESEARCH_DIR / "h15"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "model_grid_cache.pkl"

HOLD_CALENDAR_DAYS = 15
SIM_START = "2014-07-01"
DEV_END, HOLD_START = "2021-12-31", "2022-01-01"
FIRST_OOS_YEAR = 2017
TRAIN_YEARS = 3
TOP_K = 5
MIN_TRAIN_TRADES = 60
MIN_REGIME_TRADES = 15
ALWAYS_BLOCKED = {"LOW_VOL"}

MODEL_GRID = {
    "lookback": [3, 5, 10, 15, 20, 30, 40, 60],
    "vol_window": [20, 60, 120],
    "formation_days": [252, 504, 756],
    "entry_k": [1.0, 1.5, 2.0, 2.5],
    "max_vol_ratio": [None, 1.0],
}
REGIME_GRID = {
    "ema_trend": [10, 21, 50],
    "adr_window": [5, 10, 20],
    "ema_breadth": [50, 100, 200],
    "rv_window": [10, 20, 60],
    "pct_window": [126, 252, 504],
}
BASE = Config(hold_calendar_days=HOLD_CALENDAR_DAYS, max_pairs=10, max_per_stock=2,
              cost_model="futures", slippage_bps=5.0, start=SIM_START)


# ----------------------------------------------------------------------------- helpers

def model_key(c: Config) -> tuple:
    return (c.lookback, c.vol_window, c.extra["formation_days"], c.entry_k, c.max_vol_ratio)


def _norm_key(key: tuple) -> tuple:
    return tuple(None if isinstance(x, float) and np.isnan(x) else x for x in key)


def make_config(key: tuple, **kw) -> Config:
    L, vw, fd, k, v = key
    return replace(BASE, lookback=L, vol_window=vw, entry_k=k, max_vol_ratio=v, extra={"formation_days": fd},
                   label=f"L{L}_vw{vw}_f{fd}_k{k}_v{v}", **kw)


def neighbour_mean(scores: dict[tuple, float], grid: dict[str, list], key: tuple) -> float:
    dims = list(grid.values())
    pos = [dims[d].index(key[d]) for d in range(len(dims))]
    vals = []
    for delta in itertools.product((-1, 0, 1), repeat=len(dims)):
        p = [pos[d] + delta[d] for d in range(len(dims))]
        if all(0 <= p[d] < len(dims[d]) for d in range(len(dims))):
            k = tuple(dims[d][p[d]] for d in range(len(dims)))
            v = scores.get(k, np.nan)
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


class Tuner:
    def __init__(self) -> None:
        self.prices, self.market, self.meta = load_universe()
        self.bench = self.market["benchmark"]
        self.dates = self.prices.index
        self._schedules: dict[int, pd.DataFrame] = {}
        self._signals: dict[tuple, tuple] = {}
        self.regimes = self._all_regimes()

    # --- data
    def schedule(self, formation_days: int) -> pd.DataFrame:
        if formation_days not in self._schedules:
            self._schedules[formation_days] = build_schedule(self.prices, self.meta, formation_days=formation_days)
        return self._schedules[formation_days]

    def signals(self, L: int, vw: int, fd: int):
        key = (L, vw, fd)
        if key not in self._signals:
            if len(self._signals) > 12:
                self._signals.clear()
            self._signals[key] = compute_scores(self.prices, self.schedule(fd), L, vw)
        return self._signals[key]

    def _all_regimes(self) -> dict[tuple, pd.Series]:
        trend = {w: trend_indicator(self.market, w) for w in REGIME_GRID["ema_trend"]}
        adr = {w: adr_indicator(self.prices, w) for w in REGIME_GRID["adr_window"]}
        breadth = {w: breadth_indicator(self.prices, w) for w in REGIME_GRID["ema_breadth"]}
        vol = {(rv, pw): vol_indicator(self.market, rv, pw)["vol_pct"]
               for rv in REGIME_GRID["rv_window"] for pw in REGIME_GRID["pct_window"]}
        out = {}
        for key in itertools.product(*REGIME_GRID.values()):
            et, aw, eb, rv, pw = key
            ind = pd.DataFrame({"dist_ema21": trend[et], "ad_ratio": adr[aw], "pct_above_200ema": breadth[eb],
                                "vol_pct": vol[(rv, pw)]})
            out[key] = classify(ind)
        return out

    def run(self, cfg: Config, regime_key: tuple | None = None) -> dict:
        reg = self.regimes[regime_key or default_regime_key()]
        res = backtest(cfg, self.prices, reg, self.schedule(cfg.extra["formation_days"]),
                       self.signals(cfg.lookback, cfg.vol_window, cfg.extra["formation_days"]))
        res["metrics"] = perf_metrics(res["equity"]["equity"], self.bench, res["trades"])
        return res


def default_regime_key() -> tuple:
    p = RegimeParams()
    return (p.ema_trend, p.adr_window, p.ema_breadth, p.rv_window, p.pct_window)


# ----------------------------------------------------------------------------- 1. model grid

def run_model_grid(t: Tuner) -> tuple[pd.DataFrame, dict[tuple, pd.DataFrame]]:
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    rets, trades = {}, {}
    for fd in MODEL_GRID["formation_days"]:
        for L, vw in itertools.product(MODEL_GRID["lookback"], MODEL_GRID["vol_window"]):
            for k, v in itertools.product(MODEL_GRID["entry_k"], MODEL_GRID["max_vol_ratio"]):
                key = (L, vw, fd, k, v)
                res = t.run(make_config(key))
                rets[key] = res["equity"]["equity"].pct_change().fillna(0.0)
                trades[key] = res["trades"][["entry_date", "exit_date", "ret"]]
        print(f"  formation {fd}: done")
    out = (pd.DataFrame(rets), trades)
    CACHE.write_bytes(pickle.dumps(out))
    return out


def sharpe(r: pd.DataFrame | pd.Series):
    return r.mean() / r.std().replace(0, np.nan) * np.sqrt(252) if isinstance(r, pd.DataFrame) else \
        (r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan)


def select_models(rets: pd.DataFrame, trades: dict, lo: str, hi: str) -> tuple[list[tuple], dict]:
    window = rets.loc[lo:hi]
    sh = sharpe(window).to_dict()
    n_tr = {k: int(((tr["exit_date"] >= lo) & (tr["exit_date"] <= hi)).sum()) for k, tr in trades.items()}
    smooth = {k: neighbour_mean(sh, MODEL_GRID, k) for k in sh}
    ranked = sorted((k for k in sh if n_tr[k] >= MIN_TRAIN_TRADES and np.isfinite(smooth[k])), key=lambda k: -smooth[k])
    picked, seen = [], set()
    for k in ranked:
        if (k[0], k[3]) not in seen:
            seen.add((k[0], k[3]))
            picked.append(k)
        if len(picked) == TOP_K:
            break
    return picked, {k: (sh[k], smooth[k], n_tr[k]) for k in picked}


# ----------------------------------------------------------------------------- 2. regime selection

def signal_day_regime(t: Tuner, reg: pd.Series, entry_dates: pd.Series) -> np.ndarray:
    """Regime on the signal day (the trading day before the fill), matching the engine's entry_regime."""
    pos = t.dates.get_indexer(pd.DatetimeIndex(entry_dates)) - 1
    vals = reg.to_numpy(dtype=object)
    return np.where(pos >= 0, vals[np.clip(pos, 0, None)], None)


def learn_gate(rets: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    gate = {}
    for R in REGIMES:
        r = rets[labels == R]
        losing = len(r) >= MIN_REGIME_TRADES and r.mean() <= 0
        gate[R] = 0 if losing or R in ALWAYS_BLOCKED else 1
    return gate


def kept_tstat(rets: np.ndarray, labels: np.ndarray, gate: dict[str, int]) -> float:
    keep = np.array([lab is not None and gate.get(lab, 0) == 1 for lab in labels])
    r = rets[keep]
    if len(r) < 30 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(len(r)))


def select_regime(t: Tuner, pooled: pd.DataFrame) -> tuple[tuple, dict, dict]:
    scores, gates = {}, {}
    rets = pooled["ret"].to_numpy()
    for key, reg in t.regimes.items():
        labels = signal_day_regime(t, reg, pooled["entry_date"])
        gate = learn_gate(rets, labels)
        gates[key] = gate
        scores[key] = kept_tstat(rets, labels, gate)
    smooth = {k: neighbour_mean(scores, REGIME_GRID, k) for k in scores}
    default = default_regime_key()
    best = max(smooth, key=lambda k: (np.nan_to_num(smooth[k], nan=-1e9), k == default))
    return best, gates[best], {"score": scores[best], "smoothed": smooth[best]}


def trades_in(trades: dict, keys: list[tuple], lo: str, hi: str) -> pd.DataFrame:
    parts = [tr[(tr["exit_date"] >= lo) & (tr["exit_date"] <= hi)] for k, tr in ((k, trades[k]) for k in keys)]
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- 3. walk-forward

def main() -> None:
    t = Tuner()
    print("Running model grid (cached after first run)...")
    rets, trades = run_model_grid(t)
    # pandas turns a None inside tuple column labels into NaN; restore the exact config keys
    rets.columns = pd.Index([_norm_key(k) for k in rets.columns], tupleize_cols=False)
    print(f"Model grid: {rets.shape[1]} configs; regime grid: {len(t.regimes)} definitions")

    last_year = t.dates[-1].year
    streams = {"No regime gate": [], "Gate, default regime lookbacks": [], "Gate, tuned regime lookbacks (final)": []}
    per_regime_parts: dict[str, list[pd.Series]] = {R: [] for R in REGIMES}
    oos_trades, log = [], []
    year_regime: dict[int, tuple] = {}

    for year in range(FIRST_OOS_YEAR, last_year + 1):
        lo, hi = f"{year - TRAIN_YEARS}-01-01", f"{year - 1}-12-31"
        ys, ye = f"{year}-01-01", f"{year}-12-31"
        chosen, info = select_models(rets, trades, lo, hi)
        pooled = trades_in(trades, chosen, lo, hi)
        rkey, gate, rinfo = select_regime(t, pooled)
        dkey = default_regime_key()
        year_regime[year] = rkey
        dgate = learn_gate(pooled["ret"].to_numpy(), signal_day_regime(t, t.regimes[dkey], pooled["entry_date"]))

        streams["No regime gate"].append(rets.loc[ys:ye, chosen].mean(axis=1))
        tuned_parts, default_parts = [], []
        for key in chosen:
            cfg = make_config(key, end=ye)
            res = t.run(replace(cfg, pairs_by_regime={R: cfg.max_pairs * g for R, g in gate.items()}), rkey)
            tuned_parts.append(res["equity"]["equity"].pct_change().fillna(0.0).loc[ys:ye])
            tr = res["trades"]
            oos_trades.append(tr[(tr["exit_date"] >= ys) & (tr["exit_date"] <= ye)].assign(config=cfg.label))
            res_d = t.run(replace(cfg, pairs_by_regime={R: cfg.max_pairs * g for R, g in dgate.items()}), dkey)
            default_parts.append(res_d["equity"]["equity"].pct_change().fillna(0.0).loc[ys:ye])
            for R in REGIMES:
                r_only = t.run(replace(cfg, pairs_by_regime={R: cfg.max_pairs}), rkey)
                per_regime_parts[R].append(r_only["equity"]["equity"].pct_change().fillna(0.0).loc[ys:ye])
            sh, sm, n = info[key]
            log.append({"year": year, "config": cfg.label, "lookback": key[0], "vol_window": key[1],
                        "formation_days": key[2], "entry_k": key[3], "max_vol_ratio": key[4],
                        "train_sharpe": sh, "train_sharpe_smoothed": sm, "train_trades": n,
                        **dict(zip(REGIME_GRID, rkey)), "regime_tstat": rinfo["score"],
                        "blocked_regimes": ",".join(R for R, g in gate.items() if g == 0) or "-"})
        streams["Gate, tuned regime lookbacks (final)"].append(pd.concat(tuned_parts, axis=1).mean(axis=1))
        streams["Gate, default regime lookbacks"].append(pd.concat(default_parts, axis=1).mean(axis=1))
        print(f"{year}: models={[make_config(k).label for k in chosen]} regime={dict(zip(REGIME_GRID, rkey))} "
              f"blocked={[R for R, g in gate.items() if g == 0]}")

    def to_equity(parts: list[pd.Series]) -> pd.Series:
        r = pd.concat(parts)
        e = (1 + r).cumprod()
        return pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), e])

    eq = {name: to_equity(parts) for name, parts in streams.items()}
    all_trades = pd.concat(oos_trades, ignore_index=True)
    periods = {"OOS 2017-2021": ("2016-12-31", DEV_END), "OOS 2022-latest": (DEV_END, None), "OOS 2017-latest": (None, None)}
    rows = []
    for name, e in eq.items():
        for period, (s, en) in periods.items():
            m = perf_metrics(e.loc[s:en], t.bench, all_trades if name.endswith("(final)") and period == "OOS 2017-latest" else None)
            rows.append({"model": name, "period": period, **m})
    nifty = t.bench.loc["2016-12-30":]
    for period, (s, en) in periods.items():
        rows.append({"model": "Nifty 200 (buy & hold)", "period": period, **perf_metrics(nifty.loc[s:en], t.bench)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "wf_summary.csv", index=False)

    final = eq["Gate, tuned regime lookbacks (final)"]
    pd.DataFrame(eq).to_csv(OUT / "wf_equity.csv", index_label="date")
    pd.DataFrame(log).to_csv(OUT / "wf_selection_log.csv", index=False)
    all_trades.to_csv(OUT / "wf_trades.csv", index=False)
    yearly = pd.DataFrame({k: yearly_returns(v) for k, v in eq.items()})
    yearly["Nifty 200"] = yearly_returns(nifty)
    yearly.to_csv(OUT / "wf_yearly_returns.csv")
    monthly_returns_table(final).to_csv(OUT / "wf_monthly_returns.csv")
    # label each OOS day with the regime definition that was live in that year
    final_reg = pd.concat([t.regimes[k].loc[f"{y}-01-01":f"{y}-12-31"] for y, k in year_regime.items()]).reindex(final.index)
    regime_attribution(pd.DataFrame({"equity": final, "regime": final_reg}), all_trades).to_csv(OUT / "wf_regime_attribution.csv")

    reg_rows = []
    for R, parts in per_regime_parts.items():
        by_year: dict[int, list[pd.Series]] = {}
        for p in parts:
            if len(p):
                by_year.setdefault(p.index[0].year, []).append(p)
        stream = [pd.concat(v, axis=1).mean(axis=1) for _, v in sorted(by_year.items())]
        reg_rows.append({"regime": R, **perf_metrics(to_equity(stream), t.bench)})
    per_regime = pd.DataFrame(reg_rows).set_index("regime")
    per_regime.to_csv(OUT / "wf_per_regime_backtests.csv")

    # ------------------------------------------------------------------ 4. sensitivity of each lookback
    keys = list(rets.columns)
    sens_rows = []
    for period, (s, e) in {"dev 2014-21": (SIM_START, DEV_END), "holdout 2022+": (HOLD_START, None)}.items():
        sh = sharpe(rets.loc[s:e])
        for d, dim in enumerate(MODEL_GRID):
            for val in MODEL_GRID[dim]:
                sel = [k for k in keys if k[d] == val]
                sens_rows.append({"dimension": dim, "value": str(val), "period": period,
                                  "median_sharpe": float(np.nanmedian(sh[sel])), "share_positive": float((sh[sel] > 0).mean())})
    pd.DataFrame(sens_rows).to_csv(OUT / "sensitivity_model.csv", index=False)

    # regime lookbacks: gate learned on dev trades, judged on holdout trades of the models actually traded
    used = sorted({(r["lookback"], r["vol_window"], r["formation_days"], r["entry_k"], r["max_vol_ratio"]) for r in log},
                  key=str)
    used = [_norm_key(k) for k in used]
    dev_tr = trades_in(trades, used, SIM_START, DEV_END)
    hold_tr = trades_in(trades, used, HOLD_START, str(t.dates[-1].date()))
    rsens = []
    for key, reg in t.regimes.items():
        gate = learn_gate(dev_tr["ret"].to_numpy(), signal_day_regime(t, reg, dev_tr["entry_date"]))
        lab_h = signal_day_regime(t, reg, hold_tr["entry_date"])
        keep = np.array([lab is not None and gate.get(lab, 0) == 1 for lab in lab_h])
        rsens.append({**dict(zip(REGIME_GRID, key)), "holdout_kept_trades": int(keep.sum()),
                      "holdout_kept_mean_ret": float(hold_tr["ret"].to_numpy()[keep].mean()) if keep.any() else np.nan,
                      "holdout_all_mean_ret": float(hold_tr["ret"].mean())})
    rs = pd.DataFrame(rsens)
    rs.to_csv(OUT / "sensitivity_regime_full.csv", index=False)
    rmarg = []
    for dim in REGIME_GRID:
        g = rs.groupby(dim)["holdout_kept_mean_ret"].agg(["median", "mean"])
        for val, r in g.iterrows():
            rmarg.append({"dimension": dim, "value": val, "median_kept_trade_ret": r["median"], "mean_kept_trade_ret": r["mean"]})
    pd.DataFrame(rmarg).to_csv(OUT / "sensitivity_regime.csv", index=False)

    # ------------------------------------------------------------------ 5. recommended lookbacks for live use (trailing 3y to latest)
    end = t.dates[-1]
    lo = str((end - pd.DateOffset(years=TRAIN_YEARS)).date())
    chosen, info = select_models(rets, trades, lo, str(end.date()))
    rkey, gate, rinfo = select_regime(t, trades_in(trades, chosen, lo, str(end.date())))
    rec = {"trained_on": [lo, str(end.date())], "holding_period": f"{HOLD_CALENDAR_DAYS} calendar days",
           "models": [dict(zip(MODEL_GRID, k)) | {"train_sharpe": info[k][0]} for k in chosen],
           "regime_lookbacks": dict(zip(REGIME_GRID, rkey)), "blocked_regimes": [R for R, g in gate.items() if g == 0]}
    (OUT / "recommended_lookbacks.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["model", "period", "total_return", "CAGR", "sharpe", "sortino", "max_drawdown", "beta", "pct_positive_months"]
        print(summary[cols].round(3).to_string(index=False))
        print(yearly.round(3).to_string())
        print(per_regime[["total_return", "CAGR", "sharpe", "sortino", "max_drawdown", "beta"]].round(3))
        print(json.dumps(rec, indent=2, default=str))


if __name__ == "__main__":
    main()
