"""
Regime-timed Nifty 200 system with a market-neutral pair sleeve, tuned on a FIXED
training window (2015-2024) and tested once on 2025 onward.

Design (sources: P. Kaufman, "Trading Systems and Methods" -- Efficiency Ratio, stochastic /
percentile normalization, robustness testing; the pair "stress indicator" is from Kaufman's
"Alpha Trading" and his 2014 S&C article "Timing the Market with Pairs Logic"):

1. Regime factors (kaufman_factors.py) are percentile-normalized and de-correlated on the
   training window; a directional score (trend + breadth) and a volatility score, both in
   0..1, classify each day as BULLISH / BEARISH / SIDEWAYS / LOW_VOL / HIGH_VOL.
2. Index sleeve: long Nifty 200 (index futures / ETF) with a weight set by the regime profile,
   decided at day t's close, filled at t+1's close, charged INDEX_COST per unit of turnover.
3. Pair sleeve: cointegrated same-industry pairs traded on a percentile signal (percentile of
   relative strength, or Kaufman's stress indicator), 15-calendar-day hold, active only in the
   regimes of the chosen pair mode.

Scoring (Kaufman-style robustness, percentile-normalized -- no z-scores)
- Hard filter: on the training window the candidate must beat Nifty 200 buy & hold on BOTH
  Sharpe and max drawdown.
- Components: min Sharpe over the two training halves (2015-19, 2020-24), min MAR
  (CAGR / |max DD|) over the halves, and full-training Sharpe -- consistency first.
- Each component is converted to a percentile rank across candidates; the composite is
  their mean.
- Plateau: the final score of a candidate is the mean composite of itself and its
  neighbours in the parameter grid, so a broad stable region beats an isolated spike.

Outputs: research/kaufman/*
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from kaufman_factors import (
    CompositeRegime, classify_composite, composite_scores, factor_diagnostics, percentile_normalize, raw_factors,
    select_factors,
)
from perf import monthly_returns_table, perf_metrics, yearly_returns
from regime import REGIMES
from research import OUT as RESEARCH_DIR
from rs_pairs import Config, backtest, build_schedule, compute_scores
from universe import load_universe

OUT = RESEARCH_DIR / "kaufman"
OUT.mkdir(exist_ok=True)

TRAIN = ("2015-01-01", "2024-12-31")
HALVES = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2024-12-31")]
TEST_START = "2025-01-01"
INDEX_COST = 0.0007  # per unit of index-sleeve turnover: ~5 bps slippage + futures STT/charges

# Index weight per regime. Every profile is fully long in BULLISH and flat in BEARISH.
PROFILES = {
    "trend":            {"BULLISH": 1.0, "LOW_VOL": 1.0, "SIDEWAYS": 0.5, "BEARISH": 0.0, "HIGH_VOL": 0.0},
    "trend_full":       {"BULLISH": 1.0, "LOW_VOL": 1.0, "SIDEWAYS": 1.0, "BEARISH": 0.0, "HIGH_VOL": 0.0},
    "trend_hv_half":    {"BULLISH": 1.0, "LOW_VOL": 1.0, "SIDEWAYS": 1.0, "BEARISH": 0.0, "HIGH_VOL": 0.5},
    "defensive":        {"BULLISH": 1.0, "LOW_VOL": 0.5, "SIDEWAYS": 0.5, "BEARISH": 0.0, "HIGH_VOL": 0.0},
    "bear_half":        {"BULLISH": 1.0, "LOW_VOL": 1.0, "SIDEWAYS": 1.0, "BEARISH": 0.5, "HIGH_VOL": 0.0},
}
REGIME_GRID = {
    "pct_window": [252, 504, 756],
    "thresholds": [(0.55, 0.45), (0.60, 0.40), (0.65, 0.35), (0.70, 0.30)],
    "high_vol": [0.80, 0.85, 0.90],
    "low_vol": [0.15, 0.25],
    "confirm": [1, 3, 5],
}
PAIR_MODES = {
    "index only (no pairs)": set(),
    "pairs in SIDEWAYS + HIGH_VOL": {"SIDEWAYS", "HIGH_VOL"},
    "pairs in all but LOW_VOL": {"BULLISH", "BEARISH", "SIDEWAYS", "HIGH_VOL"},
}
PAIR_GRID = {
    "signal": ["rs_pct", "stress"],
    "lookback": [5, 10, 20],
    "pct_window": [126, 252],
    "entry_pct": [0.90, 0.95],
    "exit_on_median": [False, True],
    "formation_days": [504, 756],
}
PAIR_BASE = Config(hold_calendar_days=15, max_pairs=10, max_per_stock=2, cost_model="futures",
                   slippage_bps=5.0, start=TRAIN[0])


# ----------------------------------------------------------------------------- metrics & scoring

def equity_from_returns(r: pd.Series) -> pd.Series:
    r = r.dropna()
    return pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), (1 + r).cumprod()])


def window_metrics(r: pd.Series, bench: pd.Series, start: str, end: str | None) -> dict:
    return perf_metrics(equity_from_returns(r.loc[start:end]), bench)


def candidate_stats(r: pd.Series, bench: pd.Series) -> dict:
    full = window_metrics(r, bench, *TRAIN)
    halves = [window_metrics(r, bench, *h) for h in HALVES]
    mar = [h["CAGR"] / abs(h["max_drawdown"]) if h["max_drawdown"] < 0 else np.nan for h in halves]
    return {"train_sharpe": full["sharpe"], "train_maxdd": full["max_drawdown"], "train_cagr": full["CAGR"],
            "train_sortino": full["sortino"], "min_half_sharpe": min(h["sharpe"] for h in halves),
            "min_half_mar": np.nanmin(mar), "half1_sharpe": halves[0]["sharpe"], "half2_sharpe": halves[1]["sharpe"]}


def composite_score(df: pd.DataFrame, bh: dict) -> pd.Series:
    """Percentile-rank composite (no z-scores), with the beat-buy-&-hold filter applied."""
    ranks = df[["min_half_sharpe", "min_half_mar", "train_sharpe"]].rank(pct=True)
    score = ranks.mean(axis=1)
    beats = (df["train_sharpe"] > bh["sharpe"]) & (df["train_maxdd"] > bh["max_drawdown"])
    return score.where(beats, score - 1.0)  # failing candidates rank below every passing one


def plateau(df: pd.DataFrame, dims: dict[str, list], group_cols: list[str], score_col: str = "score") -> pd.Series:
    """Mean score of each candidate and its +/-1-step neighbours along the ordered dims (within its group)."""
    lookup = {tuple(r[c] for c in group_cols + list(dims)): r[score_col] for _, r in df.iterrows()}
    out = []
    for _, r in df.iterrows():
        pos = [dims[d].index(r[d]) for d in dims]
        vals = []
        for delta in itertools.product((-1, 0, 1), repeat=len(dims)):
            p = [pos[i] + delta[i] for i in range(len(dims))]
            if all(0 <= p[i] < len(list(dims.values())[i]) for i in range(len(dims))):
                key = tuple(r[c] for c in group_cols) + tuple(list(dims.values())[i][p[i]] for i in range(len(dims)))
                if key in lookup and np.isfinite(lookup[key]):
                    vals.append(lookup[key])
        out.append(np.mean(vals) if vals else np.nan)
    return pd.Series(out, index=df.index)


# ----------------------------------------------------------------------------- sleeves

def index_sleeve(regime: pd.Series, profile: dict[str, float], bench_ret: pd.Series) -> pd.Series:
    w = regime.map(lambda r: profile.get(r, 0.0) if r is not None else 0.0).astype(float)
    held = w.shift(2).fillna(0.0)   # decided at t close, filled at t+1 close, earns from t+2
    traded = w.shift(1).fillna(0.0)  # turnover happens on the fill day
    return held * bench_ret - traded.diff().abs().fillna(0.0) * INDEX_COST


class System:
    def __init__(self) -> None:
        self.prices, self.market, self.meta = load_universe()
        self.bench = self.market["benchmark"]
        self.bench_ret = self.bench.pct_change().fillna(0.0)
        self.raw = raw_factors(self.prices, self.market)
        self._sched: dict[int, pd.DataFrame] = {}
        self._sig: dict[tuple, tuple] = {}

    def schedule(self, formation_days: int) -> pd.DataFrame:
        if formation_days not in self._sched:
            self._sched[formation_days] = build_schedule(self.prices, self.meta, formation_days=formation_days)
        return self._sched[formation_days]

    def pair_returns(self, cfg: Config, regime: pd.Series) -> dict:
        fd = cfg.extra["formation_days"]
        key = (cfg.signal, cfg.lookback, cfg.pct_window, fd)
        if key not in self._sig:
            if len(self._sig) > 8:
                self._sig.clear()
            self._sig[key] = compute_scores(self.prices, self.schedule(fd), cfg.lookback, cfg.vol_window, cfg.signal, cfg.pct_window)
        return backtest(cfg, self.prices, regime, self.schedule(fd), self._sig[key])


def pair_config(p: dict, regimes_on: set[str]) -> Config:
    return replace(PAIR_BASE, signal=p["signal"], lookback=p["lookback"], pct_window=p["pct_window"],
                   entry_k=p["entry_pct"] - 0.5, exit_on_revert=p["exit_on_median"],
                   pairs_by_regime={R: PAIR_BASE.max_pairs for R in regimes_on},
                   extra={"formation_days": p["formation_days"]},
                   label="{signal}_L{lookback}_pw{pct_window}_e{entry_pct}_x{exit_on_median}_f{formation_days}".format(**p))


# ----------------------------------------------------------------------------- main

def main() -> None:
    s = System()
    bh = window_metrics(s.bench_ret, s.bench, *TRAIN)
    print(f"Buy & hold Nifty 200, training 2015-2024: Sharpe {bh['sharpe']:.2f}, max DD {bh['max_drawdown']:.1%}, CAGR {bh['CAGR']:.1%}")

    # ---- factor diagnostics & selection (training window only), per percentile window
    factor_sets, scores_by_window = {}, {}
    for w in REGIME_GRID["pct_window"]:
        pct = percentile_normalize(s.raw, w)
        corr, use = factor_diagnostics(pct, s.market, *TRAIN)
        fs = select_factors(corr, use)
        factor_sets[w] = fs
        scores_by_window[w] = composite_scores(pct, fs)
        corr.to_csv(OUT / f"factor_correlation_pct{w}.csv")
        use.to_csv(OUT / f"factor_usefulness_pct{w}.csv")
    (OUT / "selected_factors.json").write_text(json.dumps({str(k): v for k, v in factor_sets.items()}, indent=2))
    print("Selected factors:", factor_sets)

    # ---- stage 1: regime definition + index profile
    rows, regimes = [], {}
    for w, (bull, bear), hv, lv, cf in itertools.product(*REGIME_GRID.values()):
        rp = CompositeRegime(pct_window=w, bull=bull, bear=bear, high_vol=hv, low_vol=lv, confirm=cf)
        reg = classify_composite(scores_by_window[w], rp)
        regimes[rp] = reg
        for prof_name, prof in PROFILES.items():
            r = index_sleeve(reg, prof, s.bench_ret)
            rows.append({"profile": prof_name, "pct_window": w, "thresholds": (bull, bear), "high_vol": hv,
                         "low_vol": lv, "confirm": cf, **candidate_stats(r, s.bench)})
    st1 = pd.DataFrame(rows)
    st1["score"] = composite_score(st1, bh)
    st1["plateau_score"] = plateau(st1, REGIME_GRID, ["profile"])
    st1 = st1.sort_values("plateau_score", ascending=False)
    st1.to_csv(OUT / "stage1_regime_index_grid.csv", index=False)
    best1 = st1.iloc[0]
    rp = CompositeRegime(pct_window=int(best1["pct_window"]), bull=best1["thresholds"][0], bear=best1["thresholds"][1],
                         high_vol=float(best1["high_vol"]), low_vol=float(best1["low_vol"]), confirm=int(best1["confirm"]))
    regime = regimes[rp]
    profile = PROFILES[best1["profile"]]
    idx_r = index_sleeve(regime, profile, s.bench_ret)
    print(f"[stage 1] {best1['profile']} {rp} | train Sharpe {best1['train_sharpe']:.2f} maxDD {best1['train_maxdd']:.1%} "
          f"halves {best1['half1_sharpe']:.2f}/{best1['half2_sharpe']:.2f} | beat B&H in "
          f"{(st1['score'] >= 0).mean():.0%} of regime/profile candidates")

    # ---- stage 2: pair sleeve (signal + lookbacks + regimes where it trades)
    rows2, pair_r = [], {}
    pair_dims = {k: v for k, v in PAIR_GRID.items() if k != "signal"}
    rows2.append({"mode": "index only (no pairs)", "signal": "-", **{k: v[0] for k, v in pair_dims.items()},
                  **candidate_stats(idx_r, s.bench), "n_trades_train": 0})
    pair_r[("index only (no pairs)", "-")] = pd.Series(0.0, index=idx_r.index)
    for combo in itertools.product(*PAIR_GRID.values()):
        p = dict(zip(PAIR_GRID, combo))
        if p["signal"] == "stress" and p["pct_window"] != PAIR_GRID["pct_window"][0]:
            continue  # stress uses its own stochastic window; pct_window is irrelevant
        for mode_name, regimes_on in PAIR_MODES.items():
            if not regimes_on:
                continue
            cfg = pair_config(p, regimes_on)
            res = s.pair_returns(cfg, regime)
            pr = res["equity"]["equity"].pct_change().fillna(0.0).reindex(idx_r.index).fillna(0.0)
            pair_r[(mode_name, cfg.label)] = pr
            tr = res["trades"]
            n_train = int(((tr["exit_date"] >= TRAIN[0]) & (tr["exit_date"] <= TRAIN[1])).sum())
            rows2.append({"mode": mode_name, **p, "label": cfg.label, **candidate_stats(idx_r + pr, s.bench),
                          "pairs_only_train_sharpe": window_metrics(pr, s.bench, *TRAIN)["sharpe"], "n_trades_train": n_train})
    st2 = pd.DataFrame(rows2)
    st2["score"] = composite_score(st2, bh)
    st2.loc[(st2["mode"] != "index only (no pairs)") & (st2["n_trades_train"] < 100), "score"] -= 1.0  # too few trades
    stress_mask = st2["signal"] == "stress"
    st2["plateau_score"] = np.nan
    dims_pct = pair_dims
    dims_stress = {k: v for k, v in pair_dims.items() if k != "pct_window"}
    sub = st2[~stress_mask & (st2["signal"] != "-")]
    st2.loc[sub.index, "plateau_score"] = plateau(sub, dims_pct, ["mode", "signal"])
    sub = st2[stress_mask]
    st2.loc[sub.index, "plateau_score"] = plateau(sub, dims_stress, ["mode", "signal"])
    st2.loc[st2["signal"] == "-", "plateau_score"] = st2.loc[st2["signal"] == "-", "score"]
    st2 = st2.sort_values("plateau_score", ascending=False)
    st2.to_csv(OUT / "stage2_pair_grid.csv", index=False)
    best2 = st2.iloc[0]
    print(f"[stage 2] {best2['mode']} | {best2.get('label', '-')} | train Sharpe {best2['train_sharpe']:.2f} maxDD {best2['train_maxdd']:.1%}")

    # ---- final evaluation: training vs test, sleeves vs buy & hold
    pr = pair_r[(best2["mode"], best2["label"] if best2["signal"] != "-" else "-")]
    final_r = idx_r + pr
    streams = {"Final system": final_r, "Index sleeve only": idx_r, "Pair sleeve only": pr, "Nifty 200 buy & hold": s.bench_ret}
    rows = []
    for name, r in streams.items():
        for period, (a, b) in {"train 2015-2024": TRAIN, "test 2025-latest": (TEST_START, None),
                               "train half 2015-19": HALVES[0], "train half 2020-24": HALVES[1]}.items():
            m = window_metrics(r, s.bench, a, b)
            rows.append({"strategy": name, "period": period, **m})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    test_eq = pd.DataFrame({k: equity_from_returns(v.loc[TRAIN[0]:]) for k, v in streams.items()})
    test_eq.to_csv(OUT / "equity.csv", index_label="date")
    yearly = pd.DataFrame({k: yearly_returns(equity_from_returns(v.loc[TRAIN[0]:])) for k, v in streams.items()})
    yearly.to_csv(OUT / "yearly_returns.csv")
    monthly_returns_table(equity_from_returns(final_r.loc[TRAIN[0]:])).to_csv(OUT / "monthly_returns_final.csv")
    corr_sleeves = pd.DataFrame({"index": idx_r, "pairs": pr, "bench": s.bench_ret}).loc[TRAIN[0]:].corr()
    corr_sleeves.to_csv(OUT / "sleeve_correlation.csv")

    # per-regime performance (days labelled by the live regime), training and test
    reg_rows = []
    for period, (a, b) in {"train": TRAIN, "test": (TEST_START, None)}.items():
        df = pd.DataFrame({"final": final_r, "index": idx_r, "pairs": pr, "bench": s.bench_ret,
                           "regime": regime.shift(2)}).loc[a:b].dropna(subset=["regime"])  # regime that set the position
        for R in REGIMES:
            d = df[df["regime"] == R]
            row = {"period": period, "regime": R, "days": len(d), "share_of_days": len(d) / max(len(df), 1)}
            for col in ("final", "index", "pairs", "bench"):
                x = d[col]
                down = np.sqrt((np.minimum(x, 0) ** 2).mean()) if len(x) else np.nan
                row[f"{col}_ann_return"] = x.mean() * 252 if len(x) else np.nan
                row[f"{col}_sharpe"] = x.mean() / x.std() * np.sqrt(252) if len(x) > 2 and x.std() > 0 else np.nan
                row[f"{col}_sortino"] = x.mean() / down * np.sqrt(252) if down and down > 0 else np.nan
            reg_rows.append(row)
    pd.DataFrame(reg_rows).to_csv(OUT / "per_regime.csv", index=False)

    final_cfg = {"regime": asdict(rp), "factors": factor_sets[rp.pct_window], "index_profile": best1["profile"],
                 "index_weights": profile, "pair_mode": best2["mode"],
                 "pair_params": {k: best2[k] for k in PAIR_GRID} if best2["signal"] != "-" else None,
                 "hold": "15 calendar days", "costs": {"index_per_unit_turnover": INDEX_COST, "pairs": "NSE futures + 5bps/side"}}
    (OUT / "final_config.json").write_text(json.dumps(final_cfg, indent=2, default=str), encoding="utf-8")

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        cols = ["strategy", "period", "total_return", "CAGR", "sharpe", "sortino", "max_drawdown", "beta", "vol"]
        print(summary[cols].round(3).to_string(index=False))
        print(yearly.round(3).to_string())
        print(corr_sleeves.round(2))
        print(json.dumps(final_cfg, indent=2, default=str))


if __name__ == "__main__":
    main()
