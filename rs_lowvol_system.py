"""
Relative-strength + low-volatility equity system with an index-futures risk overlay and the
market-neutral pair sleeve, tuned on 2015-2024 only and tested on 2025 onward.

Goal (all four, against BOTH benchmarks): higher return, higher Sharpe, lower volatility and
smaller max drawdown than
  (i)  equal-weight buy & hold of the Nifty 200 universe, and
  (ii) the Nifty 200 index (buy & hold).

Sleeves
1. Equity sleeve (long-only, cash delivery): at each month end, rank every eligible stock by a
   composite of CROSS-SECTIONAL PERCENTILE ranks (no z-scores):
     - relative strength: return from t-L-21 to t-21 (skip the most recent month)
     - low volatility:    1 - percentile of trailing daily-return std
   optionally keeping only stocks above their 200-day EMA (Kaufman-style trend filter; when
   few stocks qualify the sleeve holds cash). Hold the top N (equal or inverse-vol weight),
   with a buffer: a current holding stays while it ranks inside the top BUFFER x N.
   Decided at month-end close, executed at the next close; cash-delivery costs on turnover.
2. Risk overlay (index futures, cheap to adjust): short h_t units of the Nifty 200, either
     - "regime": h = 1 - weight of the regime profile chosen in kaufman_system.py, or
     - "vol_target": h = 1 - min(1, target / trailing 63d vol of the equity sleeve).
   Decided at close t, filled at close t+1, INDEX_COST per unit of hedge turnover.
3. Pair sleeve: the cointegrated percentile-RS pair book selected on 2015-2024 in
   kaufman_system.py (trades only in SIDEWAYS + HIGH_VOL), added on top (futures, own margin).

Idle cash earns 0% (conservative; Indian T-bills paid ~6%).

Survivorship note: the universe is TODAY's Nifty 200, so any stock-selection backtest (and the
equal-weight buy & hold benchmark (i)) benefits from survivorship bias; benchmark (ii), the
real index, does not.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from kaufman_factors import CompositeRegime, classify_composite, composite_scores, percentile_normalize
from kaufman_system import INDEX_COST, OUT as KAUFMAN_DIR, PAIR_GRID, System, equity_from_returns, pair_config, plateau
from perf import monthly_returns_table, perf_metrics, yearly_returns
from regime import REGIMES
from research import OUT as RESEARCH_DIR
from rs_pairs import side_cost_rates

OUT = RESEARCH_DIR / "rs_lowvol"
OUT.mkdir(exist_ok=True)

TRAIN = ("2015-01-01", "2024-12-31")
HALVES = [("2015-01-01", "2019-12-31"), ("2020-01-01", "2024-12-31")]
TEST_START = "2025-01-01"
SKIP_DAYS = 21
BUFFER = 1.5
VOL_TARGET = 0.12

GRID = {
    "rs_lookback": [126, 252],
    "vol_lookback": [63, 252],
    "w_rs": [0.3, 0.5, 0.7],
    "n_hold": [20, 30, 40],
    "trend_filter": [False, True],
    "weighting": ["equal", "inv_vol"],
    "overlay": ["none", "regime", "vol_target"],
    "pairs": [False, True],
}


# ----------------------------------------------------------------------------- benchmarks

def ew_buy_and_hold(prices: pd.DataFrame, start: str, end: str | None) -> pd.Series:
    """Equal-weight buy & hold of every universe stock that has a price on the start date (no rebalancing)."""
    px = prices.loc[start:end]
    first = px.iloc[0].dropna()
    rel = px[first.index].ffill() / first
    return rel.mean(axis=1)


# ----------------------------------------------------------------------------- equity sleeve

class EquitySleeve:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices
        self.rets = prices.pct_change(fill_method=None)
        self.dates = prices.index
        self.ema200 = prices.ewm(span=200, adjust=False, min_periods=200).mean()
        self._rs = {L: prices.shift(SKIP_DAYS) / prices.shift(SKIP_DAYS + L) - 1 for L in GRID["rs_lookback"]}
        self._vol = {V: self.rets.rolling(V, min_periods=int(V * 0.8)).std() for V in GRID["vol_lookback"]}
        buy, sell, _ = side_cost_rates("cash", 5.0)
        self.buy_cost, self.sell_cost = buy, sell
        month = self.dates.to_period("M")
        self.signal_days = [i for i in range(len(self.dates) - 1) if month[i] != month[i + 1]]

    def targets(self, i: int, p: dict, held: np.ndarray) -> np.ndarray:
        """Target weights (sum <= 1) decided at the close of trading day index i."""
        px = self.prices.iloc[i].to_numpy()
        rs = self._rs[p["rs_lookback"]].iloc[i].to_numpy()
        vol = self._vol[p["vol_lookback"]].iloc[i].to_numpy()
        ok = np.isfinite(px) & np.isfinite(rs) & np.isfinite(vol) & (vol > 0)
        if p["trend_filter"]:
            ok &= px > self.ema200.iloc[i].to_numpy()
        w = np.zeros(len(px))
        if ok.sum() == 0:
            return w
        rs_pct = pd.Series(rs[ok]).rank(pct=True).to_numpy()
        lv_pct = pd.Series(-vol[ok]).rank(pct=True).to_numpy()
        comp = np.full(len(px), -np.inf)
        comp[ok] = p["w_rs"] * rs_pct + (1 - p["w_rs"]) * lv_pct
        order = np.argsort(-comp, kind="stable")
        rank = np.empty(len(px), dtype=int)
        rank[order] = np.arange(len(px))
        n = p["n_hold"]
        keep = [j for j in np.flatnonzero(held > 0) if ok[j] and rank[j] < BUFFER * n]
        chosen = list(keep)
        for j in order:
            if len(chosen) >= n or not ok[j]:
                break
            if j not in chosen:
                chosen.append(j)
        chosen = np.array(chosen[:n], dtype=int)
        if p["weighting"] == "equal":
            w[chosen] = 1.0 / n
        else:
            iv = 1.0 / vol[chosen]
            w[chosen] = iv / iv.sum() * len(chosen) / n
        return w

    def run(self, p: dict, start: str) -> tuple[pd.Series, pd.Series]:
        """Daily returns (net of costs) and turnover per execution day."""
        lo = self.dates.searchsorted(pd.Timestamp(start))
        sig = [i for i in self.signal_days if i >= lo - 1]
        r = np.zeros(len(self.dates))
        turnover = {}
        held = np.zeros(self.prices.shape[1])
        px_all = self.prices.ffill().to_numpy()
        for k, i in enumerate(sig):
            e = i + 1  # execution day
            if e >= len(self.dates):
                break
            nxt = sig[k + 1] + 1 if k + 1 < len(sig) else len(self.dates) - 1
            # drifted weights at execution vs new targets -> trade costs
            new = self.targets(i, p, held)
            buys = np.clip(new - held, 0, None).sum()
            sells = np.clip(held - new, 0, None).sum()
            cost = buys * self.buy_cost + sells * self.sell_cost
            turnover[self.dates[e]] = buys + sells
            r[e] -= cost
            # hold `new` from close e to close nxt
            idx = np.flatnonzero(new > 0)
            if nxt <= e:
                held = new
                continue
            seg = px_all[e:nxt + 1, idx] / px_all[e, idx]
            vals = seg * new[idx]
            V = (1 - new.sum()) + vals.sum(axis=1)
            r[e + 1:nxt + 1] += V[1:] / V[:-1] - 1
            held = np.zeros_like(new)
            held[idx] = vals[-1] / V[-1]
        return pd.Series(r, index=self.dates).loc[start:], pd.Series(turnover)


# ----------------------------------------------------------------------------- overlay

def hedge_returns(h: pd.Series, bench_ret: pd.Series) -> pd.Series:
    """Short h units of the index (futures): decided at t, filled at t+1, earning from t+2."""
    held = h.shift(2).fillna(0.0)
    traded = h.shift(1).fillna(0.0)
    return -held * bench_ret - traded.diff().abs().fillna(0.0) * INDEX_COST


# ----------------------------------------------------------------------------- scoring

def stats(r: pd.Series, bench: pd.Series, a: str, b: str | None) -> dict:
    return perf_metrics(equity_from_returns(r.loc[a:b]), bench)


def criteria(m: dict, bms: list[dict]) -> dict:
    return {
        "beats_return": all(m["CAGR"] > b["CAGR"] for b in bms),
        "beats_sharpe": all(m["sharpe"] > b["sharpe"] for b in bms),
        "lower_vol": all(m["vol"] < b["vol"] for b in bms),
        "lower_dd": all(m["max_drawdown"] > b["max_drawdown"] for b in bms),
    }


def main() -> None:
    s = System()
    prices, bench = s.prices, s.bench
    bench_ret = s.bench_ret
    eq = EquitySleeve(prices)

    # benchmarks
    bm = {}
    for name, (a, b) in {"train": TRAIN, "H1": HALVES[0], "H2": HALVES[1], "test": (TEST_START, None)}.items():
        ew = ew_buy_and_hold(prices, a, b)
        bm[name] = {"(i) EW buy & hold": perf_metrics(ew / ew.iloc[0], bench),
                    "(ii) Nifty 200 index": perf_metrics(bench.loc[a:b] / bench.loc[a:b].iloc[0], bench)}

    # regime + pair sleeve exactly as selected (on 2015-2024) in kaufman_system.py
    kcfg = json.loads((KAUFMAN_DIR / "final_config.json").read_text())
    rp = CompositeRegime(**kcfg["regime"])
    regime = classify_composite(composite_scores(percentile_normalize(s.raw, rp.pct_window), kcfg["factors"]), rp)
    pp = {k: (v == "True" if v in ("True", "False") else v) for k, v in kcfg["pair_params"].items()}
    pp = {k: (int(v) if k in ("lookback", "pct_window", "formation_days") else v) for k, v in pp.items()}
    pp["entry_pct"] = float(pp["entry_pct"])
    pair_cfg = pair_config(pp, {"SIDEWAYS", "HIGH_VOL"})
    pair_r = s.pair_returns(pair_cfg, regime)["equity"]["equity"].pct_change().fillna(0.0).reindex(bench_ret.index).fillna(0.0)
    regime_hedge = regime.map(lambda r: 1.0 - kcfg["index_weights"].get(r, 0.0) if r is not None else 0.0).astype(float)

    # grid (equity sleeve runs once per stock-selection setting; overlay and pairs are cheap)
    stock_keys = ["rs_lookback", "vol_lookback", "w_rs", "n_hold", "trend_filter", "weighting"]
    rows, streams = [], {}
    for combo in itertools.product(*(GRID[k] for k in stock_keys)):
        p = dict(zip(stock_keys, combo))
        er, _ = eq.run(p, TRAIN[0])
        er = er.reindex(bench_ret.loc[TRAIN[0]:].index).fillna(0.0)
        rv = er.rolling(63, min_periods=40).std() * np.sqrt(252)
        hedges = {"none": pd.Series(0.0, index=er.index),
                  "regime": regime_hedge.reindex(er.index).fillna(0.0),
                  "vol_target": (1 - np.minimum(1.0, VOL_TARGET / rv)).fillna(0.0).clip(0, 1)}
        for ov, with_pairs in itertools.product(GRID["overlay"], GRID["pairs"]):
            tot = er + hedge_returns(hedges[ov], bench_ret.loc[er.index]) + (pair_r.loc[er.index] if with_pairs else 0.0)
            key = (*combo, ov, with_pairs)
            streams[key] = tot
            m_tr, m_h1, m_h2 = stats(tot, bench, *TRAIN), stats(tot, bench, *HALVES[0]), stats(tot, bench, *HALVES[1])
            c = criteria(m_tr, list(bm["train"].values()))
            c_h = [criteria(m_h1, list(bm["H1"].values())), criteria(m_h2, list(bm["H2"].values()))]
            mar = [m["CAGR"] / abs(m["max_drawdown"]) for m in (m_h1, m_h2)]
            rows.append({**dict(zip(GRID, key)), "train_cagr": m_tr["CAGR"], "train_sharpe": m_tr["sharpe"],
                         "train_vol": m_tr["vol"], "train_maxdd": m_tr["max_drawdown"],
                         "h1_sharpe": m_h1["sharpe"], "h2_sharpe": m_h2["sharpe"], "min_half_sharpe": min(m_h1["sharpe"], m_h2["sharpe"]),
                         "min_half_mar": min(mar), "criteria_train": sum(c.values()),
                         "criteria_halves": sum(sum(x.values()) for x in c_h), **{f"train_{k}": v for k, v in c.items()}})
    grid = pd.DataFrame(rows)

    # composite percentile score; candidates meeting all 4 criteria in training rank above the rest
    ranks = grid[["min_half_sharpe", "min_half_mar", "train_sharpe", "criteria_halves"]].rank(pct=True)
    grid["score"] = ranks.mean(axis=1) + (grid["criteria_train"] == 4).astype(float)
    dims = {k: GRID[k] for k in ["rs_lookback", "vol_lookback", "w_rs", "n_hold"]}
    grid["plateau_score"] = plateau(grid, dims, ["trend_filter", "weighting", "overlay", "pairs"])
    grid["plateau_score"] = grid["plateau_score"].where(grid["criteria_train"] == 4, grid["plateau_score"] - 1)
    grid = grid.sort_values("plateau_score", ascending=False)
    grid.to_csv(OUT / "train_grid.csv", index=False)
    best = grid.iloc[0]
    best_key = tuple(best[k] for k in GRID)
    best_key = tuple(bool(v) if isinstance(v, (np.bool_,)) else (int(v) if isinstance(v, (np.integer,)) else v) for v in best_key)
    print(f"Training: {int((grid['criteria_train'] == 4).sum())} of {len(grid)} candidates meet all 4 criteria vs both benchmarks")
    print("Chosen:", dict(zip(GRID, best_key)))

    # ---- final evaluation (test run once)
    p = dict(zip(stock_keys, best_key[:6]))
    er_full, turnover = eq.run(p, TRAIN[0])
    er_full = er_full.reindex(bench_ret.loc[TRAIN[0]:].index).fillna(0.0)
    rv = er_full.rolling(63, min_periods=40).std() * np.sqrt(252)
    hedge = {"none": pd.Series(0.0, index=er_full.index), "regime": regime_hedge.reindex(er_full.index).fillna(0.0),
             "vol_target": (1 - np.minimum(1.0, VOL_TARGET / rv)).fillna(0.0).clip(0, 1)}[best_key[6]]
    hedge_r = hedge_returns(hedge, bench_ret.loc[er_full.index])
    final = er_full + hedge_r + (pair_r.loc[er_full.index] if best_key[7] else 0.0)
    comps = {"Final system": final, "Equity sleeve only": er_full, "Equity + overlay": er_full + hedge_r,
             "Pair sleeve only": pair_r.loc[er_full.index]}
    out_rows = []
    for period, (a, b) in {"train 2015-2024": TRAIN, "train H1 2015-19": HALVES[0], "train H2 2020-24": HALVES[1],
                           "test 2025-latest": (TEST_START, None)}.items():
        key = {"train 2015-2024": "train", "train H1 2015-19": "H1", "train H2 2020-24": "H2", "test 2025-latest": "test"}[period]
        for name, r in comps.items():
            m = stats(r, bench, a, b)
            out_rows.append({"strategy": name, "period": period, **m, **(criteria(m, list(bm[key].values())) if name == "Final system" else {})})
        for name, m in bm[key].items():
            out_rows.append({"strategy": name, "period": period, **m})
    summary = pd.DataFrame(out_rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    ew_full = ew_buy_and_hold(prices, TRAIN[0], None)
    eqs = pd.DataFrame({"Final system": equity_from_returns(final), "Equity sleeve only": equity_from_returns(er_full),
                        "(i) EW buy & hold": ew_full / ew_full.iloc[0], "(ii) Nifty 200 index": bench.loc[TRAIN[0]:] / bench.loc[TRAIN[0]]})
    eqs.to_csv(OUT / "equity.csv", index_label="date")
    test_eq = pd.DataFrame({"Final system": equity_from_returns(final.loc[TEST_START:]),
                            "(i) EW buy & hold": ew_buy_and_hold(prices, TEST_START, None),
                            "(ii) Nifty 200 index": bench.loc[TEST_START:] / bench.loc[TEST_START:].iloc[0]})
    test_eq.to_csv(OUT / "equity_test.csv", index_label="date")
    yearly = pd.DataFrame({"Final system": yearly_returns(equity_from_returns(final)),
                           "(i) EW buy & hold": yearly_returns(ew_full), "(ii) Nifty 200 index": yearly_returns(bench.loc["2014-12-31":])})
    yearly.to_csv(OUT / "yearly_returns.csv")
    monthly_returns_table(equity_from_returns(final)).to_csv(OUT / "monthly_returns.csv")
    turnover.to_csv(OUT / "equity_turnover.csv")
    pd.DataFrame({"hedge": hedge, "regime": regime.reindex(hedge.index)}).to_csv(OUT / "hedge_path.csv")

    reg_rows = []
    for period, (a, b) in {"train": TRAIN, "test": (TEST_START, None)}.items():
        df = pd.DataFrame({"final": final, "bench": bench_ret, "regime": regime.shift(2)}).loc[a:b].dropna(subset=["regime"])
        for R in REGIMES:
            x = df.loc[df["regime"] == R]
            row = {"period": period, "regime": R, "days": len(x)}
            for col in ("final", "bench"):
                v = x[col]
                down = np.sqrt((np.minimum(v, 0) ** 2).mean()) if len(v) else np.nan
                row[f"{col}_ann_return"] = v.mean() * 252 if len(v) else np.nan
                row[f"{col}_sharpe"] = v.mean() / v.std() * np.sqrt(252) if len(v) > 2 and v.std() > 0 else np.nan
                row[f"{col}_sortino"] = v.mean() / down * np.sqrt(252) if down and down > 0 else np.nan
            reg_rows.append(row)
    pd.DataFrame(reg_rows).to_csv(OUT / "per_regime.csv", index=False)
    (OUT / "final_config.json").write_text(json.dumps({"params": dict(zip(GRID, best_key)), "buffer": BUFFER,
                                                       "vol_target": VOL_TARGET, "pair_config": pp,
                                                       "regime": kcfg["regime"]}, indent=2, default=str), encoding="utf-8")

    # ---- every window outside training ("else testing"): mid-2013..2014 (first months with all
    # indicators warmed up) and 2025..latest, plus the full sample -- same frozen configuration
    pre = "2013-07-01"
    pr_pre = s.pair_returns(replace(pair_cfg, start=pre), regime)["equity"]["equity"].pct_change().fillna(0.0)         .reindex(bench_ret.index).fillna(0.0)
    er_all, _ = eq.run(p, pre)
    er_all = er_all.reindex(bench_ret.loc[pre:].index).fillna(0.0)
    h_all = {"none": pd.Series(0.0, index=er_all.index), "regime": regime_hedge.reindex(er_all.index).fillna(0.0),
             "vol_target": (1 - np.minimum(1.0, VOL_TARGET / (er_all.rolling(63, min_periods=40).std() * np.sqrt(252))))
             .fillna(0.0).clip(0, 1)}[best_key[6]]
    final_all = er_all + hedge_returns(h_all, bench_ret.loc[er_all.index]) + (pr_pre.loc[er_all.index] if best_key[7] else 0.0)
    win_rows = []
    for name, (a, b) in {"pre-test 2013-07..2014-12": (pre, "2014-12-31"), "train 2015-2024": TRAIN,
                         "test 2025..latest": (TEST_START, None), "full 2013-07..latest": (pre, None)}.items():
        ew = ew_buy_and_hold(prices, a, b)
        bms = {"(i) EW buy & hold": perf_metrics(ew / ew.iloc[0], bench),
               "(ii) Nifty 200 index": perf_metrics(bench.loc[a:b] / bench.loc[a:b].iloc[0], bench)}
        m = stats(final_all, bench, a, b)
        keep = ("total_return", "CAGR", "sharpe", "sortino", "vol", "max_drawdown", "beta")
        win_rows.append({"window": name, "strategy": "Final system", **{k: m[k] for k in keep}, **criteria(m, list(bms.values()))})
        for bn, bm_ in bms.items():
            win_rows.append({"window": name, "strategy": bn, **{k: bm_[k] for k in keep}})
    pd.DataFrame(win_rows).to_csv(OUT / "all_windows_summary.csv", index=False)
    ew_pre = ew_buy_and_hold(prices, pre, None)
    pd.DataFrame({"Final system": equity_from_returns(final_all), "(i) EW buy & hold": ew_pre / ew_pre.iloc[0],
                  "(ii) Nifty 200 index": bench.loc[pre:] / bench.loc[pre:].iloc[0]}).to_csv(OUT / "equity_all_windows.csv", index_label="date")

    with pd.option_context("display.width", 230, "display.max_columns", 30):
        print(pd.DataFrame(win_rows).round(3).to_string(index=False))
        cols = ["strategy", "period", "total_return", "CAGR", "sharpe", "sortino", "vol", "max_drawdown", "beta",
                "beats_return", "beats_sharpe", "lower_vol", "lower_dd"]
        print(summary[[c for c in cols if c in summary]].round(3).to_string(index=False))
        print(yearly.round(3).to_string())


if __name__ == "__main__":
    main()
