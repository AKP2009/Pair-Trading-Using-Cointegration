"""
Survivorship-free re-selection of the regime-aware cointegrated pair strategy, judged against the
Nifty 200 Total Return Index out of sample.

Protocol (fixed before running; nothing below is tuned on 2025+):
1. Universe: point-in-time Nifty 200 members (pit_universe.membership_any). Pairs are formed on each
   formation date only from stocks that were members that day; breadth regime factors count only
   stocks that were members that day. 47 of 420 historical members (delisted / merged, e.g. DHFL,
   Reliance Capital, HDFC Ltd) have no price data left on Yahoo -- residual survivorship, reported.
2. Regime definition: same procedure and grid as kaufman_system.py stage 1, on 2015-2024 only.
3. Pair strategy: same signal grid (percentile RS / Kaufman stress, lookbacks, entry percentile,
   median exit, cointegration window) x regime modes, 15-calendar-day hold, futures costs; scored on
   PAIRS-ONLY training returns with the percentile composite (min-half Sharpe, min-half MAR, full
   Sharpe) + plateau smoothing; >= 100 training trades.
4. The single selected configuration is run once on 2025-01-01 .. latest and scored against the
   Nifty 200 TRI (official file if present in data/, else price index + 1.5%/yr dividends).

Outputs: research/pit_pairs/
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from final_oos import nifty200_tri, scorecard
from kaufman_factors import (
    CompositeRegime, classify_composite, composite_scores, factor_diagnostics, percentile_normalize, raw_factors,
    select_factors,
)
from kaufman_system import (
    HALVES, PAIR_GRID, PROFILES, REGIME_GRID, TRAIN, System, candidate_stats, composite_score, equity_from_returns,
    index_sleeve, pair_config, plateau, window_metrics,
)
from perf import monthly_returns_table, perf_metrics, yearly_returns
from pit_universe import load_full_pit_universe, member_mask, membership_any
from regime import REGIMES
from research import OUT as RESEARCH_DIR
from rs_pairs import form_pairs, formation_starts

OUT = RESEARCH_DIR / "pit_pairs"
OUT.mkdir(exist_ok=True)
TEST_START = "2025-01-01"
PAIR_MODES = {
    "SIDEWAYS + HIGH_VOL": {"SIDEWAYS", "HIGH_VOL"},
    "all but LOW_VOL": {"BULLISH", "BEARISH", "SIDEWAYS", "HIGH_VOL"},
    "all regimes": set(REGIMES),
}


class PITSystem(System):
    """System on the point-in-time universe: PIT pair formation and member-masked breadth factors."""

    def __init__(self) -> None:  # noqa: D107 -- deliberately does not call System.__init__ (different data)
        self.prices, self.market, self.meta, self.missing = load_full_pit_universe()
        self.bench = self.market["benchmark"]
        self.bench_ret = self.bench.pct_change().fillna(0.0)
        self.mask = member_mask(self.prices.index, list(self.prices.columns))
        self.raw = raw_factors(self.prices, self.market, self.mask)
        self._sched: dict[int, pd.DataFrame] = {}
        self._sig: dict[tuple, tuple] = {}

    def schedule(self, formation_days: int) -> pd.DataFrame:
        if formation_days not in self._sched:
            starts = formation_starts(self.prices.index)
            out = []
            for i, s in enumerate(starts):
                members = membership_any(s - pd.Timedelta(days=1))
                pairs = form_pairs(self.prices, self.meta[self.meta["symbol"].isin(members)], s, formation_days=formation_days)
                if pairs.empty:
                    continue
                end = starts[i + 1] if i + 1 < len(starts) else self.prices.index[-1] + pd.Timedelta(days=1)
                pairs.insert(0, "period_end", end)
                pairs.insert(0, "period_start", s)
                out.append(pairs)
            self._sched[formation_days] = pd.concat(out, ignore_index=True)
        return self._sched[formation_days]


def main() -> None:
    s = PITSystem()
    print(f"PIT universe: {s.prices.shape[1]} stocks with prices; {len(s.missing)} historical members without data")
    bh = window_metrics(s.bench_ret, s.bench, *TRAIN)

    # ---- stage 1: regime definition (identical procedure to kaufman_system.py)
    factor_sets, scores_by_window = {}, {}
    for w in REGIME_GRID["pct_window"]:
        pct = percentile_normalize(s.raw, w)
        corr, use = factor_diagnostics(pct, s.market, *TRAIN)
        factor_sets[w] = select_factors(corr, use)
        scores_by_window[w] = composite_scores(pct, factor_sets[w])
    rows, regimes = [], {}
    for w, (bull, bear), hv, lv, cf in itertools.product(*REGIME_GRID.values()):
        rp = CompositeRegime(pct_window=w, bull=bull, bear=bear, high_vol=hv, low_vol=lv, confirm=cf)
        reg = classify_composite(scores_by_window[w], rp)
        regimes[rp] = reg
        for prof_name, prof in PROFILES.items():
            rows.append({"profile": prof_name, "pct_window": w, "thresholds": (bull, bear), "high_vol": hv, "low_vol": lv,
                         "confirm": cf, **candidate_stats(index_sleeve(reg, prof, s.bench_ret), s.bench)})
    st1 = pd.DataFrame(rows)
    st1["score"] = composite_score(st1, bh)
    st1["plateau_score"] = plateau(st1, REGIME_GRID, ["profile"])
    st1 = st1.sort_values("plateau_score", ascending=False)
    st1.to_csv(OUT / "stage1_regime_grid.csv", index=False)
    b1 = st1.iloc[0]
    rp = CompositeRegime(pct_window=int(b1["pct_window"]), bull=b1["thresholds"][0], bear=b1["thresholds"][1],
                         high_vol=float(b1["high_vol"]), low_vol=float(b1["low_vol"]), confirm=int(b1["confirm"]))
    regime = regimes[rp]
    print(f"[stage 1] regime {rp}; factors {factor_sets[rp.pct_window]}")

    # ---- stage 2: pair strategy, scored on pairs-only training returns
    rows2, streams = [], {}
    dims = {k: v for k, v in PAIR_GRID.items() if k != "signal"}
    for combo in itertools.product(*PAIR_GRID.values()):
        p = dict(zip(PAIR_GRID, combo))
        if p["signal"] == "stress" and p["pct_window"] != PAIR_GRID["pct_window"][0]:
            continue
        for mode, on in PAIR_MODES.items():
            cfg = pair_config(p, on)
            res = s.pair_returns(cfg, regime)
            r = res["equity"]["equity"].pct_change().fillna(0.0)
            streams[(mode, cfg.label)] = r
            tr = res["trades"]
            n_train = int(((tr["exit_date"] >= TRAIN[0]) & (tr["exit_date"] <= TRAIN[1])).sum())
            rows2.append({"mode": mode, **p, "label": cfg.label, **candidate_stats(r, s.bench), "n_trades_train": n_train})
    st2 = pd.DataFrame(rows2)
    ranks = st2[["min_half_sharpe", "min_half_mar", "train_sharpe"]].rank(pct=True)
    st2["score"] = ranks.mean(axis=1) - (st2["n_trades_train"] < 100).astype(float)
    st2["plateau_score"] = np.nan
    for sig, d in (("rs_pct", dims), ("stress", {k: v for k, v in dims.items() if k != "pct_window"})):
        sub = st2[st2["signal"] == sig]
        st2.loc[sub.index, "plateau_score"] = plateau(sub, d, ["mode", "signal"])
    st2 = st2.sort_values("plateau_score", ascending=False)
    st2.to_csv(OUT / "stage2_pair_grid.csv", index=False)
    b2 = st2.iloc[0]
    print(f"[stage 2] {b2['mode']} | {b2['label']} | train Sharpe {b2['train_sharpe']:.2f}, halves "
          f"{b2['half1_sharpe']:.2f}/{b2['half2_sharpe']:.2f}, max DD {b2['train_maxdd']:.1%}, trades {b2['n_trades_train']}")

    # ---- frozen -> out-of-sample, once
    final = streams[(b2["mode"], b2["label"])]
    tri, note = nifty200_tri(s.bench)
    rows = []
    for period, (a, b) in {"OUT-OF-SAMPLE 2025..latest": (TEST_START, None), "train 2015-2024": TRAIN,
                           "train half 2015-19": HALVES[0], "train half 2020-24": HALVES[1]}.items():
        rows.append({"period": period, **scorecard(final, tri, a, b)})
    card = pd.DataFrame(rows)
    card["criteria_met"] = card[["1_higher_return", "2_higher_sharpe", "3_lower_vol", "4_lower_maxdd"]].sum(axis=1)
    card.to_csv(OUT / "scorecard_vs_tri.csv", index=False)

    test_r = final.loc[TEST_START:]
    tri_t = tri.loc[TEST_START:]
    pd.DataFrame({"PIT pair strategy": equity_from_returns(test_r), "Nifty 200 TRI": tri_t / tri_t.iloc[0]}) \
        .to_csv(OUT / "oos_equity.csv", index_label="date")
    pd.DataFrame({"PIT pair strategy": equity_from_returns(final.loc[TRAIN[0]:]),
                  "Nifty 200 TRI": tri.loc[TRAIN[0]:] / tri.loc[TRAIN[0]:].iloc[0]}).to_csv(OUT / "full_equity.csv", index_label="date")
    monthly_returns_table(equity_from_returns(final.loc[TRAIN[0]:])).to_csv(OUT / "monthly_returns.csv")
    pd.DataFrame({"PIT pair strategy": yearly_returns(equity_from_returns(final.loc[TRAIN[0]:])),
                  "Nifty 200 TRI": yearly_returns(tri.loc["2014-12-31":])}).to_csv(OUT / "yearly_returns.csv")
    reg_rows = []
    df = pd.DataFrame({"strategy": final, "tri": tri.pct_change(), "regime": regime.shift(2)}).dropna()
    for period, (a, b) in {"train": TRAIN, "test": (TEST_START, None)}.items():
        dd = df.loc[a:b]
        for R in REGIMES:
            x = dd[dd["regime"] == R]
            row = {"period": period, "regime": R, "days": len(x)}
            for col in ("strategy", "tri"):
                v = x[col]
                down = np.sqrt((np.minimum(v, 0) ** 2).mean()) if len(v) else np.nan
                row[f"{col}_ann_return"] = v.mean() * 252 if len(v) else np.nan
                row[f"{col}_sharpe"] = v.mean() / v.std() * np.sqrt(252) if len(v) > 2 and v.std() > 0 else np.nan
                row[f"{col}_sortino"] = v.mean() / down * np.sqrt(252) if down and down > 0 else np.nan
            reg_rows.append(row)
    pd.DataFrame(reg_rows).to_csv(OUT / "per_regime.csv", index=False)
    (OUT / "final_config.json").write_text(json.dumps({
        "regime": asdict(rp), "factors": factor_sets[rp.pct_window], "pair_mode": b2["mode"],
        "pair_params": {k: b2[k] for k in PAIR_GRID}, "benchmark": note, "missing_history": s.missing,
    }, indent=2, default=str), encoding="utf-8")

    with pd.option_context("display.width", 230, "display.max_columns", 20):
        print(card[["period", "total_return", "sharpe", "vol", "max_drawdown", "tri_total_return", "tri_sharpe", "tri_vol",
                    "tri_maxdd", "criteria_met"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
