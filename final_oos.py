"""
Out-of-sample scorecard against the Nifty 200 Total Return Index (TRI).

Goal (out-of-sample period 2025-01-01 .. latest): versus the Nifty 200 TRI, achieve
  1. higher total return, 2. higher Sharpe, 3. lower volatility, 4. smaller max drawdown.

Every strategy here is FROZEN exactly as selected on the 2015-2024 training window
(kaufman_system.py / rs_lowvol_system.py); nothing is re-tuned on the test window.

TRI: if data/nifty200_tri.csv (official NSE Indices TRI, columns Date,tri) exists it is used.
Otherwise the TRI is approximated as the Nifty 200 price index with dividends reinvested daily
at DIV_YIELD per year. 1.5% is at or above the Nifty 200's recent dividend yield, so the
approximation makes the benchmark harder to beat; the breakeven yield is also reported.

Robustness checks for the pair strategy on the test window
- point-in-time universe (pit_universe.py): pairs formed only from stocks that were actually
  in the Nifty 200 on each formation date, including members that were later dropped
- slippage stress (10 and 20 bps per side) and the cash-delivery + borrow cost model
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from kaufman_factors import CompositeRegime, classify_composite, composite_scores, percentile_normalize
from kaufman_system import OUT as KAUFMAN_DIR, System, equity_from_returns, pair_config
from perf import monthly_returns_table, perf_metrics, yearly_returns
from pit_universe import load_pit_universe, membership
from regime import REGIMES
from research import OUT as RESEARCH_DIR
from rs_pairs import backtest, compute_scores, form_pairs
from universe import DATA_DIR

OUT = RESEARCH_DIR / "final_oos"
OUT.mkdir(exist_ok=True)
TEST = ("2025-01-01", None)
PRE_TEST = ("2013-07-01", "2014-12-31")
TRAIN = ("2015-01-01", "2024-12-31")
DIV_YIELD = 0.015


def nifty200_tri(bench: pd.Series) -> tuple[pd.Series, str]:
    f = DATA_DIR / "nifty200_tri.csv"
    if f.exists():
        tri = pd.read_csv(f, index_col=0, parse_dates=True)["tri"].reindex(bench.index).ffill()
        return tri, "official NSE Indices TRI"
    accrual = np.exp(np.log1p(DIV_YIELD) / 252 * np.arange(len(bench)))
    return bench * accrual, f"approximation: price index + {DIV_YIELD:.1%}/yr dividends reinvested"


def scorecard(r: pd.Series, tri: pd.Series, a: str, b: str | None) -> dict:
    m = perf_metrics(equity_from_returns(r.loc[a:b]), tri)
    t = tri.loc[a:b]
    bm = perf_metrics(t / t.iloc[0], tri)
    return {**{k: m[k] for k in ("total_return", "CAGR", "sharpe", "sortino", "vol", "max_drawdown", "beta")},
            "tri_total_return": bm["total_return"], "tri_sharpe": bm["sharpe"], "tri_vol": bm["vol"], "tri_maxdd": bm["max_drawdown"],
            "1_higher_return": m["total_return"] > bm["total_return"], "2_higher_sharpe": m["sharpe"] > bm["sharpe"],
            "3_lower_vol": m["vol"] < bm["vol"], "4_lower_maxdd": m["max_drawdown"] > bm["max_drawdown"]}


def frozen_pair_setup(s: System):
    kcfg = json.loads((KAUFMAN_DIR / "final_config.json").read_text())
    rp = CompositeRegime(**kcfg["regime"])
    regime = classify_composite(composite_scores(percentile_normalize(s.raw, rp.pct_window), kcfg["factors"]), rp)
    pp = {"signal": kcfg["pair_params"]["signal"], "lookback": int(kcfg["pair_params"]["lookback"]),
          "pct_window": int(kcfg["pair_params"]["pct_window"]), "entry_pct": float(kcfg["pair_params"]["entry_pct"]),
          "exit_on_median": str(kcfg["pair_params"]["exit_on_median"]) == "True",
          "formation_days": int(kcfg["pair_params"]["formation_days"])}
    on = {"SIDEWAYS", "HIGH_VOL"} if "SIDEWAYS" in kcfg["pair_mode"] else {"BULLISH", "BEARISH", "SIDEWAYS", "HIGH_VOL"}
    return pair_config(pp, on), regime, pp, kcfg


def pit_schedule(prices: pd.DataFrame, meta: pd.DataFrame, formation_days: int, starts: list[pd.Timestamp],
                 end: pd.Timestamp) -> pd.DataFrame:
    """Cointegration formation restricted to the point-in-time members on each formation date."""
    out = []
    for i, s in enumerate(starts):
        members = membership(s - pd.Timedelta(days=1))
        m = meta[meta["symbol"].isin(members)]
        pairs = form_pairs(prices, m, s, formation_days=formation_days)
        if pairs.empty:
            continue
        pairs.insert(0, "period_end", starts[i + 1] if i + 1 < len(starts) else end)
        pairs.insert(0, "period_start", s)
        out.append(pairs)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    s = System()
    tri, tri_note = nifty200_tri(s.bench)
    print("Benchmark:", tri_note)

    # ---- frozen candidates (returns as produced by their own scripts)
    keq = pd.read_csv(KAUFMAN_DIR / "equity.csv", index_col=0, parse_dates=True)
    req = pd.read_csv(RESEARCH_DIR / "rs_lowvol" / "equity_all_windows.csv", index_col=0, parse_dates=True)
    pair_cfg, regime, pp, kcfg = frozen_pair_setup(s)
    pair_all = s.pair_returns(replace(pair_cfg, start=PRE_TEST[0]), regime)["equity"]["equity"].pct_change().fillna(0.0)
    cands = {
        "Regime-aware cointegrated pair strategy (percentile RS)": pair_all,
        "Kaufman regime system (index timing + pairs)": keq["Final system"].pct_change(),
        "RS + low-vol stock system (hedge + pairs)": req["Final system"].pct_change(),
        "Regime index timing only": keq["Index sleeve only"].pct_change(),
    }
    rows = []
    for name, r in cands.items():
        for period, (a, b) in {"OUT-OF-SAMPLE 2025..latest": TEST, "pre-test 2013-07..2014-12": PRE_TEST, "train 2015-2024": TRAIN}.items():
            rr = r.loc[a:b].dropna()
            if rr.empty or (period.startswith("pre") and rr.index[0] > pd.Timestamp(a) + pd.Timedelta(days=10)):
                continue
            rows.append({"strategy": name, "period": period, **scorecard(r.fillna(0.0), tri, a, b)})
    card = pd.DataFrame(rows)
    card["criteria_met"] = card[["1_higher_return", "2_higher_sharpe", "3_lower_vol", "4_lower_maxdd"]].sum(axis=1)
    card.to_csv(OUT / "scorecard_vs_tri.csv", index=False)

    # ---- robustness of the pair strategy on the out-of-sample window
    rob = []
    test_r = pair_all.loc[TEST[0]:]
    tri_t = tri.loc[TEST[0]:]
    rob.append({"check": "frozen strategy (as selected)", **scorecard(pair_all, tri, *TEST)})
    for bps in (10.0, 20.0):
        r = s.pair_returns(replace(pair_cfg, start=PRE_TEST[0], slippage_bps=bps), regime)["equity"]["equity"].pct_change().fillna(0.0)
        rob.append({"check": f"slippage {bps:.0f} bps/side", **scorecard(r, tri, *TEST)})
    r = s.pair_returns(replace(pair_cfg, start=PRE_TEST[0], cost_model="cash"), regime)["equity"]["equity"].pct_change().fillna(0.0)
    rob.append({"check": "cash delivery + 8% SLB borrow costs", **scorecard(r, tri, *TEST)})

    prices_pit, _, meta_pit = load_pit_universe(TEST[0])
    starts = [pd.Timestamp(x) for x in ("2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01")]
    sched = pit_schedule(prices_pit, meta_pit, pp["formation_days"], starts, prices_pit.index[-1] + pd.Timedelta(days=1))
    sig = compute_scores(prices_pit, sched, pair_cfg.lookback, pair_cfg.vol_window, pair_cfg.signal, pair_cfg.pct_window)
    pit_res = backtest(replace(pair_cfg, start=TEST[0]), prices_pit, regime, sched, sig)
    pit_r = pit_res["equity"]["equity"].pct_change().fillna(0.0)
    rob.append({"check": "point-in-time Nifty 200 universe (fresh start 2025-01)", **scorecard(pit_r, tri, *TEST)})
    cur_fresh = s.pair_returns(replace(pair_cfg, start=TEST[0]), regime)["equity"]["equity"].pct_change().fillna(0.0)
    rob.append({"check": "today's universe (fresh start 2025-01, for comparison)", **scorecard(cur_fresh, tri, *TEST)})
    rob_df = pd.DataFrame(rob)
    rob_df["criteria_met"] = rob_df[["1_higher_return", "2_higher_sharpe", "3_lower_vol", "4_lower_maxdd"]].sum(axis=1)
    rob_df.to_csv(OUT / "pair_strategy_robustness.csv", index=False)
    sched.groupby("period_start").size().to_csv(OUT / "pit_pairs_per_period.csv")
    pit_res["trades"].to_csv(OUT / "pit_trades.csv", index=False)

    # breakeven dividend yield: TRI yield at which the TRI total return would equal the strategy's
    years = len(tri_t) / 252
    strat_tot = (1 + test_r).prod() - 1
    price_tot = s.bench.loc[TEST[0]:].iloc[-1] / s.bench.loc[TEST[0]:].iloc[0] - 1
    breakeven = ((1 + strat_tot) / (1 + price_tot)) ** (1 / years) - 1

    # ---- detail tables for the strategy
    eq = pd.DataFrame({"Pair strategy": equity_from_returns(test_r), "Nifty 200 TRI": tri_t / tri_t.iloc[0],
                       "Pair strategy (PIT universe)": equity_from_returns(pit_r.loc[TEST[0]:])})
    eq.to_csv(OUT / "oos_equity.csv", index_label="date")
    monthly_returns_table(equity_from_returns(test_r)).to_csv(OUT / "oos_monthly_returns.csv")
    pd.DataFrame({"Pair strategy": yearly_returns(equity_from_returns(test_r)),
                  "Nifty 200 TRI": yearly_returns(tri.loc["2024-12-31":])}).to_csv(OUT / "oos_yearly_returns.csv")
    reg_rows = []
    df = pd.DataFrame({"strategy": test_r, "tri": tri.pct_change().loc[TEST[0]:], "regime": regime.shift(2).loc[TEST[0]:]}).dropna()
    for R in REGIMES:
        x = df[df["regime"] == R]
        row = {"regime": R, "days": len(x)}
        for col in ("strategy", "tri"):
            v = x[col]
            down = np.sqrt((np.minimum(v, 0) ** 2).mean()) if len(v) else np.nan
            row[f"{col}_ann_return"] = v.mean() * 252 if len(v) else np.nan
            row[f"{col}_sharpe"] = v.mean() / v.std() * np.sqrt(252) if len(v) > 2 and v.std() > 0 else np.nan
            row[f"{col}_sortino"] = v.mean() / down * np.sqrt(252) if down and down > 0 else np.nan
        reg_rows.append(row)
    pd.DataFrame(reg_rows).to_csv(OUT / "oos_per_regime.csv", index=False)
    (OUT / "notes.json").write_text(json.dumps({"benchmark": tri_note, "breakeven_dividend_yield": breakeven,
                                                "pair_config": pp, "pair_regimes": kcfg["pair_mode"]}, indent=2), encoding="utf-8")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        cols = ["strategy", "period", "total_return", "sharpe", "vol", "max_drawdown", "tri_total_return", "tri_sharpe",
                "tri_vol", "tri_maxdd", "criteria_met"]
        print(card[cols].round(3).to_string(index=False))
        print(rob_df[["check", "total_return", "sharpe", "vol", "max_drawdown", "tri_total_return", "criteria_met"]].round(3).to_string(index=False))
        print(f"Breakeven TRI dividend yield: {breakeven:.2%}/yr (assumed {DIV_YIELD:.1%})")
        print("PIT pairs per period:", sched.groupby("period_start").size().to_dict())


if __name__ == "__main__":
    main()
