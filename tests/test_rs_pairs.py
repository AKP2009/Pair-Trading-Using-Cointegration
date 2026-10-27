"""
Edge-case and correctness tests for the regime engine and the RS pair backtester.

Run from the project root:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perf import monthly_returns_table, perf_metrics, yearly_returns  # noqa: E402
from regime import compute_regimes  # noqa: E402
from rs_pairs import Config, backtest, compute_scores, form_pairs, side_cost_rates  # noqa: E402

N_DAYS = 400


def _dates(n: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _pair_prices(shock_day: int = 200, shock: float = 0.10, revert_days: int = 5, seed: int = 0) -> pd.DataFrame:
    """Two stocks sharing a common factor plus small noise; A jumps by `shock` on shock_day then reverts."""
    rng = np.random.default_rng(seed)
    idx = _dates()
    common = np.cumsum(rng.normal(0, 0.01, len(idx)))
    a = common + rng.normal(0, 0.002, len(idx))
    b = common + rng.normal(0, 0.002, len(idx))
    bump = np.zeros(len(idx))
    bump[shock_day:shock_day + revert_days] = shock
    return pd.DataFrame({"A.NS": 100 * np.exp(a + bump), "B.NS": 100 * np.exp(b)}, index=idx)


def _schedule(prices: pd.DataFrame, pairs=(("A.NS", "B.NS"),)) -> pd.DataFrame:
    return pd.DataFrame([{"period_start": prices.index[0], "period_end": prices.index[-1] + pd.Timedelta(days=1),
                          "stock_a": a, "stock_b": b, "industry": "X", "adf_p": 0.01, "corr": 0.9, "ratio_vol": 0.01}
                         for a, b in pairs])


def _regimes(prices: pd.DataFrame, label: str = "SIDEWAYS") -> pd.Series:
    return pd.Series(label, index=prices.index)


CFG = Config(lookback=3, hold=5, entry_k=2.0, vol_window=60, max_pairs=5, cost_model="none")


# ----------------------------------------------------------------------------- signals & timing

def test_no_lookahead_in_scores():
    p = _pair_prices()
    sched = _schedule(p)
    s1 = compute_scores(p, sched, 5, 60)[0]
    p2 = p.copy()
    p2.iloc[250:] *= 1.5  # change the future only
    s2 = compute_scores(p2, sched, 5, 60)[0]
    pd.testing.assert_frame_equal(s1.iloc[:250], s2.iloc[:250])


def test_entry_executes_next_close_and_holds_exactly_h_days():
    p = _pair_prices(shock_day=200, revert_days=20)
    res = backtest(CFG, p, _regimes(p), _schedule(p))
    t = res["trades"]
    shock = t[t["entry_date"] == p.index[201]]
    assert len(shock) == 1, "signal at the shock close must fill exactly one day later"
    assert not (t["entry_date"] == p.index[200]).any(), "no same-close fill on the signal day"
    trade = shock.iloc[0]
    assert trade["exit_reason"] == "time" and trade["days"] == CFG.hold
    assert trade["exit_date"] == p.index[201 + CFG.hold]
    assert trade["direction"] == -1, "A outperformed -> short A / long B"


def test_calendar_hold_exits_on_first_close_at_or_after_15_days():
    p = _pair_prices(shock_day=200, revert_days=30)
    t = backtest(replace(CFG, hold_calendar_days=15), p, _regimes(p), _schedule(p))["trades"]
    for _, tr in t[t["exit_reason"] == "time"].iterrows():
        held = (tr["exit_date"] - tr["entry_date"]).days
        prev_close = p.index[p.index.get_loc(tr["exit_date"]) - 1]
        assert held >= 15, "never exit before 15 calendar days"
        assert (prev_close - tr["entry_date"]).days < 15, "exit on the FIRST close past the mark"


def test_calendar_hold_rolls_over_weekend_and_holiday():
    p = _pair_prices(shock_day=200, revert_days=30)
    entry = p.index[201]
    target = entry + pd.Timedelta(days=15)
    holiday = p.index[p.index >= target][0]
    p = p.drop(holiday)  # the day the hold would end is a market holiday
    t = backtest(replace(CFG, hold_calendar_days=15), p, _regimes(p), _schedule(p))["trades"]
    tr = t[t["entry_date"] == entry].iloc[0]
    assert tr["exit_date"] == p.index[p.index > holiday][0]


def test_reversion_profits_and_momentum_loses_on_reverting_shock():
    p = _pair_prices(shock_day=200, shock=0.10, revert_days=3)
    cfg = replace(CFG, hold=5)
    rev = backtest(cfg, p, _regimes(p), _schedule(p))["trades"]
    mom = backtest(replace(cfg, direction="momentum"), p, _regimes(p), _schedule(p))["trades"]
    shock_trade_rev = rev[rev["entry_date"] == p.index[201]].iloc[0]
    shock_trade_mom = mom[mom["entry_date"] == p.index[201]].iloc[0]
    assert shock_trade_rev["gross_pnl"] > 0
    assert shock_trade_mom["gross_pnl"] < 0


def test_formation_uses_only_pre_period_data():
    p = _pair_prices()
    meta = pd.DataFrame({"ticker": ["A.NS", "B.NS"], "industry": ["X", "X"]})
    start = p.index[300]
    f1 = form_pairs(p, meta, start, formation_days=250)
    p2 = p.copy()
    p2.loc[start:, "A.NS"] *= 3.0
    f2 = form_pairs(p2, meta, start, formation_days=250)
    pd.testing.assert_frame_equal(f1, f2)
    assert len(f1) == 1


def test_formation_needs_full_window():
    p = _pair_prices()
    meta = pd.DataFrame({"ticker": ["A.NS", "B.NS"], "industry": ["X", "X"]})
    assert form_pairs(p, meta, p.index[100], formation_days=250).empty


def test_formation_skips_cross_industry_and_bad_data():
    p = _pair_prices()
    meta = pd.DataFrame({"ticker": ["A.NS", "B.NS"], "industry": ["X", "Y"]})
    assert form_pairs(p, meta, p.index[300], formation_days=250).empty
    meta_same = pd.DataFrame({"ticker": ["A.NS", "B.NS"], "industry": ["X", "X"]})
    bad = p.copy()
    bad.iloc[150, 0] *= 3  # an unadjusted split-like print inside the window
    assert form_pairs(bad, meta_same, p.index[300], formation_days=250).empty


# ----------------------------------------------------------------------------- costs

def test_costs_reduce_pnl_and_net_equals_gross_minus_costs():
    p = _pair_prices(revert_days=20)
    for model in ("futures", "cash"):
        t = backtest(replace(CFG, cost_model=model), p, _regimes(p), _schedule(p))["trades"]
        assert (t["costs"] > 0).all()
        np.testing.assert_allclose(t["net_pnl"], t["gross_pnl"] - t["costs"])
    t0 = backtest(CFG, p, _regimes(p), _schedule(p))["trades"]
    assert (t0["costs"] == 0).all()


def test_cash_model_is_costlier_than_futures():
    fb, fs, fborrow = side_cost_rates("futures", 5)
    cb, cs, cborrow = side_cost_rates("cash", 5)
    assert cb > fb and cs > fs and cborrow > 0 and fborrow == 0


def test_equity_matches_trade_pnl_sum():
    p = _pair_prices(revert_days=20)
    res = backtest(replace(CFG, cost_model="futures"), p, _regimes(p), _schedule(p), initial_capital=1_000_000)
    eq, t = res["equity"], res["trades"]
    assert eq["equity"].iloc[-1] - 1_000_000 == pytest.approx(t["net_pnl"].sum(), rel=1e-9, abs=1e-6)


# ----------------------------------------------------------------------------- exits

def test_exit_on_revert_closes_early():
    p = _pair_prices(shock_day=200, revert_days=2)
    t = backtest(replace(CFG, hold=30, exit_on_revert=True), p, _regimes(p), _schedule(p))["trades"]
    shock = t[t["entry_date"] == p.index[201]].iloc[0]
    assert shock["exit_reason"] == "revert" and shock["days"] < 30


def test_stop_loss_triggers_on_breakdown():
    p = _pair_prices(shock_day=200, revert_days=1)
    p.loc[p.index[202]:, "A.NS"] *= np.exp(np.linspace(0.02, 0.6, len(p) - 202))  # divergence keeps widening
    t = backtest(replace(CFG, hold=60, stop_k=1.0, lookback=1, entry_k=1.5), p, _regimes(p), _schedule(p))["trades"]
    assert (t["exit_reason"] == "stop").any()


def test_open_positions_closed_at_end_of_test():
    p = _pair_prices(shock_day=N_DAYS - 4, revert_days=10)
    t = backtest(replace(CFG, hold=30), p, _regimes(p), _schedule(p))["trades"]
    assert t.iloc[-1]["exit_reason"] == "end_of_test"
    assert t.iloc[-1]["exit_date"] == p.index[-1]


def test_missing_price_forces_exit():
    p = _pair_prices(shock_day=200, revert_days=20)
    p.iloc[203, 1] = np.nan
    t = backtest(replace(CFG, hold=10), p, _regimes(p), _schedule(p))["trades"]
    assert "missing_price" in set(t["exit_reason"])


# ----------------------------------------------------------------------------- portfolio limits & gates

def _many_pairs_prices(n_stocks: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = _dates()
    common = np.cumsum(rng.normal(0, 0.01, len(idx)))
    data = {f"S{i}.NS": 100 * np.exp(common + rng.normal(0, 0.004, len(idx))) for i in range(n_stocks)}
    return pd.DataFrame(data, index=idx)


def test_max_pairs_and_max_per_stock_respected():
    p = _many_pairs_prices()
    cols = list(p.columns)
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    cfg = replace(CFG, lookback=2, entry_k=1.0, hold=10, max_pairs=3, max_per_stock=1)
    res = backtest(cfg, p, _regimes(p), _schedule(p, pairs))
    assert res["equity"]["n_open"].max() <= 3
    t = res["trades"]
    for d in p.index:
        live = t[(t["entry_date"] <= d) & (t["exit_date"] > d)]
        legs = pd.concat([live["stock_a"], live["stock_b"]])
        assert legs.value_counts().max() <= 1 if len(legs) else True


def test_regime_gate_blocks_entries():
    p = _many_pairs_prices()
    cols = list(p.columns)
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    reg = pd.Series(np.where(np.arange(len(p)) % 2 == 0, "LOW_VOL", "HIGH_VOL"), index=p.index)
    cfg = replace(CFG, lookback=2, entry_k=1.0, pairs_by_regime={"HIGH_VOL": 5, "LOW_VOL": 0})
    t = backtest(cfg, p, reg, _schedule(p, pairs))["trades"]
    assert len(t) > 0 and set(t["entry_regime"]) == {"HIGH_VOL"}


def test_unknown_regime_means_no_trading():
    p = _pair_prices(revert_days=20)
    reg = pd.Series([None] * len(p), index=p.index, dtype=object)
    res = backtest(replace(CFG, pairs_by_regime={"SIDEWAYS": 5}), p, reg, _schedule(p))
    assert res["trades"].empty and list(res["trades"].columns)  # empty but typed


def test_vol_filter_skips_volatile_divergence():
    p = _pair_prices(shock_day=200, shock=0.10, revert_days=20)
    loose = backtest(CFG, p, _regimes(p), _schedule(p))["trades"]
    strict = backtest(replace(CFG, max_vol_ratio=1.0), p, _regimes(p), _schedule(p))["trades"]
    assert (loose["entry_date"] == p.index[201]).any()
    assert not (strict["entry_date"] == p.index[201]).any()


def test_constant_prices_no_signal_no_crash():
    idx = _dates()
    p = pd.DataFrame({"A.NS": 100.0, "B.NS": 50.0}, index=idx)
    res = backtest(CFG, p, _regimes(p), _schedule(p))
    assert res["trades"].empty
    assert (res["equity"]["equity"] == res["equity"]["equity"].iloc[0]).all()


def test_inactive_pair_never_trades():
    p = _pair_prices(revert_days=20)
    sched = _schedule(p)
    sched["period_start"] = p.index[300]  # pair only becomes active after the shock
    t = backtest(CFG, p, _regimes(p), sched)["trades"]
    assert t.empty or (t["entry_date"] > p.index[300]).all()


# ----------------------------------------------------------------------------- regimes

def _market(n: int = 600, vol: float = 0.01, drift: float = 0.0, seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    bench = 1000 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    stocks = pd.DataFrame({f"S{i}": bench * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for i in range(20)}, index=idx)
    market = pd.DataFrame({"benchmark": bench, "vix": 15 + rng.normal(0, 1, n)}, index=idx)
    return stocks, market


def test_regime_no_lookahead():
    stocks, market = _market()
    r1 = compute_regimes(stocks, market)
    m2, s2 = market.copy(), stocks.copy()
    m2.iloc[400:] *= 0.5
    s2.iloc[400:] *= 2.0
    r2 = compute_regimes(s2, m2)
    pd.testing.assert_series_equal(r1["regime"].iloc[:400], r2["regime"].iloc[:400])


def test_regime_labels_and_warmup():
    stocks, market = _market()
    r = compute_regimes(stocks, market)["regime"]
    assert r.iloc[:150].isna().all(), "needs 200-EMA / percentile warm-up before labelling"
    assert set(r.dropna()) <= {"BULLISH", "BEARISH", "SIDEWAYS", "LOW_VOL", "HIGH_VOL"}


def test_vix_spike_is_high_vol_and_strong_uptrend_is_bullish():
    stocks, market = _market(drift=0.0)
    market.iloc[-5:, market.columns.get_loc("vix")] = 60.0
    assert compute_regimes(stocks, market)["regime"].iloc[-1] == "HIGH_VOL"

    up_stocks, up_market = _market(drift=0.004, vol=0.004, seed=5)
    up_stocks = up_stocks * np.exp(np.linspace(0, 1.5, len(up_stocks)))[:, None]
    lab = compute_regimes(up_stocks, up_market)["regime"].dropna()
    assert (lab == "BULLISH").mean() > 0.5


# ----------------------------------------------------------------------------- metrics

def test_perf_metrics_known_values():
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(100 * 1.001 ** np.arange(253), index=idx)
    m = perf_metrics(eq, bench=eq)
    assert m["total_return"] == pytest.approx(1.001 ** 252 - 1)
    assert m["max_drawdown"] == 0.0
    assert m["beta"] == pytest.approx(1.0)
    dd = pd.Series([100, 120, 60, 90], index=idx[:4], dtype=float)
    assert perf_metrics(dd)["max_drawdown"] == pytest.approx(-0.5)


def test_yearly_and_monthly_tables_compound_consistently():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    rng = np.random.default_rng(0)
    eq = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))), index=idx)
    y = yearly_returns(eq)
    m = monthly_returns_table(eq)
    months = m.drop(columns="Year")
    np.testing.assert_allclose((1 + months).prod(axis=1) - 1, y.values, rtol=1e-9)
