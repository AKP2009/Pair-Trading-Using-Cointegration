"""Tests for the relative-strength + low-volatility equity sleeve, hedge overlay and benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaufman_system import INDEX_COST  # noqa: E402
from rs_lowvol_system import EquitySleeve, criteria, ew_buy_and_hold, hedge_returns  # noqa: E402

P = {"rs_lookback": 126, "vol_lookback": 63, "w_rs": 0.5, "n_hold": 3, "trend_filter": False, "weighting": "equal"}


def _prices(n_days=500, n_stocks=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    drift = np.linspace(-0.0005, 0.001, n_stocks)
    vol = np.linspace(0.005, 0.03, n_stocks)
    r = rng.normal(drift, vol, size=(n_days, n_stocks))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"S{i}.NS" for i in range(n_stocks)])


def test_targets_use_no_future_data():
    px = _prices()
    i = 300
    w1 = EquitySleeve(px).targets(i, P, np.zeros(px.shape[1]))
    px2 = px.copy()
    px2.iloc[i + 1:] *= np.linspace(0.2, 5, px.shape[1])
    w2 = EquitySleeve(px2).targets(i, P, np.zeros(px.shape[1]))
    np.testing.assert_allclose(w1, w2)
    assert w1.sum() == pytest.approx(1.0) and (w1 > 0).sum() == 3


def test_selection_prefers_strong_and_calm_stocks():
    px = _prices()
    w = EquitySleeve(px).targets(400, {**P, "w_rs": 0.5}, np.zeros(px.shape[1]))
    assert w[0] == 0 or w[-1] == 0  # the weakest-drift and most volatile names are not both selected


def test_constant_prices_only_cost_returns():
    idx = pd.bdate_range("2019-01-01", periods=400)
    rng = np.random.default_rng(1)
    base = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 5)), axis=0)), index=idx[:300],
                        columns=[f"S{i}.NS" for i in range(5)])
    flat = pd.DataFrame(np.repeat(base.iloc[[-1]].to_numpy(), 100, axis=0), index=idx[300:], columns=base.columns)
    px = pd.concat([base, flat])
    sleeve = EquitySleeve(px)
    r, turnover = sleeve.run(P, str(idx[310].date()))
    nonzero = r[r != 0]
    assert (nonzero < 0).all(), "with flat prices the only P&L is trading cost"
    for d, v in nonzero.items():
        assert d in turnover.index
        assert -v <= turnover[d] * max(sleeve.buy_cost, sleeve.sell_cost) + 1e-12


def test_buffer_reduces_turnover():
    px = _prices(seed=3)
    sleeve = EquitySleeve(px)
    _, t_buf = sleeve.run(P, "2019-08-01")
    import rs_lowvol_system as m
    old = m.BUFFER
    try:
        m.BUFFER = 1.0
        _, t_nobuf = EquitySleeve(px).run(P, "2019-08-01")
    finally:
        m.BUFFER = old
    assert t_buf.iloc[1:].sum() <= t_nobuf.iloc[1:].sum() + 1e-12


def test_ew_buy_and_hold_matches_manual():
    px = _prices(n_days=50, n_stocks=3)
    px.iloc[:10, 2] = np.nan  # listed later -> excluded from a start-date buy & hold
    ew = ew_buy_and_hold(px, str(px.index[0].date()), None)
    manual = (px.iloc[:, :2] / px.iloc[0, :2]).mean(axis=1)
    pd.testing.assert_series_equal(ew, manual, check_names=False)


def test_hedge_lag_and_cost():
    idx = pd.bdate_range("2020-01-01", periods=5)
    h = pd.Series([0, 1, 1, 1, 1], index=idx, dtype=float)
    br = pd.Series(0.02, index=idx)
    r = hedge_returns(h, br)
    assert r.iloc[2] == pytest.approx(-INDEX_COST)  # hedge filled on day 2 (decided day 1)
    assert r.iloc[3] == pytest.approx(-0.02)        # and earns (short) from day 3


def test_criteria_all_four():
    m = {"CAGR": 0.2, "sharpe": 1.5, "vol": 0.12, "max_drawdown": -0.15}
    b = [{"CAGR": 0.1, "sharpe": 1.0, "vol": 0.15, "max_drawdown": -0.3}, {"CAGR": 0.19, "sharpe": 1.4, "vol": 0.13, "max_drawdown": -0.2}]
    assert all(criteria(m, b).values())
    assert not criteria({**m, "vol": 0.14}, b)["lower_vol"]
