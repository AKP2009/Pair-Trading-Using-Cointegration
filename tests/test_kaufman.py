"""Tests for the Kaufman-style factors, percentile signals and the regime-timed index sleeve."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaufman_factors import (  # noqa: E402
    CompositeRegime, classify_composite, confirm_regime, efficiency_ratio, percentile_normalize, select_factors,
)
from kaufman_system import INDEX_COST, index_sleeve  # noqa: E402
from rs_pairs import compute_scores, stochastic  # noqa: E402


def _idx(n=300):
    return pd.bdate_range("2020-01-01", periods=n)


def test_efficiency_ratio_extremes():
    idx = _idx(50)
    straight = pd.Series(np.arange(50, dtype=float) + 100, index=idx)
    zigzag = pd.Series(100 + np.tile([0.0, 1.0], 25), index=idx)
    assert efficiency_ratio(straight, 10).dropna().eq(1.0).all()
    assert efficiency_ratio(zigzag, 10).dropna().max() <= 0.11


def test_percentile_normalization_bounded_and_no_lookahead():
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({"x": rng.normal(size=300)}, index=_idx())
    p1 = percentile_normalize(raw, 100)
    assert p1.dropna().stack().between(0, 1).all()
    raw2 = raw.copy()
    raw2.iloc[200:] += 100
    pd.testing.assert_frame_equal(p1.iloc[:200], percentile_normalize(raw2, 100).iloc[:200])
    assert p1["x"].iloc[150] == pytest.approx((raw["x"].iloc[51:151] <= raw["x"].iloc[150]).mean(), abs=0.01)


def test_stochastic_and_stress_are_bounded_percentiles():
    idx = _idx()
    rng = np.random.default_rng(1)
    common = np.cumsum(rng.normal(0, 0.01, len(idx)))
    prices = pd.DataFrame({"A.NS": 100 * np.exp(common + rng.normal(0, 0.003, len(idx))),
                           "B.NS": 100 * np.exp(common + rng.normal(0, 0.003, len(idx)))}, index=idx)
    st = stochastic(prices, 20).dropna()
    assert st.stack().between(0, 1).all()
    sched = pd.DataFrame([{"period_start": idx[0], "period_end": idx[-1] + pd.Timedelta(days=1),
                           "stock_a": "A.NS", "stock_b": "B.NS"}])
    for sig in ("rs_pct", "stress"):
        score = compute_scores(prices, sched, 10, 60, sig, 100)[0].iloc[:, 0].dropna()
        assert score.between(-0.5, 0.5).all() and len(score) > 100


def test_stress_high_when_a_outruns_b():
    idx = _idx(120)
    wiggle = np.sin(np.arange(120)) * 0.1
    a = pd.Series(100.0 + wiggle, index=idx)
    a.iloc[100:] = np.linspace(101, 120, 20)  # A rallies to a 20-day high
    b = pd.Series(100.0 + wiggle, index=idx)
    b.iloc[100:] = np.linspace(99.5, 98, 20)  # B drifts to a 20-day low
    prices = pd.DataFrame({"A.NS": a, "B.NS": b}, index=idx)
    sched = pd.DataFrame([{"period_start": idx[0], "period_end": idx[-1] + pd.Timedelta(days=1), "stock_a": "A.NS", "stock_b": "B.NS"}])
    score = compute_scores(prices, sched, 20, 20, "stress", 20)[0].iloc[:, 0]
    assert score.iloc[105] > 0.4  # stress near 100 early in the divergence -> short A / long B
    # edge case: once the A-B stochastic gap has been pinned at its max for a whole window its range
    # is zero, the stress is undefined (NaN) and no new signal is generated
    assert np.isnan(score.iloc[-1])


def test_confirm_regime_filters_whipsaw():
    lab = pd.Series(["BULLISH"] * 5 + ["BEARISH"] + ["BULLISH"] * 3 + ["BEARISH"] * 4, index=_idx(13), dtype=object)
    out = confirm_regime(lab, 3)
    assert (out.iloc[:9] == "BULLISH").all(), "a 1-day flip must not switch the regime"
    assert out.iloc[11] == "BEARISH" and out.iloc[10] == "BULLISH"


def test_classify_composite_priority():
    idx = _idx(4)
    scores = pd.DataFrame({"direction": [0.9, 0.9, 0.1, 0.5], "volatility": [0.95, 0.5, 0.5, 0.1]}, index=idx)
    lab = classify_composite(scores, CompositeRegime(bull=0.6, bear=0.4, high_vol=0.8, low_vol=0.25))
    assert list(lab) == ["HIGH_VOL", "BULLISH", "BEARISH", "LOW_VOL"]


def test_select_factors_drops_correlated_and_wrong_sign():
    names = ["dist_ema21", "dist_ema50", "dist_ema200", "er_signed_20", "roc_63", "pct_above_200ema",
             "pct_above_50ema", "ad_ratio_10", "net_highs_252", "vix", "rv_20", "avgabs_14", "dispersion_20"]
    corr = pd.DataFrame(np.where(np.eye(len(names), dtype=bool), 1.0, 0.1), index=names, columns=names)
    corr.loc["dist_ema21", "er_signed_20"] = corr.loc["er_signed_20", "dist_ema21"] = 0.9
    corr.loc["rv_20", "avgabs_14"] = corr.loc["avgabs_14", "rv_20"] = 0.95
    ic = pd.DataFrame({"rank_ic": [0.05, 0.02, -0.01, 0.07, 0.01, 0.03, 0.02, 0.04, 0.06, 0.4, 0.35, 0.3, 0.1]}, index=names)
    kept = select_factors(corr, ic)
    assert "er_signed_20" in kept["direction"] and "dist_ema21" not in kept["direction"]
    assert "dist_ema200" not in kept["direction"], "negative-IC factor must be dropped"
    assert "rv_20" in kept["volatility"] and "avgabs_14" not in kept["volatility"]


def test_index_sleeve_two_day_lag_and_costs():
    idx = _idx(6)
    regime = pd.Series(["BEARISH", "BULLISH", "BULLISH", "BULLISH", "BULLISH", "BULLISH"], index=idx, dtype=object)
    ret = pd.Series(0.01, index=idx)
    r = index_sleeve(regime, {"BULLISH": 1.0, "BEARISH": 0.0}, ret)
    # decided BULLISH at day 1 close, filled day 2 close (pays cost), earns from day 3
    assert r.iloc[2] == pytest.approx(-INDEX_COST)
    assert r.iloc[3] == pytest.approx(0.01)
    assert r.iloc[:2].abs().sum() == 0
