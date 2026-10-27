"""
Market regime classification from trend, breadth and volatility.

Inputs are the Nifty 200 universe closes and the benchmark/India VIX
series from universe.py. Every indicator only uses data up to and
including day t, so the regime label for day t is known at t's close and
can gate trades executed from t+1 onward.

Indicators
Window lengths below are the defaults; RegimeParams makes every lookback tunable
(the column names keep their default-window names, e.g. dist_ema21, for compatibility).

- Trend:      distance of the benchmark from its 21-day EMA (bench / EMA21 - 1).
- Breadth:    advance/decline ratio of the universe (10-day sums of advancers
              over decliners) and % of universe stocks above their 200-day EMA.
- Volatility: India VIX and the benchmark's 20-day realized std dev
              (annualized), each turned into a trailing 1-year percentile
              rank so "high"/"low" adapts to the prevailing level.

States (first match wins)
1. HIGH_VOL  -- volatility percentile >= HIGH_VOL_PCT
2. BULLISH   -- trend+breadth vote >= +2
3. BEARISH   -- trend+breadth vote <= -2
4. LOW_VOL   -- volatility percentile <= LOW_VOL_PCT
5. SIDEWAYS  -- everything else

The vote adds +1/-1 for each of: dist from 21 EMA beyond +/-TREND_BAND,
% above 200 EMA beyond BREADTH_HI/BREADTH_LO, A/D ratio beyond ADR_HI/ADR_LO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIMES = ["BULLISH", "BEARISH", "SIDEWAYS", "LOW_VOL", "HIGH_VOL"]

EMA_TREND = 21
EMA_BREADTH = 200
ADR_WINDOW = 10
RV_WINDOW = 20
PCT_WINDOW = 252

TREND_BAND = 0.01
BREADTH_HI, BREADTH_LO = 0.60, 0.40
ADR_HI, ADR_LO = 1.2, 0.8
HIGH_VOL_PCT, LOW_VOL_PCT = 0.80, 0.25


@dataclass(frozen=True)
class RegimeParams:
    """Lookback windows of the regime indicators (defaults = the original fixed choices)."""

    ema_trend: int = EMA_TREND
    adr_window: int = ADR_WINDOW
    ema_breadth: int = EMA_BREADTH
    rv_window: int = RV_WINDOW
    pct_window: int = PCT_WINDOW


def _trailing_pct_rank(s: pd.Series, window: int = PCT_WINDOW) -> pd.Series:
    """Percentile of today's value within the trailing `window` days (inclusive), no lookahead."""
    return s.rolling(window, min_periods=window // 2).apply(lambda x: (x <= x[-1]).mean(), raw=True)


def trend_indicator(market: pd.DataFrame, ema_trend: int) -> pd.Series:
    bench = market["benchmark"]
    return bench / bench.ewm(span=ema_trend, adjust=False).mean() - 1


def adr_indicator(prices: pd.DataFrame, adr_window: int) -> pd.Series:
    daily = prices.pct_change(fill_method=None)
    adv = (daily > 0).sum(axis=1)
    dec = (daily < 0).sum(axis=1)
    return adv.rolling(adr_window).sum() / dec.rolling(adr_window).sum().replace(0, np.nan)


def breadth_indicator(prices: pd.DataFrame, ema_breadth: int) -> pd.Series:
    ema = prices.ewm(span=ema_breadth, adjust=False, min_periods=ema_breadth).mean()
    above = (prices > ema).where(ema.notna())
    return above.sum(axis=1) / above.notna().sum(axis=1).replace(0, np.nan)


def vol_indicator(market: pd.DataFrame, rv_window: int, pct_window: int) -> pd.DataFrame:
    rv = market["benchmark"].pct_change().rolling(rv_window).std() * np.sqrt(252)
    vix_pct = _trailing_pct_rank(market["vix"], pct_window)
    rv_pct = _trailing_pct_rank(rv, pct_window)
    return pd.DataFrame({"realized_vol": rv, "vix_pct": vix_pct, "rv_pct": rv_pct,
                         "vol_pct": pd.concat([vix_pct, rv_pct], axis=1).mean(axis=1)})


def compute_indicators(prices: pd.DataFrame, market: pd.DataFrame, params: RegimeParams = RegimeParams()) -> pd.DataFrame:
    out = pd.DataFrame(index=market.index)
    out["dist_ema21"] = trend_indicator(market, params.ema_trend)
    daily = prices.pct_change(fill_method=None)
    out["adv"] = (daily > 0).sum(axis=1)
    out["dec"] = (daily < 0).sum(axis=1)
    out["ad_ratio"] = adr_indicator(prices, params.adr_window)
    out["pct_above_200ema"] = breadth_indicator(prices, params.ema_breadth)
    vol = vol_indicator(market, params.rv_window, params.pct_window)
    out["realized_vol"] = vol["realized_vol"]
    out["vix"] = market["vix"]
    out[["vix_pct", "rv_pct", "vol_pct"]] = vol[["vix_pct", "rv_pct", "vol_pct"]]
    return out


def classify(ind: pd.DataFrame) -> pd.Series:
    trend_vote = np.sign(ind["dist_ema21"].where(ind["dist_ema21"].abs() > TREND_BAND, 0.0))
    breadth_vote = np.where(ind["pct_above_200ema"] > BREADTH_HI, 1, np.where(ind["pct_above_200ema"] < BREADTH_LO, -1, 0))
    adr_vote = np.where(ind["ad_ratio"] > ADR_HI, 1, np.where(ind["ad_ratio"] < ADR_LO, -1, 0))
    vote = trend_vote + breadth_vote + adr_vote

    regime = np.select(
        [ind["vol_pct"] >= HIGH_VOL_PCT, vote >= 2, vote <= -2, ind["vol_pct"] <= LOW_VOL_PCT],
        ["HIGH_VOL", "BULLISH", "BEARISH", "LOW_VOL"],
        default="SIDEWAYS",
    )
    valid = ind[["dist_ema21", "ad_ratio", "pct_above_200ema", "vol_pct"]].notna().all(axis=1)
    return pd.Series(np.where(valid, regime, None), index=ind.index, name="regime")


def compute_regimes(prices: pd.DataFrame, market: pd.DataFrame, params: RegimeParams = RegimeParams()) -> pd.DataFrame:
    ind = compute_indicators(prices, market, params)
    ind["regime"] = classify(ind)
    return ind


if __name__ == "__main__":
    from universe import DATA_DIR, load_universe

    prices, market, _ = load_universe()
    reg = compute_regimes(prices, market)
    reg.to_csv(DATA_DIR / "regimes.csv", index_label="Date")
    print(reg["regime"].value_counts(normalize=True).round(3))
    print(reg.tail())
