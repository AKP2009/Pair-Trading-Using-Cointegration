"""
Regime factor library with percentile normalization and a correlation filter.

Every raw factor is turned into a trailing percentile rank (0..1) -- no z-scores.
Factors are oriented so that a HIGH percentile means:
  trend / breadth factors  -> bullish
  volatility factors       -> volatile

Candidate factors (all trailing, known at day t's close)
Trend
  dist_ema21     benchmark / EMA21 - 1              (the original trend factor)
  dist_ema50     benchmark / EMA50 - 1
  dist_ema200    benchmark / EMA200 - 1             (long-term trend)
  er_signed_20   Kaufman Efficiency Ratio over 20 days, signed by the 20-day direction:
                 sign(P_t - P_t-20) * |P_t - P_t-20| / sum(|P_i - P_i-1|)
  roc_63         3-month rate of change
Breadth
  pct_above_200ema, pct_above_50ema   share of universe above its own EMA
  ad_ratio_10    10-day advancers / decliners
  net_highs_252  (52-week highs - 52-week lows) / stocks with data
Volatility
  vix            India VIX level
  rv_20          20-day realized vol of the benchmark
  avgabs_14      14-day mean |daily return| (close-to-close ATR proxy)
  dispersion_20  20-day mean cross-sectional std of stock returns

Factor selection (select_factors) uses the TRAINING window only: within each group,
factors are ranked by usefulness (Spearman rank-IC of the factor percentile with the
benchmark's forward 20-day return for trend/breadth, with forward 20-day realized vol
for volatility) and greedily kept unless |Spearman correlation| with an already-kept
factor exceeds MAX_FACTOR_CORR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime import REGIMES

MAX_FACTOR_CORR = 0.70
TREND_FACTORS = ["dist_ema21", "dist_ema50", "dist_ema200", "er_signed_20", "roc_63"]
BREADTH_FACTORS = ["pct_above_200ema", "pct_above_50ema", "ad_ratio_10", "net_highs_252"]
VOL_FACTORS = ["vix", "rv_20", "avgabs_14", "dispersion_20"]


def efficiency_ratio(p: pd.Series, n: int) -> pd.Series:
    """Kaufman Efficiency Ratio: |net change over n| / sum of |1-day changes| over n, in [0, 1]."""
    net = (p - p.shift(n)).abs()
    path = p.diff().abs().rolling(n).sum()
    return net / path.replace(0.0, np.nan)


def raw_factors(prices: pd.DataFrame, market: pd.DataFrame, member_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw factor values. With `member_mask`, breadth/dispersion only count stocks that were index members that day."""
    bench = market["benchmark"]
    live = member_mask.reindex(index=prices.index, columns=prices.columns).fillna(False).astype(bool)         if member_mask is not None else pd.DataFrame(True, index=prices.index, columns=prices.columns)
    f = pd.DataFrame(index=market.index)
    for n in (21, 50, 200):
        f[f"dist_ema{n}"] = bench / bench.ewm(span=n, adjust=False).mean() - 1
    f["er_signed_20"] = np.sign(bench - bench.shift(20)) * efficiency_ratio(bench, 20)
    f["roc_63"] = bench.pct_change(63)

    for n in (200, 50):
        ema = prices.ewm(span=n, adjust=False, min_periods=n).mean()
        above = (prices > ema).where(ema.notna() & live)
        f[f"pct_above_{n}ema"] = above.sum(axis=1) / above.notna().sum(axis=1).replace(0, np.nan)
    daily = prices.pct_change(fill_method=None).where(live)
    adv, dec = (daily > 0).sum(axis=1), (daily < 0).sum(axis=1)
    f["ad_ratio_10"] = adv.rolling(10).sum() / dec.rolling(10).sum().replace(0, np.nan)
    hi = prices.rolling(252, min_periods=252).max()
    lo = prices.rolling(252, min_periods=252).min()
    n_valid = (hi.notna() & live).sum(axis=1).replace(0, np.nan)
    f["net_highs_252"] = (((prices >= hi) & live).sum(axis=1) - ((prices <= lo) & live).sum(axis=1)) / n_valid

    bret = bench.pct_change()
    f["vix"] = market["vix"]
    f["rv_20"] = bret.rolling(20).std() * np.sqrt(252)
    f["avgabs_14"] = bret.abs().rolling(14).mean()
    f["dispersion_20"] = daily.std(axis=1).rolling(20).mean()
    return f


def percentile_normalize(raw: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing percentile rank of each factor within the last `window` days (inclusive of today)."""
    return raw.rolling(window, min_periods=window // 2).rank(pct=True)


def factor_diagnostics(pct: pd.DataFrame, market: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Spearman correlation matrix, usefulness table) on [start, end] only."""
    bench = market["benchmark"]
    fwd_ret = bench.shift(-20) / bench - 1
    fwd_vol = bench.pct_change().rolling(20).std().shift(-20)
    win = pct.loc[start:end]
    corr = win.corr(method="spearman")
    rows = []
    for col in pct.columns:
        group = "trend" if col in TREND_FACTORS else "breadth" if col in BREADTH_FACTORS else "volatility"
        target = fwd_vol if group == "volatility" else fwd_ret
        d = pd.concat([win[col], target.loc[start:end]], axis=1).dropna()
        ic = d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman") if len(d) > 50 else np.nan
        rows.append({"factor": col, "group": group, "rank_ic": ic,
                     "target": "fwd 20d realized vol" if group == "volatility" else "fwd 20d return"})
    return corr, pd.DataFrame(rows).set_index("factor")


def select_factors(corr: pd.DataFrame, usefulness: pd.DataFrame, max_corr: float = MAX_FACTOR_CORR) -> dict[str, list[str]]:
    """
    Greedy de-correlation. Directional factors (trend + breadth) are one group because they are
    averaged into one directional score; volatility is the other. A factor is kept only if its IC
    has the expected sign (> 0) and it is not too correlated with any factor already kept.
    """
    groups = {"direction": TREND_FACTORS + BREADTH_FACTORS, "volatility": VOL_FACTORS}
    kept: dict[str, list[str]] = {}
    for g, members in groups.items():
        ranked = usefulness.loc[members].sort_values("rank_ic", ascending=False)
        chosen: list[str] = []
        for f in ranked.index:
            if not ranked.at[f, "rank_ic"] > 0:
                continue
            if all(abs(corr.at[f, k]) <= max_corr for k in chosen):
                chosen.append(f)
        if not chosen:  # fall back to the single most useful factor
            chosen = [ranked.index[0]]
        kept[g] = chosen
    return kept


@dataclass(frozen=True)
class CompositeRegime:
    pct_window: int = 252
    bull: float = 0.60
    bear: float = 0.40
    high_vol: float = 0.80
    low_vol: float = 0.25
    confirm: int = 1  # a new regime must persist this many days before it takes effect


def composite_scores(pct: pd.DataFrame, factors: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame({"direction": pct[factors["direction"]].mean(axis=1, skipna=False),
                         "volatility": pct[factors["volatility"]].mean(axis=1, skipna=False)})


def classify_composite(scores: pd.DataFrame, p: CompositeRegime) -> pd.Series:
    d, v = scores["direction"], scores["volatility"]
    lab = np.select([v >= p.high_vol, d >= p.bull, d <= p.bear, v <= p.low_vol],
                    ["HIGH_VOL", "BULLISH", "BEARISH", "LOW_VOL"], default="SIDEWAYS")
    lab = pd.Series(np.where(d.notna() & v.notna(), lab, None), index=scores.index, dtype=object)
    return confirm_regime(lab, p.confirm) if p.confirm > 1 else lab.rename("regime")


def confirm_regime(lab: pd.Series, k: int) -> pd.Series:
    """Switch to a new regime only after it has been observed on k consecutive days (whipsaw filter)."""
    vals = lab.to_numpy(dtype=object)
    out = np.empty(len(vals), dtype=object)
    current, run_val, run_len = None, None, 0
    for i, v in enumerate(vals):
        if v == run_val:
            run_len += 1
        else:
            run_val, run_len = v, 1
        if current is None or (v is not None and run_len >= k):
            current = v
        out[i] = current
    return pd.Series(out, index=lab.index, name="regime")


assert set(REGIMES) == {"BULLISH", "BEARISH", "SIDEWAYS", "LOW_VOL", "HIGH_VOL"}
