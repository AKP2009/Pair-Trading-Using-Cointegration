"""
Regime-aware relative-strength pair trading on the Nifty 200.

No z-score of a spread level and no regression hedge ratio is used:

- Pairing: every FORMATION_MONTHS, same-industry pairs are tested for
  cointegration with a *fixed 1:1 hedge* -- an ADF unit-root test on the
  log price ratio ln(A/B) over the trailing FORMATION_DAYS. Because the
  hedge is fixed, no regression beta is estimated; a stationary log ratio
  means the two stocks' rupee values are cointegrated. Pairs must also
  have daily-return correlation >= MIN_CORR.
- Signal: relative strength RS_L = ln(A/B)_t - ln(A/B)_{t-L} (A's
  L-day return minus B's), scaled by the ratio's own volatility over the
  same horizon (daily std of d ln(A/B) over VOL_WINDOW days * sqrt(L)).
  When |RS/vol| >= entry_k the strategy shorts the outperformer and buys
  the laggard (mean reversion), with equal rupee notional on both legs.
- Exit: after a fixed holding period (`hold` trading days, or `hold_calendar_days`), optionally earlier when the
  relative strength has reverted (`exit_on_revert`) or the trade loses
  more than `stop_k` ratio-vols (`stop_k`).
- Volatility filter: a divergence that arrives with a volatility spike
  (10-day ratio vol / VOL_WINDOW-day ratio vol > `max_vol_ratio`) is
  usually news-driven and is skipped; quiet divergences tend to revert.
- Portfolio: at most `pairs_by_regime[regime]` concurrent pairs (ranked by
  |RS/vol|), each stock in at most `max_per_stock` open pairs, each leg
  sized at equity / max_pairs (100% long / 100% short at full utilization).

Timing: indicators and regime use closes up to day t. Decisions made at
t's close execute at the next day's close (1-day lag). Time
exits are scheduled in advance and execute on the close the holding
period ends.

Costs: per-side regulatory charges from costs.py's rate constants (cash
delivery + SLB borrow), or NSE stock-futures charges, plus a fixed
per-side slippage in bps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from costs import (
    DEFAULT_ANNUAL_BORROW_RATE,
    EXCHANGE_TXN_CHARGE_RATE,
    GST_RATE,
    SEBI_TURNOVER_FEE_RATE,
    STAMP_DUTY_RATE,
    STT_RATE,
)

FORMATION_DAYS = 504
FORMATION_MONTHS = 6
MIN_CORR = 0.5
MAX_ADF_P = 0.05
MAX_DAILY_JUMP = 0.40  # |log return| above this inside a formation window -> treat stock as bad data
FIRST_TRADE_DATE = "2014-01-01"

# NSE stock futures (approximate current rate card): STT on sell side only,
# lower exchange charge and stamp duty, no borrow (the short is a sold future).
FUT_STT_SELL_RATE = 0.0002
FUT_EXCHANGE_RATE = 0.0000173
FUT_STAMP_RATE = 0.00002
FUT_ROLL_DAYS = 21
SHORT_VOL_WINDOW = 10


@dataclass
class Config:
    lookback: int = 10
    hold: int = 10
    hold_calendar_days: int | None = None  # if set, overrides `hold`: exit on the first close >= this many calendar days after entry
    entry_k: float = 2.0
    vol_window: int = 60
    max_pairs: int = 20
    pairs_by_regime: dict[str, int] | None = None
    max_per_stock: int = 1
    exit_on_revert: bool = False
    stop_k: float | None = None
    signal: str = "rs_vol"  # "rs_vol" | "rs_pct" | "stress" (see compute_scores)
    pct_window: int = 252  # percentile window for signal="rs_pct"
    max_vol_ratio: float | None = None  # skip entries when short-term ratio vol / long-term vol exceeds this
    direction: str = "reversion"  # "reversion" trades against RS, "momentum" with it
    cost_model: str = "futures"  # "futures" | "cash" | "none"
    slippage_bps: float = 5.0
    start: str | None = None
    end: str | None = None
    label: str = ""
    extra: dict = field(default_factory=dict)

    def hold_trading_days(self) -> float:
        """Holding period in (approximate) trading days, for vol scaling and roll timing."""
        return self.hold_calendar_days * 252 / 365 if self.hold_calendar_days else self.hold

    def slots_for(self, regime: str | None) -> int:
        if self.pairs_by_regime is None:
            return self.max_pairs
        if regime is None:
            return 0
        return int(self.pairs_by_regime.get(regime, 0))


def side_cost_rates(cost_model: str, slippage_bps: float) -> tuple[float, float, float]:
    """(buy_rate, sell_rate, annual_borrow_rate) as fractions of traded notional."""
    slip = slippage_bps / 10_000
    if cost_model == "none":
        return 0.0, 0.0, 0.0
    if cost_model == "cash":
        exch = EXCHANGE_TXN_CHARGE_RATE * (1 + GST_RATE)
        buy = STT_RATE + exch + SEBI_TURNOVER_FEE_RATE + STAMP_DUTY_RATE + slip
        sell = STT_RATE + exch + SEBI_TURNOVER_FEE_RATE + slip
        return buy, sell, DEFAULT_ANNUAL_BORROW_RATE
    if cost_model == "futures":
        exch = FUT_EXCHANGE_RATE * (1 + GST_RATE)
        buy = exch + SEBI_TURNOVER_FEE_RATE + FUT_STAMP_RATE + slip
        sell = FUT_STT_SELL_RATE + exch + SEBI_TURNOVER_FEE_RATE + slip
        return buy, sell, 0.0
    raise ValueError(f"unknown cost_model {cost_model!r}")


# --------------------------------------------------------------------------- pair formation

def formation_starts(index: pd.DatetimeIndex, first: str = FIRST_TRADE_DATE, months: int = FORMATION_MONTHS) -> list[pd.Timestamp]:
    starts = pd.date_range(first, index[-1], freq=f"{months}MS")
    return [s for s in starts if (index >= s).any()]


def form_pairs(
    prices: pd.DataFrame,
    meta: pd.DataFrame,
    period_start: pd.Timestamp,
    formation_days: int = FORMATION_DAYS,
    max_adf_p: float = MAX_ADF_P,
    min_corr: float = MIN_CORR,
) -> pd.DataFrame:
    """Cointegrated same-industry pairs using only data strictly before `period_start`."""
    window = prices.loc[prices.index < period_start].iloc[-formation_days:]
    if len(window) < formation_days:
        return pd.DataFrame()
    window = window.loc[:, window.notna().all()]
    logp = np.log(window)
    rets = logp.diff().iloc[1:]
    window_ok = rets.abs().max() <= MAX_DAILY_JUMP
    logp, rets = logp.loc[:, window_ok], rets.loc[:, window_ok]
    corr = rets.corr()

    industry = meta.set_index("ticker")["industry"]
    rows = []
    for ind_name, tickers in industry[industry.index.isin(logp.columns)].groupby(industry).groups.items():
        for a, b in combinations(sorted(tickers), 2):
            c = corr.at[a, b]
            if c < min_corr:
                continue
            lr = (logp[a] - logp[b]).to_numpy()
            p = adfuller(lr, maxlag=1, autolag=None, regression="c")[1]
            if p <= max_adf_p:
                rows.append({"stock_a": a, "stock_b": b, "industry": ind_name, "adf_p": p, "corr": c,
                             "ratio_vol": float(np.std(np.diff(lr)))})
    return pd.DataFrame(rows)


def build_schedule(prices: pd.DataFrame, meta: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Pair list for every trading period: columns period_start, period_end + form_pairs columns."""
    starts = formation_starts(prices.index)
    out = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else prices.index[-1] + pd.Timedelta(days=1)
        pairs = form_pairs(prices, meta, s, **kwargs)
        if pairs.empty:
            continue
        pairs.insert(0, "period_end", end)
        pairs.insert(0, "period_start", s)
        out.append(pairs)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------- signals

def pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def stochastic(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Kaufman/Lane stochastic on closes: position of today's value inside its trailing window's range, in [0, 1]."""
    lo = x.rolling(window, min_periods=window).min()
    hi = x.rolling(window, min_periods=window).max()
    return (x - lo) / (hi - lo).replace(0.0, np.nan)


def compute_scores(
    prices: pd.DataFrame, schedule: pd.DataFrame, lookback: int, vol_window: int,
    signal: str = "rs_vol", pct_window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (score, daily_vol, active, vol_ratio), each dates x pair_key, from trailing data only.

    signal="rs_vol":  score = RS_L / (daily_vol * sqrt(L))  (volatility-scaled relative strength)
    signal="rs_pct":  score = percentile rank of RS_L within its trailing `pct_window` days, minus 0.5
    signal="stress":  Kaufman stress indicator minus 0.5 -- stochastic(A, L) - stochastic(B, L),
                      then the stochastic of that difference over L (range-percentile normalization)

    Percentile scores live in [-0.5, 0.5]: +0.45 means A's relative strength is at its 95th
    percentile. daily_vol = rolling std of d ln(A/B); active marks days on which the pair is in
    the current period's cointegrated list; vol_ratio = SHORT_VOL_WINDOW-day ratio vol / daily_vol.
    """
    uniq = schedule[["stock_a", "stock_b"]].drop_duplicates()
    keys = [pair_key(a, b) for a, b in uniq.itertuples(index=False)]
    logp = np.log(prices)
    lr = pd.DataFrame(
        logp[uniq["stock_a"]].to_numpy() - logp[uniq["stock_b"]].to_numpy(), index=prices.index, columns=keys
    )
    rs = lr - lr.shift(lookback)
    daily_vol = lr.diff().rolling(vol_window, min_periods=vol_window).std().replace(0.0, np.nan)
    if signal == "rs_vol":
        score = rs / (daily_vol * np.sqrt(lookback))
    elif signal == "rs_pct":
        score = rs.rolling(pct_window, min_periods=pct_window // 2).rank(pct=True) - 0.5
    elif signal == "stress":
        stoch_a = stochastic(prices[uniq["stock_a"]], lookback).to_numpy()
        stoch_b = stochastic(prices[uniq["stock_b"]], lookback).to_numpy()
        diff = pd.DataFrame(stoch_a - stoch_b, index=prices.index, columns=keys)
        score = stochastic(diff, lookback) - 0.5
    else:
        raise ValueError(f"unknown signal {signal!r}")
    vol_ratio = lr.diff().rolling(SHORT_VOL_WINDOW).std() / daily_vol

    active = np.zeros(lr.shape, dtype=bool)
    col_pos = {k: j for j, k in enumerate(keys)}
    for (s, e), grp in schedule.groupby(["period_start", "period_end"]):
        rows = (prices.index >= s) & (prices.index < e)
        cols = [col_pos[pair_key(a, b)] for a, b in zip(grp["stock_a"], grp["stock_b"])]
        active[np.ix_(rows, cols)] = True
    return score, daily_vol, pd.DataFrame(active, index=prices.index, columns=keys), vol_ratio


# --------------------------------------------------------------------------- backtest

TRADE_COLUMNS = ["pair", "stock_a", "stock_b", "direction", "entry_date", "exit_date", "days", "entry_regime",
                 "entry_score", "leg_notional", "gross_pnl", "costs", "net_pnl", "gross_ret", "ret", "exit_reason"]


@dataclass
class _Pos:
    col: int
    a: int
    b: int
    direction: int  # +1 long A / short B, -1 short A / long B
    entry_i: int
    entry_regime: object
    entry_score: float
    entry_vol: float  # daily ln(A/B) vol at signal time
    leg: float
    va: float
    vb: float
    costs: float
    days: int = 0
    gross: float = 0.0
    pending_exit: str = ""


def backtest(
    cfg: Config,
    prices: pd.DataFrame,
    regimes: pd.Series,
    schedule: pd.DataFrame,
    signals: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    initial_capital: float = 10_000_000.0,
) -> dict:
    """Simulate the portfolio over [cfg.start, cfg.end]. Returns equity frame, trade log and daily signal counts."""
    if signals is None:
        signals = compute_scores(prices, schedule, cfg.lookback, cfg.vol_window, cfg.signal, cfg.pct_window)
    score, daily_vol, active, vol_ratio = signals

    dates = prices.index
    lo = int(dates.searchsorted(pd.Timestamp(cfg.start))) if cfg.start else 0
    hi = int(dates.searchsorted(pd.Timestamp(cfg.end), side="right")) if cfg.end else len(dates)
    lo = max(lo, 1)

    col_idx = {c: i for i, c in enumerate(prices.columns)}
    px = prices.to_numpy()
    rets = prices.pct_change(fill_method=None).to_numpy()
    rets = np.where(np.isfinite(rets), rets, 0.0)
    sc = score.to_numpy()
    dv = daily_vol.to_numpy()
    act = active.to_numpy()
    if cfg.max_vol_ratio is not None:
        act = act & (vol_ratio.to_numpy() <= cfg.max_vol_ratio)
    keys = list(score.columns)
    key_ab = [(col_idx[k.split("|")[0]], col_idx[k.split("|")[1]]) for k in keys]
    reg = regimes.reindex(dates).to_numpy(dtype=object)
    sign = -1 if cfg.direction == "reversion" else 1
    buy_c, sell_c, borrow = side_cost_rates(cfg.cost_model, cfg.slippage_bps)
    roll_c = buy_c + sell_c if cfg.cost_model == "futures" else 0.0

    equity = float(initial_capital)
    positions: list[_Pos] = []
    pending: list[tuple[int, int, float, float, object]] = []  # col, direction, score, vol, regime at signal
    eq_rows: list[dict] = []
    trades: list[dict] = []
    n_signals = np.zeros(len(dates), dtype=int)

    def close(pos: _Pos, i: int, reason: str) -> float:
        long_v, short_v = (pos.va, pos.vb) if pos.direction == 1 else (pos.vb, pos.va)
        c = long_v * sell_c + short_v * buy_c
        pos.costs += c
        net = pos.gross - pos.costs
        trades.append({
            "pair": keys[pos.col], "stock_a": prices.columns[pos.a], "stock_b": prices.columns[pos.b],
            "direction": pos.direction, "entry_date": dates[pos.entry_i], "exit_date": dates[i],
            "days": pos.days, "entry_regime": pos.entry_regime, "entry_score": pos.entry_score,
            "leg_notional": pos.leg, "gross_pnl": pos.gross, "costs": pos.costs, "net_pnl": net,
            "gross_ret": pos.gross / pos.leg, "ret": net / pos.leg, "exit_reason": reason,
        })
        return c

    for i in range(lo, hi):
        day_pnl = 0.0
        day_cost = 0.0
        gap_days = (dates[i] - dates[i - 1]).days

        # 1) mark open positions to today's close
        for pos in positions:
            na, nb = pos.va * (1 + rets[i, pos.a]), pos.vb * (1 + rets[i, pos.b])
            pnl = pos.direction * ((na - pos.va) - (nb - pos.vb))
            short_v = nb if pos.direction == 1 else na
            fin = short_v * borrow * gap_days / 365.0
            pos.va, pos.vb = na, nb
            pos.days += 1
            if roll_c and pos.days % FUT_ROLL_DAYS == 0 and pos.days < cfg.hold_trading_days():
                fin += (na + nb) * roll_c
            pos.gross += pnl
            pos.costs += fin
            day_pnl += pnl
            day_cost += fin

        # 2) exits at today's close: time stop, or a signal-based exit decided yesterday
        last_day = i == hi - 1
        kept = []
        for pos in positions:
            if cfg.hold_calendar_days:
                time_up = (dates[i] - dates[pos.entry_i]).days >= cfg.hold_calendar_days
            else:
                time_up = pos.days >= cfg.hold
            reason = pos.pending_exit or ("time" if time_up else "")
            if not reason and not (np.isfinite(px[i, pos.a]) and np.isfinite(px[i, pos.b])):
                reason = "missing_price"
            if not reason and last_day:
                reason = "end_of_test"
            if reason:
                day_cost += close(pos, i, reason)
            else:
                kept.append(pos)
        positions = kept

        # 3) yesterday's entry decisions fill at today's close
        if not last_day:
            for col, direction, s, vol, sig_regime in pending:
                a, b = key_ab[col]
                if not (np.isfinite(px[i, a]) and np.isfinite(px[i, b])):
                    continue
                leg = (equity + day_pnl - day_cost) / cfg.max_pairs
                if leg <= 0:
                    continue
                c = leg * buy_c + leg * sell_c
                positions.append(_Pos(col, a, b, direction, i, sig_regime, s, vol, leg, leg, leg, costs=c))
                day_cost += c
        pending = []

        equity += day_pnl - day_cost
        gross_exp = sum(p.va + p.vb for p in positions)
        eq_rows.append({"date": dates[i], "equity": equity, "pnl": day_pnl, "costs": day_cost,
                        "n_open": len(positions), "gross_exposure": gross_exp / equity if equity > 0 else 0.0,
                        "regime": reg[i]})
        if last_day:
            break

        # 4) decisions at today's close (execute tomorrow)
        row_s = sc[i]
        for pos in positions:
            s = row_s[pos.col]
            if cfg.exit_on_revert and np.isfinite(s) and s * np.sign(pos.entry_score) <= 0:
                pos.pending_exit = "revert"
            if cfg.stop_k is not None and pos.gross - pos.costs < -cfg.stop_k * pos.entry_vol * np.sqrt(cfg.hold_trading_days()) * pos.leg:
                pos.pending_exit = pos.pending_exit or "stop"

        live = np.flatnonzero(act[i] & np.isfinite(row_s) & (np.abs(np.nan_to_num(row_s)) >= cfg.entry_k))
        n_signals[i] = len(live)
        slots = cfg.slots_for(reg[i]) - len(positions)
        if slots <= 0 or len(live) == 0:
            continue
        open_cols = {p.col for p in positions}
        exposure: dict[int, int] = {}
        for p in positions:
            exposure[p.a] = exposure.get(p.a, 0) + 1
            exposure[p.b] = exposure.get(p.b, 0) + 1
        for col in live[np.argsort(-np.abs(row_s[live]), kind="stable")]:
            if slots <= 0:
                break
            a, b = key_ab[col]
            if col in open_cols or exposure.get(a, 0) >= cfg.max_per_stock or exposure.get(b, 0) >= cfg.max_per_stock:
                continue
            pending.append((col, int(sign * np.sign(row_s[col])), float(row_s[col]), float(dv[i, col]), reg[i]))
            exposure[a] = exposure.get(a, 0) + 1
            exposure[b] = exposure.get(b, 0) + 1
            open_cols.add(col)
            slots -= 1

    eq = pd.DataFrame(eq_rows).set_index("date")
    tr = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return {"equity": eq, "trades": tr, "config": cfg, "n_signals": pd.Series(n_signals[lo:hi], index=dates[lo:hi])}


def signal_forward_returns(
    prices: pd.DataFrame, signals: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], regimes: pd.Series,
    entry_k: float, horizons: tuple[int, ...] = (1, 5, 10, 20), direction: str = "reversion",
) -> pd.DataFrame:
    """
    Event study, independent of portfolio limits and costs: for every day a pair's
    |score| first crosses entry_k (while active), the trade-direction log return of
    the pair from the next close (entry) to h days later.
    """
    score, _, active, _ = signals
    keys = list(score.columns)
    logp = np.log(prices)
    a_cols = [k.split("|")[0] for k in keys]
    b_cols = [k.split("|")[1] for k in keys]
    lr = pd.DataFrame(logp[a_cols].to_numpy() - logp[b_cols].to_numpy(), index=prices.index, columns=keys)
    trig = (score.abs() >= entry_k) & active
    first = trig & ~trig.shift(1, fill_value=False)
    sign = -1 if direction == "reversion" else 1
    out = {}
    for h in horizons:
        fwd = lr.shift(-(1 + h)) - lr.shift(-1)
        out[f"fwd_{h}d"] = (sign * np.sign(score) * fwd).where(first).stack()
    ev = pd.DataFrame(out).dropna(how="all")  # pandas>=3 stack() keeps the NaN (non-event) cells
    ev.index.names = ["date", "pair"]
    ev = ev.reset_index()
    ev["regime"] = regimes.reindex(ev["date"]).to_numpy()
    return ev


