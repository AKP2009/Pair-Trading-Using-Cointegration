"""
Performance metrics for the regime-aware pair strategy.

All annualization uses 252 trading days and a 0% risk-free rate (the
strategy is market-neutral; margin interest is not credited, which is
conservative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def perf_metrics(equity: pd.Series, bench: pd.Series | None = None, trades: pd.DataFrame | None = None) -> dict[str, float]:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()
    out: dict[str, float] = {}
    if len(rets) < 2 or equity.iloc[0] <= 0:
        return out
    years = len(rets) / ANN
    total = equity.iloc[-1] / equity.iloc[0] - 1
    out["total_return"] = float(total)
    out["CAGR"] = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    out["vol"] = float(rets.std() * np.sqrt(ANN))
    out["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(ANN)) if rets.std() > 0 else 0.0
    downside = np.sqrt((np.minimum(rets, 0.0) ** 2).mean())
    out["sortino"] = float(rets.mean() / downside * np.sqrt(ANN)) if downside > 0 else 0.0
    dd = equity / equity.cummax() - 1
    out["max_drawdown"] = float(dd.min())
    out["calmar"] = float(out["CAGR"] / abs(out["max_drawdown"])) if out["max_drawdown"] < 0 else 0.0
    out["pct_positive_days"] = float((rets > 0).mean())
    monthly = (1 + rets).resample("ME").prod() - 1
    out["pct_positive_months"] = float((monthly > 0).mean()) if len(monthly) else float("nan")

    if bench is not None:
        br = bench.pct_change().reindex(rets.index).dropna()
        r = rets.reindex(br.index)
        if len(br) > 2 and br.var() > 0 and r.var() > 0:
            out["beta"] = float(np.cov(r, br)[0, 1] / br.var())
            out["corr_bench"] = float(np.corrcoef(r, br)[0, 1])

    if trades is not None and len(trades):
        t = trades
        out["n_trades"] = int(len(t))
        out["win_rate"] = float((t["net_pnl"] > 0).mean())
        out["avg_trade_ret"] = float(t["ret"].mean())
        out["avg_trade_gross_ret"] = float(t["gross_ret"].mean())
        wins, losses = t.loc[t["net_pnl"] > 0, "net_pnl"].sum(), -t.loc[t["net_pnl"] < 0, "net_pnl"].sum()
        out["profit_factor"] = float(wins / losses) if losses > 0 else float("inf")
        out["avg_hold_days"] = float(t["days"].mean())
        out["trades_per_year"] = float(len(t) / years)
    elif trades is not None:
        out["n_trades"] = 0
    return out


def yearly_returns(equity: pd.Series) -> pd.Series:
    rets = equity.pct_change().dropna()
    return ((1 + rets).groupby(rets.index.year).prod() - 1).rename("return")


def monthly_returns_table(equity: pd.Series) -> pd.DataFrame:
    rets = equity.pct_change().dropna()
    m = (1 + rets).resample("ME").prod() - 1
    tbl = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    tbl = tbl.pivot(index="year", columns="month", values="ret")
    tbl.columns = [pd.Timestamp(2000, c, 1).strftime("%b") for c in tbl.columns]
    tbl["Year"] = yearly_returns(equity)
    return tbl


def regime_attribution(eq: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Daily-return stats grouped by the regime of each day, plus trade stats grouped by entry regime."""
    rets = eq["equity"].pct_change()
    df = pd.DataFrame({"ret": rets, "regime": eq["regime"]}).dropna()
    g = df.groupby("regime")["ret"]
    out = pd.DataFrame({
        "days": g.size(),
        "ann_return": g.mean() * ANN,
        "ann_vol": g.std() * np.sqrt(ANN),
    })
    out["sharpe"] = out["ann_return"] / out["ann_vol"]
    down = df.assign(d=np.minimum(df["ret"], 0) ** 2).groupby("regime")["d"].mean() ** 0.5
    out["sortino"] = (g.mean() / down) * np.sqrt(ANN)
    if trades is not None and len(trades):
        tg = trades.groupby("entry_regime")
        out["n_trades"] = tg.size()
        out["win_rate"] = tg["net_pnl"].apply(lambda s: (s > 0).mean())
        out["avg_trade_ret"] = tg["ret"].mean()
    return out
