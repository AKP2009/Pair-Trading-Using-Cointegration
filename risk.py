"""
Position sizing and portfolio-level risk controls for the pair trading
strategy. Complements costs.py: costs.py prices a trade, risk.py decides
how big to make it and when to force an exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_TARGET_RISK_PCT = 0.01
DEFAULT_MAX_Z_STOP = 4.0
DEFAULT_MAX_HOLD_DAYS = 40


def volatility_target_size(capital: float, spread_vol: float, target_risk_pct: float = DEFAULT_TARGET_RISK_PCT) -> float:
    """
    Size a position (as rupee notional) so that one day's spread-return
    volatility is expected to move `target_risk_pct` of allocated capital.
    Returns 0 when volatility is zero/unknown (can't size safely).
    """
    if capital <= 0 or spread_vol is None or spread_vol != spread_vol or spread_vol <= 0:
        return 0.0
    target_dollar_vol = capital * target_risk_pct
    notional = target_dollar_vol / spread_vol
    return min(notional, capital)


def should_stop_out(
    z: float,
    days_held: int,
    max_z: float = DEFAULT_MAX_Z_STOP,
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
) -> tuple[bool, str]:
    """Decide whether an open trade should be force-closed. Returns (should_stop, reason)."""
    if z is not None and z == z and abs(z) >= max_z:
        return True, f"z_breakdown(|z|={abs(z):.2f}>={max_z})"
    if days_held >= max_hold_days:
        return True, f"time_stop(days_held={days_held}>={max_hold_days})"
    return False, ""


@dataclass
class PortfolioLimits:
    """Portfolio-level exposure checker across concurrently open pairs."""

    max_concurrent_pairs: int = 3
    open_tickers: set[str] = field(default_factory=set)

    def can_open(self, stock_a: str, stock_b: str, currently_open_count: int) -> tuple[bool, str]:
        if currently_open_count >= self.max_concurrent_pairs:
            return False, f"max_concurrent_pairs_reached({self.max_concurrent_pairs})"
        if stock_a in self.open_tickers or stock_b in self.open_tickers:
            return False, "ticker_already_exposed"
        return True, ""

    def register_open(self, stock_a: str, stock_b: str) -> None:
        self.open_tickers.add(stock_a)
        self.open_tickers.add(stock_b)

    def register_close(self, stock_a: str, stock_b: str) -> None:
        self.open_tickers.discard(stock_a)
        self.open_tickers.discard(stock_b)


if __name__ == "__main__":
    size = volatility_target_size(capital=200_000, spread_vol=0.015)
    print(f"Volatility-targeted notional: {size:,.0f}")
    assert size > 0

    stop, reason = should_stop_out(z=4.5, days_held=5)
    assert stop and "z_breakdown" in reason
    stop2, reason2 = should_stop_out(z=1.0, days_held=50)
    assert stop2 and "time_stop" in reason2
    stop3, _ = should_stop_out(z=2.5, days_held=5)
    assert not stop3

    limits = PortfolioLimits(max_concurrent_pairs=2)
    ok, _ = limits.can_open("A", "B", currently_open_count=0)
    assert ok
    limits.register_open("A", "B")
    ok2, why2 = limits.can_open("A", "C", currently_open_count=1)
    assert not ok2 and why2 == "ticker_already_exposed"
    print("risk.py smoke test passed.")
