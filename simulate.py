"""
Shared day-by-day pair-trade simulation engine.

Given one trading day's prices/signal, `step()` decides whether to open,
hold, or close a position; sizes it via `risk.volatility_target_size`; and
prices entries/exits via `costs.py`. It is deliberately a pure
state-in/state-out function so it can be reused two different ways:

- walk_forward.py replays a whole historical window in a loop, folding
  each day's result into a chained equity curve.
- paper_trading.py calls it once per real calendar day, persisting
  `PositionState` to disk between separate script runs.

PnL is computed from actual integer share counts and raw price changes
(not a returns-based spread proxy), since costs.round_to_tradable_shares
gives us real share counts to work with -- this is more realistic than
the ratio-based spread return used in the original notebooks.

Borrow cost is charged daily on whichever leg is short. Since capital is
split evenly between the two legs at entry, the short leg's notional is
approximated directly from its own share count and price (not halved),
which is more accurate than assuming a 50/50 split holds over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from costs import leg_transaction_cost, round_to_tradable_shares, short_borrow_cost, slippage_cost
from risk import (
    DEFAULT_MAX_HOLD_DAYS,
    DEFAULT_MAX_Z_STOP,
    DEFAULT_TARGET_RISK_PCT,
    should_stop_out,
    volatility_target_size,
)
from costs import DEFAULT_ANNUAL_BORROW_RATE


@dataclass
class PositionState:
    direction: int = 0  # +1 = long spread (long A, short B), -1 = short spread, 0 = flat
    entry_date: Optional[object] = None
    entry_z: float = 0.0
    days_held: int = 0
    shares_a: int = 0
    shares_b: int = 0
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0


def _direction_for_action(action: str) -> int:
    if action == "LONG_SPREAD":
        return 1
    if action == "SHORT_SPREAD":
        return -1
    return 0


def _entry_costs(direction: int, notional_a: float, notional_b: float) -> float:
    if direction == 1:  # buy A, sell-short B
        return leg_transaction_cost(notional_a, is_buy=True) + leg_transaction_cost(notional_b, is_buy=False)
    return leg_transaction_cost(notional_a, is_buy=False) + leg_transaction_cost(notional_b, is_buy=True)


def _exit_costs(direction: int, notional_a: float, notional_b: float) -> float:
    if direction == 1:  # sell (delivery) A, buy-to-cover B
        return leg_transaction_cost(notional_a, is_buy=False, is_delivery_sell=True) + leg_transaction_cost(
            notional_b, is_buy=True
        )
    return leg_transaction_cost(notional_a, is_buy=True) + leg_transaction_cost(
        notional_b, is_buy=False, is_delivery_sell=True
    )


def step(
    state: PositionState,
    date: object,
    price_a: float,
    price_b: float,
    price_a_prev: float,
    price_b_prev: float,
    beta: float,
    z: float,
    baseline_action: str,
    rf_accept: bool,
    capital: float,
    spread_vol: float,
    z_exit: float = 0.5,
    target_risk_pct: float = DEFAULT_TARGET_RISK_PCT,
    annual_borrow_rate: float = DEFAULT_ANNUAL_BORROW_RATE,
    max_z_stop: float = DEFAULT_MAX_Z_STOP,
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS,
) -> tuple[PositionState, dict]:
    """Advance one trading day. Returns (new_state, day_result)."""
    result = {
        "date": date,
        "pnl": 0.0,
        "cost": 0.0,
        "notional": 0.0,
        "event": "flat",
        "stop_reason": "",
        "direction": state.direction,
    }

    if state.direction != 0:
        notional_a = state.shares_a * price_a
        notional_b = state.shares_b * price_b
        raw_pnl = state.direction * (
            state.shares_a * (price_a - price_a_prev) - state.shares_b * (price_b - price_b_prev)
        )
        short_notional = notional_b if state.direction == 1 else notional_a
        borrow = short_borrow_cost(short_notional, days=1, annual_rate=annual_borrow_rate)
        state.days_held += 1

        stop, reason = should_stop_out(z=z, days_held=state.days_held, max_z=max_z_stop, max_hold_days=max_hold_days)
        exit_signal = z is not None and z == z and abs(z) < z_exit
        if stop or exit_signal:
            exit_cost = _exit_costs(state.direction, notional_a, notional_b) + slippage_cost(
                notional_a + notional_b, spread_vol
            )
            result.update(
                pnl=raw_pnl - borrow - exit_cost,
                cost=exit_cost,
                notional=notional_a + notional_b,
                event="stop_out" if stop else "exit",
                stop_reason=reason,
                direction=state.direction,
            )
            state = PositionState()
        else:
            result.update(pnl=raw_pnl - borrow, notional=notional_a + notional_b, event="hold")
        return state, result

    direction = _direction_for_action(baseline_action)
    if direction != 0 and rf_accept:
        target_notional = volatility_target_size(capital, spread_vol, target_risk_pct)
        if target_notional > 0 and beta and beta > 0:
            sizing = round_to_tradable_shares(target_notional, price_a, price_b, beta)
            notional_a = sizing["shares_a"] * price_a
            notional_b = sizing["shares_b"] * price_b
            entry_cost = _entry_costs(direction, notional_a, notional_b) + slippage_cost(
                notional_a + notional_b, spread_vol
            )
            state = PositionState(
                direction=direction,
                entry_date=date,
                entry_z=z,
                days_held=0,
                shares_a=sizing["shares_a"],
                shares_b=sizing["shares_b"],
                entry_price_a=price_a,
                entry_price_b=price_b,
            )
            result.update(
                pnl=-entry_cost,
                cost=entry_cost,
                notional=notional_a + notional_b,
                event="entry",
                direction=direction,
            )
    return state, result


if __name__ == "__main__":
    state = PositionState()
    state, r1 = step(
        state, date=1, price_a=1500.0, price_b=800.0, price_a_prev=1500.0, price_b_prev=800.0,
        beta=1.89, z=2.3, baseline_action="SHORT_SPREAD", rf_accept=True, capital=200_000, spread_vol=0.01,
    )
    assert r1["event"] == "entry" and state.direction == -1
    state, r2 = step(
        state, date=2, price_a=1490.0, price_b=805.0, price_a_prev=1500.0, price_b_prev=800.0,
        beta=1.89, z=1.8, baseline_action="SHORT_SPREAD", rf_accept=True, capital=200_000, spread_vol=0.01,
    )
    assert r2["event"] == "hold"
    state, r3 = step(
        state, date=3, price_a=1495.0, price_b=802.0, price_a_prev=1490.0, price_b_prev=805.0,
        beta=1.89, z=0.2, baseline_action="NO_TRADE", rf_accept=False, capital=200_000, spread_vol=0.01,
    )
    assert r3["event"] == "exit" and state.direction == 0
    print("simulate.py smoke test passed.")
