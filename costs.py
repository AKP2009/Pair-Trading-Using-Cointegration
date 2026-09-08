"""
Realistic transaction cost & execution model for Indian cash-equity pair trading.

Cost assumptions (approximate, representative retail NSE delivery/intraday
charges; every constant below is a configurable estimate, not a live rate
card -- cite/adjust these in the paper as needed):

- STT (Securities Transaction Tax): 0.1% of turnover, charged on both the
  buy and the sell leg for delivery equity.
- NSE exchange transaction charge: ~0.00297% of turnover.
- SEBI turnover fee: ~0.0001% of turnover.
- Stamp duty: 0.015% of turnover, buy side only.
- GST: 18% on exchange transaction charges (brokerage assumed 0 for
  delivery trades, standard for Indian discount brokers).
- DP (depository participant) charge: a flat fee per scrip, charged only
  when a delivery holding is sold.

Short-selling note: NSE cash-market equity cannot be sold short overnight
without Securities Lending & Borrowing (SLB). This module treats the
"short" leg of a pair trade as SLB-borrowed stock and charges an explicit
annualized borrow cost via `short_borrow_cost`, rather than pretending the
short leg is free like a naive backtest would.
"""

from __future__ import annotations

STT_RATE = 0.001
EXCHANGE_TXN_CHARGE_RATE = 0.0000297
SEBI_TURNOVER_FEE_RATE = 0.000001
STAMP_DUTY_RATE = 0.00015
GST_RATE = 0.18
DP_CHARGE_FLAT = 15.0
DEFAULT_ANNUAL_BORROW_RATE = 0.08
DEFAULT_SLIPPAGE_VOL_MULTIPLIER = 0.5


def leg_transaction_cost(notional: float, is_buy: bool, is_delivery_sell: bool = False) -> float:
    """Regulatory + exchange charges for one trade leg. `notional` is the absolute rupee value traded."""
    if notional <= 0:
        return 0.0
    stt = STT_RATE * notional
    exch = EXCHANGE_TXN_CHARGE_RATE * notional
    sebi = SEBI_TURNOVER_FEE_RATE * notional
    stamp = STAMP_DUTY_RATE * notional if is_buy else 0.0
    gst = GST_RATE * exch
    dp = DP_CHARGE_FLAT if is_delivery_sell else 0.0
    return stt + exch + sebi + stamp + gst + dp


def short_borrow_cost(notional: float, days: float, annual_rate: float = DEFAULT_ANNUAL_BORROW_RATE) -> float:
    """SLB borrow cost for holding a short position of given notional for `days` calendar days."""
    if notional <= 0 or days <= 0:
        return 0.0
    return notional * annual_rate * (days / 365.0)


def slippage_cost(notional: float, spread_vol: float, multiplier: float = DEFAULT_SLIPPAGE_VOL_MULTIPLIER) -> float:
    """Slippage scaled by recent spread-return volatility instead of a fixed bps guess."""
    if notional <= 0 or spread_vol is None or spread_vol != spread_vol:
        return 0.0
    return notional * spread_vol * multiplier


def round_to_tradable_shares(capital: float, price_a: float, price_b: float, beta: float) -> dict[str, float]:
    """
    Convert a target capital allocation + hedge ratio into integer share
    quantities on both legs. Returns the realized hedge ratio and the
    tracking error introduced by rounding to whole shares.
    """
    if price_a <= 0 or price_b <= 0 or capital <= 0 or beta <= 0:
        return {"shares_a": 0, "shares_b": 0, "realized_beta": 0.0, "leftover_cash": capital, "beta_error_pct": 0.0}

    shares_a = max(int(round((capital / 2) / price_a)), 1)
    shares_b = max(int(round(shares_a * beta)), 1)

    realized_beta = shares_b / shares_a
    beta_error_pct = abs(realized_beta - beta) / beta

    spent = shares_a * price_a + shares_b * price_b
    leftover_cash = capital - spent

    return {
        "shares_a": shares_a,
        "shares_b": shares_b,
        "realized_beta": realized_beta,
        "leftover_cash": leftover_cash,
        "beta_error_pct": beta_error_pct,
    }


def round_trip_cost_pct(
    notional_a: float,
    notional_b: float,
    spread_vol: float,
    holding_days: float,
    annual_borrow_rate: float = DEFAULT_ANNUAL_BORROW_RATE,
) -> float:
    """
    Full round-trip cost as a fraction of combined notional for one pair
    trade: entry (buy+short) + holding borrow cost + exit (sell+cover) +
    slippage on both legs.
    """
    combined_notional = notional_a + notional_b
    if combined_notional <= 0:
        return 0.0

    entry_cost = leg_transaction_cost(notional_a, is_buy=True) + leg_transaction_cost(notional_b, is_buy=True)
    exit_cost = leg_transaction_cost(notional_a, is_buy=False, is_delivery_sell=True) + leg_transaction_cost(
        notional_b, is_buy=False
    )
    borrow_cost = short_borrow_cost(notional_b, holding_days, annual_borrow_rate)
    slip = slippage_cost(notional_a, spread_vol) + slippage_cost(notional_b, spread_vol)

    total = entry_cost + exit_cost + borrow_cost + slip
    return total / combined_notional


if __name__ == "__main__":
    demo = round_to_tradable_shares(capital=200_000, price_a=1500.0, price_b=800.0, beta=1.89)
    print("Share rounding:", demo)
    cost_pct = round_trip_cost_pct(notional_a=100_000, notional_b=100_000, spread_vol=0.01, holding_days=10)
    print(f"Round-trip cost: {cost_pct:.4%} of notional")
    assert cost_pct > 0
    assert demo["shares_a"] > 0 and demo["shares_b"] > 0
    print("costs.py smoke test passed.")
