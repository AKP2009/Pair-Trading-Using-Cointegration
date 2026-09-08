"""
Daily paper-trading ledger -- genuine forward-test evidence without a
broker account.

Run this once per trading day (see the Windows Task Scheduler command at
the bottom of this docstring). Each run pulls the latest close prices,
evaluates every pair's trained model for "today," advances the shared
day-by-day simulation engine (simulate.py) using the realistic cost model
(costs.py) and risk-based sizing/stop-losses (risk.py), and persists the
result to three CSVs:

- data/paper_trading_positions.csv : current open position per pair (one row per pair)
- data/paper_trading_trades.csv    : append-only closed-trade log
- data/paper_trading_equity.csv    : append-only daily mark-to-market equity curve

Reruns on a date that's already been processed are a no-op (checked via
each pair's `last_date`), so running this script twice in one day does
not double-count P&L.

To schedule on Windows (run every weekday at 18:00, after NSE close):
    schtasks /create /tn "PairTradingPaperTrade" /tr "\"<path to .venv\\Scripts\\python.exe>\" \"<path to this file>\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 18:00
"""

from __future__ import annotations

import pandas as pd

from model_pipeline import DATA_DIR, MODEL_DIR, append_latest_prices, build_features
from risk import DEFAULT_MAX_HOLD_DAYS
from simulate import PositionState, step

CAPITAL_PER_PAIR = 200_000.0
Z_ENTRY = 2.0
Z_EXIT = 0.5

POSITIONS_FILE = DATA_DIR / "paper_trading_positions.csv"
TRADES_FILE = DATA_DIR / "paper_trading_trades.csv"
EQUITY_FILE = DATA_DIR / "paper_trading_equity.csv"

POSITIONS_COLS = [
    "pair", "stock_a", "stock_b", "direction", "entry_date", "entry_z", "days_held",
    "shares_a", "shares_b", "entry_price_a", "entry_price_b", "capital", "last_date",
]


def _load_positions() -> pd.DataFrame:
    if POSITIONS_FILE.exists():
        return pd.read_csv(POSITIONS_FILE)
    return pd.DataFrame(columns=POSITIONS_COLS)


def _row_to_state(row: pd.Series) -> PositionState:
    return PositionState(
        direction=int(row["direction"]),
        entry_date=row["entry_date"] if pd.notna(row["entry_date"]) else None,
        entry_z=float(row["entry_z"]) if pd.notna(row["entry_z"]) else 0.0,
        days_held=int(row["days_held"]) if pd.notna(row["days_held"]) else 0,
        shares_a=int(row["shares_a"]) if pd.notna(row["shares_a"]) else 0,
        shares_b=int(row["shares_b"]) if pd.notna(row["shares_b"]) else 0,
        entry_price_a=float(row["entry_price_a"]) if pd.notna(row["entry_price_a"]) else 0.0,
        entry_price_b=float(row["entry_price_b"]) if pd.notna(row["entry_price_b"]) else 0.0,
    )


def main() -> None:
    registry = pd.read_csv(MODEL_DIR / "pair_model_registry.csv")
    if registry.empty:
        raise ValueError("pair_model_registry.csv is empty. Run model_pipeline.py first.")

    prices = pd.read_csv(DATA_DIR / "nifty_prices_clean.csv", index_col=0, parse_dates=True)
    all_tickers = sorted(set(prices.columns) | set(registry["stock_a"]) | set(registry["stock_b"]))
    prices = append_latest_prices(prices, all_tickers)

    positions = _load_positions()
    trade_rows: list[dict] = []
    equity_rows: list[dict] = []
    updated_positions: list[dict] = []

    for _, reg_row in registry.iterrows():
        stock_a = str(reg_row["stock_a"])
        stock_b = str(reg_row["stock_b"])
        model_name = str(reg_row["model_name"])
        beta_static = float(reg_row["beta_static"])
        threshold = float(reg_row.get("threshold", 0.55))

        if stock_a not in prices.columns or stock_b not in prices.columns:
            continue

        dataset = build_features(prices, stock_a, stock_b, beta_static)
        if dataset.empty:
            continue

        today = dataset.index[-1]
        pos_row = positions[positions["pair"] == model_name]

        if not pos_row.empty and pd.notna(pos_row.iloc[0]["last_date"]) and str(pos_row.iloc[0]["last_date"]) == today.date().isoformat():
            updated_positions.append(pos_row.iloc[0].to_dict())
            continue

        if pos_row.empty:
            state = PositionState()
            capital = CAPITAL_PER_PAIR
        else:
            state = _row_to_state(pos_row.iloc[0])
            capital = float(pos_row.iloc[0]["capital"])

        model = None
        model_path = reg_row.get("model_path")
        if model_path:
            import joblib
            from pathlib import Path

            path = Path(str(model_path))
            if path.exists():
                model = joblib.load(path)
        if model is None:
            continue

        from model_pipeline import FEATURE_COLS

        last_row = dataset.iloc[-1]
        z = float(last_row["z"])
        beta_today = float(last_row["beta"])
        spread_vol = float(last_row["spread_vol_20"])
        prob = float(model.predict_proba(last_row[FEATURE_COLS].to_frame().T)[:, 1][0])
        rf_accept = prob >= threshold
        baseline_action = "SHORT_SPREAD" if z > Z_ENTRY else ("LONG_SPREAD" if z < -Z_ENTRY else "NO_TRADE")

        price_a = float(prices.loc[today, stock_a])
        price_b = float(prices.loc[today, stock_b])
        pos_idx = prices.index.get_loc(today)
        if pos_idx == 0:
            price_a_prev, price_b_prev = price_a, price_b
        else:
            prev_date = prices.index[pos_idx - 1]
            price_a_prev = float(prices.loc[prev_date, stock_a])
            price_b_prev = float(prices.loc[prev_date, stock_b])

        was_open = state.direction != 0
        state, res = step(
            state, today, price_a, price_b, price_a_prev, price_b_prev, beta_today, z,
            baseline_action, rf_accept, capital, spread_vol, z_exit=Z_EXIT,
            max_hold_days=DEFAULT_MAX_HOLD_DAYS,
        )
        capital += res["pnl"]

        if was_open and res["event"] in ("exit", "stop_out"):
            trade_rows.append(
                {
                    "pair": model_name, "stock_a": stock_a, "stock_b": stock_b,
                    "exit_date": today.date().isoformat(), "pnl": res["pnl"], "reason": res["stop_reason"] or "z_exit",
                }
            )

        equity_rows.append(
            {"date": today.date().isoformat(), "pair": model_name, "pnl": res["pnl"], "equity": capital, "event": res["event"]}
        )

        updated_positions.append(
            {
                "pair": model_name, "stock_a": stock_a, "stock_b": stock_b, "direction": state.direction,
                "entry_date": state.entry_date.date().isoformat() if hasattr(state.entry_date, "date") else state.entry_date,
                "entry_z": state.entry_z, "days_held": state.days_held, "shares_a": state.shares_a,
                "shares_b": state.shares_b, "entry_price_a": state.entry_price_a, "entry_price_b": state.entry_price_b,
                "capital": capital, "last_date": today.date().isoformat(),
            }
        )

    if updated_positions:
        pd.DataFrame(updated_positions, columns=POSITIONS_COLS).to_csv(POSITIONS_FILE, index=False)

    if trade_rows:
        existing_trades = pd.read_csv(TRADES_FILE) if TRADES_FILE.exists() else pd.DataFrame()
        pd.concat([existing_trades, pd.DataFrame(trade_rows)], ignore_index=True).to_csv(TRADES_FILE, index=False)

    if equity_rows:
        new_equity = pd.DataFrame(equity_rows)
        existing_equity = pd.read_csv(EQUITY_FILE) if EQUITY_FILE.exists() else pd.DataFrame(columns=["date", "pair", "pnl", "equity", "event"])
        if not existing_equity.empty:
            key = existing_equity.set_index(["date", "pair"]).index
            new_key = new_equity.set_index(["date", "pair"]).index
            existing_equity = existing_equity[~key.isin(new_key)]
        combined = pd.concat([existing_equity, new_equity], ignore_index=True).sort_values(["pair", "date"])
        combined.to_csv(EQUITY_FILE, index=False)
        print(f"Paper trading updated for {len(equity_rows)} pair(s) on {equity_rows[0]['date']}.")
        print(new_equity.to_string(index=False))
    else:
        print("No pairs required an update (already processed for today, or no usable model/data).")

if __name__ == "__main__":
    main()
