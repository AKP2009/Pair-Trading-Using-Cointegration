"""
Walk-forward validation for the pair-trading strategy.

The existing model_pipeline.py trains once on a single static
train(2015-21)/valid(2022-23)/test(2024-25) split. That is weak evidence
for a research paper -- a single split can't distinguish a genuinely
robust edge from a lucky test window. This script instead retrains on an
expanding window and evaluates strictly out-of-sample on the next block,
chaining every out-of-sample block into one continuous walk-forward
equity curve, and it applies the realistic cost model (costs.py) and
risk-based position sizing/stop-losses (risk.py / simulate.py) instead of
the old flat 10bps-per-turnover assumption.

Outputs:
- data/walk_forward_stats.csv   (per-pair, per-fold metrics + aggregate rows)
- data/walk_forward_equity.csv  (chained daily equity curve, long format: date, pair, pnl, equity, event, fold)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_pipeline import DATA_DIR, FEATURE_COLS, build_features, make_rf_model, pair_slug
from simulate import PositionState, step

INITIAL_TRAIN_YEARS = 4
STEP_MONTHS = 6
CAPITAL_PER_PAIR = 200_000.0
Z_ENTRY = 2.0
Z_EXIT = 0.5
RF_THRESHOLD = 0.55
MIN_TRAIN_ROWS = 200


def performance_stats(equity: pd.Series) -> dict[str, float]:
    """Sharpe / CAGR / Vol / MaxDD from a daily equity curve (same metric names as baseline_backtest_stats.csv)."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "FinalEquity": float(equity.iloc[-1]) if len(equity) else 0.0}

    rets = equity.pct_change().fillna(0.0)
    ann_factor = 252
    years = len(equity) / ann_factor
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = float(rets.std() * np.sqrt(ann_factor))
    sharpe = float((rets.mean() * ann_factor) / vol) if vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_dd = float(drawdown.min())
    return {"CAGR": float(cagr), "Vol": vol, "Sharpe": sharpe, "MaxDD": max_dd, "FinalEquity": float(equity.iloc[-1])}


def _trade_win_rate(events: list[str], pnls: list[float]) -> tuple[int, float]:
    trade_id = 0
    ids = []
    for ev in events:
        if ev == "entry":
            trade_id += 1
        ids.append(trade_id if ev in ("entry", "hold", "exit", "stop_out") else 0)
    df = pd.DataFrame({"trade_id": ids, "pnl": pnls})
    df = df[df["trade_id"] > 0]
    if df.empty:
        return 0, float("nan")
    trade_pnls = df.groupby("trade_id")["pnl"].sum()
    return int(len(trade_pnls)), float((trade_pnls > 0).mean())


def walk_forward_pair(prices: pd.DataFrame, pair_row: pd.Series) -> tuple[list[dict], list[dict]]:
    stock_a = str(pair_row["stock_a"])
    stock_b = str(pair_row["stock_b"])
    beta_static = float(pair_row["beta_a_on_b"])
    model_name = pair_slug(stock_a, stock_b)

    dataset = build_features(prices, stock_a, stock_b, beta_static)
    if dataset.empty:
        return [], []

    train_start = dataset.index.min()
    train_end = train_start + pd.DateOffset(years=INITIAL_TRAIN_YEARS)
    end_date = dataset.index.max()

    state = PositionState()
    capital = CAPITAL_PER_PAIR
    fold_id = 0
    equity_rows: list[dict] = []
    fold_rows: list[dict] = []

    while train_end < end_date:
        test_start = train_end
        test_end = min(test_start + pd.DateOffset(months=STEP_MONTHS), end_date)

        train_df = dataset.loc[train_start:train_end]
        test_df = dataset.loc[test_start:test_end]
        test_df = test_df[test_df.index > train_end]

        if len(train_df) < MIN_TRAIN_ROWS or test_df.empty:
            train_end = test_end
            fold_id += 1
            continue

        model = make_rf_model()
        model.fit(train_df[FEATURE_COLS], train_df["y"])
        probs = model.predict_proba(test_df[FEATURE_COLS])[:, 1]

        fold_start_capital = capital

        for i, (dt, row) in enumerate(test_df.iterrows()):
            z = float(row["z"])
            beta_today = float(row["beta"])
            spread_vol = float(row["spread_vol_20"])
            rf_accept = bool(probs[i] >= RF_THRESHOLD)
            baseline_action = "SHORT_SPREAD" if z > Z_ENTRY else ("LONG_SPREAD" if z < -Z_ENTRY else "NO_TRADE")

            price_a = float(prices.loc[dt, stock_a])
            price_b = float(prices.loc[dt, stock_b])
            pos = prices.index.get_loc(dt)
            if pos == 0:
                price_a_prev, price_b_prev = price_a, price_b
            else:
                prev_dt = prices.index[pos - 1]
                price_a_prev = float(prices.loc[prev_dt, stock_a])
                price_b_prev = float(prices.loc[prev_dt, stock_b])

            state, res = step(
                state, dt, price_a, price_b, price_a_prev, price_b_prev, beta_today, z,
                baseline_action, rf_accept, capital, spread_vol, z_exit=Z_EXIT,
            )
            capital += res["pnl"]
            equity_rows.append(
                {"date": dt, "pair": model_name, "fold": fold_id, "pnl": res["pnl"], "equity": capital, "event": res["event"]}
            )

        fold_slice = [r for r in equity_rows if r["fold"] == fold_id]
        fold_equity = pd.Series([fold_start_capital] + [r["equity"] for r in fold_slice])
        stats = performance_stats(fold_equity)
        n_trades, win_rate = _trade_win_rate([r["event"] for r in fold_slice], [r["pnl"] for r in fold_slice])
        fold_rows.append(
            {
                "pair": model_name, "fold": fold_id, "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(), "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(), "n_trades": n_trades, "win_rate": win_rate, **stats,
            }
        )

        train_end = test_end
        fold_id += 1

    return equity_rows, fold_rows


def main() -> None:
    prices = pd.read_csv(DATA_DIR / "nifty_prices_clean.csv", index_col=0, parse_dates=True)
    pairs = pd.read_csv(DATA_DIR / "selected_pairs.csv")
    if pairs.empty:
        raise ValueError("selected_pairs.csv is empty. Run pair selection first.")

    all_equity_rows: list[dict] = []
    all_fold_rows: list[dict] = []

    for _, pair_row in pairs.iterrows():
        equity_rows, fold_rows = walk_forward_pair(prices, pair_row)
        all_equity_rows.extend(equity_rows)
        all_fold_rows.extend(fold_rows)

    if not all_equity_rows:
        raise ValueError("No walk-forward folds produced. Check data history length vs INITIAL_TRAIN_YEARS/STEP_MONTHS.")

    equity_df = pd.DataFrame(all_equity_rows).sort_values(["pair", "date"])
    fold_df = pd.DataFrame(all_fold_rows)

    aggregate_rows = []
    for pair_name, grp in equity_df.groupby("pair"):
        chained_equity = pd.Series([CAPITAL_PER_PAIR] + grp["equity"].tolist())
        stats = performance_stats(chained_equity)
        n_trades, win_rate = _trade_win_rate(grp["event"].tolist(), grp["pnl"].tolist())
        aggregate_rows.append({"pair": pair_name, "fold": "ALL", "n_trades": n_trades, "win_rate": win_rate, **stats})

    portfolio_daily = equity_df.pivot_table(index="date", columns="pair", values="pnl", aggfunc="sum").fillna(0.0)
    portfolio_capital = CAPITAL_PER_PAIR * portfolio_daily.shape[1]
    portfolio_equity = portfolio_capital + portfolio_daily.sum(axis=1).cumsum()
    portfolio_stats = performance_stats(pd.Series([portfolio_capital] + portfolio_equity.tolist()))
    aggregate_rows.append({"pair": "PORTFOLIO", "fold": "ALL", "n_trades": None, "win_rate": None, **portfolio_stats})

    stats_out = pd.concat([fold_df, pd.DataFrame(aggregate_rows)], ignore_index=True)
    stats_file = DATA_DIR / "walk_forward_stats.csv"
    stats_out.to_csv(stats_file, index=False)

    portfolio_equity_rows = pd.DataFrame({"date": portfolio_equity.index, "pair": "PORTFOLIO", "fold": "ALL", "pnl": portfolio_daily.sum(axis=1).values, "equity": portfolio_equity.values, "event": ""})
    equity_out = pd.concat([equity_df, portfolio_equity_rows], ignore_index=True)
    equity_file = DATA_DIR / "walk_forward_equity.csv"
    equity_out.to_csv(equity_file, index=False)

    print("Walk-forward validation complete")
    print(f"Saved stats: {stats_file}")
    print(f"Saved equity curve: {equity_file}")
    print(pd.DataFrame(aggregate_rows).to_string(index=False))


if __name__ == "__main__":
    main()
