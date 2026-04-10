from __future__ import annotations

import json
from pathlib import Path
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_BANK_DIR = MODEL_DIR / "pair_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_BANK_DIR.mkdir(parents=True, exist_ok=True)


def build_features(
    prices: pd.DataFrame,
    stock_a: str,
    stock_b: str,
    beta_static: float,
    beta_window: int = 252,
    z_window: int = 30,
    horizon: int = 10,
) -> pd.DataFrame:
    """Build model features and target label for one selected pair."""
    beta_roll = prices[stock_a].rolling(beta_window).cov(prices[stock_b]) / prices[stock_b].rolling(beta_window).var()
    beta = beta_roll.fillna(beta_static)

    spread = prices[stock_a] - beta * prices[stock_b]
    spread_mean = spread.rolling(z_window).mean()
    spread_std = spread.rolling(z_window).std()
    z = (spread - spread_mean) / spread_std

    ret_a = prices[stock_a].pct_change()
    ret_b = prices[stock_b].pct_change()
    spread_ret = ret_a - beta * ret_b

    feat = pd.DataFrame(index=prices.index)
    feat["z"] = z
    feat["z_chg_5"] = z - z.shift(5)
    feat["z_chg_20"] = z - z.shift(20)
    feat["corr_30"] = prices[stock_a].rolling(30).corr(prices[stock_b])
    feat["spread_vol_20"] = spread_ret.rolling(20).std()
    feat["spread_vol_60"] = spread_ret.rolling(60).std()
    feat["beta"] = beta
    feat["spread"] = spread

    # Target: does spread move closer to rolling mean in next N days?
    future_spread = spread.shift(-horizon)
    dist_now = (spread - spread_mean).abs()
    dist_future = (future_spread - spread_mean).abs()
    feat["y"] = (dist_future < dist_now).astype(int)

    return feat.dropna()


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.loc["2015-01-01":"2021-12-31"].copy()
    valid_df = df.loc["2022-01-01":"2023-12-31"].copy()
    test_df = df.loc["2024-01-01":"2025-12-31"].copy()
    return train_df, valid_df, test_df


def append_latest_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Append newer close data from Yahoo Finance after last available date."""
    if prices.empty:
        return prices

    last_dt = pd.to_datetime(prices.index.max())
    start = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    if pd.Timestamp(start) >= pd.Timestamp(end):
        return prices

    unique_tickers = list(dict.fromkeys(tickers))

    try:
        raw_new = yf.download(
            tickers=unique_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
    except Exception:
        return prices

    if raw_new is None or len(raw_new) == 0:
        return prices

    if "Close" in raw_new:
        close_new = raw_new["Close"].copy()
    else:
        close_new = raw_new.copy()

    if isinstance(close_new.columns, pd.MultiIndex):
        close_new.columns = close_new.columns.get_level_values(0)

    close_new = close_new.loc[:, ~close_new.columns.duplicated()]

    close_new.index = pd.to_datetime(close_new.index)
    close_new = close_new[[c for c in unique_tickers if c in close_new.columns]]

    if close_new.empty:
        return prices

    prices = prices.loc[:, ~prices.columns.duplicated()]
    combined = pd.concat([prices, close_new], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = combined.ffill().dropna()
    return combined


def metric_bundle(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def pair_slug(stock_a: str, stock_b: str) -> str:
    return f"{stock_a.replace('.NS', '')}__{stock_b.replace('.NS', '')}".replace("-", "_")


def train_one_pair(
    prices: pd.DataFrame,
    pair_row: pd.Series,
    threshold: float = 0.55,
) -> tuple[dict[str, object], pd.DataFrame] | None:
    stock_a = str(pair_row["stock_a"])
    stock_b = str(pair_row["stock_b"])
    beta_static = float(pair_row["beta_a_on_b"])

    dataset = build_features(prices, stock_a, stock_b, beta_static)
    feature_cols = ["z", "z_chg_5", "z_chg_20", "corr_30", "spread_vol_20", "spread_vol_60"]
    train_df, valid_df, test_df = split_data(dataset)

    if train_df.empty or valid_df.empty or test_df.empty:
        return None

    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_valid, y_valid = valid_df[feature_cols], valid_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=15,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    valid_pred = (valid_prob >= threshold).astype(int)
    test_pred = (test_prob >= threshold).astype(int)

    valid_metrics = metric_bundle(y_valid, valid_pred, valid_prob)
    test_metrics = metric_bundle(y_test, test_pred, test_prob)

    model_name = pair_slug(stock_a, stock_b)
    model_file = MODEL_BANK_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_file)

    last = dataset.iloc[-1]
    x_live = last[feature_cols].to_frame().T
    live_prob = float(model.predict_proba(x_live)[:, 1][0])
    z_now = float(last["z"])
    z_entry = 2.0

    if z_now > z_entry:
        baseline_action = "SHORT_SPREAD"
    elif z_now < -z_entry:
        baseline_action = "LONG_SPREAD"
    else:
        baseline_action = "NO_TRADE"

    if baseline_action != "NO_TRADE" and live_prob >= threshold:
        final_action = baseline_action
    else:
        final_action = "NO_TRADE"

    meta: dict[str, object] = {
        "model_name": model_name,
        "stock_a": stock_a,
        "stock_b": stock_b,
        "beta_static": beta_static,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "model_path": str(model_file),
        "dataset_last_date": dataset.index[-1].date().isoformat(),
        "live_zscore": z_now,
        "live_probability": live_prob,
        "baseline_action": baseline_action,
        "final_action": final_action,
        "validation_accuracy": valid_metrics["accuracy"],
        "validation_precision": valid_metrics["precision"],
        "validation_recall": valid_metrics["recall"],
        "validation_f1": valid_metrics["f1"],
        "validation_roc_auc": valid_metrics["roc_auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
    }

    live_row = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "date": dataset.index[-1].date().isoformat(),
                "stock_a": stock_a,
                "stock_b": stock_b,
                "zscore": z_now,
                "rf_probability": live_prob,
                "baseline_action": baseline_action,
                "final_action": final_action,
                "threshold": threshold,
                "validation_f1": valid_metrics["f1"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        ]
    )

    return meta, live_row


def main() -> None:
    prices = pd.read_csv(DATA_DIR / "nifty_prices_clean.csv", index_col=0, parse_dates=True)
    pairs = pd.read_csv(DATA_DIR / "selected_pairs.csv")

    if pairs.empty:
        raise ValueError("selected_pairs.csv is empty. Run pair selection first.")

    all_tickers = sorted(set(prices.columns.tolist()) | set(pairs["stock_a"].tolist()) | set(pairs["stock_b"].tolist()))
    prices = append_latest_prices(prices, all_tickers)

    registry_rows: list[dict[str, object]] = []
    live_rows: list[pd.DataFrame] = []
    threshold = 0.55

    for _, pair_row in pairs.iterrows():
        result = train_one_pair(prices, pair_row, threshold=threshold)
        if result is None:
            continue
        meta, live_row = result
        registry_rows.append(meta)
        live_rows.append(live_row)

    if not registry_rows:
        raise ValueError("No pair models were trained. Check selected pairs and data splits.")

    registry = pd.DataFrame(registry_rows).sort_values(
        ["validation_f1", "validation_roc_auc", "test_f1"],
        ascending=[False, False, False],
    )

    registry_file = MODEL_DIR / "pair_model_registry.csv"
    registry_json = MODEL_DIR / "pair_model_registry.json"
    registry.to_csv(registry_file, index=False)
    registry_json.write_text(registry.to_json(orient="records", indent=2), encoding="utf-8")

    best = registry.iloc[0]
    best_live = pd.DataFrame([r.iloc[0] for r in live_rows if r.iloc[0]["model_name"] == best["model_name"]])
    if best_live.empty:
        best_live = pd.DataFrame([best["model_name"]])

    live_out = pd.DataFrame([
        {
            "date": best["dataset_last_date"],
            "stock_a": best["stock_a"],
            "stock_b": best["stock_b"],
            "zscore": best["live_zscore"],
            "rf_probability": best["live_probability"],
            "baseline_action": best["baseline_action"],
            "final_action": best["final_action"],
            "threshold": best["threshold"],
            "model_name": best["model_name"],
            "validation_f1": best["validation_f1"],
            "test_f1": best["test_f1"],
            "test_roc_auc": best["test_roc_auc"],
        }
    ])

    live_file = DATA_DIR / "live_prediction.csv"
    live_out.to_csv(live_file, index=False)

    # also keep a simple best-model artifact name for direct loading
    best_model_path = Path(str(best["model_path"]))
    best_model_copy = MODEL_DIR / "rf_pair_model.joblib"
    if best_model_path.exists():
        joblib.dump(joblib.load(best_model_path), best_model_copy)

    meta_file = MODEL_DIR / "rf_pair_model_meta.json"
    meta_file.write_text(json.dumps(best.to_dict(), indent=2, default=str), encoding="utf-8")

    print("Model training complete")
    print(f"Saved registry: {registry_file}")
    print(f"Saved registry json: {registry_json}")
    print(f"Saved best model: {best_model_copy}")
    print(f"Saved best model metadata: {meta_file}")
    print(f"Saved live prediction: {live_file}")
    print("Best model metrics:")
    print(json.dumps({
        "validation_f1": float(best["validation_f1"]),
        "validation_roc_auc": float(best["validation_roc_auc"]),
        "test_f1": float(best["test_f1"]),
        "test_roc_auc": float(best["test_roc_auc"]),
    }, indent=2))
    print("Live decision:")
    print(live_out.to_string(index=False))


if __name__ == "__main__":
    main()
