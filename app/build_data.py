"""
Export the data bundle the Streamlit dashboard (app/streamlit_app.py) reads.

The research scripts save summary tables but not everything a demo needs: the final strategy's
full pair schedule, its trade log, its daily book and the regime scores behind it. This script
re-runs the FROZEN final configuration (research/pit_pairs/final_config.json) -- nothing is
re-tuned -- checks that it reproduces research/pit_pairs/full_equity.csv, and writes compact
files to app/data/.

Run from the project root (needs the full research environment, not just the dashboard's):
    .venv\\Scripts\\python.exe app\\build_data.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from final_oos import nifty200_tri  # noqa: E402
from kaufman_factors import (  # noqa: E402
    CompositeRegime, classify_composite, composite_scores, factor_diagnostics, percentile_normalize, select_factors,
)
from kaufman_system import TRAIN, equity_from_returns, pair_config  # noqa: E402
from pit_pair_system import PAIR_MODES, PITSystem  # noqa: E402
from pit_universe import checkpoints, load_snapshots, membership_any  # noqa: E402
from rs_pairs import pair_key, side_cost_rates  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
CFG_FILE = ROOT / "research" / "pit_pairs" / "final_config.json"


def company_names(symbols: list[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    snap = load_snapshots().sort_values("as_of")
    for sym, txt in snap.groupby("canonical")["name_industry"].last().items():
        t = str(txt)
        for cut in ("Ltd.", "Limited"):
            if cut in t:
                t = t.split(cut)[0] + cut
                break
        names[sym] = t
    for f in ("nifty500_constituents.csv", "nifty200_constituents.csv"):
        df = pd.read_csv(ROOT / "data" / f)
        names.update(dict(zip(df["Symbol"], df["Company Name"])))
    return {s: names.get(s, s) for s in symbols}


def main() -> None:
    cfg = json.loads(CFG_FILE.read_text(encoding="utf-8"))
    s = PITSystem()

    # ---- regime: the frozen definition (window, thresholds, selected factors)
    rp = CompositeRegime(**cfg["regime"])
    pct = percentile_normalize(s.raw, rp.pct_window)
    scores = composite_scores(pct, cfg["factors"])
    regime = classify_composite(scores, rp)
    corr, use = factor_diagnostics(pct, s.market, *TRAIN)
    if select_factors(corr, use) != cfg["factors"]:
        raise SystemExit("factor selection does not reproduce final_config.json")
    corr.to_csv(OUT / "factor_correlation.csv")
    use.to_csv(OUT / "factor_usefulness.csv")

    # ---- pair strategy: the frozen parameters, run once over the whole history
    pp = cfg["pair_params"]
    p = {"signal": pp["signal"], "lookback": int(pp["lookback"]), "pct_window": int(pp["pct_window"]),
         "entry_pct": float(pp["entry_pct"]), "exit_on_median": str(pp["exit_on_median"]) == "True",
         "formation_days": int(pp["formation_days"])}
    pcfg = pair_config(p, PAIR_MODES[cfg["pair_mode"]])
    res = s.pair_returns(pcfg, regime)
    r = res["equity"]["equity"].pct_change().fillna(0.0)

    # ---- reproduction check against the research output
    ref = pd.read_csv(ROOT / "research" / "pit_pairs" / "full_equity.csv", index_col=0, parse_dates=True)["PIT pair strategy"]
    mine = equity_from_returns(r.loc[TRAIN[0]:])
    gap = float((mine.reindex(ref.index) - ref).abs().max())
    print(f"max |equity - research/pit_pairs/full_equity.csv| = {gap:.2e}")
    if not gap < 1e-9:
        raise SystemExit("re-run does not reproduce the research equity curve; refusing to export")

    tri, tri_note = nifty200_tri(s.bench)
    sched = s.schedule(p["formation_days"]).copy()
    trades = res["trades"].copy()
    syms = sorted({c.replace(".NS", "") for c in s.prices.columns})
    names = company_names(syms)
    industry = dict(zip(s.meta["symbol"], s.meta["industry"]))

    # pairs: one row per (formation period, pair)
    for col in ("stock_a", "stock_b"):
        sched[col] = sched[col].str.replace(".NS", "", regex=False)
    sched["pair"] = sched["stock_a"] + " / " + sched["stock_b"]
    sched.to_csv(OUT / "final_pairs_by_period.csv", index=False)

    # trades
    for col in ("stock_a", "stock_b"):
        trades[col] = trades[col].str.replace(".NS", "", regex=False)
    trades["pair"] = trades["stock_a"] + " / " + trades["stock_b"]
    trades["industry"] = trades["stock_a"].map(industry)
    trades["long_leg"] = np.where(trades["direction"] == 1, trades["stock_a"], trades["stock_b"])
    trades["short_leg"] = np.where(trades["direction"] == 1, trades["stock_b"], trades["stock_a"])
    trades["entry_rs_pct"] = trades["entry_score"] + 0.5
    trades = trades.drop(columns=["direction", "entry_score"])
    trades.to_csv(OUT / "final_trades.csv", index=False)

    # daily book
    eq = res["equity"].copy()
    eq["n_signals"] = res["n_signals"].reindex(eq.index)
    eq["benchmark"] = s.bench.reindex(eq.index)
    eq["tri"] = tri.reindex(eq.index)
    eq.index.name = "date"
    eq.to_parquet(OUT / "final_daily.parquet")

    # regime scores + every candidate factor's percentile (for the regime page)
    reg = pd.concat([scores, regime.rename("regime"), pct.add_prefix("pct_"),
                     s.market[["benchmark", "vix"]]], axis=1)
    reg.index.name = "date"
    reg.to_parquet(OUT / "regime_daily.parquet")

    # prices of every stock that ever appears in a final-strategy pair
    used = sorted(set(sched["stock_a"]) | set(sched["stock_b"]))
    px = s.prices[[f"{u}.NS" for u in used]].astype("float32")
    px.columns = used
    px.index.name = "date"
    px.to_parquet(OUT / "pair_prices.parquet")

    # point-in-time membership through time
    rows = []
    have = set(syms)
    for cp in checkpoints():
        mem = membership_any(cp)
        rows.append({"date": cp, "members": len(mem), "with_prices": len(mem & have), "without_prices": len(mem - have)})
    pd.DataFrame(rows).to_csv(OUT / "pit_membership_counts.csv", index=False)

    stocks = pd.DataFrame({"symbol": syms})
    stocks["company"] = stocks["symbol"].map(names)
    stocks["industry"] = stocks["symbol"].map(industry)
    current = set(pd.read_csv(ROOT / "data" / "nifty200_constituents.csv")["Symbol"])
    stocks["in_index_today"] = stocks["symbol"].isin(current)
    stocks["in_final_pairs"] = stocks["symbol"].isin(used)
    stocks.to_csv(OUT / "stocks.csv", index=False)

    # latest-day snapshot of the live pair list: the signal each pair shows today
    score, _, active, _ = s._sig[(pcfg.signal, pcfg.lookback, pcfg.pct_window, p["formation_days"])]
    last = score.index[-1]
    live = sched[sched["period_start"] == sched["period_start"].max()]
    snap = []
    for a, b, ind in live[["stock_a", "stock_b", "industry"]].itertuples(index=False):
        k = pair_key(f"{a}.NS", f"{b}.NS")
        snap.append({"stock_a": a, "stock_b": b, "industry": ind, "rs_pct": float(score.at[last, k] + 0.5),
                     "rs_pct_prev": float(score[k].iloc[-2] + 0.5), "active": bool(active.at[last, k])})
    pd.DataFrame(snap).to_csv(OUT / "live_signals.csv", index=False)

    # the app recomputes the signal for one pair from prices; make sure its formula matches
    k0 = pair_key(f"{live.iloc[0]['stock_a']}.NS", f"{live.iloc[0]['stock_b']}.NS")
    lr = np.log(px[live.iloc[0]["stock_a"]].astype(float)) - np.log(px[live.iloc[0]["stock_b"]].astype(float))
    rs = lr - lr.shift(p["lookback"])
    app_score = rs.rolling(p["pct_window"], min_periods=p["pct_window"] // 2).rank(pct=True) - 0.5
    diff = (app_score - score[k0]).abs().max()
    print(f"app-side signal formula vs engine (float32 prices): max |diff| = {diff:.4f}")

    # phase-1 Random Forest filters: feature importances (so the app does not need scikit-learn)
    reg_models = pd.read_csv(ROOT / "models" / "pair_model_registry.csv")
    imp = {}
    for name, cols in zip(reg_models["model_name"], reg_models["feature_cols"]):
        model = joblib.load(ROOT / "models" / "pair_models" / f"{name}.joblib")
        imp[name] = dict(zip(ast.literal_eval(cols), map(float, model.feature_importances_)))
    pd.DataFrame(imp).to_csv(OUT / "rf_feature_importance.csv", index_label="feature")

    meta = {
        "built": pd.Timestamp.now().isoformat(timespec="seconds"),
        "cost_rates_ex_slippage": {m: dict(zip(("buy", "sell", "annual_borrow"), side_cost_rates(m, 0.0)))
                                   for m in ("futures", "cash")},
        "data_start": str(s.prices.index[0].date()), "data_end": str(last.date()),
        "regime_today": regime.iloc[-1], "direction_today": float(scores["direction"].iloc[-1]),
        "volatility_today": float(scores["volatility"].iloc[-1]),
        "benchmark_note": tri_note, "n_stocks_with_prices": len(syms), "n_missing_history": len(s.missing),
        "missing_history": s.missing, "config": cfg, "trade_count": int(len(trades)),
        "equity_check_max_abs_diff": gap,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"wrote {len(sched)} pair-periods, {len(trades)} trades, {len(used)} stocks -> {OUT}")


if __name__ == "__main__":
    main()
