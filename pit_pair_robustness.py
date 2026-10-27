"""
Robustness of the frozen point-in-time pair strategy (pit_pair_system.py) on the out-of-sample window.
Diagnostics only -- nothing here feeds back into the selection.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from final_oos import DIV_YIELD, nifty200_tri, scorecard
from kaufman_factors import CompositeRegime, classify_composite, composite_scores, percentile_normalize
from kaufman_system import PAIR_GRID, pair_config
from pit_pair_system import OUT, PAIR_MODES, TEST_START, PITSystem

CRIT = ["1_higher_return", "2_higher_sharpe", "3_lower_vol", "4_lower_maxdd"]


def main() -> None:
    s = PITSystem()
    cfg_json = json.loads((OUT / "final_config.json").read_text())
    rp = CompositeRegime(**cfg_json["regime"])
    regime = classify_composite(composite_scores(percentile_normalize(s.raw, rp.pct_window), cfg_json["factors"]), rp)
    tri, _ = nifty200_tri(s.bench)

    def params(row) -> dict:
        p = {k: row[k] for k in PAIR_GRID}
        p["lookback"], p["pct_window"], p["formation_days"] = int(p["lookback"]), int(p["pct_window"]), int(p["formation_days"])
        p["entry_pct"] = float(p["entry_pct"])
        p["exit_on_median"] = str(p["exit_on_median"]) == "True"
        return p

    best = pd.read_csv(OUT / "stage2_pair_grid.csv").iloc[0]
    cfg = pair_config(params(best), PAIR_MODES[best["mode"]])
    rows = []
    for name, c in {"frozen strategy": cfg, "slippage 10 bps/side": replace(cfg, slippage_bps=10.0),
                    "slippage 20 bps/side": replace(cfg, slippage_bps=20.0),
                    "cash delivery + 8% SLB borrow": replace(cfg, cost_model="cash")}.items():
        r = s.pair_returns(c, regime)["equity"]["equity"].pct_change().fillna(0.0)
        rows.append({"check": name, **scorecard(r, tri, TEST_START, None)})
        if name == "frozen strategy":
            frozen_r = r
    # stricter benchmark: TRI with a 2.5%/yr dividend yield (well above the Nifty 200's ~1.2-1.5%)
    strict = s.bench * np.exp(np.log1p(0.025) / 252 * np.arange(len(s.bench)))
    rows.append({"check": "benchmark TRI at 2.5%/yr dividends", **scorecard(frozen_r, strict, TEST_START, None)})
    rob = pd.DataFrame(rows)
    rob["criteria_met"] = rob[CRIT].sum(axis=1)
    rob.to_csv(OUT / "robustness.csv", index=False)

    # how many of the training top-N (by plateau score) also meet all four out of sample?
    grid = pd.read_csv(OUT / "stage2_pair_grid.csv")
    nb = []
    for rank, (_, row) in enumerate(grid.head(20).iterrows(), start=1):
        c = pair_config(params(row), PAIR_MODES[row["mode"]])
        r = s.pair_returns(c, regime)["equity"]["equity"].pct_change().fillna(0.0)
        sc = scorecard(r, tri, TEST_START, None)
        nb.append({"train_rank": rank, "mode": row["mode"], "label": row["label"], "train_sharpe": row["train_sharpe"],
                   "oos_return": sc["total_return"], "oos_sharpe": sc["sharpe"], "oos_vol": sc["vol"], "oos_maxdd": sc["max_drawdown"],
                   "oos_criteria_met": sum(sc[k] for k in CRIT)})
    nb = pd.DataFrame(nb)
    nb.to_csv(OUT / "top20_oos_diagnostic.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(rob[["check", "total_return", "sharpe", "vol", "max_drawdown", "tri_total_return", "tri_sharpe", "criteria_met"]].round(3).to_string(index=False))
        print(nb.round(3).to_string(index=False))
        print("top-20 meeting all four OOS:", int((nb["oos_criteria_met"] == 4).sum()), "| median OOS Sharpe", round(nb["oos_sharpe"].median(), 3))


if __name__ == "__main__":
    main()
