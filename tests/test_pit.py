"""Tests for point-in-time Nifty 200 membership and the TRI benchmark helper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final_oos import nifty200_tri, scorecard  # noqa: E402
from pit_universe import (  # noqa: E402
    _industry_from_snapshot_text, canonical, load_changes, member_mask, membership, membership_any,
)


@pytest.mark.parametrize("date", ["2014-01-01", "2016-01-01", "2020-07-01", "2022-07-01", "2023-01-01",
                                  "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01"])
def test_membership_size(date):
    n = len(membership_any(date))
    assert 199 <= n <= 202  # DVR share classes (201) and the approximate 2023-24 stretch (202) are known exceptions


def test_reviews_are_applied_on_effective_date():
    ch = load_changes()
    eff = pd.Timestamp("2025-03-28")
    g = ch[ch["effective_date"] == eff]
    before, after = membership(eff - pd.Timedelta(days=1)), membership(eff)
    assert set(g.loc[g["action"] == "include", "symbol"]) <= after
    assert not (set(g.loc[g["action"] == "include", "symbol"]) & before)
    assert set(g.loc[g["action"] == "exclude", "symbol"]) <= before


def test_reliable_window_guard():
    with pytest.raises(ValueError):
        membership("2020-01-01")


def test_canonical_renames():
    assert canonical("ZOMATO") == "ETERNAL" and canonical("LTI") == "LTM" and canonical("INFY") == "INFY"


def test_forward_and_backward_reconstructions_differ_by_one_review():
    # Jan-2023 is only reachable forward (Mar-2022 list + Sep-2022 review); rolling today's list back to it would
    # need the missing Mar-2023 review. The gap must look like ONE review: balanced in/out, about a dozen names each.
    from pit_universe import _membership_backward
    fwd = membership_any("2023-01-15")
    bwd = _membership_backward(pd.Timestamp("2023-01-15"))
    only_f, only_b = fwd - bwd, bwd - fwd
    assert abs(len(only_f) - len(only_b)) <= 3 and len(only_f) <= 15
    assert "HDFC" in only_f  # merged into HDFC Bank in Jul-2023 -- an ad-hoc change, not a review


def test_member_mask_switches_on_review_date():
    idx = pd.bdate_range("2025-03-20", "2025-04-04")
    inc = load_changes()
    sym = inc[(inc["effective_date"] == pd.Timestamp("2025-03-28")) & (inc["action"] == "include")]["symbol"].iloc[0]
    m = member_mask(idx, [f"{sym}.NS"])
    assert not m.loc[:"2025-03-27"].any().any() and m.loc["2025-04-01":].all().all()


def test_industry_mapping_old_taxonomy():
    assert _industry_from_snapshot_text("Lupin Ltd. PHARMA") == "Healthcare"
    assert _industry_from_snapshot_text("ICICI Bank Ltd. BANKS") == "Financial Services"
    assert _industry_from_snapshot_text("Larsen & Toubro Infotech Ltd. IT") == "Information Technology"


def test_tri_approximation_and_scorecard():
    idx = pd.bdate_range("2025-01-01", periods=253)
    bench = pd.Series(100.0, index=idx)
    tri, note = nifty200_tri(bench) if not Path("data/nifty200_tri.csv").exists() else (bench * 1.0, "official")
    if note.startswith("approximation"):
        assert tri.iloc[-1] / tri.iloc[0] - 1 == pytest.approx(0.015, rel=0.01)
    r = pd.Series(0.0004, index=idx)
    r.iloc[::7] = -0.0002
    sc = scorecard(r, bench * np.linspace(1, 0.97, len(idx)), "2025-01-01", None)
    assert sc["1_higher_return"] and sc["3_lower_vol"] is not None
