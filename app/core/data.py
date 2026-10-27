"""Cached loaders. Research outputs are read where the research scripts wrote them; the
dashboard-only bundle (app/build_data.py) lives in app/data/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent
BUNDLE = APP_DIR / "data"
RESEARCH = ROOT / "research"
DATA = ROOT / "data"

TEST_START = pd.Timestamp("2025-01-01")
TRAIN = ("2015-01-01", "2024-12-31")


@st.cache_data(show_spinner=False)
def meta() -> dict:
    return json.loads((BUNDLE / "meta.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def research_csv(rel: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(RESEARCH / rel, **kwargs)


@st.cache_data(show_spinner=False)
def data_csv(rel: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA / rel, **kwargs)


@st.cache_data(show_spinner=False)
def research_json(rel: str) -> dict:
    return json.loads((RESEARCH / rel).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def bundle_csv(rel: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(BUNDLE / rel, **kwargs)


@st.cache_data(show_spinner=False)
def daily() -> pd.DataFrame:
    return pd.read_parquet(BUNDLE / "final_daily.parquet")


@st.cache_data(show_spinner=False)
def regime_daily() -> pd.DataFrame:
    return pd.read_parquet(BUNDLE / "regime_daily.parquet")


@st.cache_data(show_spinner=False)
def pair_prices() -> pd.DataFrame:
    return pd.read_parquet(BUNDLE / "pair_prices.parquet").astype("float64")


@st.cache_data(show_spinner=False)
def pairs_by_period() -> pd.DataFrame:
    return pd.read_csv(BUNDLE / "final_pairs_by_period.csv", parse_dates=["period_start", "period_end"])


@st.cache_data(show_spinner=False)
def trades() -> pd.DataFrame:
    t = pd.read_csv(BUNDLE / "final_trades.csv", parse_dates=["entry_date", "exit_date"])
    t["period"] = np.where(t["exit_date"] >= TEST_START, "Out-of-sample", "Training")
    return t


@st.cache_data(show_spinner=False)
def stocks() -> pd.DataFrame:
    return pd.read_csv(BUNDLE / "stocks.csv")


@st.cache_data(show_spinner=False)
def scorecard() -> pd.DataFrame:
    return research_csv("pit_pairs/scorecard_vs_tri.csv")


@st.cache_data(show_spinner=False)
def equity_full() -> pd.DataFrame:
    return research_csv("pit_pairs/full_equity.csv", index_col=0, parse_dates=True)


@st.cache_data(show_spinner=False)
def equity_oos() -> pd.DataFrame:
    return research_csv("pit_pairs/oos_equity.csv", index_col=0, parse_dates=True)


def drawdown(eq: pd.Series) -> pd.Series:
    eq = eq.dropna()
    return eq / eq.cummax() - 1


def company(symbol: str) -> str:
    s = stocks().set_index("symbol")["company"]
    return str(s.get(symbol, symbol))


def pair_signal(a: str, b: str, lookback: int = 20, pct_window: int = 126) -> pd.DataFrame:
    """The final strategy's signal for one pair, recomputed from prices (build_data.py checks it matches the engine)."""
    px = pair_prices()
    lr = np.log(px[a]) - np.log(px[b])
    rs = lr - lr.shift(lookback)
    rs_pct = rs.rolling(pct_window, min_periods=pct_window // 2).rank(pct=True)
    return pd.DataFrame({"price_a": px[a], "price_b": px[b], "log_ratio": lr, "rs": rs, "rs_pct": rs_pct})
