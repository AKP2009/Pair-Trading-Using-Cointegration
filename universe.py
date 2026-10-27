"""
Nifty 200 universe loader.

Downloads (and caches) adjusted daily closes for every Nifty 200
constituent, plus the Nifty 200 benchmark (^CNX200) and India VIX
(^INDIAVIX), from Yahoo Finance.

Constituents come from NSE's official list (data/nifty200_constituents.csv,
https://archives.nseindia.com/content/indices/ind_nifty200list.csv), whose
"Industry" column is used as the sector grouping for pair formation.

Survivorship-bias note: this uses *today's* Nifty 200 membership for the
whole history. Stocks that were dropped from the index (often after
underperforming) are missing, and recent IPOs only enter once they have
enough history. Pair trading is market-neutral so this bias is smaller
than for a long-only strategy, but it is not zero -- state it in the paper.

Outputs:
- data/nifty200_prices.csv     (Date x ticker adjusted close, NaN before listing)
- data/nifty200_market.csv     (Date x [benchmark, vix])
- data/nifty200_meta.csv       (ticker, symbol, company, industry)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CONSTITUENTS_FILE = DATA_DIR / "nifty200_constituents.csv"
PRICES_FILE = DATA_DIR / "nifty200_prices.csv"
MARKET_FILE = DATA_DIR / "nifty200_market.csv"
META_FILE = DATA_DIR / "nifty200_meta.csv"

START_DATE = "2012-01-01"
BENCHMARK_TICKER = "^CNX200"
VIX_TICKER = "^INDIAVIX"


def load_constituents() -> pd.DataFrame:
    df = pd.read_csv(CONSTITUENTS_FILE)
    df = df.rename(columns={"Company Name": "company", "Industry": "industry", "Symbol": "symbol"})
    df["ticker"] = df["symbol"].astype(str) + ".NS"
    return df[["ticker", "symbol", "company", "industry"]]


def _download_close(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def download_universe(start: str = START_DATE, end: str | None = None) -> None:
    meta = load_constituents()
    tickers = meta["ticker"].tolist()

    close = _download_close(tickers, start, end)
    close = close.dropna(how="all")
    missing = [t for t in tickers if t not in close.columns or close[t].notna().sum() == 0]
    if missing:
        print(f"No data for {len(missing)} ticker(s): {missing}")
    close = close[[t for t in tickers if t not in missing]]

    market = _download_close([BENCHMARK_TICKER, VIX_TICKER], start, end)
    market = market.rename(columns={BENCHMARK_TICKER: "benchmark", VIX_TICKER: "vix"})
    # Trading calendar = days the benchmark printed; VIX gaps are forward-filled.
    market = market.dropna(subset=["benchmark"])
    market["vix"] = market["vix"].ffill()

    close = close.reindex(market.index)
    # Forward-fill only short gaps (holidays/suspensions); never back-fill pre-listing history.
    close = close.ffill(limit=5)

    close.to_csv(PRICES_FILE, index_label="Date")
    market.to_csv(MARKET_FILE, index_label="Date")
    meta[~meta["ticker"].isin(missing)].to_csv(META_FILE, index=False)

    print(f"Saved {close.shape[1]} tickers x {close.shape[0]} days -> {PRICES_FILE}")
    print(f"Saved benchmark + VIX -> {MARKET_FILE}")


def load_universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (prices, market, meta) from the cached CSVs, downloading them first if absent."""
    if not (PRICES_FILE.exists() and MARKET_FILE.exists() and META_FILE.exists()):
        download_universe()
    prices = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    market = pd.read_csv(MARKET_FILE, index_col=0, parse_dates=True)
    meta = pd.read_csv(META_FILE)
    return prices, market, meta


if __name__ == "__main__":
    download_universe()
