"""
Nifty 200 Total Return Index (TRI) loader.

The TRI reinvests dividends, so it is the fair benchmark for any strategy whose own returns
include dividends (Yahoo's adjusted closes used for the stocks do). Source: NSE Indices'
historical-data endpoint (the same call the open-source `nsepython` package makes), fetched
year by year and cached to data/nifty200_tri.csv.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import requests

from universe import DATA_DIR

TRI_FILE = DATA_DIR / "nifty200_tri.csv"
URL = "https://niftyindices.com/Backpage.aspx/getTotalReturnIndexString"
HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Content-Type": "application/json; charset=UTF-8",
    "Origin": "https://niftyindices.com",
    "Referer": "https://niftyindices.com/reports/historical-data",
}


def _fetch(name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cinfo = "{'name':'%s','startDate':'%s','endDate':'%s','indexName':'%s'}" % (
        name, start.strftime("%d-%b-%Y"), end.strftime("%d-%b-%Y"), name)
    resp = requests.post(URL, headers=HEADERS, json={"cinfo": cinfo}, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame.from_records(json.loads(resp.json()["d"]))


def download_tri(name: str = "NIFTY 200", start: str = "2012-01-01", end: str | None = None) -> pd.Series:
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    parts = []
    for y in range(pd.Timestamp(start).year, end_ts.year + 1):
        a = max(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(start))
        b = min(pd.Timestamp(f"{y}-12-31"), end_ts)
        df = _fetch(name, a, b)
        if not df.empty:
            parts.append(df)
        time.sleep(0.5)
    raw = pd.concat(parts, ignore_index=True)
    date_col = next(c for c in raw.columns if "date" in c.lower())
    val_col = next(c for c in raw.columns if "total" in c.lower() or "tri" in c.lower())
    s = pd.Series(pd.to_numeric(raw[val_col].astype(str).str.replace(",", ""), errors="coerce").to_numpy(),
                  index=pd.to_datetime(raw[date_col], format="mixed", dayfirst=True), name="tri")
    s = s[~s.index.duplicated()].sort_index().dropna()
    s.to_csv(TRI_FILE, index_label="Date")
    return s


def load_tri() -> pd.Series:
    if not TRI_FILE.exists():
        return download_tri()
    return pd.read_csv(TRI_FILE, index_col=0, parse_dates=True)["tri"]


if __name__ == "__main__":
    tri = download_tri()
    print(tri.head(), tri.tail(), len(tri), sep="\n")
