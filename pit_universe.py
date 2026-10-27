"""
Point-in-time (PIT) Nifty 200 membership, reconstructed from NSE Indices' semi-annual review
press releases (data/nifty200_index_changes.csv, parsed from the "Nifty 200" sections of the
"Replacements in indices" releases, Sep 2022 .. Sep 2026).

membership(d) starts from today's constituent list and undoes every review that became
effective after d. Check: the reconstruction has exactly 200 members at every review from
Mar 2025 onward (the out-of-sample window). Before Mar 2024 it shows 202, i.e. at least one
ad-hoc change (merger / suspension) is not in the semi-annual files, so dates before
2024-03-28 are flagged as approximate and are not used.

The PIT universe removes the look-ahead of using TODAY's members during the test window:
stocks that were in the index in Jan 2025 but dropped out later (often after falling) are
included again, and later additions are excluded until they actually joined.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from universe import DATA_DIR, load_universe

CHANGES_FILE = DATA_DIR / "nifty200_index_changes.csv"
N500_FILE = DATA_DIR / "nifty500_constituents.csv"
EXTRA_PRICES_FILE = DATA_DIR / "nifty200_pit_extra_prices.csv"
RELIABLE_FROM = pd.Timestamp("2024-03-28")

# Industry for former members that are no longer in the Nifty 500 list (NSE industry classes).
MANUAL_INDUSTRY = {
    "ABFRL": "Consumer Services", "APLLTD": "Healthcare", "GUJGASLTD": "Oil Gas & Consumable Fuels",
    "ICICIPRULI": "Financial Services", "METROPOLIS": "Healthcare", "MINDTREE": "Information Technology",
    "PEL": "Financial Services", "PGHH": "Fast Moving Consumer Goods",
}


SNAPSHOTS_FILE = DATA_DIR / "nifty200_snapshots_2013_2022.csv"
ALL_EXTRA_PRICES_FILE = DATA_DIR / "nifty200_pit_all_extra_prices.csv"
SNAPSHOT_UNTIL = pd.Timestamp("2022-03-31")

# Old NSE symbol -> current symbol (renames / name changes; Yahoo only keeps the current one).
RENAMES = {
    "ADANIGAS": "ATGL", "ADANITRANS": "ADANIENSOL", "AMARAJABAT": "ARE&M", "CADILAHC": "ZYDUSLIFE",
    "CASTROL": "CASTROLIND", "CROMPGREAV": "CGPOWER", "DALMIABHA": "DALBHARAT", "IBULHSGFIN": "SAMMAANCAP",
    "IDFCBANK": "IDFCFIRSTB", "INFRATEL": "INDUSTOWER", "L&TFH": "LTF", "LTI": "LTM", "MCDOWELL-N": "UNITDSPR",
    "MOTHERSUMI": "MOTHERSON", "NIITTECH": "COFORGE", "SRTRANSFIN": "SHRIRAMFIN", "TATAGLOBAL": "TATACONSUM", "LTIM": "LTM",
    "ZOMATO": "ETERNAL", "STRTECH": "STLTECH", "MAX": "MFSL", "JUBILANT": "JUBLPHARMA", "RNAM": "NAM-INDIA",
    "WELSPUNIND": "WELSPUNLIV", "IBVENTURES": "DHANI", "GMRINFRA": "GMRAIRPORT", "TATAMOTORS": "TMPV",
}


def canonical(sym: str) -> str:
    return RENAMES.get(sym, sym)


def load_changes() -> pd.DataFrame:
    ch = pd.read_csv(CHANGES_FILE, parse_dates=["effective_date"])
    return ch[ch["action"].isin(["include", "exclude"])]


def membership(date: str | pd.Timestamp, current: set[str] | None = None) -> set[str]:
    """NSE symbols in the Nifty 200 at the close of `date` (changes take effect on their effective date)."""
    d = pd.Timestamp(date)
    if d < RELIABLE_FROM:
        raise ValueError(f"PIT membership is only reliable from {RELIABLE_FROM.date()}")
    if current is None:
        current = set(pd.read_csv(DATA_DIR / "nifty200_constituents.csv")["Symbol"])
    mem = set(current)
    ch = load_changes()
    for eff in sorted(ch["effective_date"].unique(), reverse=True):
        if pd.Timestamp(eff) <= d:
            break
        g = ch[ch["effective_date"] == eff]
        mem = (mem - set(g.loc[g["action"] == "include", "symbol"])) | set(g.loc[g["action"] == "exclude", "symbol"])
    return mem


def industry_map() -> dict[str, str]:
    n500 = pd.read_csv(N500_FILE)
    out = dict(zip(n500["Symbol"], n500["Industry"]))
    out.update(MANUAL_INDUSTRY)
    cur = pd.read_csv(DATA_DIR / "nifty200_constituents.csv")
    out.update(dict(zip(cur["Symbol"], cur["Industry"])))
    return out


def load_pit_universe(start: str = "2025-01-01") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (prices, market, meta) where prices also covers every stock that was a member at any time
    from `start` on (downloading former members once and caching them), and meta has industries.
    """
    prices, market, meta = load_universe()
    dates = [pd.Timestamp(start)] + [pd.Timestamp(e) for e in load_changes()["effective_date"].unique()
                                     if pd.Timestamp(e) > pd.Timestamp(start)]
    ever = set().union(*(membership(d) for d in dates))
    have = {t.replace(".NS", "") for t in prices.columns}
    missing = sorted(ever - have)
    if missing:
        if EXTRA_PRICES_FILE.exists():
            extra = pd.read_csv(EXTRA_PRICES_FILE, index_col=0, parse_dates=True)
        else:
            raw = yf.download([f"{s}.NS" for s in missing], start="2012-01-01", auto_adjust=True, progress=False,
                              group_by="column")
            extra = raw["Close"] if "Close" in raw else raw
            extra.index = pd.to_datetime(extra.index).tz_localize(None)
            extra = extra.dropna(how="all", axis=1)
            extra.to_csv(EXTRA_PRICES_FILE, index_label="Date")
        extra = extra.reindex(prices.index).ffill(limit=5)
        prices = pd.concat([prices, extra[[c for c in extra.columns if c not in prices.columns]]], axis=1)
    ind = industry_map()
    meta = pd.DataFrame({"ticker": prices.columns, "symbol": [c.replace(".NS", "") for c in prices.columns]})
    meta["industry"] = meta["symbol"].map(ind)
    meta["company"] = meta["symbol"]
    return prices, market, meta


if __name__ == "__main__":
    for d in ["2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01"]:
        print(d, len(membership(d)))
    p, m, meta = load_pit_universe()
    print(p.shape, meta["industry"].isna().sum(), "without industry")


# ----------------------------------------------------------------------------- full history (2013 .. today)

_OLD_INDUSTRY_KEYWORDS = [
    (("BANK", "FINANC", "INSURANCE", "NBFC"), "Financial Services"),
    (("PHARMA", "HEALTH"), "Healthcare"),
    (("SOFTWARE", "COMPUTERS"), "Information Technology"),
    (("AUTO",), "Automobile and Auto Components"),
    (("CEMENT",), "Construction Materials"),
    (("CONSTRUCTION", "INFRA"), "Construction"),
    (("METAL", "MINING", "STEEL"), "Metals & Mining"),
    (("POWER",), "Power"),
    (("OIL", "GAS", "ENERGY", "PETRO"), "Oil Gas & Consumable Fuels"),
    (("CONSUMER", "FMCG", "FOOD", "BEVERAGE", "TOBACCO"), "Fast Moving Consumer Goods"),
    (("MEDIA", "ENTERTAINMENT"), "Media Entertainment & Publication"),
    (("TELECOM",), "Telecommunication"),
    (("FERTILISER", "PESTICIDE", "CHEMICAL"), "Chemicals"),
    (("TEXTILE",), "Textiles"),
    (("INDUSTRIAL MANUFACTURING", "CAPITAL GOODS", "ELECTRICAL"), "Capital Goods"),
    (("REALTY", "REAL ESTATE"), "Realty"),
    (("SERVICES",), "Services"),
]


def _industry_from_snapshot_text(text: str) -> str:
    """Map the industry tail of a constituent-PDF row ('Lupin Ltd. PHARMA') onto today's NSE industry names."""
    tail = str(text).split("Ltd.")[-1].split("Limited")[-1].upper()
    if tail.strip() in ("IT", " IT"):
        return "Information Technology"
    for keys, name in _OLD_INDUSTRY_KEYWORDS:
        if any(k in tail for k in keys):
            return name
    if tail.strip().endswith(" IT") or tail.strip() == "IT":
        return "Information Technology"
    return "Diversified"


def load_snapshots() -> pd.DataFrame:
    snap = pd.read_csv(SNAPSHOTS_FILE, parse_dates=["as_of"])
    snap["canonical"] = snap["symbol"].map(canonical)
    return snap


def membership_any(date: str | pd.Timestamp) -> set[str]:
    """
    Canonical NSE symbols in the Nifty 200 at the close of `date`, for any date from Dec 2013:
    - up to 2022-03-31: the latest official month-end constituent list (NSE Indices archives)
    - 2022-04 .. 2023-03: the Mar-2022 list rolled forward through the Sep-2022 review
    - 2023-04 .. 2024-03-27: today's list rolled back (approximate: 202 names, one ad-hoc change missing)
    - from 2024-03-28: today's list rolled back through the semi-annual reviews (exact)
    """
    d = pd.Timestamp(date)
    if d >= RELIABLE_FROM:
        return membership(d)
    if d >= pd.Timestamp("2023-03-31"):
        return _membership_backward(d)
    snap = load_snapshots()
    if d <= SNAPSHOT_UNTIL:
        asof = snap.loc[snap["as_of"] <= d, "as_of"].max()
        if pd.isna(asof):
            raise ValueError(f"no constituent snapshot on or before {d.date()}")
        return set(snap.loc[snap["as_of"] == asof, "canonical"])
    mem = set(snap.loc[snap["as_of"] == SNAPSHOT_UNTIL, "canonical"])
    ch = load_changes()
    for eff in sorted(ch["effective_date"].unique()):
        if pd.Timestamp(eff) > d:
            break
        g = ch[ch["effective_date"] == eff]
        mem = (mem - set(g.loc[g["action"] == "exclude", "symbol"].map(canonical))) | set(g.loc[g["action"] == "include", "symbol"].map(canonical))
    return mem


def _membership_backward(d: pd.Timestamp) -> set[str]:
    mem = set(pd.read_csv(DATA_DIR / "nifty200_constituents.csv")["Symbol"])
    ch = load_changes()
    for eff in sorted(ch["effective_date"].unique(), reverse=True):
        if pd.Timestamp(eff) <= d:
            break
        g = ch[ch["effective_date"] == eff]
        mem = (mem - set(g.loc[g["action"] == "include", "symbol"])) | set(g.loc[g["action"] == "exclude", "symbol"])
    return {canonical(s) for s in mem}


def checkpoints() -> list[pd.Timestamp]:
    """Dates on which membership can change (snapshot dates, then review effective dates)."""
    snap_dates = sorted(load_snapshots()["as_of"].unique())
    eff = sorted(pd.Timestamp(e) for e in load_changes()["effective_date"].unique() if pd.Timestamp(e) > SNAPSHOT_UNTIL)
    return [pd.Timestamp(x) for x in snap_dates] + eff


def member_mask(index: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """Daily boolean frame: True where the stock was a Nifty 200 member on that day."""
    syms = [t.replace(".NS", "") for t in tickers]
    mask = pd.DataFrame(False, index=index, columns=tickers)
    cps = checkpoints()
    for i, cp in enumerate(cps):
        nxt = cps[i + 1] if i + 1 < len(cps) else index[-1] + pd.Timedelta(days=1)
        mem = membership_any(cp)
        rows = (index > cp) & (index <= nxt) if i > 0 else (index <= nxt)
        cols = [t for t, s in zip(tickers, syms) if s in mem]
        mask.loc[rows, cols] = True
    return mask


def load_full_pit_universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    (prices, market, meta, missing): prices for every Nifty 200 member since Dec 2013 that Yahoo still
    carries (current + former members, renames mapped to today's symbols), industries for all of them,
    and the list of historical members with no price data (delisted / merged -- residual survivorship).
    """
    prices, market, _ = load_pit_universe("2025-01-01")
    if ALL_EXTRA_PRICES_FILE.exists():
        extra = pd.read_csv(ALL_EXTRA_PRICES_FILE, index_col=0, parse_dates=True).reindex(prices.index).ffill(limit=5)
        prices = pd.concat([prices, extra[[c for c in extra.columns if c not in prices.columns]]], axis=1)
    snap = load_snapshots()
    ever = set(snap["canonical"]) | {canonical(s) for s in load_changes()["symbol"]}
    have = {c.replace(".NS", "") for c in prices.columns}
    missing = sorted(ever - have)
    ind = industry_map()
    last_txt = snap.sort_values("as_of").groupby("canonical")["name_industry"].last()
    meta = pd.DataFrame({"ticker": prices.columns, "symbol": [c.replace(".NS", "") for c in prices.columns]})
    meta["industry"] = [ind.get(s) or _industry_from_snapshot_text(last_txt.get(s, "")) for s in meta["symbol"]]
    meta["company"] = meta["symbol"]
    return prices, market, meta, missing
