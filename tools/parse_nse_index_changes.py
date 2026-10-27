"""Source: NSE Indices 'Replacements in indices' press releases
(https://www.niftyindices.com/Press_Release/ind_prsDDMMYYYY.pdf). Output feeds data/nifty200_index_changes.csv.

Parse NSE Indices press releases for NIFTY 200 replacements. Usage: python -I parse_prs.py <pdf_dir> <out_csv>"""
import os, re, sys, hashlib
import pdfplumber
import pandas as pd

pdf_dir, out_csv = sys.argv[1], sys.argv[2]
HDR = r"(?m)^\s*(?:[a-z]{1,2}|\d{1,2})\)\s*Nifty 200\s*$"
seen, rows = set(), []
for f in sorted(os.listdir(pdf_dir)):
    path = os.path.join(pdf_dir, f)
    h = hashlib.md5(open(path, "rb").read()).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    try:
        with pdfplumber.open(path) as pdf:
            txt = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        print("unreadable", f, e); continue
    flat = re.sub(r"\s+", " ", txt)
    eff = re.search(r"effective from ([A-Z][a-z]+ \d{1,2}, \d{4})", flat)
    eff = eff.group(1) if eff else None
    # every section header like "x) Nifty 200" (exact name, not Nifty 200 Momentum etc.)
    for m in re.finditer(HDR, txt, flags=re.I):
        start = m.end()
        nxt = re.search(r"(?m)^\s*(?:[a-z]{1,2}|\d{1,2})\)\s*\S", txt[start:])
        sec = txt[start:start + nxt.start()] if nxt else txt[start:start + 4000]
        mode = None
        for line in sec.splitlines():
            l = line.strip()
            if re.search(r"being excluded", l, re.I): mode = "exclude"; continue
            if re.search(r"being included", l, re.I): mode = "include"; continue
            mm = re.match(r"^\d+\s+(.+?)\s+([A-Z0-9&\-]+)$", l)
            if mode and mm and not l.startswith("Sr."):
                rows.append({"file": f, "effective": eff, "action": mode, "company": mm.group(1), "symbol": mm.group(2)})
    if re.search(HDR, txt, flags=re.I) is None and "Nifty 200 " in flat and re.search(r"Nifty 200[ ,]", flat):
        # ad-hoc releases often list affected indices inline, e.g. "...from NIFTY 100, NIFTY 200, NIFTY 500 ..."
        rows.append({"file": f, "effective": eff, "action": "ADHOC_MENTION", "company": flat[:300], "symbol": ""})
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(len(rows), "rows from", len(seen), "unique pdfs")
