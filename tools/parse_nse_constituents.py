"""Source: NSE Indices monthly archives
(https://www.niftyindices.com/Indices_-_Market_Capitalisation_and_Weightage/indices_dataMonYYYY.zip).
Output feeds data/nifty200_snapshots_2013_2022.csv.

Parse NIFTY 200 constituent PDFs inside NSE monthly archives. Usage: python -I parse_monthly.py <zip_dir> <out_csv>"""
import io, os, re, sys, zipfile
import pdfplumber
import pandas as pd

zdir, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(os.listdir(zdir)):
    if not f.endswith(".zip"):
        continue
    z = zipfile.ZipFile(os.path.join(zdir, f))
    names = [n for n in z.namelist() if re.search(r"(NIFTY|CNX)[ _]?200[_ ]", n, re.I) and n.lower().endswith(".pdf")
             and not re.search(r"(alpha|quality|momentum|value|low|equal)", n, re.I)]
    if not names:
        print("no N200 pdf in", f, z.namelist()[:6]); continue
    with pdfplumber.open(io.BytesIO(z.read(names[0]))) as pdf:
        txt = "\n".join(p.extract_text() or "" for p in pdf.pages)
    asof = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", txt)
    syms = []
    for line in txt.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9&\-]{1,15})\s+(.*\S)\s+([\d,]+\.\d+)\s+([\d,]+)\s+(\d+\.\d+)$", line.strip())
        if m:
            syms.append((m.group(1), float(m.group(5)), m.group(2)))
    for s, w, txt2 in syms:
        rows.append({"archive": f, "as_of": asof.group(1) if asof else None, "symbol": s, "weight_pct": w, "name_industry": txt2})
    print(f, names[0], asof.group(1) if asof else "?", len(syms), "weights sum", round(sum(x[1] for x in syms), 1))
pd.DataFrame(rows).to_csv(out, index=False)
