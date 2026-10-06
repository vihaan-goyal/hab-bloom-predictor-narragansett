"""
Determine EMPIRICALLY which column is chlorophyll-a in ug/L:
  `Chlorophyll` (used by the bloom label) or `Corrected_Chlorophyll`.

Three independent lines of evidence, no guessing, no waiting on email:

  A. DISTRIBUTIONS vs known Long Island Sound values.
     Published LIS chlorophyll-a: roughly 1-30 ug/L typical, blooms 20-40+.
     Whichever column lives in that range is the one in ug/L.

  B. SATELLITE AS A RULER.
     MODIS L3 `chlor_a` is reported in mg m^-3, which is EXACTLY equal to ug/L.
     So the satellite is an independent measurement in KNOWN units. Regress each
     in-situ column against co-located satellite chl: the column in ug/L should
     have a slope near 1 and comparable magnitude.
     CAVEAT: MODIS overestimates chl in turbid Case-2 coastal water (western LIS
     is turbid), so expect satellite to run high. This is an order-of-magnitude
     test, which is all we need to settle a 4.7x question.

  C. ERDDAP METADATA (definitive).
     The source data came from an ERDDAP server, and ERDDAP carries a `units`
     attribute on every variable, set by the data provider. Go read it.

Run from repo root:
  python scripts/data_checks/identify_chl_units.py
"""

import json
import sys
import numpy as np
import polars as pl
from pathlib import Path

FINAL = Path("data/hab_features_final.csv")
MATCHED = Path("data/matched_labels.csv")
MODIS = Path("data/modis_station_daily.csv")

ERDDAPS = [
    "http://merlin.dms.uconn.edu:8080/erddap",
]

LIS_NOTE = """
  Reference (published Long Island Sound chlorophyll-a):
    typical open-Sound      ~ 2 - 10 ug/L
    western Sound / summer  ~ 5 - 20 ug/L
    bloom conditions        ~ 20 - 40+ ug/L
    winter-spring diatom blooms in the western Narrows can exceed 30 ug/L.
"""


def sect(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ---------------------------------------------------------------- A
sect("A. DISTRIBUTIONS -- which column looks like LIS chlorophyll-a?")
f = pl.scan_csv(FINAL, schema_overrides={"station_name": pl.String},
                infer_schema_length=50000, ignore_errors=True).select(
    "station_name", "date", "Chlorophyll", "Corrected_Chlorophyll").collect()

for c in ("Chlorophyll", "Corrected_Chlorophyll"):
    s = f[c].drop_nulls()
    if s.len() == 0:
        continue
    ps = [float(s.quantile(p)) for p in (0.10, 0.50, 0.90, 0.99)]
    print(f"  {c:<24} n={s.len():>9,}   "
          f"p10={ps[0]:6.2f}  med={ps[1]:6.2f}  p90={ps[2]:7.2f}  p99={ps[3]:7.2f}")
print(LIS_NOTE)

# ---------------------------------------------------------------- B
sect("B. SATELLITE RULER -- MODIS chlor_a is in mg/m3 == ug/L exactly")

sat = None
if MATCHED.exists():
    m = pl.scan_csv(MATCHED, infer_schema_length=10000, ignore_errors=True).collect()
    if {"insitu_chl", "satellite_chl"}.issubset(set(m.columns)):
        mm = m.filter(pl.col("insitu_chl").is_not_null()
                      & pl.col("satellite_chl").is_not_null()
                      & (pl.col("insitu_chl") > 0) & (pl.col("satellite_chl") > 0))
        if mm.height > 20:
            sat = mm
            iq = mm["insitu_chl"]
            sq = mm["satellite_chl"]
            print(f"  matched_labels.csv: {mm.height:,} co-located pairs")
            print(f"    insitu_chl     med={float(iq.median()):7.2f}  "
                  f"p90={float(iq.quantile(0.9)):7.2f}")
            print(f"    satellite_chl  med={float(sq.median()):7.2f}  "
                  f"p90={float(sq.quantile(0.9)):7.2f}")
            x = sq.to_numpy().astype(float)
            y = iq.to_numpy().astype(float)
            k = float((x * y).sum() / (x * x).sum())
            print(f"    best fit: insitu = {k:.3f} * satellite   "
                  f"(r={np.corrcoef(x, y)[0,1]:+.3f})")
            print(f"    ratio med(insitu)/med(satellite) = "
                  f"{float(iq.median())/float(sq.median()):.3f}")
            print("\n    NOTE: which column is `insitu_chl` built from? If it is the")
            print("    RAW Chlorophyll and it already sits near the satellite scale,")
            print("    then RAW is the ug/L column and Corrected is something else.")

# direct: join both in-situ columns to MODIS per station-date
if MODIS.exists():
    md = pl.scan_csv(MODIS, schema_overrides={"station_name": pl.String},
                     infer_schema_length=10000, ignore_errors=True).collect()
    satcol = next((c for c in md.columns
                   if "chl" in c.lower() and "valid" not in c.lower()), None)
    if satcol and "date" in md.columns:
        md = md.with_columns(
            pl.when(pl.col("station_name").str.contains(r"^\d+$"))
              .then(pl.col("station_name").str.zfill(2))
              .otherwise(pl.col("station_name")).alias("station_name"))
        ff = f.with_columns(
            pl.when(pl.col("station_name").str.contains(r"^\d+$"))
              .then(pl.col("station_name").str.zfill(2))
              .otherwise(pl.col("station_name")).alias("station_name"))
        j = ff.join(md.select("station_name", "date", satcol),
                    on=["station_name", "date"], how="inner").filter(
            pl.col(satcol).is_not_null() & (pl.col(satcol) > 0))
        print(f"\n  direct MODIS join: {j.height:,} station-date matches "
              f"(satellite col = '{satcol}')")
        s = j[satcol].to_numpy().astype(float)
        for c in ("Chlorophyll", "Corrected_Chlorophyll"):
            sub = j.filter(pl.col(c).is_not_null() & (pl.col(c) > 0))
            if sub.height < 20:
                print(f"    {c:<22} too few overlapping rows ({sub.height})")
                continue
            sv = sub[satcol].to_numpy().astype(float)
            iv = sub[c].to_numpy().astype(float)
            k = float((sv * iv).sum() / (sv * sv).sum())
            r = float(np.corrcoef(np.log(sv), np.log(iv))[0, 1])
            print(f"    {c:<22} n={sub.height:>6,}  med={np.median(iv):7.2f}  "
                  f"slope_vs_sat={k:6.3f}  r(log)={r:+.3f}")
        print(f"    {'MODIS satellite':<22} n={j.height:>6,}  "
              f"med={np.median(s):7.2f}  (known units: ug/L)")
        print("\n    The column whose magnitude and slope sit closest to the")
        print("    satellite is the one measured in ug/L. Remember MODIS reads")
        print("    HIGH in turbid water, so expect satellite > true.")

# ---------------------------------------------------------------- C
sect("C. ERDDAP METADATA -- the provider's own declared units (definitive)")
try:
    import requests
except ImportError:
    print("  requests not installed; skipping")
    sys.exit(0)

sess = requests.Session()
sess.headers.update({"User-Agent": "hab-bloom-predictor/1.0 (research)"})

found = False
for base in ERDDAPS:
    try:
        r = sess.get(f"{base}/info/index.json?itemsPerPage=1000", timeout=60)
        r.raise_for_status()
        tbl = r.json()["table"]
        cols = tbl["columnNames"]
        di = cols.index("Dataset ID")
        ids = [row[di] for row in tbl["rows"]]
    except Exception as exc:  # noqa: BLE001
        print(f"  {base}: could not list datasets ({exc})")
        continue

    print(f"  {base}: {len(ids)} datasets; scanning for chlorophyll variables...")
    for ds in ids:
        try:
            ri = sess.get(f"{base}/info/{ds}/index.json", timeout=30)
            if ri.status_code != 200:
                continue
            rows = ri.json()["table"]["rows"]
        except Exception:  # noqa: BLE001
            continue

        # variables whose name mentions chl
        chl_vars = {v for t, v, *_ in rows
                    if t == "variable" and "chl" in str(v).lower()}
        if not chl_vars:
            continue
        for t, v, attr, _dt, val in rows:
            if t == "attribute" and v in chl_vars and attr in (
                    "units", "long_name", "comment", "standard_name"):
                print(f"    [{ds}] {v}.{attr} = {val}")
                found = True

if not found:
    print("\n  No chlorophyll units found on the ERDDAP servers listed.")
    print("  The CT DEEP chlorophyll may come from a different server. If you")
    print("  know the source URL, add it to ERDDAPS at the top of this script.")

print("\n" + "=" * 70)
print("HOW TO CONCLUDE")
print("=" * 70)
print("  If ERDDAP declares units for both columns -> that settles it outright.")
print("  Otherwise: the column sitting in the 1-30 ug/L band AND tracking the")
print("  satellite's magnitude is the chlorophyll-a in ug/L. If that is the RAW")
print("  column, your label and your paper's '>10 ug/L' claim are correct as-is,")
print("  and Corrected_Chlorophyll is some other quantity (and can be ignored).")