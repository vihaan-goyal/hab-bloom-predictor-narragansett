"""Consolidate RIDEM Narragansett Bay Fixed-Site Monitoring Network corrected
sonde files (15-min cadence) into one tidy surface-sonde CSV.

Input : data/raw/narragansett/extracted/nbfsmn.YYYY/<station dir>/*.corrected.xlsx
Output: data/narragansett_surface_15min.csv
        station, datetime, temp_c, salinity_psu, do_pct, do_mgl, ph, chl_ugl

Surface block only (matches the LIS surface-sample convention). Header row is
located by the cell 'Site' in column 0; the Chl column is the one whose units
row reads 'ug/L' (not RFU). Run from repo root, BASE conda env.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

OUT = "data/narragansett_surface_15min.csv"
FILES = sorted(glob.glob(
    "data/raw/narragansett/extracted/**/*corrected*.xlsx", recursive=True))

frames = []
for f in FILES:
    m = re.search(r"[/\\](B\d+w?|F\d+)\.\w+\.", f)
    station = m.group(1) if m else os.path.basename(f).split(".")[0]
    try:
        xl = pd.ExcelFile(f, engine="calamine")
    except Exception as e:
        print(f"SKIP (unreadable: {type(e).__name__}): {f}"); continue
    sheet = next((s for s in xl.sheet_names
                  if s.lower() not in ("notes",) and "flow" not in s.lower()), None)
    if sheet is None:
        print(f"SKIP (no data sheet): {f}"); continue
    try:
        raw = xl.parse(sheet, header=None)
    except Exception as e:
        print(f"SKIP (parse fail {type(e).__name__}): {f}"); continue
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        print(f"SKIP (empty sheet): {f}"); continue
    hdr_rows = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Site"]
    if len(hdr_rows) == 0:
        print(f"SKIP (no 'Site' header): {f}"); continue
    h = hdr_rows[0]
    names = raw.iloc[h].astype(str).str.strip()
    if h + 1 >= len(raw):
        print(f"SKIP (no units row): {f}"); continue
    units = raw.iloc[h + 1].astype(str).str.strip().str.lower()

    def first_col(name, unit_sub=None):
        # header text may sit on the names row (2023 layout) or the units row
        # (2021/22 layout, e.g. 'Date & Time'); accept either
        for j in range(len(names)):
            if (names[j].lower().startswith(name.lower())
                    or units[j].startswith(name.lower())):
                if unit_sub is None or unit_sub in units[j] or unit_sub in names[j].lower():
                    return j
        return None

    cols = {
        "datetime":     first_col("Date & Time"),
        "temp_c":       first_col("Temp"),
        "salinity_psu": first_col("Salinity"),
        "do_pct":       first_col("DO%"),
        "do_mgl":       first_col("DO Conc"),
        "ph":           first_col("pH", "nbs"),
        "chl_ugl":      first_col("Chl", "ug/l"),
    }
    if cols["datetime"] is None or cols["chl_ugl"] is None:
        print(f"SKIP (missing datetime/chl): {f}"); continue
    if h + 2 >= len(raw):
        print(f"SKIP (empty body): {f}"); continue
    body = raw.iloc[h + 2:]
    out = pd.DataFrame({k: body.iloc[:, j] if j is not None else np.nan
                        for k, j in cols.items()})
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"])
    for c in out.columns:
        if c != "datetime":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out.insert(0, "station", station)
    frames.append(out)
    n_chl = out["chl_ugl"].notna().sum()
    print(f"{station:5s} {os.path.basename(f):40s} rows={len(out):7d} chl_rows={n_chl:7d} "
          f"span={out.datetime.min().date()}..{out.datetime.max().date()}")

df = pd.concat(frames, ignore_index=True).sort_values(["station", "datetime"])
df = df.drop_duplicates(subset=["station", "datetime"], keep="last")
df.to_csv(OUT, index=False)
print(f"\nTOTAL rows={len(df)}  stations={df.station.nunique()}  "
      f"chl coverage={df.chl_ugl.notna().mean():.1%}  -> {OUT}")
