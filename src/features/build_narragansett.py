"""Consolidate RIDEM NBFSMN corrected sonde files (15-min) into one tidy CSV.

v2: parses BOTH sonde blocks (surface and bottom) plus density. Each
corrected workbook lays blocks side by side, each starting with a 'Site'
column; block 1 = surface, block 2 = bottom. Header text may sit on the
names row (2023 layout) or the units row (2021/22 layout).

Output: data/narragansett_surface_15min.csv
  station, datetime, temp_c, salinity_psu, do_pct, do_mgl, ph, chl_ugl,
  density_gcm3, bot_temp_c, bot_salinity_psu, bot_do_pct, bot_do_mgl,
  bot_chl_ugl, bot_density_gcm3
Run from repo root, BASE conda env.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

OUT = "data/narragansett_surface_15min.csv"
FILES = sorted(glob.glob(
    "data/raw/narragansett/extracted/**/*corrected*.xls", recursive=True)
    + glob.glob(
    "data/raw/narragansett/extracted/**/*corrected*.xlsx", recursive=True))

VARS = {   # output name -> (header prefix, required unit substring or None)
    "datetime":     ("Date & Time", None),
    "temp_c":       ("Temp", None),
    "salinity_psu": ("Salinity", None),
    "do_pct":       ("DO%", None),
    "do_mgl":       ("DO Conc", None),
    "ph":           ("pH", "nbs"),
    "chl_ugl":      ("Chl", "ug/l"),
    "density_gcm3": ("Density", None),
}

def parse_block(raw, names, units, lo, hi):
    def col(name, unit_sub):
        for j in range(lo, hi):
            if (names[j].lower().startswith(name.lower())
                    or units[j].startswith(name.lower())):
                if unit_sub is None or unit_sub in units[j] or unit_sub in names[j].lower():
                    return j
        return None
    cols = {k: col(p, u) for k, (p, u) in VARS.items()}
    if cols["datetime"] is None:
        # old layout: datetime column named 'Time' with units 'Date & Hour'
        for j in range(lo, hi):
            if names[j].lower() == "time" and "date" in units[j]:
                cols["datetime"] = j; break
    if cols["chl_ugl"] is None:
        # old layout: 'Chl' name with units row 'ug/L' handled, but some files
        # leave units blank -- accept bare 'Chl' as last resort
        for j in range(lo, hi):
            if names[j].lower().startswith("chl"):
                cols["chl_ugl"] = j; break
    if cols["datetime"] is None or cols["chl_ugl"] is None:
        return None
    body = raw.iloc[:, :]
    out = pd.DataFrame({k: body.iloc[:, j] if j is not None else np.nan
                        for k, j in cols.items()})
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"])
    for c in out.columns:
        if c != "datetime":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

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
        # 2003-2011 layout: header row starts with 'Time', no 'Site' column
        c0 = raw.iloc[:, 0].astype(str).str.strip().str.lower()
        hdr_rows = raw.index[c0 == "time"]
    if len(hdr_rows) == 0 or hdr_rows[0] + 2 >= len(raw):
        print(f"SKIP (no usable header): {f}"); continue
    h = hdr_rows[0]
    names = raw.iloc[h].astype(str).str.strip()
    units = raw.iloc[h + 1].astype(str).str.strip().str.lower()
    body = raw.iloc[h + 2:]
    site_cols = [j for j in range(raw.shape[1]) if names[j] == "Site"]
    bounds = site_cols + [raw.shape[1]]
    blocks = []
    for b in range(len(site_cols)):
        blk = parse_block(body, names, units, bounds[b], bounds[b + 1])
        if blk is not None:
            blocks.append(blk)
    if not blocks:
        print(f"SKIP (no parsable block): {f}"); continue
    surf = blocks[0]
    if len(blocks) > 1:
        bot = blocks[1].rename(columns={c: "bot_" + c for c in blocks[1].columns
                                        if c != "datetime"})
        bot = bot.drop(columns=[c for c in ("bot_ph",) if c in bot.columns])
        surf = surf.merge(bot, on="datetime", how="outer")
    surf.insert(0, "station", station)
    frames.append(surf)
    print(f"{station:5s} {os.path.basename(f):42s} rows={len(surf):7d} "
          f"chl={surf.chl_ugl.notna().sum():7d} "
          f"bot={'y' if len(blocks) > 1 else 'n'} "
          f"span={surf.datetime.min().date()}..{surf.datetime.max().date()}")

df = pd.concat(frames, ignore_index=True).sort_values(["station", "datetime"])
df = df.drop_duplicates(subset=["station", "datetime"], keep="last")
df.to_csv(OUT, index=False)
bot_cov = df.bot_temp_c.notna().mean() if "bot_temp_c" in df else 0
print(f"\nTOTAL rows={len(df)}  stations={df.station.nunique()}  "
      f"chl coverage={df.chl_ugl.notna().mean():.1%}  bottom coverage={bot_cov:.1%}")
