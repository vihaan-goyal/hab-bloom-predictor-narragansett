"""
fetch_chesapeake.py -- pull Maryland DNR "Eyes on the Bay" continuous-monitoring
sonde data and write the tidy input for src/transfer/transfer_eval.py.
--------------------------------------------------------------------------
SOURCE   Maryland DNR Continuous Monitoring Program (Shallow Water Monitoring),
         YSI 6600/EXO sondes, 15-minute cadence, 2001-present.
         Query page:  https://eyesonthebay.dnr.maryland.gov/contmon/ContMon.cfm
         The page's "Download Raw" option submits (GET) to:

  https://eyesonthebay.dnr.maryland.gov/contmon/JustDownload.cfm
      ?station=<3-letter code>&parameter=wtemp&parameter=Salinity
      &parameter=DO&parameter=ph&parameter=DOpctSat&parameter=TChlPreCal
      &StartDate=YYYY-MM-DD&EndDate=YYYY-MM-DD&outputtype=2

         Response: CSV with header
           Sample_date,Sample_Time,DateTime,Station,Layer,Salinity_ppt,pH,
           DO_mg/L,DO_pctSat,Chl_ug/L,Temp_C
         (columns appear only for the parameters requested). No login, no
         cookie, no CAPTCHA -- a plain GET works from curl/requests.
         Station codes + record spans were scraped from the <select name=
         "station"> options on ContMon.cfm (150 stations). The other download
         link on the station table (ConMon_CBPDownload.cfm -> CBP DataHub
         export) returns only the monthly discrete calibration samples, NOT
         the sonde record, so it is not used. (The CBP DataHub REST API at
         https://datahub.chesapeakebay.net/API is an alternative bulk source
         for the same sonde data; not needed here.)

STATIONS (default set): every station whose record spans >= 5 years inside
         2010-2025; where a site has a surface and a bottom sonde only the
         surface one is kept (AES not AEB, PPT not PPB, GOO not GOB).
YEARS    2010-2025 (one request per station-year; the server returns a full
         year of 15-min rows, ~35k lines, in a few seconds).

UNITS    Chl_ug/L   = "TChlPreCal": total chlorophyll from the sonde in-vivo
                      FLUORESCENCE sensor, pre-calibration, reported in ug/L.
                      Not extracted chlorophyll-a; not inter-comparable across
                      sensors -- hence the harness per-station p75 label.
         Salinity_ppt = derived from specific conductance (practical salinity,
                      numerically PSU); passed through unchanged as salinity_psu.
         Temp_C, DO_mg/L, DO_pctSat, pH passed through.
QUALITY  values coerced to numeric; rows with non-numeric / missing chl dropped;
         chl < 0 or chl > 1000 treated as sentinel/garbage and dropped; any
         value <= -99 in the other columns set to NaN. If a station reports
         more than one Layer, the most common layer is kept.

OUTPUT   data/transfer/raw/chesapeake/<STN>_<YEAR>.csv   raw server responses
         data/transfer/chesapeake_15min.csv   tidy: station, datetime, chl_ugl,
                                              temp_c, salinity_psu, do_mgl,
                                              do_pct, ph

Usage (fork root, BASE env):
    python -m src.transfer.fetch_chesapeake            # fetch + build
    python -m src.transfer.fetch_chesapeake --build-only
    python -m src.transfer.fetch_chesapeake --stations JUG OPC --years 2015 2016
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import requests

URL = "https://eyesonthebay.dnr.maryland.gov/contmon/JustDownload.cfm"
PARAMS = ["wtemp", "Salinity", "DO", "ph", "DOpctSat", "TChlPreCal"]
RAW_DIR = "data/transfer/raw/chesapeake"
OUT = "data/transfer/chesapeake_15min.csv"

# code: (name, first_year, last_year) -- from the ContMon.cfm station <select>
STATIONS = {
    "OPC": ("Bush River - Otter Point Creek", 2003, 2026),
    "MTI": ("Patuxent River - Mataponi", 2003, 2026),
    "JUG": ("Patuxent River - Jug Bay", 2003, 2026),
    "IPL": ("Patuxent River - Iron Pot Landing", 2003, 2026),
    "SPS": ("Chesapeake Bay Seg 3 - Sandy Point South Beach", 2004, 2026),
    "SGC": ("Potomac River - St. Georges Creek", 2006, 2026),
    "LMN": ("Wicomico River - Little Monie Creek", 2006, 2026),
    "SUS": ("Susquehanna River - Havre de Grace", 2007, 2026),
    "PUB": ("Coastal Bays - Public Landing", 2005, 2020),
    "NPC": ("Coastal Bays - Newport Creek", 2006, 2020),
    "MSC": ("Patapsco River - Masonville Cove Pier", 2013, 2026),
    "RIV": ("Back River - Riverside", 2014, 2026),
    "GYK": ("Coastal Bays - Greys Creek", 2008, 2020),
    "COR": ("Corsica River - Sycamore Point", 2005, 2017),
    "PPT": ("Corsica River - Possum Point Surface", 2006, 2017),
    "MAT": ("Potomac River - Mattawoman", 2004, 2015),
    "BUD": ("Sassafras River - Budds Landing", 2007, 2018),
    "GOO": ("Chesapeake Bay Seg 4 - Gooses Reef Surface", 2010, 2018),
    "FLT": ("Chesapeake Bay Seg 1 - Susquehanna Flats", 2007, 2017),
    "AWS": ("Patapsco River - Aquarium West", 2016, 2026),
    "AES": ("Patapsco River - Aquarium East Surface", 2016, 2026),
    "MAB": ("Potomac River - Mallows Bay Buoy", 2018, 2026),
    "HAU": ("Choptank River - Harris Creek Upstream", 2013, 2020),
    "HAD": ("Choptank River - Harris Creek Downstream", 2013, 2020),
}
YEARS = list(range(2010, 2026))


def fetch_one(stn, year, session, retries=3):
    path = os.path.join(RAW_DIR, f"{stn}_{year}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, "cached"
    q = [("station", stn)] + [("parameter", p) for p in PARAMS] + [
        ("StartDate", f"{year}-01-01"), ("EndDate", f"{year}-12-31"), ("outputtype", "2")]
    msg = ""
    for k in range(retries):
        try:
            r = session.get(URL, params=q, timeout=600)
            if r.status_code == 200 and r.text.startswith("Sample_date"):
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(r.text)
                return path, f"{len(r.text.splitlines()) - 1} rows"
            msg = f"status={r.status_code} head={r.text[:80]!r}"
        except requests.RequestException as e:
            msg = repr(e)
        time.sleep(3 * (k + 1))
    # write an empty marker so reruns skip it; delete the file to retry
    open(path, "w").close()
    return path, f"FAILED ({msg})"


def fetch_all(stations, years):
    os.makedirs(RAW_DIR, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (research; hab-bloom-predictor)"
    for stn in stations:
        _, y0, y1 = STATIONS.get(stn, (stn, 2001, 2026))
        for y in years:
            if y < y0 or y > y1:
                continue
            t = time.time()
            path, note = fetch_one(stn, y, s)
            print(f"{stn} {y}: {note} ({time.time() - t:.1f}s)", flush=True)


def build():
    parts = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*_*.csv"))):
        if os.path.getsize(path) == 0:
            continue
        stn = os.path.basename(path).split("_")[0]
        if stn not in STATIONS:
            continue
        d = pd.read_csv(path, dtype=str)
        d["station"] = stn
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    ren = {"Chl_ug/L": "chl_ugl", "Temp_C": "temp_c", "Salinity_ppt": "salinity_psu",
           "DO_mg/L": "do_mgl", "DO_pctSat": "do_pct", "pH": "ph"}
    for c in ren:
        if c not in raw:
            raw[c] = np.nan
    t = raw.rename(columns=ren)
    t["datetime"] = pd.to_datetime(t["DateTime"], format="%m/%d/%Y %H:%M", errors="coerce")
    n0 = len(t)
    for c in ren.values():
        t[c] = pd.to_numeric(t[c], errors="coerce")
        t.loc[t[c] <= -99, c] = np.nan            # -999 style sentinels
    t = t.dropna(subset=["datetime"])
    n_chl = int(t["chl_ugl"].notna().sum())
    t = t[t["chl_ugl"].notna() & (t["chl_ugl"] >= 0) & (t["chl_ugl"] <= 1000)]
    # one layer per station (most common)
    if "Layer" in t:
        keep = t.groupby("station")["Layer"].agg(
            lambda s: s.mode().iloc[0] if s.notna().any() else None)
        t = t[t["Layer"].isna() | (t["Layer"] == t["station"].map(keep))]
    t = t.sort_values(["station", "datetime"]).drop_duplicates(["station", "datetime"])
    cols = ["station", "datetime", "chl_ugl", "temp_c", "salinity_psu", "do_mgl", "do_pct", "ph"]
    out = t[cols].copy()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT, index=False)
    print(f"raw rows={n0:,}  rows with numeric chl={n_chl:,} ({n_chl / n0:.1%})  "
          f"kept={len(out):,}")
    print(f"stations={out.station.nunique()} : {sorted(out.station.unique())}")
    print(f"date range {out.datetime.min()} .. {out.datetime.max()}")
    print(t.groupby("station").agg(n=("chl_ugl", "size"), first=("datetime", "min"),
                                   last=("datetime", "max"), chl_med=("chl_ugl", "median"),
                                   sal_med=("salinity_psu", "median"),
                                   layer=("Layer", lambda s: s.mode().iloc[0])).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="*", default=list(STATIONS))
    ap.add_argument("--years", nargs="*", type=int, default=YEARS)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    if not a.build_only:
        fetch_all(a.stations, a.years)
    build()
