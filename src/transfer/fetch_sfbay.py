"""
fetch_sfbay.py -- USGS NWIS 15-min water quality for San Francisco Bay / Delta
-------------------------------------------------------------------------------
Builds data/transfer/sfbay_15min.csv for src/transfer/transfer_eval.py.

SOURCE   USGS NWIS Instantaneous Values web service, RDB format, one site and
         one calendar quarter per request (5 windows/year: Q1 is split at the
         spring DST-start day, see windows(); a 6-parameter request longer
         than ~3 months also returns HTTP 503 from this service):
           https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites=<id>
             &parameterCd=00010,00300,00480,00095,32316,32318
             &startDT=<yyyy-mm-dd>&endDT=<yyyy-mm-dd>
         (301 -> https://nwis.waterservices.usgs.gov/nwis/iv/ ; followed.)
         Site discovery / period of record from the NWIS site service:
           https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd=ca
             &parameterCd=32316,32318,32295&siteStatus=all&hasDataTypeCd=iv
             &seriesCatalogOutput=true&outputDataTypeCd=iv
         Raw RDB cached under data/transfer/raw/sfbay/<site>_<year>Q<q>.rdb
         (gitignored); existing files are not re-downloaded.
         Note: on 2026-09-03 the IV service was down (HTTP 503) for ~1 h; the
         OGC API (api.waterdata.usgs.gov/ogcapi/v0/collections/continuous)
         was tried meanwhile but rate-limits at ~10 requests per 20-30 min,
         far too slow for ~300 series-years.  Its partial cache is left under
         raw/sfbay/ogc_partial/ and is not used.

PARAMETER CODES
  00010 temperature (deg C)                -> temp_c
  00300 dissolved oxygen (mg/L)            -> do_mgl
  00480 salinity (ppt, ~PSU)               -> salinity_psu (preferred)
  00095 specific conductance (uS/cm @25C)  -> salinity_psu when 00480 absent
  32316 chlorophyll fluorescence (ug/L)    -> chl_ugl    (all selected sites)
  32318 chlorophyll relative fluor. (RFU)  -> chl_rfu    (fallback only; none
                                              of the selected sites needs it)
  Where a site has two series for one parameter (two sensors / redeployment)
  the values are averaged per timestamp.

SITES  (all have 32316 chl in ug/L for >= 4 years; chosen from the catalog
        above; the Bay-proper bridge sites -- Alcatraz 374938122251801,
        Dumbarton 373015122071000, Richmond-San Rafael 375607122264701 --
        only report chlorophyll from 2023-09 on and were excluded; Mallard
        Island 11185185 and Benicia 380339122034900 have no NWIS chl series)
  11455508         Suisun Bay at Van Sickle Island nr Pittsburg     2016-09..2025-08
  381142122015801  First Mallard Branch nr Fairfield (Suisun Marsh) 2018-05..2025
  11337190         San Joaquin R at Jersey Point                    2016-07..2025
  11455478         Sacramento R at Decker Island nr Rio Vista       2013-01..2019-02
  11455315         Cache Slough at S Liberty Island nr Rio Vista    2014-10..2025
  11455350         Cache Slough at Ryer Island (00095 only, no 00480) 2013-02..2018-07
  11447890         Sacramento R above Delta Cross Channel           2014-03..2025
  11336790         Little Potato Slough at Terminous                2019-07..2025

YEARS  2013-01-01 .. 2025-12-31 requested (the catalog shows no chlorophyll
       before 2013 at any of these sites).  Timestamps are NWIS local clock
       time (tz_cd PST/PDT) and are used as-is for the station-day grouping.

CONDUCTANCE -> SALINITY
  00095 is specific conductance already normalised to 25 C, so PSS-78
  reduces to the 25 C polynomial of Schemel (2001, IEP Newsletter 14(1):17-18),
  the standard Bay-Delta conversion:
      R = SC / 53087        (SC in uS/cm; 53087 = C(35,25,0))
      S = 0.0120 - 0.2174 R^0.5 + 25.3283 R + 13.7714 R^1.5
          - 6.4788 R^2 + 2.5842 R^2.5
  Clipped at 0.  Below ~1 PSU (Delta freshwater) the polynomial is only
  approximate but the error is < 0.05 PSU.  00480 is used wherever present;
  the conversion fills gaps and Ryer Island (no 00480 series).

Usage (fork root, BASE env):
    python -m src.transfer.fetch_sfbay                 # download + parse
    python -m src.transfer.fetch_sfbay --parse-only
    python -m src.transfer.fetch_sfbay --fetch-only --sites 11455508 11337190
"""
import argparse
import io
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

RAW_DIR = "data/transfer/raw/sfbay"
OUT = "data/transfer/sfbay_15min.csv"
IV_URL = ("https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites={site}"
          "&parameterCd=00010,00300,00480,00095,32316,32318"
          "&startDT={d0}&endDT={d1}")
YEARS = range(2013, 2026)


def windows(y):
    """Request windows for year y: 5 chunks.  The IV service returns HTTP 503
    for any window that contains the spring DST-start day (2nd Sunday of
    March, local 02:00 does not exist), so Q1 is split there and the request
    resumes at 03:00 local on that day (the first 2 h of that day are lost)."""
    d = 8
    while pd.Timestamp(y, 3, d).weekday() != 6:
        d += 1
    return [("Q1a", f"{y}-01-01", f"{y}-03-{d-1:02d}"),
            ("Q1b", f"{y}-03-{d:02d}T03:00", f"{y}-03-31"),
            ("Q2", f"{y}-04-01", f"{y}-06-30"),
            ("Q3", f"{y}-07-01", f"{y}-09-30"),
            ("Q4", f"{y}-10-01", f"{y}-12-31")]
SITES = {
    "11455508": "VanSickle",
    "381142122015801": "FirstMallard",
    "11337190": "JerseyPoint",
    "11455478": "DeckerIsland",
    "11455315": "CacheSloughLiberty",
    "11455350": "CacheSloughRyer",
    "11447890": "SacAbDCC",
    "11336790": "LittlePotato",
}
PARM = {"00010": "temp_c", "00300": "do_mgl", "00480": "salinity_psu",
        "00095": "spc_uscm", "32316": "chl_ugl", "32318": "chl_rfu"}
NA = ["", "Ice", "Eqp", "Ssn", "Dis", "***", "Bkw", "Mnt", "Rat", "Fld", "Zfl", "Tst"]


def sc_to_psu(sc):
    """Schemel (2001) 25 C form of PSS-78; sc in uS/cm."""
    r = np.asarray(sc, dtype=float) / 53087.0
    s = (0.0120 - 0.2174 * r ** 0.5 + 25.3283 * r + 13.7714 * r ** 1.5
         - 6.4788 * r ** 2 + 2.5842 * r ** 2.5)
    return np.clip(s, 0, None)


def fetch(site, y, tag, d0, d1, retries=4):
    path = os.path.join(RAW_DIR, f"{site}_{y}{tag}.rdb")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = IV_URL.format(site=site, d0=d0, d1=d1)
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            print(f"  {site} {y}{tag}: {len(data)/1e6:.2f} MB", flush=True)
            time.sleep(0.5)
            return path
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            if code == 404:  # no data for this site/window
                with open(path, "wb") as f:
                    f.write(b"# no data (HTTP 404)\n")
                print(f"  {site} {y}{tag}: no data", flush=True)
                return path
            wait = 10 * (k + 1)
            print(f"  {site} {y}{tag}: attempt {k+1} failed: {e}; sleeping {wait}s", flush=True)
            time.sleep(wait)
    return None


def parse_rdb(path):
    """NWIS IV RDB -> wide frame (datetime + one column per parameter)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l for l in f if not l.startswith("#")]
    if len(lines) < 3:
        return None
    hdr = lines[0].rstrip("\n").split("\t")
    body = "".join(lines[2:])  # skip the RDB format line (5s 15s 20d ...)
    df = pd.read_csv(io.StringIO(body), sep="\t", names=hdr, dtype=str,
                     na_values=NA, keep_default_na=True)
    if "datetime" not in df:
        return None
    out = {"datetime": df["datetime"]}
    # value columns are <ts_id>_<parm> or <ts_id>_<parm>_<suffix>; QC flags end in _cd
    for parm, name in PARM.items():
        cols = [c for c in hdr if not c.endswith("_cd")
                and (c.endswith(f"_{parm}") or f"_{parm}_" in c)]
        if not cols:
            continue
        vals = df[cols].apply(pd.to_numeric, errors="coerce")
        out[name] = vals.mean(axis=1)  # average duplicate sensors / ts_ids
    return pd.DataFrame(out)


def main(parse_only=False, fetch_only=False, sites=None):
    os.makedirs(RAW_DIR, exist_ok=True)
    frames = []
    for site, short in SITES.items():
        if sites and site not in sites:
            continue
        print(f"[{site} {short}]", flush=True)
        for y in YEARS:
            for tag, d0, d1 in windows(y):
                path = os.path.join(RAW_DIR, f"{site}_{y}{tag}.rdb")
                if not parse_only:
                    path = fetch(site, y, tag, d0, d1)
                    if path is None:
                        print(f"  {site} {y}{tag}: giving up", file=sys.stderr, flush=True)
                        continue
                if fetch_only or not os.path.exists(path):
                    continue
                d = parse_rdb(path)
                if d is None or ("chl_ugl" not in d and "chl_rfu" not in d):
                    continue
                d.insert(0, "station", short)
                frames.append(d)
    if fetch_only:
        print("fetch done", flush=True)
        return
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")  # local clock time
    df = df.dropna(subset=["datetime"])
    for c in PARM.values():
        if c not in df:
            df[c] = np.nan
    # salinity: 00480 where present, else PSS-78 (Schemel 2001) from 00095
    from_sc = sc_to_psu(df["spc_uscm"])
    df["sal_source"] = np.where(df["salinity_psu"].notna(), "00480",
                                np.where(df["spc_uscm"].notna(), "00095->PSS78", "none"))
    df["salinity_psu"] = df["salinity_psu"].where(df["salinity_psu"].notna(), from_sc)
    df = df.dropna(subset=["chl_ugl"])
    df = df.sort_values(["station", "datetime"]).drop_duplicates(["station", "datetime"])
    cols = ["station", "datetime", "chl_ugl", "temp_c", "salinity_psu", "do_mgl", "sal_source"]
    df[cols].to_csv(OUT, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    print(f"wrote {OUT}: {len(df):,} rows, {df.station.nunique()} stations, "
          f"{df.datetime.min()} .. {df.datetime.max()}")
    print(df.groupby("station").agg(
        n=("chl_ugl", "size"), start=("datetime", "min"), end=("datetime", "max"),
        chl_med=("chl_ugl", "median"), sal_med=("salinity_psu", "median"),
        temp_ok=("temp_c", lambda s: s.notna().mean()),
        do_ok=("do_mgl", lambda s: s.notna().mean())).round(3).to_string())
    print(df.sal_source.value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--sites", nargs="*", help="subset of site numbers")
    a = ap.parse_args()
    main(a.parse_only, a.fetch_only, a.sites)
