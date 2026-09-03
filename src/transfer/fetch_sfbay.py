"""
fetch_sfbay.py -- USGS 15-min water quality for San Francisco Bay / Delta
-------------------------------------------------------------------------------
Builds data/transfer/sfbay_15min.csv for src/transfer/transfer_eval.py.

SOURCE   USGS Water Data for the Nation OGC API (the legacy NWIS "iv" service
         at waterservices.usgs.gov returned HTTP 503 "service temporarily
         unavailable" on 2026-09-03, so it is not used):
           series list per site:
             https://api.waterdata.usgs.gov/ogcapi/v0/collections/
               time-series-metadata/items?f=json&monitoring_location_id=USGS-<site>
           values, one series x one calendar year per request (<= 35,136 rows,
           under the 50,000 page limit; `offset` paging is still handled):
             https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/
               items?f=csv&time_series_id=<id>&datetime=<y>-01-01T00:00:00Z/
               <y+1>-01-01T00:00:00Z&limit=50000
         Site discovery / period of record came from the NWIS site service
         series catalog (still up):
           https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd=ca
             &parameterCd=32316,32318,32295&siteStatus=all&hasDataTypeCd=iv
             &seriesCatalogOutput=true&outputDataTypeCd=iv
         Raw CSVs cached under data/transfer/raw/sfbay/<site>_<parm>_<ts>_<year>.csv
         (gitignored); existing files are not re-downloaded (0-byte = no data).

PARAMETER CODES (statistic 00011 = instantaneous)
  00010 temperature (deg C)                -> temp_c
  00300 dissolved oxygen (mg/L)            -> do_mgl
  00480 salinity (ppt, ~PSU)               -> salinity_psu (preferred)
  00095 specific conductance (uS/cm @25C)  -> salinity_psu when 00480 absent
  32316 chlorophyll fluorescence (ug/L)    -> chl_ugl    (all selected sites)
  32318 chlorophyll relative fluor. (RFU)  -> chl_rfu    (fallback only; none
                                              of the selected sites needs it)
  (00301 DO %sat and 00400 pH are optional for the harness and are skipped to
   keep the download to ~5 series per site.)

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

YEARS  2012-01-01 .. 2025-12-31 requested (the catalog shows no chlorophyll
       before 2013 at any of these sites).  API timestamps are UTC; they are
       shifted to Pacific Standard Time (UTC-8, no DST, the NWIS convention)
       before the harness groups readings into station-days.

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
    python -m src.transfer.fetch_sfbay            # download + parse
    python -m src.transfer.fetch_sfbay --parse-only
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

RAW_DIR = "data/transfer/raw/sfbay"
OUT = "data/transfer/sfbay_15min.csv"
API = "https://api.waterdata.usgs.gov/ogcapi/v0/collections"
META_URL = API + "/time-series-metadata/items?f=json&monitoring_location_id=USGS-{site}&limit=200"
VAL_URL = (API + "/continuous/items?f=csv&time_series_id={ts}"
           "&datetime={y}-01-01T00:00:00Z/{y1}-01-01T00:00:00Z&limit={limit}&offset={offset}"
           "&properties=time,value")
LIMIT = 50000
YEARS = range(2012, 2026)
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
UTC_OFFSET_H = -8  # Pacific Standard Time


def sc_to_psu(sc):
    """Schemel (2001) 25 C form of PSS-78; sc in uS/cm."""
    r = np.asarray(sc, dtype=float) / 53087.0
    s = (0.0120 - 0.2174 * r ** 0.5 + 25.3283 * r + 13.7714 * r ** 1.5
         - 6.4788 * r ** 2 + 2.5842 * r ** 2.5)
    return np.clip(s, 0, None)


def get(url, retries=8, timeout=600):
    """GET with backoff; the API rate-limits (HTTP 429) so requests are paced
    ~1/s and 429s wait Retry-After (or 30 s x attempt)."""
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            time.sleep(1.0)
            return data
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            ra = None
            if code == 429:
                try:
                    ra = float(e.headers.get("Retry-After", ""))
                except (ValueError, AttributeError):
                    ra = None
            wait = ra if ra else (30 * (k + 1) if code == 429 else 10 * (k + 1))
            print(f"    attempt {k+1} failed: {e}; sleeping {wait:.0f}s", flush=True)
            time.sleep(wait)
    return None


def list_series(site):
    """Instantaneous series at a site for the wanted parameter codes."""
    path = os.path.join(RAW_DIR, f"{site}_meta.json")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        d = json.load(open(path, encoding="utf-8"))
    else:
        raw = get(META_URL.format(site=site))
        if raw is None:
            return []
        open(path, "wb").write(raw)
        d = json.loads(raw)
    keep = {}
    for f in d.get("features", []):
        p = f["properties"]
        if p.get("parameter_code") in PARM and p.get("statistic_id") == "00011" and p.get("begin"):
            s = dict(ts=f["id"], parm=p["parameter_code"],
                     begin=int(p["begin"][:4]), end=int(p["end"][:4]) if p.get("end") else 2026)
            # keep only the longest series per parameter (the API rate-limits;
            # duplicate sensors add requests, not years)
            k = keep.get(s["parm"])
            if k is None or s["end"] - s["begin"] > k["end"] - k["begin"]:
                keep[s["parm"]] = s
    if "00480" in keep and "00095" in keep:
        del keep["00095"]
    order = ["32316", "32318", "00010", "00480", "00095", "00300"]
    return [keep[p] for p in order if p in keep]


def fetch_series_year(site, s, y):
    path = os.path.join(RAW_DIR, f"{site}_{s['parm']}_{s['ts'][:8]}_{y}.csv")
    if os.path.exists(path):
        return path
    chunks, offset = [], 0
    while True:
        raw = get(VAL_URL.format(ts=s["ts"], y=y, y1=y + 1, limit=LIMIT, offset=offset))
        if raw is None:
            print(f"  {site} {s['parm']} {y}: giving up", file=sys.stderr, flush=True)
            return None
        n = raw.count(b"\n") - 1 if raw else 0
        chunks.append(raw if offset == 0 else raw.split(b"\n", 1)[1] if b"\n" in raw else b"")
        if n < LIMIT:
            break
        offset += LIMIT
    data = b"".join(chunks)
    open(path, "wb").write(data)
    print(f"  {site} {s['parm']} {y}: {len(data)/1e6:.1f} MB", flush=True)
    return path


def read_csv(path, parm):
    if os.path.getsize(path) == 0:
        return None
    d = pd.read_csv(path, usecols=["time", "value"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d["parameter_code"] = parm
    return d.dropna(subset=["value"])


def main(parse_only=False, fetch_only=False, sites=None):
    os.makedirs(RAW_DIR, exist_ok=True)
    frames = []
    for site, short in SITES.items():
        if sites and site not in sites:
            continue
        series = list_series(site)
        print(f"[{site} {short}] {len(series)} series: "
              + ", ".join(f"{s['parm']}:{s['begin']}-{s['end']}" for s in series), flush=True)
        for s in series:
            for y in YEARS:
                if y < s["begin"] or y > s["end"]:
                    continue
                path = os.path.join(RAW_DIR, f"{site}_{s['parm']}_{s['ts'][:8]}_{y}.csv")
                if not parse_only:
                    path = fetch_series_year(site, s, y)
                if path is None or not os.path.exists(path) or fetch_only:
                    continue
                d = read_csv(path, s["parm"])
                if d is not None and len(d):
                    d["station"] = short
                    frames.append(d)
    if fetch_only:
        print("fetch done", flush=True)
        return
    lng = pd.concat(frames, ignore_index=True)
    lng["datetime"] = (pd.to_datetime(lng["time"], utc=True, errors="coerce")
                       + pd.Timedelta(hours=UTC_OFFSET_H)).dt.tz_localize(None)
    lng = lng.dropna(subset=["datetime"])
    lng["var"] = lng["parameter_code"].map(PARM)
    # average duplicate series (two sensors / redeployments) at the same time
    wide = (lng.groupby(["station", "datetime", "var"])["value"].mean()
            .unstack("var").reset_index())
    for c in PARM.values():
        if c not in wide:
            wide[c] = np.nan
    from_sc = sc_to_psu(wide["spc_uscm"])
    wide["sal_source"] = np.where(wide["salinity_psu"].notna(), "00480",
                                  np.where(wide["spc_uscm"].notna(), "00095->PSS78", "none"))
    wide["salinity_psu"] = wide["salinity_psu"].where(wide["salinity_psu"].notna(), from_sc)
    wide = wide.dropna(subset=["chl_ugl"]).sort_values(["station", "datetime"])
    cols = ["station", "datetime", "chl_ugl", "temp_c", "salinity_psu", "do_mgl", "sal_source"]
    wide[cols].to_csv(OUT, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    print(f"wrote {OUT}: {len(wide):,} rows, {wide.station.nunique()} stations, "
          f"{wide.datetime.min()} .. {wide.datetime.max()}")
    print(wide.groupby("station").agg(
        n=("chl_ugl", "size"), start=("datetime", "min"), end=("datetime", "max"),
        chl_med=("chl_ugl", "median"), sal_med=("salinity_psu", "median"),
        temp_ok=("temp_c", lambda s: s.notna().mean()),
        do_ok=("do_mgl", lambda s: s.notna().mean())).round(3).to_string())
    print(wide.sal_source.value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--sites", nargs="*", help="subset of site numbers (for parallel fetch)")
    a = ap.parse_args()
    main(a.parse_only, a.fetch_only, a.sites)
