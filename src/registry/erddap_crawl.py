"""
erddap_crawl.py -- find every public sub-daily chlorophyll sonde on the ERDDAP federation
---------------------------------------------------------------------------------------
Milestone 2, step 1. Server list: awesome-erddap `erddaps.json` (60 public
ERDDAPs; cached at data/registry/erddaps.json). For each server:
  1. tabledap/allDatasets.csv  -> table datasets that are not trajectories
  2. info/<id>/index.json      -> keep if a variable looks like chlorophyll
                                  and time/latitude/longitude exist; record
                                  temp / salinity / DO / station-id variables
  3. cadence probe             -> last 7 days before maxTime: median spacing
                                  of time (per station if a station var exists);
                                  keep if <= 60 min
Outputs (incremental, resumable):
  data/registry/insitu_catalog.csv  server, dataset_id, title, n_stations,
      cadence_min, start, end, years, chl_var, temp_var, sal_var, do_var,
      station_var, lat_min, lat_max, lon_min, lon_max, url, already_covered
  data/registry/crawl_log.csv       server, url, status, n_datasets,
      n_candidates, n_kept, seconds
`already_covered` marks datasets from networks the project already ingests
(NERRS via IOOS, USGS NWIS, RIDEM, Eyes on the Bay, Cefas, IMOS, GLERL).
Politeness: sequential per server, 1 s pause, research User-Agent, backoff on
429/5xx, 2 attempts then skip. Run from fork root, BASE env:
    python -m src.registry.erddap_crawl [--servers name1,name2] [--max-datasets N]
"""
import argparse
import io
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (hab-bloom-predictor research; student project)"}
REG_URL = "https://raw.githubusercontent.com/IrishMarineInstitute/awesome-erddap/master/erddaps.json"
REG = "data/registry/erddaps.json"
CAT = "data/registry/insitu_catalog.csv"
LOG = "data/registry/crawl_log.csv"
CHL = re.compile(r"chl|chlor|chlorophyll|fluorescence", re.I)
TEMP = re.compile(r"^(sea_water_temperature|water_temp|temp|temperature|wtemp|sea_surface_temperature)$", re.I)
SAL = re.compile(r"salin|sea_water_practical_salinity|psal", re.I)
DO = re.compile(r"dissolved_oxygen|^do$|do_mgl|oxygen", re.I)
STATION = re.compile(r"^(station|station_id|platform|platform_id|site|site_id|buoy|mooring|wmo_platform_code)$", re.I)
COVERED = re.compile(r"nerrs_|nwis|usgs|ridem|narragansett|eyesonthebay|maryland|cefas|smartbuoy|imos|anmn|nrs|glerl|cwq_", re.I)
EXCLUDE_TYPES = {"Trajectory", "TrajectoryProfile", "Profile", "Grid", "Swath"}
# servers that serve only moving platforms, animal tags, cruise bottles or model output
SKIP_SERVERS = {"VOTO", "NGDAC", "OTN", "ATN-IOOS", "ALAMO", "SOCAT", "IOC-IODE-OA", "Bio-Oracle", "DIVER", "OSMC"}
MIN_YEARS, MIN_CADENCE_MIN = 0.5, 1.0


def get(url, tries=2, timeout=120):
    wait = 5
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(wait); wait *= 3; continue
            return r
        except requests.RequestException:
            if i < tries - 1:
                time.sleep(wait); wait *= 3
    return None


def all_datasets(base):
    u = (f"{base}tabledap/allDatasets.csv?datasetID,title,minTime,maxTime,minLongitude,maxLongitude,"
         f"minLatitude,maxLatitude,dataStructure,cdm_data_type")
    r = get(u)
    if r is None or r.status_code != 200:
        return None
    try:
        d = pd.read_csv(io.StringIO(r.text), skiprows=[1])
    except Exception:
        return None
    d = d[(d.dataStructure == "table") & ~d.cdm_data_type.isin(EXCLUDE_TYPES)]
    return d


def variables(base, ds):
    r = get(f"{base}info/{ds}/index.json", timeout=60)
    if r is None or r.status_code != 200:
        return None
    try:
        rows = r.json()["table"]["rows"]
    except Exception:
        return None
    var = {}
    for rt, vname, aname, dtype, val in rows:
        if rt == "variable":
            var.setdefault(vname, {"type": dtype})
        elif rt == "attribute" and vname in var and aname in ("standard_name", "long_name", "units", "cf_role"):
            var[vname][aname] = str(val)
    return var


def pick(var, rx):
    for name, a in var.items():
        blob = " ".join([name, a.get("standard_name", ""), a.get("long_name", "")])
        if rx.search(blob):
            return name
    return None


def cadence_probe(base, ds, station_var, max_time):
    if pd.isna(max_time):   # blank maxTime: ask the server for the real last timestamp
        r0 = get(f'{base}tabledap/{ds}.csv?time&orderByMax("time")', timeout=120)
        try:
            max_time = pd.read_csv(io.StringIO(r0.text), skiprows=[1])["time"].iloc[-1]
        except Exception:
            max_time = None
    t1 = pd.Timestamp(max_time) if pd.notna(max_time) else pd.Timestamp.utcnow()
    t1 = t1.tz_convert(None) if t1.tzinfo else t1
    t0 = t1 - pd.Timedelta(days=7)
    cols = "time" + (f",{station_var}" if station_var else "")
    u = f"{base}tabledap/{ds}.csv?{cols}&time>={t0.strftime('%Y-%m-%dT%H:%M:%SZ')}&time<={t1.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    r = get(u, timeout=120)
    if r is None or r.status_code != 200:
        return None, 0
    try:
        d = pd.read_csv(io.StringIO(r.text), skiprows=[1])
    except Exception:
        return None, 0
    if len(d) < 3:
        return None, 0
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"])
    if station_var and station_var in d:
        n_st = d[station_var].nunique()
        gaps = d.sort_values("time").groupby(station_var)["time"].diff().dt.total_seconds().dropna()
    else:
        n_st = 1
        gaps = d.sort_values("time")["time"].diff().dt.total_seconds().dropna()
    gaps = gaps[gaps > 0]
    if len(gaps) < 2:
        return None, n_st
    return float(np.median(gaps) / 60.0), int(n_st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", default="all")
    ap.add_argument("--max-datasets", type=int, default=100000)
    a = ap.parse_args()
    os.makedirs("data/registry", exist_ok=True)
    if not os.path.exists(REG):
        open(REG, "w", encoding="utf-8").write(get(REG_URL).text)
    servers = [s for s in json.load(open(REG, encoding="utf-8")) if s.get("public", True)]
    if a.servers != "all":
        servers = [s for s in servers if s["short_name"] in a.servers.split(",")]
    done_servers = set(pd.read_csv(LOG).server) if os.path.exists(LOG) else set()
    for s in servers:
        name, base = s["short_name"], s["url"].rstrip("/") + "/"
        if name in done_servers or name in SKIP_SERVERS:
            continue
        t_start = time.time()
        print(f"== {name} {base}", flush=True)
        cat = all_datasets(base)
        if cat is None:
            pd.DataFrame([dict(server=name, url=base, status="unreachable", n_datasets=0, n_candidates=0,
                               n_kept=0, seconds=round(time.time() - t_start))]).to_csv(
                LOG, mode="a", header=not os.path.exists(LOG), index=False)
            continue
        cand = cat[cat.title.str.contains("chl|chlor|fluor|water quality|sonde|buoy|mooring|station|sensor",
                                          case=False, na=False)]
        if len(cand) == 0:
            cand = cat
        cand = cand.head(a.max_datasets)
        kept = 0
        for _, row in cand.iterrows():
          try:
            var = variables(base, row.datasetID)
            time.sleep(1)
            if not var or "time" not in var or "latitude" not in var or "longitude" not in var:
                continue
            chl = pick(var, CHL)
            if not chl:
                continue
            station_var = next((v for v, at in var.items() if at.get("cf_role") == "timeseries_id"), None) or pick(var, STATION)
            cad, n_st = cadence_probe(base, row.datasetID, station_var, row.maxTime)
            time.sleep(1)
            if pd.isna(row.maxTime) and cad is not None:
                r0 = get(f'{base}tabledap/{row.datasetID}.csv?time&orderByMax("time")', timeout=60)
                try: row.maxTime = pd.read_csv(io.StringIO(r0.text), skiprows=[1])["time"].iloc[-1]
                except Exception: pass
            if cad is None or cad > 60 or cad < MIN_CADENCE_MIN:
                print(f"   skip {row.datasetID}: cadence={cad}", flush=True)
                continue
            tmax = pd.Timestamp(row.maxTime) if pd.notna(row.maxTime) else pd.Timestamp.utcnow().tz_localize(None)
            tmin = pd.Timestamp(row.minTime)
            tmax = tmax.tz_convert(None) if tmax.tzinfo else tmax; tmin = tmin.tz_convert(None) if tmin.tzinfo else tmin
            years = (tmax - tmin).days / 365.25
            if years < MIN_YEARS:
                print(f"   skip {row.datasetID}: span {years:.2f} y", flush=True)
                continue
            rec = dict(server=name, dataset_id=row.datasetID, title=str(row.title)[:120], n_stations=n_st,
                       cadence_min=round(cad, 1), start=str(tmin)[:10], end=str(tmax)[:10],
                       years=round(years, 1), chl_var=chl, temp_var=pick(var, TEMP), sal_var=pick(var, SAL),
                       do_var=pick(var, DO), station_var=station_var, lat_min=row.minLatitude, lat_max=row.maxLatitude,
                       lon_min=row.minLongitude, lon_max=row.maxLongitude, url=f"{base}tabledap/{row.datasetID}",
                       already_covered=bool(COVERED.search(row.datasetID + " " + str(row.title))))
            pd.DataFrame([rec]).to_csv(CAT, mode="a", header=not os.path.exists(CAT), index=False)
            kept += 1
            print(f"   KEEP {row.datasetID}: {n_st} st, {cad:.0f} min, {years:.1f} y, chl={chl}", flush=True)
          except Exception as e:
            print(f"   error {row.datasetID}: {str(e)[:80]}", flush=True)
        pd.DataFrame([dict(server=name, url=base, status="ok", n_datasets=len(cat), n_candidates=len(cand),
                           n_kept=kept, seconds=round(time.time() - t_start))]).to_csv(
            LOG, mode="a", header=not os.path.exists(LOG), index=False)
    print("crawl complete")


if __name__ == "__main__":
    main()
