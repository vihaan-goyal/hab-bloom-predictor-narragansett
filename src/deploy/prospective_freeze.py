"""
prospective_freeze.py -- ONE-SHOT: seed the per-site history files from the
cached pulls and freeze each station's bloom threshold (own 75th percentile of
daily-mean chlorophyll) before the first prospective issuance.
-------------------------------------------------------------------------------
Purpose : the p75 written here is the outcome definition for the whole
          prospective protocol; recomputed only by a dated protocol amendment
          (1.0 frozen 2026-09-06T00:37Z; 1.1 re-freeze 2026-09-06, Scripps -> _eco).
Inputs  : data/registry/sites/<id>.csv          (erddap_top, run_catalog cache)
          live ERDDAP pull for sites with seed_from (protocol 1.1 (a); raw cached
          under data/prospective/raw/freeze_<version>/; history REPLACED, not appended)
          data/transfer/raw/nerrs/erddap/nerrs_kac{ss,h3}wq.csv (UTC, QARTOD)
          data/transfer/chesapeake_15min.csv     (Eastern clock -> UTC)
          data/buoy_eco_fl/all_buoys_eco_fl.parquet (LIS ECO-FL, night-only;
          copied from the parent repo when absent)
Outputs : data/prospective/history/<site_id>.csv (seeded via append_history)
          data/prospective/site_p75.csv          (refuses to overwrite unless --force)
Run     : python -m src.deploy.prospective_freeze [--force]
          from the fork root with C:/Users/vihaa/anaconda3/python.exe
"""
import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

from src.deploy import live_feeds as lf
from src.deploy import prospective_sites as ps

SITES_CACHE = os.path.join(ps.ROOT, "data", "registry", "sites")
NERRS_RAW = os.path.join(ps.ROOT, "data", "transfer", "raw", "nerrs", "erddap")
CHES_15MIN = os.path.join(ps.ROOT, "data", "transfer", "chesapeake_15min.csv")
LIS_PARQUET = os.path.join(ps.ROOT, "data", "buoy_eco_fl", "all_buoys_eco_fl.parquet")
LIS_PARQUET_SRC = r"C:\Users\vihaa\hab-bloom-predictor\data\buoy_eco_fl\all_buoys_eco_fl.parquet"
LIS_STATION_MAP = {"WLIS_WQ_SFC": "WLIS_ECO_FL", "EXRX_WQ_SFC": "EXRX_ECO_FL"}
EXPECTED_P75 = {"WLIS_ECO_FL": 257, "EXRX_ECO_FL": 417, "MSC": 23, "AWS": 26, "AES": 25, "MAB": 11, "RIV": 80,
                "SPS": 19, "kac_ss": 3.1, "kac_h3": 2.7, "scripps-pier-automated-shore-sta-1": 1.3,   # 1.1: _eco channel
                "newport-pier-automated-shore-sta": 13, "edu_ucsc_scwharf1": 27, "oa2-mbari-buoy": 10,
                "mlml_mlml_sea": 3.8, "tiburon-water-tibc1": 3.8, "edu_calpoly_marine_morro": 2.9,
                "edu_humboldt_humboldt": 4.0, "indian-river-lagoon-banana-river": 4.4,
                "indian-river-lagoon-vero-beach-i": 3.9}
P75_COLS = ["site_group", "site_id", "station", "chl_p75_site", "chl_units", "n_station_days", "history_first",
            "history_last", "min_readings", "t_star_site", "frozen_at_utc", "model_version", "code_version",
            "protocol_version"]


def seed_erddap_top(site):
    if site.get("seed_from"):                          # protocol 1.1 (a): channel changed -> live re-seed
        end = pd.Timestamp.utcnow().tz_localize(None) if pd.Timestamp.utcnow().tzinfo else pd.Timestamp.utcnow()
        frame, status, note = lf.fetch_site(site, pd.Timestamp(site["seed_from"]), end, f"freeze_{ps.PROTOCOL_VERSION}")
        print(f"  {site['site_id']}: live seed {site['chl_var']} from {site['seed_from']}: {status} "
              f"({len(frame)} readings){' - ' + note if note else ''}", file=sys.stderr)
        return frame if status == "ok" else None
    p = os.path.join(SITES_CACHE, f"{site['site_id']}.csv")
    if not os.path.exists(p):
        return None
    return lf._finish(pd.read_csv(p))         # run_catalog contract cache: no qc column, no QARTOD filter


def seed_nerrs(site):
    p = os.path.join(NERRS_RAW, f"nerrs_{site['station'].replace('_', '')}wq.csv")
    if not os.path.exists(p):
        return None
    return lf.parse_nerrs_frame(pd.read_csv(p, skiprows=[1], low_memory=False), site["station"])


_CHES = {}


def seed_chesapeake(site):
    if "df" not in _CHES:
        _CHES["df"] = pd.read_csv(CHES_15MIN) if os.path.exists(CHES_15MIN) else None
    d = _CHES["df"]
    if d is None:
        return None
    d = d[d.station == site["station"]]
    out = pd.DataFrame({"station": site["station"], "datetime": lf.eastern_to_utc(d["datetime"]),
                        "chl": d["chl_ugl"], "temp": d["temp_c"], "sal": d["salinity_psu"], "do": d["do_mgl"]})
    return lf._finish(out)


def seed_lis(site):
    if not os.path.exists(LIS_PARQUET):
        if not os.path.exists(LIS_PARQUET_SRC):
            return None
        os.makedirs(os.path.dirname(LIS_PARQUET), exist_ok=True)
        shutil.copyfile(LIS_PARQUET_SRC, LIS_PARQUET)
    p = pd.read_parquet(LIS_PARQUET, columns=["station", "time", "Avg_FL"])
    p["station"] = p["station"].map(LIS_STATION_MAP)
    p = p[p.station == site["site_id"]]
    out = pd.DataFrame({"station": site["site_id"], "datetime": pd.to_datetime(p["time"], utc=True).dt.tz_localize(None),
                        "chl": p["Avg_FL"], "temp": np.nan, "sal": np.nan, "do": np.nan})
    return lf.night_only(lf._finish(out)).reset_index(drop=True)


SEEDERS = {"erddap_top": seed_erddap_top, "nerrs": seed_nerrs, "chesapeake": seed_chesapeake, "lis_buoy": seed_lis}


def freeze_site(site, pa, stamps):
    seed = SEEDERS[site["site_group"]](site)
    if seed is None or len(seed) == 0:
        print(f"  {site['site_id']}: no cached history found", file=sys.stderr)
        hist = lf.load_history(site)
    elif site.get("seed_from"):
        hist = lf.write_history(site, seed)            # channel changed: old-channel history discarded
    else:
        hist = lf.append_history(site, seed)
    day = ps.build_site_daily(site, hist, pa)
    rows = []
    for st, g in day.groupby("station"):
        rows.append(dict(site_group=site["site_group"], site_id=site["site_id"], station=st,
                         chl_p75_site=float(g.chl.quantile(0.75)), chl_units=site["chl_units"], n_station_days=len(g),
                         history_first=g.date.min().date(), history_last=g.date.max().date(),
                         min_readings=site["min_readings"], t_star_site=site["t_star_site"], **stamps))
    return rows


def main():
    ap = argparse.ArgumentParser(description="freeze per-station p75 thresholds (one shot)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing site_p75.csv")
    a = ap.parse_args()
    if os.path.exists(ps.P75_PATH) and not a.force:
        sys.exit(f"{ps.P75_PATH} exists; the threshold table is frozen. Use --force to redo it.")
    pa = ps.load_pa()
    stamps = dict(frozen_at_utc=pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), model_version=ps.model_version(),
                  code_version=ps.code_version(), protocol_version=ps.PROTOCOL_VERSION)
    rows = []
    for site in ps.SITES:
        print(f"freezing {site['site_group']}/{site['site_id']}", flush=True)
        rows += freeze_site(site, pa, stamps)
    tab = pd.DataFrame(rows).reindex(columns=P75_COLS)
    os.makedirs(ps.PROSPECTIVE_DIR, exist_ok=True)
    tab.to_csv(ps.P75_PATH, index=False)
    show = tab[["site_group", "site_id", "station", "chl_p75_site", "n_station_days", "history_first", "history_last",
                "min_readings", "t_star_site"]].copy()
    show["expected_approx"] = show.site_id.map(EXPECTED_P75)
    show["chl_p75_site"] = show.chl_p75_site.round(2)
    print("\n" + show.to_markdown(index=False))
    print(f"\nwrote {ps.P75_PATH}: {len(tab)} stations | {stamps}")


if __name__ == "__main__":
    main()
