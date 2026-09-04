"""
satellite_fetch.py -- daily satellite chlorophyll + SST at every sonde station
-----------------------------------------------------------------------------
Milestone 0/1 of the "going global" plan. For each station in
data/transfer/stations_latlon.csv, pulls a 3x3-pixel box around the station
from each candidate product and stores the per-day MEDIAN of valid pixels.

Products (finest first). ERDDAP griddap, no login, research User-Agent:
  olci300   Copernicus OCEANCOLOUR_GLO_BGC_L3_MY_009_103, 300 m OLCI, 2016-
            (via the `copernicusmarine` toolbox; needs `copernicusmarine login`)
  olci750   CoastWatch S3A OLCI 750 m sectors -- archive starts 2025-08-29 only,
            so it cannot serve the 2016-2023 test; kept for future live use
  dineof2k  ERD noaacwNPPN20S3ASCIDINEOF2kmDaily, gap-filled VIIRS+OLCI 2 km, 2018-
  viirs4k   ERD erdVH2018chla1day, NASA S-NPP VIIRS SMI 4 km, 2012-2022-07
            (nesdisVHNSQchlaDaily was tried first: its time axis has month-long
            gaps on this server, so it was dropped)
  olci4k    CoastWatch noaacwS3AOLCIchlaDaily, 4 km, 2019-
  sst       ERD jplMURSST41, MUR 1 km daily SST, 2002-  (always pulled)

Stages:
  --stage viability   one summer month (2021-07) per station x product: counts
                      valid days/pixels in the 3x3 box -> data/transfer/satellite_viability.csv
  --stage pull        full window per station for every viable product
                      -> data/transfer/satellite/<product>/<source>__<station>.csv
                         (date, value, n_pixels)
Options: --sources, --products, --start, --end, --box (half-width in pixels).
Requests are cached; rerunning skips finished station-years. CoastWatch returns
502/403 when overloaded: backoff up to --max-wait seconds, then skip and log.

Run from fork root, BASE env:
    python -m src.transfer.satellite_fetch --stage viability
    python -m src.transfer.satellite_fetch --stage pull --start 2016-01-01 --end 2023-12-31
"""
import argparse
import io
import os
import sys
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (hab-bloom-predictor research; student project)"}
ERD = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
CW = "https://coastwatch.noaa.gov/erddap/griddap"
STATIONS = "data/transfer/stations_latlon.csv"
VIAB = "data/transfer/satellite_viability.csv"
CACHE = "data/transfer/satellite"
SECTORS = "data/transfer/satellite/coastwatch_olci_datasets.csv"

# name: (base, datasetID, var, has_altitude_dim, pixel_deg, start_year)
PRODUCTS = {
    "viirs4k":  (ERD, "erdVH2018chla1day", "chla", False, 0.0417, 2012),   # NASA VIIRS SMI, 2012-2022-07; lat axis DESCENDING
    "dineof2k": (ERD, "noaacwNPPN20S3ASCIDINEOF2kmDaily", "chlor_a", True, 0.02, 2018),
    "olci4k":   (CW, "noaacwS3AOLCIchlaDaily", "chlor_a", True, 0.0375, 2019),   # data begin 2019-06-06
    "olci750":  (CW, None, "chlor_a", True, 0.0075, 2020),      # sector resolved per station
    "olci300":  (None, "cmems_obs-oc_glo_bgc-plankton_my_l3-olci-300m_P1D", "CHL", False, 0.003, 2016),
    "sst":      (ERD, "jplMURSST41", "analysed_sst", False, 0.01, 2002),
}


def sector_for(lat, lon):
    if not os.path.exists(SECTORS):
        return None
    s = pd.read_csv(SECTORS)
    s = s[s.datasetID.str.contains("Sector") & s.datasetID.str.startswith("noaacwS3AOLCIchla")]
    hit = s[(s.minLatitude <= lat) & (lat <= s.maxLatitude) & (s.minLongitude <= lon) & (lon <= s.maxLongitude)]
    return None if hit.empty else hit.datasetID.iloc[0]


def erddap_box(base, ds, var, has_alt, lat, lon, half, t0, t1, max_wait):
    alt = "[(0.0)]" if has_alt else ""
    lat_a, lat_b = (lat + half, lat - half) if ds == "erdVH2018chla1day" else (lat - half, lat + half)
    url = (f"{base}/{ds}.csv?{var}[({t0}):({t1})]{alt}"
           f"[({lat_a}):({lat_b})][({lon - half}):({lon + half})]")
    wait, waited = 30, 0
    while True:
        try:
            r = requests.get(url, headers=UA, timeout=600); code = r.status_code
        except requests.RequestException as e:
            r, code = None, str(e)[:60]
        if r is not None and code == 200:
            d = pd.read_csv(io.StringIO(r.text), skiprows=[1])
            d["date"] = pd.to_datetime(d["time"]).dt.normalize()
            return d.rename(columns={var: "value"})[["date", "latitude", "longitude", "value"]]
        if r is not None and code == 404:          # outside the dataset's time/space range: nothing to fetch
            return pd.DataFrame(columns=["date", "latitude", "longitude", "value"])
        if waited >= max_wait:
            print(f"    giving up ({code}) {ds} {t0}..{t1}", file=sys.stderr); return None
        time.sleep(wait); waited += wait; wait = min(wait * 2, 300)


def copernicus_box(ds_id, var, lat, lon, half, t0, t1):
    try:
        import copernicusmarine as cm
    except ImportError:
        print("    copernicusmarine not installed; skipping olci300", file=sys.stderr); return None
    try:
        ds = cm.open_dataset(dataset_id=ds_id, variables=[var],
                             minimum_longitude=lon - half, maximum_longitude=lon + half,
                             minimum_latitude=lat - half, maximum_latitude=lat + half,
                             start_datetime=t0, end_datetime=t1)
        d = ds[var].to_dataframe().reset_index()
    except Exception as e:
        print(f"    copernicus failed: {str(e)[:120]}", file=sys.stderr); return None
    d["date"] = pd.to_datetime(d["time"]).dt.normalize()
    return d.rename(columns={var: "value"})[["date", "latitude", "longitude", "value"]]


def daily_median(box):
    if box is None:
        return None
    if box.empty:
        return pd.DataFrame(columns=["date", "value", "n_pixels"])
    g = box.dropna(subset=["value"]).groupby("date")["value"]
    return g.agg(value="median", n_pixels="size").reset_index()


def fetch(product, st, t0, t1, box_px, max_wait):
    base, ds, var, has_alt, px, _ = PRODUCTS[product]
    half = px * box_px
    if product == "olci300":
        return daily_median(copernicus_box(ds, var, st.lat, st.lon, half, t0, t1))
    if product == "olci750":
        ds = sector_for(st.lat, st.lon)
        if ds is None:
            return pd.DataFrame(columns=["date", "value", "n_pixels"])
    return daily_median(erddap_box(base, ds, var, has_alt, st.lat, st.lon, half, t0, t1, max_wait))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["viability", "pull"], required=True)
    ap.add_argument("--sources", default="all")
    ap.add_argument("--products", default="viirs4k,dineof2k,olci4k,olci750,olci300,sst")
    ap.add_argument("--start", default="2016-01-01"); ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--box", type=int, default=1, help="half-width in pixels (1 -> 3x3)")
    ap.add_argument("--max-wait", type=int, default=900)
    a = ap.parse_args()
    stations = pd.read_csv(STATIONS).dropna(subset=["lat", "lon"])
    if a.sources != "all":
        stations = stations[stations.source.isin(a.sources.split(","))]
    products = [p for p in a.products.split(",") if p in PRODUCTS]

    if a.stage == "viability":
        # incremental + resumable: one row appended per station-product as it completes
        done = pd.read_csv(VIAB) if os.path.exists(VIAB) else pd.DataFrame(columns=["source", "station", "product", "valid_days"])
        done = done[done.valid_days >= 0]
        finished = set(zip(done.source, done.station, done["product"]))
        for _, st in stations.iterrows():
            for p in products:
                if p == "sst" or (st.source, st.station, p) in finished:
                    continue
                d = fetch(p, st, "2021-07-01", "2021-07-31", a.box, a.max_wait)
                row = dict(source=st.source, station=st.station, product=p, days_checked=31,
                           valid_days=-1 if d is None else len(d),
                           max_pixels=-1 if d is None else (int(d.n_pixels.max()) if len(d) else 0))
                pd.DataFrame([row]).to_csv(VIAB, mode="a", header=not os.path.exists(VIAB), index=False)
                print(f"{st.source:12s} {st.station:14s} {p:9s} valid_days={row['valid_days']:2d} pixels={row['max_pixels']}", flush=True)
        print(f"viability rows now in {VIAB}  (-1 = server unreachable, rerun to retry)")
        return

    viab = pd.read_csv(VIAB) if os.path.exists(VIAB) else None
    years = range(int(a.start[:4]), int(a.end[:4]) + 1)
    for _, st in stations.iterrows():
        for p in products:
            if viab is not None and p != "sst":
                v = viab[(viab.source == st.source) & (viab.station == st.station) & (viab["product"] == p)]
                if len(v) and v.valid_days.iloc[0] == 0:
                    continue
            if p == "olci750" and sector_for(st.lat, st.lon) is None:
                continue
            os.makedirs(f"{CACHE}/{p}", exist_ok=True)
            out = f"{CACHE}/{p}/{st.source}__{st.station}.csv"
            done = (pd.read_csv(out, parse_dates=["date"]) if os.path.exists(out)
                    else pd.DataFrame(columns=["date", "value", "n_pixels"]))
            done_years = set(pd.to_datetime(done.date).dt.year.unique()) if len(done) else set()
            parts = [done]
            for y in years:
                if y in done_years or y < PRODUCTS[p][5]:
                    continue
                t0, t1 = max(a.start, f"{y}-01-01"), min(a.end, f"{y}-12-31")
                if p == "olci4k" and y == 2019: t0 = max(t0, "2019-06-06")
                if p == "viirs4k" and y == 2022: t1 = min(t1, "2022-07-25")
                d = fetch(p, st, t0, t1, a.box, a.max_wait)
                if d is None:
                    continue
                parts.append(d)
                print(f"{st.source:12s} {st.station:14s} {p:9s} {y}: {len(d):3d} valid days", flush=True)
            allp = pd.concat(parts, ignore_index=True)[["date", "value", "n_pixels"]]
            allp.drop_duplicates("date").sort_values("date").to_csv(out, index=False)


if __name__ == "__main__":
    main()
