"""
fetch_imos.py -- IMOS National Reference Station moorings -> transfer harness input
---------------------------------------------------------------------------------
Builds data/transfer/imos_15min.csv (HOURLY cadence, despite the file name the
harness expects; run the harness with --min-readings 6) from the IMOS/AODN
"hourly time series" FV02 products of the Australian National Mooring Network
(ANMN) National Reference Stations (NRS).

SOURCE  AWS Open Data mirror of the AODN archive (public HTTPS, no credentials):
        https://imos-data.s3.ap-southeast-2.amazonaws.com/<key>     (keys in FILES)
        Same files on THREDDS:
        https://thredds.aodn.org.au/thredds/catalog/IMOS/ANMN/NRS/<site>/hourly_timeseries/catalog.html
        Product used: IMOS_ANMN-NRS_B?OST?Z_<start>_<site>_FV02_hourly-timeseries-including-non-QC_END-<end>_C-<created>.nc
        File metadata: included_values_flagged_as = "No_QC_performed, Good_data,
        Probably_good_data", i.e. IMOS flags 0/1/2 kept, 3 (probably bad) and
        4 (bad) DROPPED, out-of-water samples removed, then binned to 1 h
        (CPHL = hourly MEDIAN of the ~15-min WQM/FLNTU bursts; TEMP/PSAL/DOX =
        hourly MEAN).  WHY NOT the strictly-QC'd variant ("hourly-timeseries",
        flags 1-2 only): in the source FV01 files the fluorometric CPHL is
        left at flag 0 (no QC performed) for nearly all deployments, so that
        variant keeps <2% of the CPHL record at MAI/ROT/ESP/PH100 (checked
        2026-09-03).  TEMP/PSAL/DOX are QC'd (flag 1-2) in both variants.
        Every observation carries instrument_index -> (instrument_id,
        NOMINAL_DEPTH, source FV01 file), so the surface-most instrument can be
        selected per hour.

STATIONS (site code, region, CPHL coverage at the depth band used)
  NRSMAI  Maria Island, Tasmania        2008-2017  WQM 20 m (site 90 m)
  NRSROT  Rottnest Island, WA           2010-2026  WQM 22-24 m (site 45 m)
  NRSNSI  North Stradbroke Island, QLD  2010-2019  WQM 20 m (site 60 m)
  NRSESP  Esperance, WA                 2008-2012  WQM 21-25 m (site 50 m; closed 2013)
  NRSYON  Yongala, QLD (GBR lagoon)     2015-2023  WQM/FLNTU 0.5-2 m (site 30 m)
  NRSDAR  Darwin, NT                    2014-2025  WQM/FLNTU 0.1-2 m (site 22 m)
  PH100   Port Hacking 100 m, NSW       2010,2013-2015 WQM 15 m (NSW sub-facility;
          IMOS/ANMN/NRS/NRSPHB holds only ship profiles, the moored record is
          IMOS/ANMN/NSW/PH100).  Only 4 years -> included but below the
          5-year target.
  NRSKAI  Kangaroo Island: NO calibrated CPHL above 30 m (only CHLU counts at
          depth) -> dropped.  Only the CPHL variable is used (CHLF/CHLU from
          older/other sensors are ignored to avoid mixing sensor scales).

INSTRUMENT / DEPTH SELECTION
  Instruments are grouped into depth bands by NOMINAL_DEPTH: <5 m, 5-15 m,
  15-30 m.  Per station the SHALLOWEST band with >= MIN_YEARS calendar years
  of >= MIN_HOURS_PER_YEAR valid CPHL hours is used; within it, the shallowest
  instrument with valid CPHL in each hour.  Rationale: at YON and DAR the
  ~1 m and ~18-28 m (near-bottom) fluorometers are essentially uncorrelated
  hour-to-hour (r = 0.18 and 0.03), so splicing them per hour would mix water
  masses; the ~1 m instrument only exists from 2015 (YON) / 2014 (DAR), so
  those stations start then.  If the chosen CPHL instrument has no TEMP/PSAL/
  DO in an hour, they are filled from the next-shallowest instrument (<= 30 m)
  in the same hour.  Realised depths are saved to
  data/transfer/raw/imos/imos_provenance.csv.

UNITS
  CPHL  mg m-3 == ug/L  -> chl_ugl (fluorometric "artificial chlorophyll")
  TEMP  degC            -> temp_c
  PSAL  PSU (file metadata says "S m-1"; values are practical salinity ~35)
  DO    DOX1 umol/L * 0.032 = mg/L ; else DOX2 umol/kg * 1.025 * 0.032 = mg/L
        (O2 molar mass 32 g/mol; seawater density ~1.025 kg/L)

RAW FILES are cached under data/transfer/raw/imos/<site>_hourly_nonqc.nc (gitignored).
Run (fork root, BASE env):
    python -m src.transfer.fetch_imos            # download-if-missing + parse
    python -m src.transfer.transfer_eval --source imos --min-readings 6
"""
import os
import sys
import urllib.request

import netCDF4 as nc
import numpy as np
import pandas as pd

S3 = "https://imos-data.s3.ap-southeast-2.amazonaws.com/"
RAW_DIR = "data/transfer/raw/imos"
OUT = "data/transfer/imos_15min.csv"
MAX_DEPTH_M = 30.0
BANDS = [(-1, 5), (5, 15), (15, 30)]      # (lo, hi] nominal-depth bands, shallow first
MIN_YEARS = 5
MIN_HOURS_PER_YEAR = 1000
O2_UMOL_L_TO_MG_L = 0.032          # 32 g/mol
SEAWATER_DENSITY_KG_L = 1.025

FILES = {
    "NRSMAI": "IMOS/ANMN/NRS/NRSMAI/hourly_timeseries/IMOS_ANMN-NRS_BOSTUZ_20080411_NRSMAI_FV02_hourly-timeseries-including-non-QC_END-20260421_C-20260815.nc",
    "NRSROT": "IMOS/ANMN/NRS/NRSROT/hourly_timeseries/IMOS_ANMN-NRS_BOSTUZ_20081120_NRSROT_FV02_hourly-timeseries-including-non-QC_END-20260724_C-20260801.nc",
    "NRSNSI": "IMOS/ANMN/NRS/NRSNSI/hourly_timeseries/IMOS_ANMN-NRS_BOSTUZ_20101213_NRSNSI_FV02_hourly-timeseries-including-non-QC_END-20260106_C-20260510.nc",
    "NRSESP": "IMOS/ANMN/NRS/NRSESP/hourly_timeseries/IMOS_ANMN-NRS_BOSTUZ_20081127_NRSESP_FV02_hourly-timeseries-including-non-QC_END-20131205_C-20220622.nc",
    "NRSYON": "IMOS/ANMN/NRS/NRSYON/hourly_timeseries/IMOS_ANMN-NRS_BFOSTUZ_20080623_NRSYON_FV02_hourly-timeseries-including-non-QC_END-20240304_C-20240420.nc",
    "NRSDAR": "IMOS/ANMN/NRS/NRSDAR/hourly_timeseries/IMOS_ANMN-NRS_BFOSTUZ_20090816_NRSDAR_FV02_hourly-timeseries-including-non-QC_END-20251118_C-20260509.nc",
    "PH100":  "IMOS/ANMN/NSW/PH100/hourly_timeseries/IMOS_ANMN-NSW_BOSTUZ_20091029_PH100_FV02_hourly-timeseries-including-non-QC_END-20260616_C-20260808.nc",
    # "NRSKAI": no calibrated CPHL above 30 m -- see docstring
}


def download(site):
    path = os.path.join(RAW_DIR, f"{site}_hourly_nonqc.nc")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(RAW_DIR, exist_ok=True)
    url = S3 + FILES[site]
    print(f"[{site}] downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def _var(ds, name, n):
    if name not in ds.variables:
        return np.full(n, np.nan)
    return np.ma.filled(ds.variables[name][:].astype("float64"), np.nan)


def pick_band(chl):
    """Shallowest depth band with >= MIN_YEARS years of >= MIN_HOURS_PER_YEAR CPHL hours."""
    for lo, hi in BANDS:
        sub = chl[(chl.depth > lo) & (chl.depth <= hi)]
        if sub.empty:
            continue
        good_years = (sub.groupby(sub.datetime.dt.year).size() >= MIN_HOURS_PER_YEAR).sum()
        if good_years >= MIN_YEARS:
            return lo, hi
    lo, hi = BANDS[-1]
    return -1, hi        # fall back to everything <= MAX_DEPTH_M


def parse(site, path):
    ds = nc.Dataset(path)
    t = nc.num2date(ds.variables["TIME"][:], ds.variables["TIME"].units,
                    only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    idx = np.asarray(ds.variables["instrument_index"][:]).astype(int)
    ndep = np.ma.filled(ds.variables["NOMINAL_DEPTH"][:].astype("float64"), np.nan)
    n = len(idx)
    df = pd.DataFrame({
        "datetime": pd.to_datetime(t),
        "depth": ndep[idx],
        "cphl": _var(ds, "CPHL", n),
        "temp": _var(ds, "TEMP", n),
        "psal": _var(ds, "PSAL", n),
        "dox1": _var(ds, "DOX1", n),
        "dox2": _var(ds, "DOX2", n),
    })
    ds.close()
    do = df["dox1"] * O2_UMOL_L_TO_MG_L
    df["do"] = do.where(do.notna(), df["dox2"] * SEAWATER_DENSITY_KG_L * O2_UMOL_L_TO_MG_L)
    df = df[df.depth <= MAX_DEPTH_M].copy()

    chl_all = df.dropna(subset=["cphl"])
    lo, hi = pick_band(chl_all)
    chl = chl_all[(chl_all.depth > lo) & (chl_all.depth <= hi)].sort_values(["datetime", "depth"])
    chl = chl.drop_duplicates("datetime", keep="first").copy()
    aux = df.sort_values(["datetime", "depth"])
    for c in ("temp", "psal", "do"):
        fill = aux.dropna(subset=[c]).drop_duplicates("datetime", keep="first").set_index("datetime")[c]
        chl[c] = chl[c].where(chl[c].notna(), chl["datetime"].map(fill))
    out = pd.DataFrame({
        "station": site,
        "datetime": chl["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S").values,
        "chl_ugl": chl["cphl"].round(4).values,
        "temp_c": chl["temp"].round(3).values,
        "salinity_psu": chl["psal"].round(3).values,
        "do_mgl": chl["do"].round(3).values,
    })
    depth_counts = chl["depth"].round(1).value_counts().sort_index()
    prov = dict(station=site, band=f"({lo},{hi}] m", n_hours=len(out),
                start=str(chl.datetime.min().date()), end=str(chl.datetime.max().date()),
                n_years_with_data=int(chl.datetime.dt.year.nunique()),
                depths_m=";".join(f"{d:g}m:{k}" for d, k in depth_counts.items()),
                pct_temp=round(100 * out.temp_c.notna().mean(), 1),
                pct_sal=round(100 * out.salinity_psu.notna().mean(), 1),
                pct_do=round(100 * out.do_mgl.notna().mean(), 1),
                chl_median=float(out.chl_ugl.median()), chl_p95=float(out.chl_ugl.quantile(0.95)))
    return out, prov


def main(sites=None):
    sites = sites or list(FILES)
    parts, provs = [], []
    for s in sites:
        try:
            path = download(s)
        except Exception as e:  # noqa: BLE001
            print(f"[{s}] download failed: {e}", flush=True)
            continue
        out, prov = parse(s, path)
        print(f"[{s}] band={prov['band']} hours={len(out):,} {prov['start']}..{prov['end']} "
              f"years={prov['n_years_with_data']} depths={prov['depths_m']} chl med={prov['chl_median']:.2f} "
              f"p95={prov['chl_p95']:.2f} T%={prov['pct_temp']:.0f} S%={prov['pct_sal']:.0f} "
              f"DO%={prov['pct_do']:.0f}", flush=True)
        parts.append(out)
        provs.append(prov)
    df = pd.concat(parts, ignore_index=True).sort_values(["station", "datetime"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.DataFrame(provs).to_csv(os.path.join(RAW_DIR, "imos_provenance.csv"), index=False)
    print(f"wrote {OUT}: rows={len(df):,} stations={df.station.nunique()}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
