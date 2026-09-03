"""
fetch_lake_erie.py -- western Lake Erie (freshwater cyanobacteria) buoys -> tidy 15-min table
------------------------------------------------------------------------------------------
Builds data/transfer/lake_erie_15min.csv for src/transfer/transfer_eval.py.

SOURCE  NOAA GLERL / CIGLR western Lake Erie HAB monitoring buoys, archived at NCEI
        (parent accession GLERL-CIGLR-HAB-LakeErie-water-qual; ESSD paper
        https://essd.copernicus.org/articles/15/3853/2023/). Continuous moored-buoy
        files (NOT the weekly grab samples, which are in the small-boat accessions):
          WE02  NCEI 0190201  https://www.ncei.noaa.gov/data/oceans/archive/arc0140/0190201/1.1/data/0-data/
          WE04  NCEI 0190729  https://www.ncei.noaa.gov/data/oceans/archive/arc0140/0190729/1.1/data/0-data/
          WE08  NCEI 0194301  https://www.ncei.noaa.gov/data/oceans/archive/arc0142/0194301/1.1/data/0-data/
          WE13  NCEI 0194302  https://www.ncei.noaa.gov/data/oceans/archive/arc0142/0194302/1.1/data/0-data/
        files: <STATION>_<YEAR>_annual_summary.csv (one per deployment season)
YEARS   2014-2018 (WE13 from 2015). Buoys deployed ~May-Oct/Nov only.
CADENCE 15 min (WE02 2014 is hourly). Timestamps UTC.
SENSOR  YSI EXO2 sonde ~1 m below surface: water_temperature (degC),
        chlorophylla (RFU, fluorescence), organic_dissolved_oxygen (mg/L),
        organic_dissolved_oxygen_saturation (%), pH, specific_conductivity (uS/cm or mS/cm).
        WE02 2014 reports chlorophylla_rfu AND chlorophylla (ug/L); WE04 2014 has no EXO
        chlorophyll (only a C6 sensor on a different scale) and drops out.
UNITS   chl_ugl = RFU * 4.0. YSI EXO chlorophyll ug/L is a linear rescale of RFU; the
        in-dataset ratio (WE02 2014, both columns present) is 4.10 +/- 0.32 (median 3.96),
        so 4.0 is used as the nominal factor. This only matters for the abs10 label and
        zeroshot_raw; the p75 label and zeroshot_qm are scale-free.
QC      Files from 2015 on carry a per-variable QARTOD flag string "a b c d"
        (1 pass, 2 not evaluated, 3 suspect, 4 fail, 9 missing). Any '4' or '9' in a
        variable's flag string -> that value set to NaN. 2014 files have no flags.
        Sentinel 'NAN'/'NA' and negative chl -> NaN.
FRESH   salinity_psu = 0.0 for every row (freshwater; harness convention).

Also probed: GLOS Seagull ERDDAP (seagull-erddap.glos.org, datasets obs_80/81/82/215 =
GLERLWE2/4/8/13) -- only WE2 2024-09-03..2024-10-18 is served; the others are empty.
GLERL rtMonSQL.php returns an empty body to curl. Not used.

Usage (fork root, BASE env):
    python -m src.transfer.fetch_lake_erie
"""
import os
import urllib.request

import numpy as np
import pandas as pd

RAW = "data/transfer/raw/lake_erie/ncei"
OUT = "data/transfer/lake_erie_15min.csv"
BASE = "https://www.ncei.noaa.gov/data/oceans/archive"
RFU_TO_UGL = 4.0

SOURCES = {  # station -> (archive path, years)
    "WE02": ("arc0140/0190201", [2014, 2015, 2016, 2017, 2018]),
    "WE04": ("arc0140/0190729", [2014, 2015, 2016, 2017, 2018]),
    "WE08": ("arc0142/0194301", [2014, 2015, 2016, 2017, 2018]),
    "WE13": ("arc0142/0194302", [2015, 2016, 2017, 2018]),
}


def download():
    os.makedirs(RAW, exist_ok=True)
    for st, (path, years) in SOURCES.items():
        for y in years:
            f = f"{st}_{y}_annual_summary.csv"
            dst = os.path.join(RAW, f)
            if os.path.exists(dst) and os.path.getsize(dst) > 10_000:
                continue
            url = f"{BASE}/{path}/1.1/data/0-data/{f}"
            print("fetch", url)
            urllib.request.urlretrieve(url, dst)


def _num(s):
    return pd.to_numeric(s.replace({"NAN": np.nan, "NA": np.nan}), errors="coerce")


def _apply_flags(df, col):
    """NaN-out values whose QARTOD flag string contains a fail (4) or missing (9)."""
    fc = f"{col}_flags"
    if fc in df.columns:
        bad = df[fc].astype(str).str.contains(r"\b[49]\b", regex=True)
        df.loc[bad, col] = np.nan


def parse_file(path, station, year):
    df = pd.read_csv(path, skiprows=[1, 2], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # chlorophyll (RFU): 2014 WE02 has rfu + ug/L columns; use the RFU one for scale consistency
    chl_col = "chlorophylla_rfu" if "chlorophylla_rfu" in df.columns else "chlorophylla"
    if chl_col not in df.columns:
        return None
    for c in (chl_col, "water_temperature", "organic_dissolved_oxygen",
              "organic_dissolved_oxygen_saturation", "pH", "ph"):
        if c in df.columns:
            df[c] = _num(df[c])
            _apply_flags(df, c)
    ph_col = "pH" if "pH" in df.columns else ("ph" if "ph" in df.columns else None)
    out = pd.DataFrame({
        "station": station,
        "datetime": pd.to_datetime(df["timestamp"], errors="coerce", format="mixed"),
        "chl_rfu": df[chl_col],
        "temp_c": df["water_temperature"] if "water_temperature" in df.columns else np.nan,
        "salinity_psu": 0.0,
        "do_mgl": df["organic_dissolved_oxygen"] if "organic_dissolved_oxygen" in df.columns else np.nan,
        "do_pct": df["organic_dissolved_oxygen_saturation"] if "organic_dissolved_oxygen_saturation" in df.columns else np.nan,
        "ph": df[ph_col] if ph_col else np.nan,
    })
    out.loc[out.chl_rfu < 0, "chl_rfu"] = np.nan
    out.loc[(out.temp_c < -2) | (out.temp_c > 40), "temp_c"] = np.nan
    out.loc[(out.do_mgl < 0) | (out.do_mgl > 30), "do_mgl"] = np.nan
    out["chl_ugl"] = out.chl_rfu * RFU_TO_UGL
    out = out.dropna(subset=["datetime"])
    out = out[out.datetime.dt.year == year]
    return out


def main():
    download()
    parts = []
    for st, (_, years) in SOURCES.items():
        for y in years:
            p = os.path.join(RAW, f"{st}_{y}_annual_summary.csv")
            d = parse_file(p, st, y)
            if d is None:
                print(f"  {st} {y}: no EXO chlorophyll column -- skipped"); continue
            print(f"  {st} {y}: rows={len(d):,} chl={d.chl_ugl.notna().mean():.1%} "
                  f"cadence={d.datetime.diff().dt.total_seconds().median()/60:.0f}min "
                  f"{d.datetime.min():%Y-%m-%d}..{d.datetime.max():%Y-%m-%d}")
            parts.append(d)
    df = pd.concat(parts).sort_values(["station", "datetime"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["station", "datetime"])
    cols = ["station", "datetime", "chl_ugl", "temp_c", "salinity_psu", "do_mgl", "do_pct", "ph"]
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df[cols].to_csv(OUT, index=False)
    n = len(df); nchl = df.chl_ugl.notna().sum()
    print(f"\nwrote {OUT}: rows={n:,} with_chl={nchl:,} ({nchl/n:.1%}) "
          f"stations={df.station.nunique()} years={df.datetime.str[:4].min()}-{df.datetime.str[:4].max()}")
    print(df.groupby("station").chl_ugl.describe()[["count", "50%", "75%", "max"]].round(2))


if __name__ == "__main__":
    main()
