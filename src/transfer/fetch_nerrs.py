"""
fetch_nerrs.py -- NERRS SWMP 15-min sonde data -> data/transfer/nerrs_15min.csv
-------------------------------------------------------------------------------
Cross-site transfer input for src/transfer/transfer_eval.py (source "nerrs").
Narragansett Bay (NAR) is the training bay and is deliberately excluded.

WHY TWO SOURCES.  Chlorophyll fluorescence (ChlFluor) is an *optional* SWMP
parameter and the only open bulk feeds that carry it are:
  (1) NCEI accession 0052765 (NERRS SWMP 1995-01 .. 2011-08), plain HTTPS dir:
      https://www.ncei.noaa.gov/data/oceans/archive/arc0023/0052765/2.2/data/
        0-data/NERRS_CDMO_Archive_8.3.11/<station>wq<year>.csv
      Same layout as CDMO exports (F_<param> flag columns, local standard time).
      A 2009 sweep of all 107 wq stations found ChlFluor populated only at
      Chesapeake Bay MD (cbm), Mission-Aransas TX (mar), South Slough OR (sos),
      Waquoit Bay MA (wqb) and, from 2010, Wells ME inlet (welin). Chl starts
      2008/2009 at these stations, so the usable span is 2008 .. 2011-08.
  (2) IOOS sensors ERDDAP (mirror of CDMO via AOOS) for Kachemak Bay AK (kac):
      https://erddap.sensors.ioos.us/erddap/tabledap/nerrs_<station>wq.csv
      ChlFluor at kacss 2011-2025, kach3 2012-2025 (gaps), kacsd 2014-2018.
      QARTOD aggregate flags; times UTC (shifted here to AKST, UTC-9).
Sources tried and rejected: NCEI accession 0200366 (1994-2024) -> "not yet
available for download"; CDMO /waf/swmp_data_archives yearly CSVs (1995-2026)
-> ChlFluor column present but 100 % blank at every station checked
(CDMO strips the optional parameter from that export); CDMO SOAP web services
-> "Invalid ip" without registration.

FLAG HANDLING.  NCEI/CDMO F_ codes: 0 passed QAQC, 1 suspect, 4 historical,
5 corrected; negatives = missing / rejected / out of sensor range. A value is
kept only if its flag is 0, 4 or 5 (blank flag with a value present is kept);
suspect (1) and all negative codes are set to NaN. Applied per parameter
(ChlFluor, Temp, Sal, DO_mgl, DO_pct, pH). ERDDAP QARTOD qc_agg: 1 pass,
2 not evaluated, 3 suspect, 4 fail -> chl kept for 1, 2 (or missing flag).

STATIONS (name = <reserve>_<site>):
  cbm_ip cbm_mc cbm_oc cbm_rr | mar_ab mar_ce mar_cw mar_mb mar_sc
  sos_ch sos_se sos_va sos_wi | wqb_cr wqb_mh wqb_mp wqb_sl | wel_in
  kac_ss kac_h3 kac_sd
Raw downloads are cached under data/transfer/raw/nerrs/{ncei,erddap}/ (gitignored).

Usage (fork root, BASE env):
    python -m src.transfer.fetch_nerrs            # download (cached) + build
    python -m src.transfer.transfer_eval --source nerrs --min-readings 48
"""
import os

import pandas as pd
import requests

RAW = "data/transfer/raw/nerrs"
OUT = "data/transfer/nerrs_15min.csv"

NCEI_BASE = ("https://www.ncei.noaa.gov/data/oceans/archive/arc0023/0052765/2.2/"
             "data/0-data/NERRS_CDMO_Archive_8.3.11/")
NCEI_STATIONS = ["cbmip", "cbmmc", "cbmoc", "cbmrr",
                 "marab", "marce", "marcw", "marmb", "marsc",
                 "sosch", "sosse", "sosva", "soswi",
                 "wqbcr", "wqbmh", "wqbmp", "wqbsl",
                 "welin"]
NCEI_YEARS = range(2008, 2012)

ERDDAP_BASE = "https://erddap.sensors.ioos.us/erddap/tabledap/"
ERDDAP_STATIONS = ["kacss", "kach3", "kacsd"]
ERDDAP_VARS = ["time", "mass_concentration_of_chlorophyll_in_sea_water",
               "mass_concentration_of_chlorophyll_in_sea_water_qc_agg",
               "sea_water_temperature", "sea_water_practical_salinity",
               "mass_concentration_of_oxygen_in_sea_water",
               "fractional_saturation_of_oxygen_in_sea_water",
               "sea_water_ph_reported_on_total_scale"]
ERDDAP_UTC_OFFSET_H = -9          # Kachemak Bay: Alaska standard time

KEEP_FLAGS = {0, 4, 5}            # CDMO F_ codes accepted
ERDDAP_KEEP = {1, 2}              # QARTOD aggregate accepted


def _download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return True
    r = requests.get(url, timeout=600)
    if r.status_code != 200:
        print(f"  skip {url} (HTTP {r.status_code})")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return True


def _flagged(df, col, flag_col):
    """Return column as float with rejected/suspect flags set to NaN."""
    v = pd.to_numeric(df[col], errors="coerce")
    if flag_col not in df:
        return v
    code = df[flag_col].astype(str).str.extract(r"<\s*(-?\d+)\s*>")[0]
    code = pd.to_numeric(code, errors="coerce")
    ok = code.isin(KEEP_FLAGS) | (code.isna() & v.notna())
    return v.where(ok)


def load_ncei(station, year):
    fn = f"{station}wq{year}.csv"
    path = os.path.join(RAW, "ncei", fn)
    if not _download(NCEI_BASE + fn, path):
        return None
    d = pd.read_csv(path, low_memory=False)
    d.columns = [c.strip() for c in d.columns]
    out = pd.DataFrame({
        "station": f"{station[:3]}_{station[3:]}",
        "datetime": pd.to_datetime(d["DateTimeStamp"], format="%m/%d/%Y %H:%M", errors="coerce"),
        "chl_ugl": _flagged(d, "ChlFluor", "F_ChlFluor"),
        "temp_c": _flagged(d, "Temp", "F_Temp"),
        "salinity_psu": _flagged(d, "Sal", "F_Sal"),
        "do_mgl": _flagged(d, "DO_mgl", "F_DO_mgl"),
        "do_pct": _flagged(d, "DO_Pct", "F_DO_Pct"),
        "ph": _flagged(d, "pH", "F_pH"),
    })
    return out.dropna(subset=["datetime"])


def load_erddap(station):
    ds = f"nerrs_{station}wq"
    path = os.path.join(RAW, "erddap", ds + ".csv")
    url = f"{ERDDAP_BASE}{ds}.csv?" + ",".join(ERDDAP_VARS)
    if not _download(url, path):
        return None
    d = pd.read_csv(path, skiprows=[1])          # row 1 = units
    chl = pd.to_numeric(d[ERDDAP_VARS[1]], errors="coerce")
    qc = pd.to_numeric(d[ERDDAP_VARS[2]], errors="coerce")
    chl = chl.where(qc.isin(ERDDAP_KEEP) | qc.isna())
    t = pd.to_datetime(d["time"], utc=True) + pd.Timedelta(hours=ERDDAP_UTC_OFFSET_H)
    return pd.DataFrame({
        "station": f"{station[:3]}_{station[3:]}",
        "datetime": t.dt.tz_localize(None),
        "chl_ugl": chl,
        "temp_c": pd.to_numeric(d[ERDDAP_VARS[3]], errors="coerce"),
        "salinity_psu": pd.to_numeric(d[ERDDAP_VARS[4]], errors="coerce"),
        "do_mgl": pd.to_numeric(d[ERDDAP_VARS[5]], errors="coerce"),
        "do_pct": pd.to_numeric(d[ERDDAP_VARS[6]], errors="coerce"),
        "ph": pd.to_numeric(d[ERDDAP_VARS[7]], errors="coerce"),
    })


def main():
    parts = []
    for st in NCEI_STATIONS:
        for yr in NCEI_YEARS:
            p = load_ncei(st, yr)
            if p is not None:
                parts.append(p)
                print(f"  ncei   {st} {yr}: rows={len(p):6d} chl%={p.chl_ugl.notna().mean():.2f}")
    for st in ERDDAP_STATIONS:
        p = load_erddap(st)
        if p is not None:
            parts.append(p)
            print(f"  erddap {st}: rows={len(p):7d} chl%={p.chl_ugl.notna().mean():.2f} "
                  f"span={p.datetime.min().date()}..{p.datetime.max().date()}")
    df = pd.concat(parts, ignore_index=True)
    n_all = len(df)
    df = df.dropna(subset=["chl_ugl"]).sort_values(["station", "datetime"])
    df = df.drop_duplicates(["station", "datetime"])
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: rows={len(df):,} of {n_all:,} raw ({len(df)/n_all:.1%} with accepted chl) "
          f"stations={df.station.nunique()}")
    summ = df.assign(yr=df.datetime.str[:4]).groupby("station").agg(
        n=("chl_ugl", "size"), first=("datetime", "min"), last=("datetime", "max"),
        chl_med=("chl_ugl", "median"), chl_p75=("chl_ugl", lambda s: s.quantile(.75)),
        years=("yr", "nunique"))
    print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
