"""
fetch_cefas.py -- Cefas SmartBuoy (UK shelf seas) -> data/transfer/cefas_15min.csv
-----------------------------------------------------------------------------------
Builds the tidy sub-daily input for src/transfer/transfer_eval.py:
    station, datetime, chl_ugl, temp_c, salinity_psu, do_mgl, do_pct

WHAT IS (AND IS NOT) OPENLY DOWNLOADABLE  (checked 2026-09-03)
  The long-running NMMP SmartBuoy archive (Liverpool Bay 2002-, West Gabbard /
  WestGabbard2 2001-, Warp 2000-, Dowsing 2009-, Celtic Deep 2009-12; 30-min
  bursts) is catalogued on the Cefas Data Hub as holdings 3040-3053
  (https://data.cefas.co.uk/#/View/3047 etc.) but those holdings carry NO
  recordsets: the data are served only through the WaveNet/SmartBuoy portal
  (https://smartbuoy.cefas.co.uk -> API https://smartbuoy-api.cefas.co.uk/api).
  That API's archive routes (download/criteria, download/platforms,
  download/parameters) answer 401 without a JWT; Account/Sign-In needs an
  existing account and Account/Register requires an e-mail + password, so the
  archive cannot be fetched without a human-registered login. Only the live
  "summary" route (latest 30-min values for WESTGAB2, LIVBAY, TH1) is public.
  No open mirror was found: EMODnet Physics ERDDAP has no Cefas/Gabbard/Dowsing/
  Liverpool/Warp datasets, data.gov.uk / DASSH records point back to the Hub,
  BODC banks the series (e.g. series 747094, 947312) behind its own login.

  What IS open (OGL v3) with 30-min chlorophyll fluorescence, T, S, O2 at 1 m:
    holding 18641  Shelf Sea Biogeochemistry - Celtic Deep 2 SmartBuoy
                   DOI 10.14466/CefasDataHub.39   recordset 9797
                   https://data-api.cefas.co.uk/api/export/9797?format=csv
                   CELTDEEP2 51.138N 6.562W, 2014-01-01 .. 2015-08-15
    holding 18640  Shelf Sea Biogeochemistry - CANDYFLOSS SmartBuoy
                   DOI 10.14466/CefasDataHub.37   recordset 9795
                   https://data-api.cefas.co.uk/api/export/9795?format=csv
                   CANDYFLOSS (central Celtic Sea), 2014-03-27 .. 2015-08-23
  Other SmartBuoy recordsets on the Hub have no chlorophyll (3376 Dowsing
  "Emeco Test Case" May-2010 T/S/O2 only; 3814 North Dogger 2007-08 daily
  T/S/O2 only; 12945 temperature-only UK-shelf compilation) and are not used.

FORMAT OF THE SOURCE CSVs (long format, one row per burst mean per sensor)
  dateTime ("M/D/YYYY h:MM:SS AM/PM", UTC), deployment (e.g. CELTDEEP2/008),
  deployment_group, lat, lon, depth (m), value, stdev, n (samples in the
  5-min burst; 1 Hz), sensor, sensor_serial, par, unit.
  par codes used:  FLUORS  Seapoint chlorophyll fluorometer, "arb. unit"
                   TEMP    Aanderaa 3919B conductivity sensor, degC
                   SAL     PSS-78
                   O2CONC  Aanderaa optode 3835/3830, mmol m-3
  (FTU turbidity, PAR_A, TEMP_alt/SAL_alt, and the discrete nutrient bottles
   NITRIT/TOXN/SILICA are ignored.)

UNITS / CONVERSIONS
  chl_ugl      = FLUORS as delivered. The fluorometers were standardised with
                 fluorosphere beads, NOT calibrated to extracted chl-a, so the
                 values are relative (0.02-8.6 here). The harness's p75 bloom
                 definition and quantile-mapped zero-shot are scale-free; the
                 abs10 (>10 ug/L) definition is meaningless for this column.
  do_mgl       = O2CONC [mmol m-3] * 31.998 / 1000   (mmol m-3 == umol L-1)
  do_pct       = NaN (no O2SAT rows in these recordsets)
  temp_c, salinity_psu straight through.

QC HANDLING
  The Hub CSVs carry no flag column (Cefas applies its SOP QA before
  publication; SSB salinity/nutrients validated via QUASIMEME). Applied here:
  depth == 1 m only; non-finite values dropped; FLUORS < 0 dropped; SAL outside
  20-40 dropped; TEMP outside -2..35 dropped; O2CONC <= 0 dropped; duplicate
  (station, datetime, par) rows averaged (two optodes overlap on CANDYFLOSS).

Usage (fork root, BASE env):
    python -m src.transfer.fetch_cefas
    python -m src.transfer.transfer_eval --source cefas --min-readings 12
"""
import os
import numpy as np
import pandas as pd
import requests

RAW = "data/transfer/raw/cefas"
OUT = "data/transfer/cefas_15min.csv"
API = "https://data-api.cefas.co.uk/api/export/{rs}?format=csv"
RECORDSETS = {  # recordset id -> (station label, cached filename)
    9797: ("CELTDEEP2", "CELTDEEP2_2014_2015_rs9797.csv"),
    9795: ("CANDYFLOSS", "CANDYFLOSS_2014_2015_rs9795.csv"),
}
PAR = {"FLUORS": "chl_ugl", "TEMP": "temp_c", "SAL": "salinity_psu", "O2CONC": "do_mgl"}


def fetch(rs, fname):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, fname)
    if not os.path.exists(path):
        r = requests.get(API.format(rs=rs), timeout=900)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    return path


def parse(path, station):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["par"].isin(PAR) & (df["depth"] == 1.0)].copy()
    df["datetime"] = pd.to_datetime(df["dateTime"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["datetime", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[np.isfinite(df["value"])]
    ok = ((df.par == "FLUORS") & (df.value >= 0)) | \
         ((df.par == "TEMP") & df.value.between(-2, 35)) | \
         ((df.par == "SAL") & df.value.between(20, 40)) | \
         ((df.par == "O2CONC") & (df.value > 0))
    df = df[ok]
    wide = df.pivot_table(index="datetime", columns="par", values="value", aggfunc="mean")
    wide = wide.rename(columns=PAR).reset_index()
    for c in PAR.values():
        if c not in wide: wide[c] = np.nan
    wide["do_mgl"] = wide["do_mgl"] * 31.998 / 1000.0
    wide["do_pct"] = np.nan
    wide.insert(0, "station", station)
    return wide[["station", "datetime", "chl_ugl", "temp_c", "salinity_psu", "do_mgl", "do_pct"]]


def main():
    parts = [parse(fetch(rs, fn), st) for rs, (st, fn) in RECORDSETS.items()]
    out = pd.concat(parts).sort_values(["station", "datetime"]).reset_index(drop=True)
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: rows={len(out):,} stations={out.station.nunique()} "
          f"{out.datetime.min()} .. {out.datetime.max()}")
    for st, g in out.groupby("station"):
        dt = pd.to_datetime(g.datetime).diff().dt.total_seconds().div(60)
        print(f"  {st}: rows={len(g):,} chl%={g.chl_ugl.notna().mean()*100:.1f} "
              f"temp%={g.temp_c.notna().mean()*100:.1f} sal%={g.salinity_psu.notna().mean()*100:.1f} "
              f"do%={g.do_mgl.notna().mean()*100:.1f} median step={dt.median():.0f} min "
              f"chl median={g.chl_ugl.median():.2f} p75={g.chl_ugl.quantile(.75):.2f} max={g.chl_ugl.max():.2f}")


if __name__ == "__main__":
    main()
