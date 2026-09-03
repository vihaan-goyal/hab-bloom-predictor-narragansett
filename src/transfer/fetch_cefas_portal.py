"""
fetch_cefas_portal.py -- parse the Cefas SmartBuoy portal export (full archive)
------------------------------------------------------------------------------
Source: https://smartbuoy.cefas.co.uk/download (free login; export requested
2026-09-03 by the project owner: 7 platforms, post-recovery data, CSV,
28/08/2002 - 03/09/2026, parameters temperature, salinity, fluorescence MT+SP,
oxygen % saturation). Zip saved to data/transfer/raw/cefas/portal/ and
unzipped there. Licence PDFs (OGL) ship with the data.

Platforms -> station ids:
  west-gabbard-smartbuoy (2002-2016)          wgab
  west-gabbard-2-smartbuoy (2016-2026)        wgab2   (~2 km from wgab)
  liverpool-bay-coastal-observatory (2002-)   lbay
  dowsing-smartbuoy (2009-2019)               dows
  oyster-ground-smartbuoy (2006-2013)         oyst
  celtic-deep-smartbuoy (2009-2012)           cdeep
  celtic-deep-smartbuoy-site-2 (2012-2015)    cdeep2

Layout: one CSV per platform, 30-min rows, 'Time (GMT)' as dd/mm/yyyy HH:MM.
Column mapping (first non-null wins):
  chl_ugl      Fluorescence (SP) at 1 m, else Fluorescence (MT) at 1 m
               [arbitrary units, bead-standardised; NOT ug/L -- only the
               p75 label is meaningful downstream]
  temp_c       Temperature at 1 m, else at -1 m
  salinity_psu Salinity (PSS78) at 1 m, else at -1 m, else internally calc.
  do_pct       Oxygen percent saturation at 1 m
  do_mgl       derived from do_pct, temp_c, salinity_psu via Garcia & Gordon
               (1992) solubility (Benson-Krause coefficients), so the model's
               DO features are populated rather than median-imputed.

Output: data/transfer/cefas_full_15min.csv (harness contract). Then:
    python -m src.transfer.transfer_eval --source cefas_full --min-readings 12
"""
import glob
import os

import numpy as np
import pandas as pd

RAW = "data/transfer/raw/cefas/portal"
OUT = "data/transfer/cefas_full_15min.csv"
STATIONS = {
    "west-gabbard-smartbuoy": "wgab",
    "west-gabbard-2-smartbuoy": "wgab2",
    "liverpool-bay-coastal-observatory": "lbay",
    "dowsing-smartbuoy": "dows",
    "oyster-ground-smartbuoy": "oyst",
    "celtic-deep-smartbuoy": "cdeep",
    "celtic-deep-smartbuoy-site-2": "cdeep2",
}


def o2_solubility_mgl(t, s):
    """Garcia & Gordon 1992 (Benson-Krause fit): O2 at 100% saturation.
    Returns mg/L (umol/kg x 32 g/mol / 1000 x seawater density ~1.025)."""
    ts = np.log((298.15 - t) / (273.15 + t))
    a = [5.80871, 3.20291, 4.17887, 5.10006, -9.86643e-2, 3.80369]
    b = [-7.01577e-3, -7.70028e-3, -1.13864e-2, -9.51519e-3]
    c0 = -2.75915e-7
    lnc = (a[0] + a[1]*ts + a[2]*ts**2 + a[3]*ts**3 + a[4]*ts**4 + a[5]*ts**5
           + s*(b[0] + b[1]*ts + b[2]*ts**2 + b[3]*ts**3) + c0*s**2)
    return np.exp(lnc) * 32.0 / 1000.0 * 1.025


def first(df, names):
    out = pd.Series(np.nan, index=df.index)
    for n in names:
        if n in df.columns:
            out = out.fillna(pd.to_numeric(df[n], errors="coerce"))
    return out


def main():
    parts = []
    for path in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        key = os.path.basename(path)[:-4]
        if key not in STATIONS:
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        d = pd.DataFrame({
            "station": STATIONS[key],
            "datetime": pd.to_datetime(df["Time (GMT)"], format="%d/%m/%Y %H:%M", errors="coerce"),
            "chl_ugl": first(df, ["Fluorescence (SP) (arb. unit) at 1 m",
                                  "Fluorescence (MT) (arb. unit) at 1 m"]),
            "temp_c": first(df, ["Temperature (°C) at 1 m", "Temperature (°C) at -1 m"]),
            "salinity_psu": first(df, ["Salinity (PSS78) at 1 m", "Salinity (PSS78) at -1 m",
                                       "Salinity (internally calculated) (PSS78) at 1 m"]),
            "do_pct": first(df, ["Oxygen percent saturation (%) at 1 m"]),
        })
        d = d.dropna(subset=["datetime"])
        d = d[d.chl_ugl.notna() | d.temp_c.notna()]
        d.loc[d.chl_ugl < 0, "chl_ugl"] = np.nan
        ok = d.temp_c.notna() & d.salinity_psu.notna() & d.do_pct.notna()
        d["do_mgl"] = np.nan
        d.loc[ok, "do_mgl"] = (d.loc[ok, "do_pct"] / 100.0
                               * o2_solubility_mgl(d.loc[ok, "temp_c"].values,
                                                   d.loc[ok, "salinity_psu"].values))
        parts.append(d)
        print(f"{key:40s} rows={len(d):>8,} chl={d.chl_ugl.notna().mean():.0%} "
              f"do={d.do_mgl.notna().mean():.0%} {d.datetime.min().date()}..{d.datetime.max().date()}")
    out = pd.concat(parts).sort_values(["station", "datetime"])
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} rows, {out.station.nunique()} stations")


if __name__ == "__main__":
    main()
