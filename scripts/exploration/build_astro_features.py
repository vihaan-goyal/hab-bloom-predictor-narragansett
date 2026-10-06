"""
build_astro_features.py

Generates daily astronomical features for the LIS HAB pipeline, computed offline
with pyephem (no network, no USNO API calls). Output is keyed by date so it merges
into the existing daily feature matrix the same way the other exogenous features do.

Features produced:
  moon_illum_frac  : illuminated fraction of the Moon, 0..1
  spring_neap      : abs(2*illum - 1). ~1 near new/full moon (spring tides,
                     strong currents/mixing), ~0 near quarters (neap tides).
                     This is the ONE feature orthogonal to seasonality and the
                     only one worth expecting a real test from.
  photoperiod_hrs  : day length in hours. Almost certainly redundant with your
                     month / climatology features. Included so the ablation can
                     confirm-and-reject it rather than leaving it untested.

Usage:
    conda activate hab
    python scripts/exploration/build_astro_features.py --start 1993-01-01 --end 2025-12-31 \
        --out data/astro_features_daily.csv

Then merge on your date column, e.g.:
    astro = pd.read_csv("data/astro_features_daily.csv", parse_dates=["date"])
    df = df.merge(astro, on="date", how="left")
and add ["spring_neap", "photoperiod_hrs"] to the feature list in your ablation.
"""

import argparse
import datetime as dt

import ephem
import pandas as pd

# Representative point for Long Island Sound. Moon illumination is effectively
# location independent; day length varies negligibly across the LIS latitude band.
LIS_LAT = "41.1"
LIS_LON = "-72.9"


def astro_row(day: dt.date, moon, sun, obs):
    noon = dt.datetime(day.year, day.month, day.day, 12)

    moon.compute(noon)
    illum = moon.phase / 100.0
    spring_neap = abs(2.0 * illum - 1.0)

    obs.date = dt.datetime(day.year, day.month, day.day, 0, 0)
    try:
        rise = obs.next_rising(sun)   # this day's sunrise
        obs.date = rise
        set_ = obs.next_setting(sun)  # the sunset that follows that sunrise
        photoperiod = float((set_ - rise) * 24.0)
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        photoperiod = float("nan")  # never triggers at LIS latitude, kept for safety

    return {
        "date": pd.Timestamp(day),
        "moon_illum_frac": round(illum, 4),
        "spring_neap": round(spring_neap, 4),
        "photoperiod_hrs": round(photoperiod, 3),
    }


def build(start: str, end: str) -> pd.DataFrame:
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)

    moon, sun = ephem.Moon(), ephem.Sun()
    obs = ephem.Observer()
    obs.lat, obs.lon = LIS_LAT, LIS_LON
    obs.horizon = "-0:34"  # standard atmospheric refraction for sunrise/sunset

    rows, day = [], start_d
    while day <= end_d:
        rows.append(astro_row(day, moon, sun, obs))
        day += dt.timedelta(days=1)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1993-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default="data/astro_features_daily.csv")
    args = ap.parse_args()

    df = build(args.start, args.end)
    df.to_csv(args.out, index=False)

    # Sanity readout. spring_neap should be uncorrelated with season (that is the
    # whole point); photoperiod should be almost perfectly explained by season
    # (which is why it will not add anything your month features do not already).
    import numpy as np

    doy = df["date"].dt.dayofyear.to_numpy()
    season = np.cos(2 * np.pi * (doy - 172) / 365.25)  # peaks at summer solstice
    print(f"rows: {len(df)}")
    print(df[["moon_illum_frac", "spring_neap", "photoperiod_hrs"]].describe().round(3))
    print(f"\nspring_neap  vs season corr: {np.corrcoef(df['spring_neap'], season)[0,1]:+0.4f}  (want ~0)")
    print(f"photoperiod  vs season corr: {np.corrcoef(df['photoperiod_hrs'], season)[0,1]:+0.4f}  (want ~1, confirms redundancy)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()