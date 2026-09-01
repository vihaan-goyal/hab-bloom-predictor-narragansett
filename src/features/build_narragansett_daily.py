"""Daily feature table for the Narragansett fork.

Aggregates data/narragansett_surface_15min.csv to station-days (>= 48 chl
readings), builds two feature tiers, and the LIS-convention forward label
(any daily-mean chl > 10 ug/L within `horizon` days, right-censored -> NaN).

Tier A (LIS-analog)  : daily-mean chl, chl lags/rolls/trend/anomaly/climatology,
                       temp, sal, DO + 1-day lags, month.
Tier B (sonde-native): + within-day structure the LIS boat data can never see:
                       diel DO swing, night DO minimum, chl daily max/std,
                       day-over-day chl rate & acceleration, pH, DO%sat, temp range.

Output: data/narragansett_daily_features.csv
Run from repo root, BASE conda env.
"""
import numpy as np
import pandas as pd

HORIZON = 7
BLOOM = 10.0
MIN_READINGS = 48

df = pd.read_csv("data/narragansett_surface_15min.csv")
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", format="mixed")
df = df.dropna(subset=["datetime"])
df["date"] = df["datetime"].dt.normalize()
df["hour"] = df["datetime"].dt.hour
night = df[(df.hour >= 22) | (df.hour <= 5)]

day = df.groupby(["station", "date"]).agg(
    chl=("chl_ugl", "mean"), chl_max=("chl_ugl", "max"), chl_std=("chl_ugl", "std"),
    temp=("temp_c", "mean"), temp_range=("temp_c", lambda s: s.max() - s.min()),
    sal=("salinity_psu", "mean"), do=("do_mgl", "mean"),
    do_min=("do_mgl", "min"), do_range=("do_mgl", lambda s: s.max() - s.min()),
    do_pct=("do_pct", "mean"), ph=("ph", "mean"), n=("chl_ugl", "count"),
).reset_index()
ndo = night.groupby(["station", "date"])["do_mgl"].min().rename("do_night_min")
day = day.merge(ndo, on=["station", "date"], how="left")
day = day[day.n >= MIN_READINGS].sort_values(["station", "date"]).reset_index(drop=True)

g = day.groupby("station")
for k in (1, 2, 3, 4):
    day[f"chl_lag{k}"] = g["chl"].shift(k)
day["do_lag1"] = g["do"].shift(1)
day["temp_lag1"] = g["temp"].shift(1)
for k in (1, 2, 3, 4):
    day[f"sal_lag{k}"] = g["sal"].shift(k)
for w in (3, 6, 9, 14, 21):
    day[f"chl_roll{w}_mean"] = g["chl"].transform(
        lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean())
day["chl_trend"] = day["chl"] - day["chl_roll6_mean"]
day["chl_rate_1d"] = g["chl"].diff(1)
day["chl_accel"] = g["chl_rate_1d"].diff(1)
day["month"] = day["date"].dt.month
day["doy"] = day["date"].dt.dayofyear

# station x DOY-bin climatology (train-safe enough for feature use; 24 bins)
day["doy_bin"] = (day["doy"] - 1) // 15
clim = day.groupby(["station", "doy_bin"])["chl"].transform("mean")
day["chl_climatology"] = clim
day["chl_anomaly"] = day["chl"] - clim

# forward label, LIS convention (right-censored NaN)
day["bloom_fwd"] = np.nan
for st, grp in day.groupby("station"):
    idx = grp.index; dates = grp["date"].values; chl = grp["chl"].values
    last = dates.max()
    lab = np.full(len(grp), np.nan)
    for i in range(len(grp)):
        end = dates[i] + np.timedelta64(HORIZON, "D")
        m = (dates > dates[i]) & (dates <= end)
        if m.any() and (chl[m] > BLOOM).any():
            lab[i] = 1
        elif end <= last:
            lab[i] = 0
    day.loc[idx, "bloom_fwd"] = lab

day.drop(columns=["doy_bin"]).to_csv("data/narragansett_daily_features.csv", index=False)
lab = day["bloom_fwd"]
print(f"station-days={len(day)}  labeled={lab.notna().sum()}  "
      f"positive rate={lab.mean():.3f}  years={sorted(day.date.dt.year.unique())}")
