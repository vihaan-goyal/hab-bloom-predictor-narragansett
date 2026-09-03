#!/usr/bin/env python
"""
predict_anywhere.py -- run the Narragansett bloom model on YOUR water body
=========================================================================
Give it a CSV of sonde readings from any site and it returns, for every
station and day, the probability that daily-mean chlorophyll will exceed a
bloom level within the next 7 days. No training, no internet, no Narragansett
data needed: just this file and release/narragansett_bloom_model.joblib.

INPUT CSV (any cadence: 15-min, hourly, or one row per day)
    station   any id                       required
    datetime  ISO, e.g. 2018-07-01 00:15   required
    chl       chlorophyll (any fluorometer units)   required
    temp      water temperature, deg C     optional
    sal       salinity, PSU (0 for lakes)  optional
    do        dissolved oxygen, mg/L       optional
Missing optional columns are imputed with training medians (skill drops a
little; see the findings note §19 for what each site lost).

WHAT IT DOES
  1. Aggregates to station-days (>= --min-readings readings per day).
  2. Builds the 23 tier-A features (chl lags/rolling means/trend/anomaly/
     station climatology, DO, temperature, salinity lags, month).
  3. Rescales your chlorophyll onto the Narragansett scale by quantile
     matching, because every fluorometer reads differently. Without this
     the model never alerts on foreign data.
  4. Applies the frozen model and writes probabilities + alerts.

USAGE
    python predict_anywhere.py readings.csv
    python predict_anywhere.py readings.csv --date 2018-08-15 --threshold 0.5
    python predict_anywhere.py daily_means.csv --min-readings 1

Needs: python >= 3.9, pandas, numpy, scikit-learn, joblib.
Expected skill on new sites (onset-only, own-site 75th-percentile label):
lift 1.3-2.5x over always-alert, AUC 0.6-0.86 (findings §19). Your first
21 days per station carry NaN rolling features and score lower.

Vihaan Goyal, Westhill High School. github.com/vihaan-goyal/hab-bloom-predictor-narragansett
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "release", "narragansett_bloom_model.joblib")


def build_daily(df, min_readings):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "chl"])
    df["date"] = df["datetime"].dt.normalize()
    for c in ("temp", "sal", "do"):
        if c not in df:
            df[c] = np.nan
            print(f"note: no '{c}' column; imputing with training median", file=sys.stderr)
    day = df.groupby(["station", "date"]).agg(
        chl=("chl", "mean"), temp=("temp", "mean"), sal=("sal", "mean"),
        do=("do", "mean"), n=("chl", "count")).reset_index()
    day = day[day.n >= min_readings].sort_values(["station", "date"]).reset_index(drop=True)
    g = day.groupby("station")
    for k in (1, 2, 3, 4):
        day[f"chl_lag{k}"] = g["chl"].shift(k)
        day[f"sal_lag{k}"] = g["sal"].shift(k)
    day["do_lag1"] = g["do"].shift(1)
    day["temp_lag1"] = g["temp"].shift(1)
    for w in (3, 6, 9, 14, 21):
        day[f"chl_roll{w}_mean"] = g["chl"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean())
    day["chl_trend"] = day["chl"] - day["chl_roll6_mean"]
    day["month"] = day["date"].dt.month
    day["doy_bin"] = (day["date"].dt.dayofyear - 1) // 15
    day["chl_climatology"] = day.groupby(["station", "doy_bin"])["chl"].transform("mean")
    day["chl_anomaly"] = day["chl"] - day["chl_climatology"]
    return day.drop(columns=["doy_bin"])


def rescale_chl(day, ref_quantiles):
    """Quantile-map this site's chlorophyll columns onto the Narragansett scale."""
    src = np.sort(day["chl"].dropna().values)
    grid = np.linspace(0, 1, len(ref_quantiles))

    def qm(x):
        q = np.searchsorted(src, x, side="right") / len(src)
        return np.interp(np.clip(q, 0, 1), grid, ref_quantiles)

    out = day.copy()
    for c in ("chl", "chl_lag1", "chl_lag2", "chl_lag3", "chl_lag4",
              "chl_roll3_mean", "chl_roll6_mean", "chl_roll9_mean",
              "chl_roll14_mean", "chl_roll21_mean", "chl_climatology"):
        v = out[c].values.astype(float); ok = ~np.isnan(v)
        v2 = v.copy(); v2[ok] = qm(v[ok]); out[c] = v2
    out["chl_trend"] = out["chl"] - out["chl_roll6_mean"]
    out["chl_anomaly"] = out["chl"] - out["chl_climatology"]
    return out


def main():
    ap = argparse.ArgumentParser(description="Narragansett bloom model, any site")
    ap.add_argument("csv", help="readings: station, datetime, chl[, temp, sal, do]")
    ap.add_argument("--date", default=None, help="report this day (default: last day in file)")
    ap.add_argument("--threshold", type=float, default=None, help="alert if prob >= this (default: model's 0.50)")
    ap.add_argument("--min-readings", type=int, default=12, help="readings per station-day to keep (1 for daily data)")
    ap.add_argument("--out", default="bloom_predictions.csv")
    ap.add_argument("--no-rescale", action="store_true", help="skip quantile rescaling (only if your chl is already on the RIDEM sonde ug/L scale)")
    a = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"model file not found: {MODEL_PATH}")
    pack = joblib.load(MODEL_PATH)
    model, feats, med = pack["model"], pack["features"], pack["medians"]
    t = a.threshold if a.threshold is not None else pack["threshold"]

    raw = pd.read_csv(a.csv)
    missing = [c for c in ("station", "datetime", "chl") if c not in raw.columns]
    if missing:
        sys.exit(f"input is missing required columns: {missing}")
    day = build_daily(raw, a.min_readings)
    if len(day) == 0:
        sys.exit("no station-days survived --min-readings; lower it (use 1 for daily data)")
    if len(day) < 60:
        print("warning: fewer than 60 station-days; the rescaling and climatology are unreliable", file=sys.stderr)
    scored = day if a.no_rescale else rescale_chl(day, pack["chl_quantiles"])
    X = scored[feats].fillna(pd.Series(med)).fillna(0.0).values
    day["bloom_prob"] = model.predict_proba(X)[:, 1]
    day["alert"] = day.bloom_prob >= t
    day["warmup"] = day.chl_roll21_mean.isna()      # first ~3 weeks per station

    out = day[["station", "date", "chl", "temp", "sal", "do", "bloom_prob", "alert", "warmup"]]
    out.to_csv(a.out, index=False)

    target = pd.Timestamp(a.date) if a.date else day.date.max()
    latest = (day[day.date <= target].sort_values("date").groupby("station").tail(1)
              .sort_values("bloom_prob", ascending=False))
    latest["days_old"] = (target - latest.date).dt.days
    print(f"Model: {pack['trained_on']}")
    print(f"Horizon {pack['horizon_days']} d | threshold {t:.2f} | "
          f"{'no rescale' if a.no_rescale else 'chl rescaled to training scale'}")
    print(f"Scored {len(day):,} station-days, {day.station.nunique()} stations, "
          f"{day.date.min().date()}..{day.date.max().date()} -> {a.out}\n")
    print(f"Latest day at or before {target.date()}:")
    show = latest[["station", "date", "days_old", "chl", "bloom_prob", "alert", "warmup"]].copy()
    show["date"] = show.date.dt.date
    show["chl"] = show.chl.round(2); show["bloom_prob"] = show.bloom_prob.round(3)
    print(show.to_string(index=False))
    print(f"\nAlerts: {int(latest.alert.sum())} of {len(latest)} stations. "
          f"Read bloom_prob as P(chl exceeds a bloom level within {pack['horizon_days']} days).")


if __name__ == "__main__":
    main()
