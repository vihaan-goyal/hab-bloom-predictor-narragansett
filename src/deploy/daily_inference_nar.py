"""
daily_inference_nar.py -- per-date bloom probabilities for Narragansett Bay
--------------------------------------------------------------------------
Narragansett analogue of the LIS daily_inference.py. For a target date D:

  1. Loads data/narragansett_daily_features.csv (built by
     src/features/build_narragansett_daily.py; label = daily-mean sonde chl
     > 10 ug/L within HORIZON days, right-censored -> NaN).
  2. Trains HistGradientBoosting on tier-A (LIS-analog) features using only
     rows whose label window closes on or before D (date <= D - HORIZON).
     Walk-forward: no future information.
  3. Scores each station's latest station-day at or before D (older than
     --max-stale days -> STALE, not scored).
  4. Alert = P(bloom within HORIZON d) >= t*. t* = 0.50 is the value chosen
     on the 2021-22 validation years for GB tier A
     (data/narragansett_model_results.csv); it is not re-tuned here.
  5. Writes data/narragansett_daily_predictions.csv.

Reference skill for this recipe (test 2023, onset-only, today's chl <= 10):
  precision 0.696, POD 0.600, base rate 0.347, lift 2.00, AUC 0.839.
Stations already above 10 ug/L are flagged "blooming"; for them the
probability is a persistence question, not a forecast.

Usage (from repo root, BASE conda env):
    python src/deploy/daily_inference_nar.py --date 2023-07-19
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES_PATH = "data/narragansett_daily_features.csv"
OUTPUT_PATH = "data/narragansett_daily_predictions.csv"
HORIZON = 7
BLOOM = 10.0
T_STAR = 0.50          # val-chosen for GB tier A -- do NOT tune here
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']


def parse_args():
    p = argparse.ArgumentParser(description="Narragansett bloom early warning")
    p.add_argument("--date", type=str, default=None,
                   help="Target date YYYY-MM-DD (default: last date in data)")
    p.add_argument("--t-star", type=float, default=T_STAR)
    p.add_argument("--max-stale", type=int, default=7,
                   help="skip stations whose latest day is older than this")
    return p.parse_args()


def main():
    a = parse_args()
    print(f"Loading {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    target = pd.Timestamp(a.date) if a.date else df.date.max()
    print(f"Target date: {target.date()}   t* = {a.t_star:.2f}   "
          f"horizon = {HORIZON}d   model = HistGB tier A")

    # ---- walk-forward training set: label window fully observed by target
    cutoff = target - pd.Timedelta(days=HORIZON)
    train = df[(df.date <= cutoff)].dropna(subset=["bloom_fwd"])
    if len(train) < 500:
        raise SystemExit(f"only {len(train)} labelled rows before "
                         f"{cutoff.date()} -- pick a later date")
    med = train[TIER_A].median(numeric_only=True)
    X = train[TIER_A].fillna(med).values
    y = train.bloom_fwd.astype(int).values
    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50,
        l2_regularization=1.0, random_state=42, class_weight="balanced")
    model.fit(X, y)
    print(f"Trained HistGB on {len(train):,} station-days through "
          f"{cutoff.date()} (bloom rate {y.mean():.3f})")

    # ---- latest station-day at or before target
    latest = (df[df.date <= target].sort_values("date")
              .groupby("station").tail(1).copy())
    latest["days_old"] = (target - latest.date).dt.days
    stale = latest[latest.days_old > a.max_stale]
    score = latest[latest.days_old <= a.max_stale].copy()
    if len(score):
        score["bloom_prob"] = model.predict_proba(
            score[TIER_A].fillna(med).values)[:, 1]
    else:
        score["bloom_prob"] = np.nan
    score["alert"] = score.bloom_prob >= a.t_star
    score["status"] = np.where(score.chl > BLOOM, "blooming", "onset risk")
    score = score.sort_values("bloom_prob", ascending=False)

    out = score[["station", "date", "days_old", "chl", "bloom_prob",
                 "alert", "status"]].rename(columns={"chl": "chl_today"})
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}\n")

    print(f"Stations scored: {len(score)}   stale (> {a.max_stale}d, "
          f"skipped): {len(stale)}")
    onset = score[score.status == "onset risk"]
    print(f"Alerts at t*={a.t_star:.2f}: {int(score.alert.sum())} of "
          f"{len(score)}  (onset-risk stations alerted: "
          f"{int(onset.alert.sum())} of {len(onset)})\n")
    print("Stations by bloom probability (chl in ug/L, sonde fluorescence):")
    show = out.copy()
    show["date"] = show.date.dt.date
    show["chl_today"] = show.chl_today.round(1)
    show["bloom_prob"] = show.bloom_prob.round(3)
    print(show.to_string(index=False))
    if len(stale):
        print(f"\nStale stations (no day within {a.max_stale}d): "
              + ", ".join(sorted(stale.station)))


if __name__ == "__main__":
    main()
