"""
export_model.py -- freeze the Narragansett bloom model for use anywhere
----------------------------------------------------------------------
Trains the reference HistGradientBoosting model (tier-A features, label =
daily-mean chl > 10 ug/L within 7 d) on ALL labelled Narragansett station-days
and saves everything predict_anywhere.py needs into one file:

  release/narragansett_bloom_model.joblib
    model          fitted HistGradientBoostingClassifier
    features       tier-A column order
    medians        train medians for imputation
    chl_quantiles  1,001 quantiles of Narragansett daily chl (for rescaling a
                   foreign fluorometer onto the training scale)
    horizon_days   7
    threshold      0.50 (val-chosen operating point)
    trained_on     description string

Run once from fork root, BASE env:  python src/deploy/export_model.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']
GB_KW = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50,
             l2_regularization=1.0, random_state=42, class_weight="balanced")
OUT = "release/narragansett_bloom_model.joblib"


def main():
    nar = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
    lab = nar.dropna(subset=["bloom_fwd"])
    med = lab[TIER_A].median(numeric_only=True)
    model = HistGradientBoostingClassifier(**GB_KW).fit(
        lab[TIER_A].fillna(med).values, lab.bloom_fwd.astype(int).values)
    q = np.quantile(nar.chl.dropna().values, np.linspace(0, 1, 1001))
    os.makedirs("release", exist_ok=True)
    joblib.dump(dict(model=model, features=TIER_A, medians=med.to_dict(),
                     chl_quantiles=q, horizon_days=7, threshold=0.50,
                     trained_on=(f"RIDEM Narragansett Bay sondes, {lab.station.nunique()} stations, "
                                 f"{lab.date.dt.year.min()}-{lab.date.dt.year.max()}, "
                                 f"{len(lab):,} station-days, label chl>10 ug/L within 7 d")),
                OUT, compress=3)
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e3:.0f} kB); rows={len(lab):,} pos={lab.bloom_fwd.mean():.3f}")


if __name__ == "__main__":
    main()
