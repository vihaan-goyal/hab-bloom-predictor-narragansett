"""Group ablation on the Narragansett GB model (tier C features, onset focus).

Removes one feature group at a time, refits HistGB on train (<=2020), scores
test 2023 onset-only (today <= 10) at t*=0.50. Delta vs full model = the
group's weight. Output: data/narragansett_ablation.csv
Run from repo root, BASE conda env.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BLOOM, TSTAR = 10.0, 0.50
TIER = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
        'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
        'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
        'chl_anomaly', 'chl_climatology',
        'do', 'do_lag1', 'temp', 'temp_lag1',
        'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month',
        'chl_max', 'chl_std', 'chl_rate_1d', 'chl_accel',
        'do_min', 'do_range', 'do_night_min', 'do_pct', 'ph', 'temp_range',
        'strat_dens', 'strat_temp', 'strat_sal', 'bot_do', 'bot_do_min']
GROUPS = {
    "chl history (lags+rolls+trend+anom)": ['chl_lag1','chl_lag2','chl_lag3','chl_lag4',
        'chl_roll3_mean','chl_roll6_mean','chl_roll9_mean','chl_roll14_mean',
        'chl_roll21_mean','chl_trend','chl_anomaly'],
    "current chl (+within-day max/std)": ['chl','chl_max','chl_std'],
    "chl climatology": ['chl_climatology'],
    "chl rate + acceleration": ['chl_rate_1d','chl_accel'],
    "DO (all surface)": ['do','do_lag1','do_min','do_range','do_night_min','do_pct'],
    "temperature": ['temp','temp_lag1','temp_range'],
    "salinity": ['sal','sal_lag1','sal_lag2','sal_lag3','sal_lag4'],
    "month": ['month'],
    "pH": ['ph'],
    "stratification + bottom DO": ['strat_dens','strat_temp','strat_sal','bot_do','bot_do_min'],
}

df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
df["year"] = df.date.dt.year
lab = df.dropna(subset=["bloom_fwd"])
feats_all = [f for f in TIER if f in lab.columns]
train, test = lab[lab.year <= 2020], lab[lab.year == 2023].reset_index(drop=True)
onset = (test.chl <= BLOOM).values
yt = test.bloom_fwd.astype(int).values

def run(feats):
    med = train[feats].median(numeric_only=True)
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                       min_samples_leaf=50, l2_regularization=1.0,
                                       random_state=42, class_weight="balanced")
    m.fit(train[feats].fillna(med).values, train.bloom_fwd.astype(int).values)
    p = m.predict_proba(test[feats].fillna(med).values)[:, 1]
    y, pp = yt[onset], p[onset]
    a = (pp >= TSTAR).astype(int)
    tp = ((a == 1) & (y == 1)).sum(); fp = ((a == 1) & (y == 0)).sum()
    fn = ((a == 0) & (y == 1)).sum()
    return dict(auc=roc_auc_score(y, pp),
                precision=tp / (tp + fp) if tp + fp else np.nan,
                pod=tp / (tp + fn) if tp + fn else np.nan)

base = run(feats_all)
print(f"FULL onset: AUC={base['auc']:.3f} P={base['precision']:.3f} POD={base['pod']:.3f}")
rows = [dict(removed="none", **base, delta_auc=0.0, delta_precision=0.0)]
for name, g in GROUPS.items():
    feats = [f for f in feats_all if f not in g]
    if len(feats) == len(feats_all):
        continue
    r = run(feats)
    rows.append(dict(removed=name, **r, delta_auc=r["auc"] - base["auc"],
                     delta_precision=r["precision"] - base["precision"]))
    print(f"- {name:38s} dAUC={r['auc']-base['auc']:+.3f} dP={r['precision']-base['precision']:+.3f}")
pd.DataFrame(rows).to_csv("data/narragansett_ablation.csv", index=False)
