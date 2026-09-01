"""Station-clustered bootstrap CIs for the fork's test-2023 numbers.

Convention matches the LIS repo (bootstrap_ci.py): resample clusters with
replacement, n_boot=2000, seed=42. Test is a single year, so the LIS
station-year cluster reduces to station. Models are fit ONCE on train; only
test clusters are resampled (no refit inside the loop).

Output: data/narragansett_bootstrap_cis.csv + printed table.
Run from repo root, BASE conda env.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BLOOM = 10.0
N_BOOT, SEED = 2000, 42
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']
TSTAR = {"LR": 0.35, "GB": 0.50}   # val-selected in train_narragansett.py

df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
df["year"] = df.date.dt.year
lab = df.dropna(subset=["bloom_fwd"])
train, test = lab[lab.year <= 2020], lab[lab.year == 2023].reset_index(drop=True)

med = train[TIER_A].median(numeric_only=True)
Xtr = train[TIER_A].fillna(med).values
ytr = train.bloom_fwd.astype(int).values
Xte = test[TIER_A].fillna(med).values
yte = test.bloom_fwd.astype(int).values

sc = StandardScaler().fit(Xtr)
preds = {
    "LR": LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000,
                             random_state=42).fit(sc.transform(Xtr), ytr)
          .predict_proba(sc.transform(Xte))[:, 1],
    "GB": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                         max_iter=300, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=42,
                                         class_weight="balanced").fit(Xtr, ytr)
          .predict_proba(Xte)[:, 1],
}
onset = (test.chl <= BLOOM).values
persist_alert = (test.chl > BLOOM).astype(int).values
stations = test.station.values
clusters = np.unique(stations)
print(f"test n={len(test)}  clusters(stations)={len(clusters)}  onset n={onset.sum()}")

def stat_block(idx):
    out = {}
    y = yte[idx]
    for mn, p in preds.items():
        pi, a = p[idx], (p[idx] >= TSTAR[mn]).astype(int)
        om = onset[idx]
        for tag, yy, aa, pp in ((f"{mn}_all", y, a, pi),
                                (f"{mn}_onset", y[om], a[om], pi[om])):
            tp = ((aa == 1) & (yy == 1)).sum(); fp = ((aa == 1) & (yy == 0)).sum()
            fn = ((aa == 0) & (yy == 1)).sum()
            prec = tp / (tp + fp) if tp + fp else np.nan
            base = yy.mean() if len(yy) else np.nan
            out[f"{tag}_auc"] = roc_auc_score(yy, pp) if 0 < yy.mean() < 1 else np.nan
            out[f"{tag}_pod"] = tp / (tp + fn) if tp + fn else np.nan
            out[f"{tag}_precision"] = prec
            out[f"{tag}_lift"] = prec / base if base else np.nan
    pa = persist_alert[idx]
    tp = ((pa == 1) & (y == 1)).sum(); fp = ((pa == 1) & (y == 0)).sum()
    prec = tp / (tp + fp) if tp + fp else np.nan
    out["persistence_all_precision"] = prec
    out["persistence_all_lift"] = prec / y.mean() if y.mean() else np.nan
    return out

point = stat_block(np.arange(len(test)))
rng = np.random.default_rng(SEED)
cluster_rows = {c: np.where(stations == c)[0] for c in clusters}
boot = []
for _ in range(N_BOOT):
    draw = rng.choice(clusters, size=len(clusters), replace=True)
    idx = np.concatenate([cluster_rows[c] for c in draw])
    boot.append(stat_block(idx))
bdf = pd.DataFrame(boot)

rows = []
for k, v in point.items():
    lo, hi = np.nanpercentile(bdf[k], [2.5, 97.5])
    rows.append(dict(metric=k, point=v, lo=lo, hi=hi))
out = pd.DataFrame(rows)
out.to_csv("data/narragansett_bootstrap_cis.csv", index=False)
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
