"""Train the locked-spec model family on Narragansett daily features.

Feature tiers:
  A = LIS-analog features only (what the LIS model had)
  B = A + sonde-native features (diel DO swing, night DO min, chl rate/accel,
      within-day chl max/std, pH, DO%sat, temp range)

Models: LogisticRegression (locked spec: C=0.05, balanced, scaler, median
impute) and XGBoost (light regularization) for comparison.
Baselines: always-alert, persistence (today's chl > 10), station-DOY climatology.
Split: temporal -- train <= TRAIN_END year, val = VAL_YEARS, test = TEST_YEARS.
Threshold: chosen on val (max F1); exactly one test evaluation per model.

Output: data/narragansett_model_results.csv + printed table.
Run from repo root, BASE conda env.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BLOOM = 10.0
TRAIN_MAX = 2020
VAL_YEARS = (2021, 2022)
TEST_YEARS = (2023,)

TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']
TIER_B = TIER_A + ['chl_max', 'chl_std', 'chl_rate_1d', 'chl_accel',
                   'do_min', 'do_range', 'do_night_min', 'do_pct',
                   'ph', 'temp_range']

df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
df["year"] = df.date.dt.year
lab = df.dropna(subset=["bloom_fwd"]).copy()
train = lab[lab.year <= TRAIN_MAX]
val = lab[lab.year.isin(VAL_YEARS)]
test = lab[lab.year.isin(TEST_YEARS)]
print(f"train n={len(train)} pos={train.bloom_fwd.mean():.3f} years<= {TRAIN_MAX}")
print(f"val   n={len(val)} pos={val.bloom_fwd.mean():.3f} {VAL_YEARS}")
print(f"test  n={len(test)} pos={test.bloom_fwd.mean():.3f} {TEST_YEARS}")
if len(train) == 0 or len(test) == 0:
    raise SystemExit("empty split -- adjust years")

def metrics(y, alert):
    tp = int(((alert == 1) & (y == 1)).sum()); fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum()); tn = int(((alert == 0) & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    prec = 1 - far if not np.isnan(far) else np.nan
    base = (tp + fn) / (tp + fp + fn + tn)
    return dict(tp=tp, fp=fp, fn=fn, pod=pod, far=far, precision=prec,
                base_rate=base, lift=prec / base if base else np.nan)

def fit_predict(features, model_name):
    Xtr = train[features].copy(); med = Xtr.median(numeric_only=True)
    def prep(X): return X[features].fillna(med).values
    ytr = train.bloom_fwd.astype(int).values
    if model_name == "LR":
        sc = StandardScaler().fit(prep(train))
        m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000,
                               random_state=42).fit(sc.transform(prep(train)), ytr)
        pv = m.predict_proba(sc.transform(prep(val)))[:, 1]
        pt = m.predict_proba(sc.transform(prep(test)))[:, 1]
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                           max_iter=300, min_samples_leaf=50,
                                           l2_regularization=1.0, random_state=42,
                                           class_weight="balanced")
        m.fit(prep(train), ytr)
        pv = m.predict_proba(prep(val))[:, 1]
        pt = m.predict_proba(prep(test))[:, 1]
    yv = val.bloom_fwd.astype(int).values; yt = test.bloom_fwd.astype(int).values
    ts = np.arange(0.05, 0.96, 0.05)
    f1s = []
    for t in ts:
        mm = metrics(yv, (pv >= t).astype(int))
        f1 = (2 * mm["precision"] * mm["pod"] / (mm["precision"] + mm["pod"])
              if mm["precision"] and mm["pod"] and not np.isnan(mm["far"]) else 0)
        f1s.append(f1 or 0)
    t_star = float(ts[int(np.argmax(f1s))])
    r = metrics(yt, (pt >= t_star).astype(int))
    r.update(model=model_name, auc_val=roc_auc_score(yv, pv),
             auc_test=roc_auc_score(yt, pt), t_star=t_star)
    om = (test.chl <= BLOOM).values
    ro = metrics(yt[om], (pt[om] >= t_star).astype(int))
    ro.update(model=model_name + "_onset", auc_val=np.nan,
              auc_test=roc_auc_score(yt[om], pt[om]), t_star=t_star)
    return [r, ro]

rows = []
for tier, feats in (("A_LIS_analog", TIER_A), ("B_sonde_native", TIER_B)):
    for mn in ("LR", "GB"):
        for r in fit_predict(feats, mn):
            r["features"] = tier; rows.append(r)

# baselines on test
yt = test.bloom_fwd.astype(int).values
rows.append(dict(model="always_alert", features="-", t_star=np.nan,
                 auc_val=np.nan, auc_test=np.nan,
                 **metrics(yt, np.ones(len(test), dtype=int))))
rows.append(dict(model="persistence", features="chl>10 today", t_star=np.nan,
                 auc_val=np.nan, auc_test=roc_auc_score(yt, test.chl.values),
                 **metrics(yt, (test.chl > BLOOM).astype(int).values)))
clim_rate = (train.assign(bin=(train.date.dt.dayofyear - 1) // 15)
             .groupby(["station", "bin"]).bloom_fwd.mean())
tb = test.assign(bin=(test.date.dt.dayofyear - 1) // 15)
pr = tb.set_index(["station", "bin"]).index.map(clim_rate).values
pr = np.where(pd.isna(pr), float(train.bloom_fwd.mean()), pr.astype(float))
rows.append(dict(model="climatology", features="station-DOY rate", t_star=0.5,
                 auc_val=np.nan, auc_test=roc_auc_score(yt, pr),
                 **metrics(yt, (pr >= 0.5).astype(int))))

out = pd.DataFrame(rows)[["model", "features", "auc_val", "auc_test", "t_star",
                          "pod", "far", "precision", "base_rate", "lift",
                          "tp", "fp", "fn"]]
om = (test.chl <= BLOOM).values
rows_onset_base = dict(model="always_alert_onset", features="-", t_star=np.nan,
                       auc_val=np.nan, auc_test=np.nan,
                       **metrics(yt[om], np.ones(om.sum(), dtype=int)))
out = pd.concat([out, pd.DataFrame([rows_onset_base])[out.columns]], ignore_index=True)
out.to_csv("data/narragansett_model_results.csv", index=False)
print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
