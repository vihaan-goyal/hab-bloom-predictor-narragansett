"""
acceleration_feature.py
------------------------
Adds two new chlorophyll trajectory features:
  - chl_acceleration: change in 9-day rolling mean over the past 7 days
                      (positive = building up, near zero = plateau, negative = declining)
  - chl_roll14_mean:  14-day rolling mean (slower buildup signal)

Retrains the ensemble with these features added and compares precision/recall/F1
against the baseline (without acceleration features).

Run from repo root:
    python src/models/acceleration_feature.py
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    precision_recall_curve,
)
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load + recompute features
# ---------------------------------------------------------------------------
print("Loading data/hab_features_daily.csv...")
df = pd.read_csv("data/hab_features_daily.csv")
df['date'] = pd.to_datetime(df['date'])

for n, min_p in [(3, 2), (6, 3), (9, 5)]:
    df[f'chl_roll{n}_mean'] = (
        df.groupby('station_name')['Chlorophyll']
          .transform(lambda x: x.rolling(n, min_periods=min_p).mean())
    )

df['chl_trend'] = (
    df.groupby('station_name')['Chlorophyll']
      .transform(lambda x: x.rolling(4, min_periods=3)
                 .apply(lambda v: np.polyfit(range(len(v)), v, 1)[0]))
)

df['bloom_28d'] = 0
for station, grp in df.groupby('station_name'):
    idx   = grp.index
    dates = grp['date'].values
    chl   = grp['Chlorophyll'].values
    labels = np.zeros(len(grp), dtype=int)
    for i in range(len(grp)):
        mask = (dates > dates[i]) & (dates <= dates[i] + np.timedelta64(28, 'D'))
        if mask.any() and (chl[mask] > 10).any():
            labels[i] = 1
    df.loc[idx, 'bloom_28d'] = labels

# ---------------------------------------------------------------------------
# Add new features
# chl_acceleration: change in 9-day rolling mean over past 7 rows (days)
# chl_roll14_mean:  14-day rolling mean
# ---------------------------------------------------------------------------
print("Computing acceleration features...")

df['chl_roll14_mean'] = (
    df.groupby('station_name')['Chlorophyll']
      .transform(lambda x: x.rolling(14, min_periods=7).mean())
)

df['chl_acceleration'] = (
    df.groupby('station_name')['chl_roll9_mean']
      .transform(lambda x: x.diff(7))
)

print(f"chl_acceleration: {df['chl_acceleration'].notna().sum():,} non-null values")
print(f"chl_roll14_mean:  {df['chl_roll14_mean'].notna().sum():,} non-null values")
print(f"chl_acceleration mean: {df['chl_acceleration'].mean():.3f}  "
      f"std: {df['chl_acceleration'].std():.3f}")

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
FEATURES_BASE = [
    'Chlorophyll', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
    'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean', 'chl_trend',
    'chl_anomaly', 'chl_climatology',
    'do_lag1', 'temp_lag1', 'sal_lag1',
    'sea_water_temperature', 'sea_water_salinity',
    'oxygen_concentration_in_sea_water',
    'month', 'latitude_x', 'longitude_x',
    'nox_lag2', 'dip_lag2', 'dip_change', 'dip_x_month',
    'neighbor_chl3_mean', 'neighbor_chl3_lag1',
]

FEATURES_ACCEL = FEATURES_BASE + ['chl_acceleration', 'chl_roll14_mean']

# Filter to only columns that exist
FEATURES_BASE  = [f for f in FEATURES_BASE  if f in df.columns]
FEATURES_ACCEL = [f for f in FEATURES_ACCEL if f in df.columns]

print(f"\nBase features:  {len(FEATURES_BASE)}")
print(f"Accel features: {len(FEATURES_ACCEL)} "
      f"(+{len(FEATURES_ACCEL)-len(FEATURES_BASE)} new)")

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
train = df[df['date'].dt.year <= 2019]
val   = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)]
test  = df[df['date'].dt.year >= 2023]

def prepare(split, features):
    rows = split[features + ['bloom_28d']].dropna(subset=['bloom_28d'])
    X = rows[features].copy().reset_index(drop=True)
    y = rows['bloom_28d'].copy().reset_index(drop=True)
    return X, y

X_train_b, y_train_b = prepare(train, FEATURES_BASE)
X_val_b,   y_val_b   = prepare(val,   FEATURES_BASE)
X_test_b,  y_test_b  = prepare(test,  FEATURES_BASE)

X_train_a, y_train_a = prepare(train, FEATURES_ACCEL)
X_val_a,   y_val_a   = prepare(val,   FEATURES_ACCEL)
X_test_a,  y_test_a  = prepare(test,  FEATURES_ACCEL)

print(f"\nBase   -- Train: {len(X_train_b):,} | Val: {len(X_val_b):,} | Test: {len(X_test_b):,}")
print(f"Accel  -- Train: {len(X_train_a):,} | Val: {len(X_val_a):,} | Test: {len(X_test_a):,}")
print(f"(Accel has fewer rows due to 14-day warmup for rolling features)")

# ---------------------------------------------------------------------------
# Train and evaluate both models
# ---------------------------------------------------------------------------
def train_ensemble(X_tr, y_tr, X_v, X_te, features):
    MED = X_tr.median()

    xgb_m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=1.0, eval_metric='auc',
        random_state=42, verbosity=0,
    )
    xgb_m.fit(X_tr.fillna(MED), y_tr,
              eval_set=[(X_v.fillna(MED), y_v := y_tr[:len(X_v)])],
              verbose=False)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr.fillna(MED))
    X_v_s  = sc.transform(X_v.fillna(MED))
    X_te_s = sc.transform(X_te.fillna(MED))

    lr_m = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_m.fit(X_tr_s, y_tr)

    xgb_v  = xgb_m.predict_proba(X_v.fillna(MED))[:, 1]
    lr_v   = lr_m.predict_proba(X_v_s)[:, 1]
    ens_v  = 0.80 * lr_v  + 0.20 * xgb_v

    xgb_te = xgb_m.predict_proba(X_te.fillna(MED))[:, 1]
    lr_te  = lr_m.predict_proba(X_te_s)[:, 1]
    ens_te = 0.80 * lr_te + 0.20 * xgb_te

    return ens_v, ens_te, xgb_m, lr_m, sc, MED

def best_f1_threshold(y_true, probs):
    prec, rec, thresh = precision_recall_curve(y_true, probs)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = f1.argmax()
    t = float(thresh[idx]) if idx < len(thresh) else 0.5
    return t, float(prec[idx]), float(rec[idx]), float(f1[idx])

def eval_at_thresh(y_true, probs, thresh):
    preds = (probs >= thresh).astype(int)
    return {
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall':    recall_score(y_true, preds, zero_division=0),
        'f1':        f1_score(y_true, preds, zero_division=0),
        'auc':       roc_auc_score(y_true, probs),
        'ap':        average_precision_score(y_true, probs),
    }

# Fit correctly -- pass the right y_val to each
MED_b = X_train_b.median()
MED_a = X_train_a.median()

xgb_b = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=1.0, eval_metric='auc', random_state=42, verbosity=0)
xgb_b.fit(X_train_b.fillna(MED_b), y_train_b,
          eval_set=[(X_val_b.fillna(MED_b), y_val_b)], verbose=False)

sc_b = StandardScaler()
lr_b = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_b.fit(sc_b.fit_transform(X_train_b.fillna(MED_b)), y_train_b)

ens_val_b  = 0.80*lr_b.predict_proba(sc_b.transform(X_val_b.fillna(MED_b)))[:,1] \
           + 0.20*xgb_b.predict_proba(X_val_b.fillna(MED_b))[:,1]
ens_test_b = 0.80*lr_b.predict_proba(sc_b.transform(X_test_b.fillna(MED_b)))[:,1] \
           + 0.20*xgb_b.predict_proba(X_test_b.fillna(MED_b))[:,1]

xgb_a = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=1.0, eval_metric='auc', random_state=42, verbosity=0)
xgb_a.fit(X_train_a.fillna(MED_a), y_train_a,
          eval_set=[(X_val_a.fillna(MED_a), y_val_a)], verbose=False)

sc_a = StandardScaler()
lr_a = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_a.fit(sc_a.fit_transform(X_train_a.fillna(MED_a)), y_train_a)

ens_val_a  = 0.80*lr_a.predict_proba(sc_a.transform(X_val_a.fillna(MED_a)))[:,1] \
           + 0.20*xgb_a.predict_proba(X_val_a.fillna(MED_a))[:,1]
ens_test_a = 0.80*lr_a.predict_proba(sc_a.transform(X_test_a.fillna(MED_a)))[:,1] \
           + 0.20*xgb_a.predict_proba(X_test_a.fillna(MED_a))[:,1]

# Best-F1 threshold from val
thresh_b, _, _, _ = best_f1_threshold(y_val_b, ens_val_b)
thresh_a, _, _, _ = best_f1_threshold(y_val_a, ens_val_a)

print(f"\nBase  best-F1 threshold (val): {thresh_b:.3f}")
print(f"Accel best-F1 threshold (val): {thresh_a:.3f}")

m_b = eval_at_thresh(y_test_b, ens_test_b, thresh_b)
m_a = eval_at_thresh(y_test_a, ens_test_a, thresh_a)

print("\n" + "=" * 60)
print("TEST SET RESULTS (2023-2025)")
print("=" * 60)
print(f"\n{'Metric':<12}  {'Baseline':>10}  {'+ Acceleration':>14}  {'Delta':>8}")
print("-" * 50)
for k in ['auc', 'ap', 'precision', 'recall', 'f1']:
    delta = m_a[k] - m_b[k]
    marker = " <--" if k == 'precision' else ""
    print(f"{k:<12}  {m_b[k]:>10.4f}  {m_a[k]:>14.4f}  {delta:>+8.4f}{marker}")

print(f"\nBase  threshold: {thresh_b:.3f}")
print(f"Accel threshold: {thresh_a:.3f}")

# Also check SHAP importance of new features (XGBoost only, quick check)
print("\n-- XGBoost feature importance (top 10, acceleration model) ----------")
import pandas as pd
imp = pd.Series(xgb_a.feature_importances_, index=FEATURES_ACCEL)
print(imp.sort_values(ascending=False).head(10).to_string())

# ---------------------------------------------------------------------------
# Figure: precision-recall curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
prec_b, rec_b, _ = precision_recall_curve(y_test_b, ens_test_b)
prec_a, rec_a, _ = precision_recall_curve(y_test_a, ens_test_a)
ap_b = average_precision_score(y_test_b, ens_test_b)
ap_a = average_precision_score(y_test_a, ens_test_a)

ax.plot(rec_b, prec_b, 'b-', lw=2, label=f'Baseline AP={ap_b:.3f}')
ax.plot(rec_a, prec_a, 'g-', lw=2, label=f'+Acceleration AP={ap_a:.3f}')
ax.axhline(y_test_b.mean(), color='gray', linestyle='--', alpha=0.6,
           label=f'No-skill ({y_test_b.mean()*100:.1f}%)')

ax.scatter([eval_at_thresh(y_test_b, ens_test_b, thresh_b)['recall']],
           [eval_at_thresh(y_test_b, ens_test_b, thresh_b)['precision']],
           color='blue', s=80, zorder=5, label=f'Baseline best F1 (t={thresh_b:.2f})')
ax.scatter([m_a['recall']], [m_a['precision']],
           color='green', s=80, zorder=5, label=f'+Accel best F1 (t={thresh_a:.2f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall: baseline vs +acceleration features\n'
             '(test set 2023-2025)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/model/acceleration_pr_curve.png", dpi=150, bbox_inches='tight')
print("\nSaved figures/model/acceleration_pr_curve.png")

# Check at fixed threshold 0.60 to compare apples to apples
m_b_60 = eval_at_thresh(y_test_b, ens_test_b, 0.60)
m_a_60 = eval_at_thresh(y_test_a, ens_test_a, 0.60)
print(f"\nAt fixed threshold 0.60:")
print(f"  Baseline:     precision={m_b_60['precision']:.3f}  recall={m_b_60['recall']:.3f}  F1={m_b_60['f1']:.3f}")
print(f"  +Accel:       precision={m_a_60['precision']:.3f}  recall={m_a_60['recall']:.3f}  F1={m_a_60['f1']:.3f}")