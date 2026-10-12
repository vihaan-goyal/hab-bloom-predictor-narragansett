"""
precision_features.py
---------------------
Adds three features targeting the false positive profile identified
in false_positive_analysis.py:

  1. chl_spike_ratio:         current CHL / recent rolling mean
                              FPs spike suddenly; TPs build steadily
  2. chl_lag_consistency:     average recent lags / current CHL
                              high = sustained buildup (TP signal)
                              low  = sudden spike (FP signal)
  3. station_month_bloom_rate: historical bloom rate for this
                              station x month combination
                              captures "this station never blooms
                              in January" -- kills winter FPs

Retrains LR with these features added and compares to baseline.

Run from repo root:
    python src/models/precision_features.py
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load + recompute base features
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
# Add new precision-targeting features
# ---------------------------------------------------------------------------
print("Computing precision features...")

# 1. Spike ratio: current CHL relative to 9-day rolling mean
#    FPs spike suddenly above baseline; TPs build steadily
#    Values >> 1.0 = sudden spike (FP signal)
#    Values ~= 1.0 = steady / sustained (TP signal)
df['chl_spike_ratio'] = df['Chlorophyll'] / (df['chl_roll9_mean'] + 0.1)

# 2. Lag consistency: are recent lags building toward current level?
#    FPs have lower lags than current CHL (sudden spike)
#    TPs have lags close to or higher than current CHL (sustained buildup)
#    High value = recent history matches current (sustained = TP signal)
#    Low value  = recent history much lower than current (spike = FP signal)
df['chl_lag_consistency'] = (
    (df['chl_lag1'].fillna(df['Chlorophyll']) +
     df['chl_lag2'].fillna(df['Chlorophyll'])) /
    (2 * df['Chlorophyll'] + 0.1)
)

# 3. Station-month bloom rate: historical bloom frequency for this
#    station in this calendar month (computed on TRAIN only to avoid leakage)
#    This kills winter false positives -- Jan at A4 has near-zero historical
#    bloom rate so the model learns to down-weight January alerts
df['month'] = df['date'].dt.month  # ensure month column exists

# Compute from train period only (1993-2019)
train_mask = df['date'].dt.year <= 2019
train_rates = (
    df[train_mask]
    .groupby(['station_name', 'month'])['bloom_28d']
    .mean()
    .reset_index()
    .rename(columns={'bloom_28d': 'station_month_bloom_rate'})
)
df = df.merge(train_rates, on=['station_name', 'month'], how='left')
# Fill NaN for station-month combos not seen in train with global train mean
global_train_rate = df[train_mask]['bloom_28d'].mean()
df['station_month_bloom_rate'] = df['station_month_bloom_rate'].fillna(global_train_rate)

print(f"chl_spike_ratio:          {df['chl_spike_ratio'].notna().sum():,} non-null")
print(f"chl_lag_consistency:      {df['chl_lag_consistency'].notna().sum():,} non-null")
print(f"station_month_bloom_rate: {df['station_month_bloom_rate'].notna().sum():,} non-null")

# Quick sanity check on station_month_bloom_rate
print("\nSample station-month bloom rates (should be low in Jan/Feb):")
sample = train_rates[train_rates['station_name'] == 'A4'].sort_values('month')
if len(sample) > 0:
    print(sample.to_string(index=False))

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
BASE_FEATURES = [
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

NEW_FEATURES = [
    'chl_spike_ratio',
    'chl_lag_consistency',
    'station_month_bloom_rate',
]

BASE_FEATURES  = [f for f in BASE_FEATURES  if f in df.columns]
AUGMENTED      = [f for f in BASE_FEATURES + NEW_FEATURES if f in df.columns]

print(f"\nBase features:      {len(BASE_FEATURES)}")
print(f"Augmented features: {len(AUGMENTED)} (+{len(AUGMENTED)-len(BASE_FEATURES)} new)")

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

X_train_b, y_train_b = prepare(train, BASE_FEATURES)
X_val_b,   y_val_b   = prepare(val,   BASE_FEATURES)
X_test_b,  y_test_b  = prepare(test,  BASE_FEATURES)

X_train_a, y_train_a = prepare(train, AUGMENTED)
X_val_a,   y_val_a   = prepare(val,   AUGMENTED)
X_test_a,  y_test_a  = prepare(test,  AUGMENTED)

MED_b = X_train_b.median()
MED_a = X_train_a.median()

print(f"\nBase      -- Train: {len(X_train_b):,} | Val: {len(X_val_b):,} | Test: {len(X_test_b):,}")
print(f"Augmented -- Train: {len(X_train_a):,} | Val: {len(X_val_a):,} | Test: {len(X_test_a):,}")

# ---------------------------------------------------------------------------
# Train both models
# ---------------------------------------------------------------------------
def fit_lr(X_tr, y_tr, X_v, X_te, MED):
    sc = StandardScaler()
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(sc.fit_transform(X_tr.fillna(MED)), y_tr)
    p_v  = lr.predict_proba(sc.transform(X_v.fillna(MED)))[:, 1]
    p_te = lr.predict_proba(sc.transform(X_te.fillna(MED)))[:, 1]
    return p_v, p_te, lr, sc

print("\nFitting baseline LR...")
p_val_b, p_test_b, lr_b, sc_b = fit_lr(X_train_b, y_train_b,
                                         X_val_b, X_test_b, MED_b)

print("Fitting augmented LR...")
p_val_a, p_test_a, lr_a, sc_a = fit_lr(X_train_a, y_train_a,
                                         X_val_a, X_test_a, MED_a)

# ---------------------------------------------------------------------------
# Find best-F1 threshold on val, evaluate on test
# ---------------------------------------------------------------------------
def best_f1_thresh(y, p):
    prec, rec, thresh = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = f1.argmax()
    return float(thresh[idx]) if idx < len(thresh) else 0.5

t_b = best_f1_thresh(y_val_b, p_val_b)
t_a = best_f1_thresh(y_val_a, p_val_a)

def eval_at(y, p, t):
    preds = (p >= t).astype(int)
    tp = int(((preds==1)&(y==1)).sum())
    fp = int(((preds==1)&(y==0)).sum())
    fn = int(((preds==0)&(y==1)).sum())
    tn = int(((preds==0)&(y==0)).sum())
    return {
        'auc':       roc_auc_score(y, p),
        'ap':        average_precision_score(y, p),
        'precision': precision_score(y, preds, zero_division=0),
        'recall':    recall_score(y, preds, zero_division=0),
        'f1':        f1_score(y, preds, zero_division=0),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'far': fp/(fp+tn) if (fp+tn)>0 else 0,
    }

m_b_val  = eval_at(y_val_b,  p_val_b,  t_b)
m_a_val  = eval_at(y_val_a,  p_val_a,  t_a)
m_b_test = eval_at(y_test_b, p_test_b, t_b)
m_a_test = eval_at(y_test_a, p_test_a, t_a)

# Also evaluate both at fixed 0.60
m_b_60 = eval_at(y_test_b, p_test_b, 0.60)
m_a_60 = eval_at(y_test_a, p_test_a, 0.60)

print("\n" + "=" * 65)
print("RESULTS COMPARISON")
print("=" * 65)
print(f"\n{'Metric':<12}  {'Base (val t)':>12}  {'Aug (val t)':>12}  {'Delta':>8}")
print("-" * 50)
for k in ['auc', 'ap', 'precision', 'recall', 'f1']:
    delta = m_a_test[k] - m_b_test[k]
    marker = " <--" if k == 'precision' else ""
    print(f"{k:<12}  {m_b_test[k]:>12.4f}  {m_a_test[k]:>12.4f}  {delta:>+8.4f}{marker}")
print(f"\nBase threshold (val): {t_b:.3f}  |  Augmented threshold (val): {t_a:.3f}")

print(f"\n-- At fixed threshold 0.60 --")
print(f"{'Metric':<12}  {'Baseline':>10}  {'Augmented':>10}  {'Delta':>8}")
print("-" * 45)
for k in ['precision', 'recall', 'f1']:
    delta = m_a_60[k] - m_b_60[k]
    marker = " <--" if k == 'precision' else ""
    print(f"{k:<12}  {m_b_60[k]:>10.4f}  {m_a_60[k]:>10.4f}  {delta:>+8.4f}{marker}")
print(f"Base:      TP={m_b_60['tp']}  FP={m_b_60['fp']}  FN={m_b_60['fn']}")
print(f"Augmented: TP={m_a_60['tp']}  FP={m_a_60['fp']}  FN={m_a_60['fn']}")

# ---------------------------------------------------------------------------
# LR coefficients for new features
# ---------------------------------------------------------------------------
print("\n-- Coefficients of new features in augmented LR --")
coef_df = pd.Series(lr_a.coef_[0], index=AUGMENTED)
for feat in NEW_FEATURES:
    if feat in coef_df:
        direction = "reduces FP (good)" if coef_df[feat] < 0 else "increases FP risk"
        print(f"  {feat:<35} coef={coef_df[feat]:>+8.4f}  ({direction})")

# ---------------------------------------------------------------------------
# Full threshold sweep for augmented model
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("AUGMENTED MODEL THRESHOLD SWEEP (test set)")
print("=" * 65)
print(f"{'Thresh':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'FAR':>7}")
print("-" * 45)
for t in np.arange(0.10, 0.91, 0.05):
    m = eval_at(y_test_a, p_test_a, t)
    marker = "  <-- best F1" if abs(t - t_a) < 0.03 else ""
    print(f"  {t:>5.2f}  {m['precision']:>10.3f}  {m['recall']:>8.3f}  "
          f"{m['f1']:>6.3f}  {m['far']:>7.3f}{marker}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
prec_b, rec_b, _ = precision_recall_curve(y_test_b, p_test_b)
prec_a, rec_a, _ = precision_recall_curve(y_test_a, p_test_a)

ax.plot(rec_b, prec_b, 'b-', lw=2,
        label=f'Baseline AP={average_precision_score(y_test_b, p_test_b):.3f}')
ax.plot(rec_a, prec_a, 'g-', lw=2,
        label=f'+Precision features AP={average_precision_score(y_test_a, p_test_a):.3f}')
ax.axhline(y_test_b.mean(), color='gray', linestyle='--', alpha=0.6,
           label=f'No-skill ({y_test_b.mean()*100:.1f}%)')
ax.scatter([m_b_60['recall']], [m_b_60['precision']],
           color='blue',  s=100, zorder=5, label=f'Baseline t=0.60')
ax.scatter([m_a_60['recall']], [m_a_60['precision']],
           color='green', s=100, zorder=5, label=f'Augmented t=0.60')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Baseline vs +precision features\n(test set 2023-2025)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/model/precision_features_pr_curve.png", dpi=150, bbox_inches='tight')
print("\nSaved figures/model/precision_features_pr_curve.png")