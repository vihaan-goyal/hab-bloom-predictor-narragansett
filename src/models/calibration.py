"""
calibration.py
--------------
Fits isotonic regression calibration on val set (2020-2022) ensemble
probabilities, then evaluates calibrated vs uncalibrated on test set
(2023-2025).

Calibration does not change AUC -- it reshapes the probability distribution
so that p=0.6 actually means ~60% of predictions are true blooms. This makes
threshold choices more reliable and can improve precision at a given threshold.

Run from repo root:
    python src/models/calibration.py
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    precision_score, recall_score,
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

FEATURES_ALL = [
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
FEATURES = [f for f in FEATURES_ALL if f in df.columns]

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
train = df[df['date'].dt.year <= 2019]
val   = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)]
test  = df[df['date'].dt.year >= 2023]

def prepare(split):
    rows = split[FEATURES + ['bloom_28d']].dropna(subset=['bloom_28d'])
    X = rows[FEATURES].copy()
    y = rows['bloom_28d'].copy()
    return X.reset_index(drop=True), y.reset_index(drop=True)

X_train, y_train = prepare(train)
X_val,   y_val   = prepare(val)
X_test,  y_test  = prepare(test)

MED = X_train.median()

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Train bloom rate: {y_train.mean()*100:.1f}%")
print(f"Val   bloom rate: {y_val.mean()*100:.1f}%")
print(f"Test  bloom rate: {y_test.mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# Fit base ensemble on train
# ---------------------------------------------------------------------------
print("\nFitting base ensemble on train 1993-2019...")

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=1.0, eval_metric='auc',
    random_state=42, verbosity=0,
)
xgb_model.fit(
    X_train.fillna(MED), y_train,
    eval_set=[(X_val.fillna(MED), y_val)],
    verbose=False,
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train.fillna(MED))
X_v_s  = scaler.transform(X_val.fillna(MED))
X_te_s = scaler.transform(X_test.fillna(MED))

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_tr_s, y_train)

# Raw ensemble probabilities
xgb_val_p  = xgb_model.predict_proba(X_val.fillna(MED))[:, 1]
lr_val_p   = lr_model.predict_proba(X_v_s)[:, 1]
ens_val    = 0.80 * lr_val_p + 0.20 * xgb_val_p

xgb_test_p = xgb_model.predict_proba(X_test.fillna(MED))[:, 1]
lr_test_p  = lr_model.predict_proba(X_te_s)[:, 1]
ens_test   = 0.80 * lr_test_p + 0.20 * xgb_test_p

print(f"Uncalibrated Val AUC:  {roc_auc_score(y_val,  ens_val):.4f}")
print(f"Uncalibrated Test AUC: {roc_auc_score(y_test, ens_test):.4f}")

# ---------------------------------------------------------------------------
# Fit isotonic calibration on val set probabilities
# Isotonic regression maps the raw ensemble scores to better-calibrated
# probabilities by fitting a monotone step function on the val set.
# It uses only val -- never test -- so the test evaluation stays honest.
# ---------------------------------------------------------------------------
print("\nFitting isotonic calibration on val set probabilities...")

from sklearn.isotonic import IsotonicRegression

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(ens_val, y_val)

ens_val_cal  = iso.predict(ens_val)
ens_test_cal = iso.predict(ens_test)

print(f"Calibrated Val AUC:    {roc_auc_score(y_val,  ens_val_cal):.4f}")
print(f"Calibrated Test AUC:   {roc_auc_score(y_test, ens_test_cal):.4f}")

# ---------------------------------------------------------------------------
# Helper: best-F1 metrics
# ---------------------------------------------------------------------------
def sweep_metrics(y_true, probs, label):
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)
        rows.append({'threshold': round(t, 2), 'precision': prec,
                     'recall': rec, 'f1': f1})
    df_s = pd.DataFrame(rows)
    best = df_s.loc[df_s['f1'].idxmax()]
    print(f"\n{label}")
    print(f"  AUC:          {roc_auc_score(y_true, probs):.4f}")
    print(f"  AP:           {average_precision_score(y_true, probs):.4f}")
    print(f"  Best-F1 thresh: {best['threshold']:.2f}")
    print(f"  Precision:    {best['precision']:.4f}")
    print(f"  Recall:       {best['recall']:.4f}")
    print(f"  F1:           {best['f1']:.4f}")
    return df_s, best

print("\n" + "=" * 60)
print("COMPARISON: UNCALIBRATED vs CALIBRATED (test set 2023-2025)")
print("=" * 60)

sweep_uncal, best_uncal = sweep_metrics(y_test, ens_test,     "Uncalibrated ensemble")
sweep_cal,   best_cal   = sweep_metrics(y_test, ens_test_cal, "Calibrated ensemble (isotonic on val)")

# ---------------------------------------------------------------------------
# Full threshold table for calibrated model
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("FULL THRESHOLD SWEEP — CALIBRATED ENSEMBLE (test 2023-2025)")
print("=" * 72)
print(f"{'Thresh':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'FAR':>7}")
print("-" * 50)
thresholds = np.arange(0.05, 0.96, 0.05)
for _, r in sweep_cal.iterrows():
    preds = (ens_test_cal >= r['threshold']).astype(int)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    marker = "  <-- best F1" if r['threshold'] == best_cal['threshold'] else ""
    print(f"  {r['threshold']:>5.2f}  {r['precision']:>10.3f}  {r['recall']:>8.3f}  "
          f"{r['f1']:>6.3f}  {far:>7.3f}{marker}")

# ---------------------------------------------------------------------------
# Reliability diagrams (calibration curves)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: reliability diagram
ax = axes[0]
n_bins = 8
prob_true_u, prob_pred_u = calibration_curve(y_test, ens_test,     n_bins=n_bins, strategy='quantile')
prob_true_c, prob_pred_c = calibration_curve(y_test, ens_test_cal, n_bins=n_bins, strategy='quantile')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
ax.plot(prob_pred_u, prob_true_u, 'b-o', markersize=6, label='Uncalibrated')
ax.plot(prob_pred_c, prob_true_c, 'g-o', markersize=6, label='Calibrated (isotonic)')
ax.set_xlabel('Mean predicted probability', fontsize=12)
ax.set_ylabel('Fraction of positives', fontsize=12)
ax.set_title('Reliability diagram\n(test set 2023-2025)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Right: precision-recall curves
ax2 = axes[1]
prec_u, rec_u, _ = precision_recall_curve(y_test, ens_test)
prec_c, rec_c, _ = precision_recall_curve(y_test, ens_test_cal)
ap_u = average_precision_score(y_test, ens_test)
ap_c = average_precision_score(y_test, ens_test_cal)

ax2.plot(rec_u, prec_u, 'b-', lw=2, label=f'Uncalibrated AP={ap_u:.3f}')
ax2.plot(rec_c, prec_c, 'g-', lw=2, label=f'Calibrated AP={ap_c:.3f}')
ax2.axhline(y_test.mean(), color='gray', linestyle='--', alpha=0.6,
            label=f'No-skill ({y_test.mean()*100:.1f}%)')

# Mark best-F1 points
p_u = precision_score(y_test, (ens_test     >= best_uncal['threshold']).astype(int), zero_division=0)
r_u = recall_score(y_test,    (ens_test     >= best_uncal['threshold']).astype(int), zero_division=0)
p_c = precision_score(y_test, (ens_test_cal >= best_cal['threshold']).astype(int),   zero_division=0)
r_c = recall_score(y_test,    (ens_test_cal >= best_cal['threshold']).astype(int),   zero_division=0)
ax2.scatter([r_u], [p_u], color='blue',  s=80, zorder=5,
            label=f'Uncal best F1 (t={best_uncal["threshold"]:.2f})')
ax2.scatter([r_c], [p_c], color='green', s=80, zorder=5,
            label=f'Cal best F1 (t={best_cal["threshold"]:.2f})')

ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve\n(test set 2023-2025)', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("figures/model/calibration.png", dpi=150, bbox_inches='tight')
print("\nSaved figures/model/calibration.png")

# ---------------------------------------------------------------------------
# Save calibrated test probabilities
# ---------------------------------------------------------------------------
pd.DataFrame({
    'ens_uncalibrated': ens_test,
    'ens_calibrated':   ens_test_cal,
    'y_true':           y_test.values,
}).to_csv("data/calibrated_test_probs.csv", index=False)
print("Saved data/calibrated_test_probs.csv")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Uncalibrated: precision={best_uncal['precision']:.3f}  "
      f"recall={best_uncal['recall']:.3f}  F1={best_uncal['f1']:.3f}  "
      f"thresh={best_uncal['threshold']:.2f}")
print(f"  Calibrated:   precision={best_cal['precision']:.3f}  "
      f"recall={best_cal['recall']:.3f}  F1={best_cal['f1']:.3f}  "
      f"thresh={best_cal['threshold']:.2f}")
delta = best_cal['precision'] - best_uncal['precision']
print(f"  Precision delta: {delta:+.3f}")
print(f"\n  Note: AUC should be identical or nearly identical.")
print(f"  If precision improved, the model was overconfident -- calibration fixed it.")
print(f"  If precision did not improve, the model was already well-calibrated.")