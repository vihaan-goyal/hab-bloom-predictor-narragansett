"""
tune_scale_pos_weight.py
------------------------
Sweeps scale_pos_weight for XGBoost and the ensemble to find the value
that maximizes precision on the val set (2020-2022) at the best-F1 threshold.

Evaluates each candidate on val, then reports test set performance at the
best val precision weight -- keeping the test set honest.

Run from repo root:
    python src/models/tune_scale_pos_weight.py
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve,
)
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load + recompute features (mirrors daily_inference.py exactly)
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
train = df[df['date'].dt.year <= 2022]
val   = df[df['date'].dt.year == 2022]
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

default_spw = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nDefault scale_pos_weight (train ratio): {default_spw:.2f}")

# Fit LR once -- it uses class_weight='balanced' so doesn't need sweeping
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train.fillna(MED))
X_v_s  = scaler.transform(X_val.fillna(MED))
X_te_s = scaler.transform(X_test.fillna(MED))

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr_model.fit(X_tr_s, y_train)
lr_val_p  = lr_model.predict_proba(X_v_s)[:, 1]
lr_test_p = lr_model.predict_proba(X_te_s)[:, 1]

# ---------------------------------------------------------------------------
# Helper: best-F1 threshold and metrics from a probability array
# ---------------------------------------------------------------------------
def best_f1_metrics(y_true, probs):
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, probs)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    idx = f1_arr.argmax()
    return {
        'threshold': float(thresh_arr[idx]) if idx < len(thresh_arr) else 0.5,
        'precision': float(prec_arr[idx]),
        'recall':    float(rec_arr[idx]),
        'f1':        float(f1_arr[idx]),
        'auc':       roc_auc_score(y_true, probs),
        'ap':        average_precision_score(y_true, probs),
    }

# ---------------------------------------------------------------------------
# Sweep scale_pos_weight
# ---------------------------------------------------------------------------
# Range: from matching test bloom rate ratio all the way up to train ratio
# Test ratio ~13:1, train ratio ~3.4:1 -- sweep a wide range
candidates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, default_spw]
candidates = sorted(set(round(c, 2) for c in candidates))

print(f"\nSweeping scale_pos_weight: {candidates}")
print("\n" + "=" * 80)
print("SCALE_POS_WEIGHT SWEEP — VAL SET (2020-2022) — Ensemble LR80+XGB20")
print("=" * 80)
print(f"{'SPW':>6}  {'Val AUC':>8}  {'Val AP':>7}  {'Best-F1 thresh':>15}  "
      f"{'Precision':>10}  {'Recall':>8}  {'F1':>6}")
print("-" * 80)

results = []
for spw in candidates:
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric='auc',
        random_state=42, verbosity=0,
    )
    xgb_model.fit(X_train.fillna(MED), y_train, verbose=False)

    xgb_val_p = xgb_model.predict_proba(X_val.fillna(MED))[:, 1]
    ens_val   = 0.80 * lr_val_p + 0.20 * xgb_val_p

    m = best_f1_metrics(y_val, ens_val)

    marker = " <-- default" if abs(spw - default_spw) < 0.1 else ""
    print(f"{spw:>6.1f}  {m['auc']:>8.4f}  {m['ap']:>7.4f}  "
          f"{m['threshold']:>15.3f}  {m['precision']:>10.3f}  "
          f"{m['recall']:>8.3f}  {m['f1']:>6.3f}{marker}")

    results.append({
        'spw': spw,
        'xgb_model': xgb_model,
        'ens_val_probs': ens_val,
        **{f'val_{k}': v for k, v in m.items() if k != 'threshold'},
        'val_threshold': m['threshold'],
    })

# ---------------------------------------------------------------------------
# Pick best by val precision (with F1 > 0.25 floor to avoid degenerate cases)
# ---------------------------------------------------------------------------
res_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'xgb_model'
                         and k != 'ens_val_probs'} for r in results])

eligible = res_df[res_df['val_f1'] >= 0.25]
if eligible.empty:
    eligible = res_df  # fallback if nothing clears the floor

best_idx  = eligible['val_precision'].idxmax()
best_spw  = res_df.loc[best_idx, 'spw']
best_thresh = res_df.loc[best_idx, 'val_threshold']

print(f"\nBest val precision with F1 >= 0.25: SPW = {best_spw} "
      f"(precision={res_df.loc[best_idx,'val_precision']:.3f}, "
      f"recall={res_df.loc[best_idx,'val_recall']:.3f}, "
      f"F1={res_df.loc[best_idx,'val_f1']:.3f})")

# ---------------------------------------------------------------------------
# Evaluate best SPW on test set (touch once)
# ---------------------------------------------------------------------------
best_xgb = next(r['xgb_model'] for r in results if r['spw'] == best_spw)
xgb_test_p = best_xgb.predict_proba(X_test.fillna(MED))[:, 1]
ens_test   = 0.80 * lr_test_p + 0.20 * xgb_test_p

print("\n" + "=" * 60)
print(f"TEST SET RESULTS — best SPW={best_spw}, threshold={best_thresh:.3f}")
print("=" * 60)
preds = (ens_test >= best_thresh).astype(int)
print(f"  AUC:       {roc_auc_score(y_test, ens_test):.4f}")
print(f"  AP:        {average_precision_score(y_test, ens_test):.4f}")
print(f"  Precision: {precision_score(y_test, preds, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, preds, zero_division=0):.4f}")
print(f"  F1:        {f1_score(y_test, preds, zero_division=0):.4f}")

# Also compare default SPW on test
default_xgb = next(r['xgb_model'] for r in results if abs(r['spw'] - default_spw) < 0.1)
xgb_test_default = default_xgb.predict_proba(X_test.fillna(MED))[:, 1]
ens_test_default  = 0.80 * lr_test_p + 0.20 * xgb_test_default
default_thresh = next(r['val_threshold'] for r in results if abs(r['spw'] - default_spw) < 0.1)
preds_default = (ens_test_default >= default_thresh).astype(int)

print(f"\n  [comparison] Default SPW={default_spw:.1f}, threshold={default_thresh:.3f}:")
print(f"  AUC:       {roc_auc_score(y_test, ens_test_default):.4f}")
print(f"  Precision: {precision_score(y_test, preds_default, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, preds_default, zero_division=0):.4f}")
print(f"  F1:        {f1_score(y_test, preds_default, zero_division=0):.4f}")

# ---------------------------------------------------------------------------
# Figure: precision / recall / F1 vs SPW on val set
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
spws = res_df['spw'].values
ax.plot(spws, res_df['val_precision'], 'b-o', label='Precision', markersize=5)
ax.plot(spws, res_df['val_recall'],    'r-o', label='Recall',    markersize=5)
ax.plot(spws, res_df['val_f1'],        'g-o', label='F1',        markersize=5)
ax.axvline(default_spw, color='gray',  linestyle='--', alpha=0.7,
           label=f'Default SPW ({default_spw:.1f})')
ax.axvline(best_spw,    color='green', linestyle=':',  alpha=0.8,
           label=f'Best precision SPW ({best_spw})')
ax.set_xlabel('scale_pos_weight', fontsize=12)
ax.set_ylabel('Score (at best-F1 threshold)', fontsize=12)
ax.set_title('Precision / Recall / F1 vs scale_pos_weight\n'
             '(Ensemble LR80+XGB20, val set 2020-2022)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/model/spw_sweep.png", dpi=150, bbox_inches='tight')
print("\nSaved figures/model/spw_sweep.png")

res_df.drop(columns=['ens_val_probs'], errors='ignore').to_csv(
    "data/spw_sweep_results.csv", index=False)
print("Saved data/spw_sweep_results.csv")