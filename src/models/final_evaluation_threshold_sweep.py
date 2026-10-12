"""
final_evaluation_threshold_sweep.py
------------------------------------
Fits LR on train (1993-2019), then runs a full threshold sweep on the TEST
set (2023-2025).

Uses hab_features_daily.csv and bloom_28d label -- matching daily_inference.py.

Run from repo root:
    python src/models/final_evaluation_threshold_sweep.py
"""

import glob
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    precision_score, recall_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

# Minimal thread-through for the label robustness check. Defaults reproduce the
# locked pipeline exactly (bloom label = chl > 10 ug/L, canonical preds path).
_ap = argparse.ArgumentParser()
_ap.add_argument('--bloom-threshold', type=float, default=10.0,
                 help='chlorophyll-a ug/L cutoff defining a bloom day (locked value: 10)')
_ap.add_argument('--preds-out', default='data/test_predictions.csv',
                 help='where to dump per-row test predictions')
_args, _ = _ap.parse_known_args()
BLOOM_THRESHOLD = _args.bloom_threshold

# ---------------------------------------------------------------------------
# Load + recompute features
# ---------------------------------------------------------------------------
print("Loading data/hab_features_tidal.csv...")
df = pd.read_csv("data/hab_features_tidal.csv")
df['date'] = pd.to_datetime(df['date'])


def load_percent_saturation():
    """percent_saturation lives only in the raw ERDDAP surface (depth_code='S')
    extracts, not in the feature CSVs. Concatenate every deep_wq_S_*.csv, drop
    the units row, and reduce to one value per (date, station_name).

    The ERDDAP timestamps store local midnight encoded as UTC (e.g. EDT midnight
    -> 04:00:00Z), so the UTC calendar date equals the local sample date that
    keys the feature files."""
    frames = []
    for f in sorted(glob.glob('data/raw/deep_wq_extra/deep_wq_S_*.csv')):
        # row 0 = header, row 1 = units ("UTC", "mg/L", ...) -> skip units.
        frames.append(pd.read_csv(
            f, skiprows=[1],
            usecols=['station_name', 'time', 'percent_saturation']))
    ps = pd.concat(frames, ignore_index=True)
    ps = ps[ps['station_name'].notna()].copy()
    ps['station_name'] = ps['station_name'].astype(str)
    ps['date'] = (pd.to_datetime(ps['time'], utc=True)
                    .dt.tz_localize(None).dt.normalize())
    ps['percent_saturation'] = pd.to_numeric(ps['percent_saturation'],
                                             errors='coerce')
    return (ps.dropna(subset=['percent_saturation'])
              .groupby(['date', 'station_name'], as_index=False)
              ['percent_saturation'].mean())


if 'percent_saturation' not in df.columns:
    print("Merging percent_saturation from data/raw/deep_wq_extra/deep_wq_S_*.csv...")
    ps = load_percent_saturation()
    df['station_name'] = df['station_name'].astype(str)
    df = df.merge(ps, on=['date', 'station_name'], how='left')
    print(f"  percent_saturation coverage: "
          f"{df['percent_saturation'].notna().mean() * 100:.1f}%")

print("Merging max_gust_3d from data/gust_features_daily.csv...")
gust = pd.read_csv("data/gust_features_daily.csv", usecols=['date', 'max_gust_3d'])
gust['date'] = pd.to_datetime(gust['date'])
df = df.merge(gust, on='date', how='left')
print(f"  max_gust_3d coverage: {df['max_gust_3d'].notna().mean() * 100:.1f}%")

for n, min_p in [(3, 2), (6, 3), (9, 5), (14, 7), (21, 10)]:
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
        if mask.any() and (chl[mask] > BLOOM_THRESHOLD).any():
            labels[i] = 1
    df.loc[idx, 'bloom_28d'] = labels

# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------
FEATURES_ALL = [
    'Chlorophyll', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
    'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
    'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
    'chl_anomaly', 'chl_climatology',
    'do_lag1', 'temp_lag1', 'sal_lag1',
    'sal_lag2', 'sal_lag3', 'sal_lag4',
    'sea_water_temperature', 'sea_water_salinity',
    'oxygen_concentration_in_sea_water',
    'month', 'latitude_x', 'longitude_x',
    'nox_lag2', 'dip_lag2', 'dip_change', 'dip_x_month',
    'neighbor_chl3_mean', 'neighbor_chl3_lag1',
    'tidal_gt_anom', 'tidal_msl_anom',
    'percent_saturation',
    'max_gust_3d',
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
# Fit LR on train 1993-2022
# ---------------------------------------------------------------------------
print("\nFitting LR on train 1993-2019 (incl. tidal anomaly features)...")

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train.fillna(MED))
X_v_s  = scaler.transform(X_val.fillna(MED))
X_te_s = scaler.transform(X_test.fillna(MED))

lr_model = LogisticRegression(class_weight='balanced', C=0.05, max_iter=1000, random_state=42)
lr_model.fit(X_tr_s, y_train)

# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------
lr_val_p  = lr_model.predict_proba(X_v_s)[:, 1]
lr_test_p = lr_model.predict_proba(X_te_s)[:, 1]

# --- dump test predictions for bootstrap_ci.py ---
_test_meta = test.loc[test[FEATURES + ['bloom_28d']].dropna(subset=['bloom_28d']).index]
pd.DataFrame({
    "station_name": _test_meta['station_name'].astype(str).values,
    "date":         _test_meta['date'].values,
    "y_true":       y_test.values,
    "y_prob":       lr_test_p,
}).to_csv(_args.preds_out, index=False)
print(f"Saved {_args.preds_out} ({len(y_test):,} rows)")

print(f"\nLR Val AUC  (2020-2022):      {roc_auc_score(y_val,  lr_val_p):.4f}")
print(f"LR Test AUC (2023-2025): {roc_auc_score(y_test, lr_test_p):.4f}")
print(f"LR Test AP  (2023-2025): {average_precision_score(y_test, lr_test_p):.4f}")

# ---------------------------------------------------------------------------
# Threshold sweep -- LR on test set
# ---------------------------------------------------------------------------
thresholds = np.arange(0.10, 0.91, 0.05)
probs = lr_test_p

rows = []
for t in thresholds:
    preds = (probs >= t).astype(int)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    far  = fp / (fp + tn) if (fp + tn) > 0 else 0
    rows.append({
        'threshold': round(t, 2),
        'precision': round(prec, 3),
        'recall':    round(rec, 3),
        'f1':        round(f1, 3),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'false_alarm_rate': round(far, 3),
    })

sweep = pd.DataFrame(rows)
best_t = sweep.loc[sweep['f1'].idxmax(), 'threshold']

print("\n" + "=" * 72)
print("THRESHOLD SWEEP — LR TEST SET (2023-2025)")
print("=" * 72)
print(f"{'Thresh':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  "
      f"{'TP':>5}  {'FP':>5}  {'FN':>5}  {'FAR':>7}")
print("-" * 72)
for _, r in sweep.iterrows():
    marker = ""
    if r['threshold'] == 0.50:
        marker = "  <-- default"
    elif r['threshold'] == best_t:
        marker = "  <-- best F1"
    print(f"  {r['threshold']:>5.2f}  {r['precision']:>10.3f}  {r['recall']:>8.3f}  "
          f"{r['f1']:>6.3f}  {r['TP']:>5,}  {r['FP']:>5,}  {r['FN']:>5,}  "
          f"{r['false_alarm_rate']:>7.3f}{marker}")

sweep.to_csv("data/threshold_sweep_results.csv", index=False)
print("\nSaved data/threshold_sweep_results.csv")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(sweep['threshold'], sweep['precision'], 'b-o', label='Precision', markersize=4)
ax.plot(sweep['threshold'], sweep['recall'],    'r-o', label='Recall',    markersize=4)
ax.plot(sweep['threshold'], sweep['f1'],        'g-o', label='F1',        markersize=4)
ax.axvline(0.50,   color='gray',  linestyle='--', alpha=0.7, label='Default (0.5)')
ax.axvline(best_t, color='green', linestyle=':',  alpha=0.8, label=f'Best F1 ({best_t:.2f})')
ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision / Recall / F1 vs Threshold\n(LR, test 2023-2025)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(0, 1)

prec_curve, rec_curve, _ = precision_recall_curve(y_test, probs)
ap = average_precision_score(y_test, probs)
ax2 = axes[1]
ax2.plot(rec_curve, prec_curve, 'b-', lw=2, label=f'LR AP = {ap:.3f}')
ax2.axhline(y_test.mean(), color='gray', linestyle='--', alpha=0.7,
            label=f'No-skill baseline ({y_test.mean()*100:.1f}%)')
p_default = precision_score(y_test, (probs >= 0.50).astype(int), zero_division=0)
r_default = recall_score(y_test,    (probs >= 0.50).astype(int), zero_division=0)
ax2.scatter([r_default], [p_default], color='gray',  s=80, zorder=5, label='Threshold 0.50')
p_best = precision_score(y_test, (probs >= best_t).astype(int), zero_division=0)
r_best = recall_score(y_test,    (probs >= best_t).astype(int), zero_division=0)
ax2.scatter([r_best], [p_best], color='green', s=80, zorder=5, label=f'Best F1 ({best_t:.2f})')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve\n(LR, test 2023-2025)', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("figures/model/threshold_sweep.png", dpi=150, bbox_inches='tight')
print("Saved figures/model/threshold_sweep.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("SUMMARY AT KEY THRESHOLDS (LR)")
print("=" * 50)
for t in [0.30, 0.40, 0.50, best_t]:
    r = sweep[sweep['threshold'] == round(t, 2)]
    if r.empty:
        continue
    r = r.iloc[0]
    label = "(best F1)" if t == best_t else ""
    print(f"  t={t:.2f}: precision={r['precision']:.3f}  recall={r['recall']:.3f}  "
          f"F1={r['f1']:.3f}  FAR={r['false_alarm_rate']:.3f}  {label}")