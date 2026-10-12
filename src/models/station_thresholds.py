"""
station_thresholds.py
---------------------
Fits per-station optimal decision thresholds on the val set (2020-2022),
then evaluates on the test set (2023-2025).

Rationale: eastern LIS stations have very low bloom rates (~1-5%) while
western stations (A4, B3, C1) have much higher rates (~20-40%). A single
global threshold over-flags eastern stations and under-flags western ones.
Per-station thresholds correct for this.

Run from repo root:
    python src/models/station_thresholds.py
"""

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
# Splits -- keep station_name and date for per-station analysis
# ---------------------------------------------------------------------------
train = df[df['date'].dt.year <= 2019]
val   = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)]
test  = df[df['date'].dt.year >= 2023]

def prepare(split, keep_meta=False):
    rows = split[FEATURES + ['bloom_28d', 'station_name', 'date']].dropna(subset=['bloom_28d'])
    X = rows[FEATURES].copy().reset_index(drop=True)
    y = rows['bloom_28d'].copy().reset_index(drop=True)
    meta = rows[['station_name', 'date']].copy().reset_index(drop=True)
    return X, y, meta

X_train, y_train, meta_train = prepare(train)
X_val,   y_val,   meta_val   = prepare(val)
X_test,  y_test,  meta_test  = prepare(test)

MED = X_train.median()

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Train bloom rate: {y_train.mean()*100:.1f}%")
print(f"Val   bloom rate: {y_val.mean()*100:.1f}%")
print(f"Test  bloom rate: {y_test.mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# Fit ensemble on train
# ---------------------------------------------------------------------------
print("\nFitting ensemble on train 1993-2019...")

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

# Ensemble probabilities
xgb_val_p  = xgb_model.predict_proba(X_val.fillna(MED))[:, 1]
lr_val_p   = lr_model.predict_proba(X_v_s)[:, 1]
ens_val    = 0.80 * lr_val_p + 0.20 * xgb_val_p

xgb_test_p = xgb_model.predict_proba(X_test.fillna(MED))[:, 1]
lr_test_p  = lr_model.predict_proba(X_te_s)[:, 1]
ens_test   = 0.80 * lr_test_p + 0.20 * xgb_test_p

# Global best-F1 threshold on val (baseline to beat)
from sklearn.metrics import precision_recall_curve as prc
prec_v, rec_v, thresh_v = prc(y_val, ens_val)
f1_v = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-9)
global_thresh = float(thresh_v[f1_v.argmax()])
print(f"Global best-F1 threshold (val): {global_thresh:.3f}")

# ---------------------------------------------------------------------------
# Per-station threshold fitting on val set
# Strategy: find threshold that maximizes F1 on val for each station.
# Fallback to global threshold if station has < MIN_BLOOMS positive examples.
# ---------------------------------------------------------------------------
MIN_BLOOMS  = 3    # minimum positive examples to fit a per-station threshold
GLOBAL_FALLBACK = global_thresh

station_thresholds = {}
station_val_stats  = []

val_probs_series  = pd.Series(ens_val,  index=meta_val.index)
test_probs_series = pd.Series(ens_test, index=meta_test.index)

print("\nFitting per-station thresholds on val set...")
for station in sorted(meta_val['station_name'].unique()):
    val_mask  = meta_val['station_name'] == station
    y_s       = y_val[val_mask]
    p_s       = val_probs_series[val_mask]

    n_pos = int(y_s.sum())
    n_tot = len(y_s)
    bloom_rate = n_pos / n_tot if n_tot > 0 else 0

    if n_pos < MIN_BLOOMS:
        # Not enough bloom examples -- use global threshold
        thresh = GLOBAL_FALLBACK
        method = 'global'
    else:
        prec_s, rec_s, thresh_s = prc(y_s, p_s)
        f1_s = 2 * prec_s * rec_s / (prec_s + rec_s + 1e-9)
        best_idx = f1_s.argmax()
        thresh = float(thresh_s[best_idx]) if best_idx < len(thresh_s) else GLOBAL_FALLBACK
        method = 'station'

    station_thresholds[station] = thresh
    station_val_stats.append({
        'station': station,
        'n_val':   n_tot,
        'n_blooms': n_pos,
        'bloom_rate': round(bloom_rate * 100, 1),
        'threshold': round(thresh, 3),
        'method':  method,
    })

stats_df = pd.DataFrame(station_val_stats).sort_values('bloom_rate', ascending=False)
print(f"\nStations with custom threshold: "
      f"{(stats_df['method']=='station').sum()} / {len(stats_df)}")
print(f"Stations using global fallback: "
      f"{(stats_df['method']=='global').sum()} / {len(stats_df)}")
print("\nTop 15 stations by bloom rate (val):")
print(stats_df[['station','n_val','n_blooms','bloom_rate','threshold','method']]
      .head(15).to_string(index=False))

# ---------------------------------------------------------------------------
# Apply per-station thresholds to test set
# ---------------------------------------------------------------------------
preds_global  = (ens_test >= global_thresh).astype(int)
preds_station = np.zeros(len(y_test), dtype=int)

for i, station in enumerate(meta_test['station_name']):
    thresh = station_thresholds.get(station, GLOBAL_FALLBACK)
    preds_station[i] = int(ens_test[i] >= thresh)

# ---------------------------------------------------------------------------
# Compare global vs station-adaptive on test set
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST SET COMPARISON (2023-2025)")
print("=" * 60)

def report(name, y_true, preds, probs):
    print(f"\n{name}")
    print(f"  AUC:       {roc_auc_score(y_true, probs):.4f}")
    print(f"  Precision: {precision_score(y_true, preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, preds, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_true, preds, zero_division=0):.4f}")
    tp = int(((preds==1)&(y_true==1)).sum())
    fp = int(((preds==1)&(y_true==0)).sum())
    fn = int(((preds==0)&(y_true==1)).sum())
    tn = int(((preds==0)&(y_true==0)).sum())
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}  FAR={far:.3f}")

report(f"Global threshold ({global_thresh:.3f})", y_test, preds_global,  ens_test)
report("Station-adaptive thresholds",             y_test, preds_station, ens_test)

# ---------------------------------------------------------------------------
# Per-station precision on test set
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PER-STATION TEST PRECISION (stations with >= 3 bloom events in test)")
print("=" * 70)
print(f"{'Station':>10}  {'N test':>7}  {'Blooms':>7}  {'Bloom%':>7}  "
      f"{'Thresh':>7}  {'Prec(global)':>13}  {'Prec(station)':>14}")
print("-" * 70)

per_station_rows = []
for station in sorted(meta_test['station_name'].unique()):
    test_mask = meta_test['station_name'] == station
    y_s   = y_test[test_mask]
    p_s   = ens_test[test_mask]
    n_pos = int(y_s.sum())
    n_tot = len(y_s)
    if n_pos < 3:
        continue

    thresh_g = global_thresh
    thresh_s = station_thresholds.get(station, GLOBAL_FALLBACK)

    preds_g = (p_s >= thresh_g).astype(int)
    preds_s = (p_s >= thresh_s).astype(int)

    prec_g = precision_score(y_s, preds_g, zero_division=0)
    prec_s = precision_score(y_s, preds_s, zero_division=0)

    per_station_rows.append({
        'station': station, 'n_test': n_tot, 'n_blooms': n_pos,
        'bloom_rate': n_pos/n_tot*100,
        'threshold': thresh_s,
        'prec_global': prec_g, 'prec_station': prec_s,
        'delta': prec_s - prec_g,
    })
    print(f"{station:>10}  {n_tot:>7}  {n_pos:>7}  {n_pos/n_tot*100:>6.1f}%  "
          f"{thresh_s:>7.3f}  {prec_g:>13.3f}  {prec_s:>14.3f}")

per_df = pd.DataFrame(per_station_rows)
if len(per_df) > 0:
    print(f"\nMean precision (stations with >=3 blooms):")
    print(f"  Global:  {per_df['prec_global'].mean():.3f}")
    print(f"  Station: {per_df['prec_station'].mean():.3f}")
    print(f"  Delta:   {per_df['delta'].mean():+.3f}")

# ---------------------------------------------------------------------------
# Save threshold table
# ---------------------------------------------------------------------------
thresh_out = pd.DataFrame([
    {'station': k, 'threshold': v} for k, v in station_thresholds.items()
]).sort_values('station')
thresh_out.to_csv("data/station_thresholds.csv", index=False)
print("\nSaved data/station_thresholds.csv")

# ---------------------------------------------------------------------------
# Figure: threshold per station vs bloom rate
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['steelblue' if m == 'station' else 'lightgray'
          for m in stats_df['method']]
ax.scatter(stats_df['bloom_rate'], stats_df['threshold'],
           c=colors, alpha=0.8, edgecolors='gray', linewidths=0.5, s=60)
ax.axhline(global_thresh, color='red', linestyle='--', alpha=0.7,
           label=f'Global threshold ({global_thresh:.3f})')
ax.set_xlabel('Station bloom rate in val set (%)', fontsize=12)
ax.set_ylabel('Per-station threshold', fontsize=12)
ax.set_title('Station-adaptive thresholds vs bloom rate\n'
             '(blue = custom, gray = global fallback)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Label top western stations
for _, row in stats_df[stats_df['station'].isin(['A4','B3','C1','A2','02','01'])].iterrows():
    ax.annotate(row['station'],
                (row['bloom_rate'], row['threshold']),
                textcoords='offset points', xytext=(5, 3), fontsize=9)

plt.tight_layout()
plt.savefig("figures/model/station_thresholds.png", dpi=150, bbox_inches='tight')
print("Saved figures/model/station_thresholds.png")