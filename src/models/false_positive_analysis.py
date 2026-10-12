"""
false_positive_analysis.py
--------------------------
Analyzes what false positives have in common vs true positives.
If a consistent pattern exists, it can be engineered into a feature
to improve precision.

Run from repo root:
    python src/models/false_positive_analysis.py
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
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
    rows = split[FEATURES + ['bloom_28d', 'station_name', 'date']].dropna(subset=['bloom_28d'])
    X    = rows[FEATURES].copy().reset_index(drop=True)
    y    = rows['bloom_28d'].copy().reset_index(drop=True)
    meta = rows[['station_name', 'date']].copy().reset_index(drop=True)
    return X, y, meta

X_train, y_train, _          = prepare(train)
X_val,   y_val,   _          = prepare(val)
X_test,  y_test,  meta_test  = prepare(test)
MED = X_train.median()

# ---------------------------------------------------------------------------
# Fit LR
# ---------------------------------------------------------------------------
print("Fitting LR on train...")
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train.fillna(MED))
X_v_s  = scaler.transform(X_val.fillna(MED))
X_te_s = scaler.transform(X_test.fillna(MED))

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_tr_s, y_train)

p_test = lr.predict_proba(X_te_s)[:, 1]
THRESHOLD = 0.60
preds = (p_test >= THRESHOLD).astype(int)

tp_mask = (preds == 1) & (y_test == 1)
fp_mask = (preds == 1) & (y_test == 0)
fn_mask = (preds == 0) & (y_test == 1)

print(f"\nAt threshold {THRESHOLD}:")
print(f"  TP: {tp_mask.sum()}  FP: {fp_mask.sum()}  FN: {fn_mask.sum()}")
print(f"  Precision: {precision_score(y_test, preds, zero_division=0):.3f}")
print(f"  Recall:    {recall_score(y_test, preds, zero_division=0):.3f}")

# ---------------------------------------------------------------------------
# Feature comparison: FP vs TP
# ---------------------------------------------------------------------------
X_fp = X_test[fp_mask].copy()
X_tp = X_test[tp_mask].copy()

print("\n" + "=" * 70)
print("FEATURE MEANS: FALSE POSITIVES vs TRUE POSITIVES")
print("=" * 70)
print(f"{'Feature':<30}  {'FP mean':>10}  {'TP mean':>10}  {'Delta':>10}  {'FP>TP?':>8}")
print("-" * 70)

comparison = []
for feat in FEATURES:
    fp_mean = X_fp[feat].mean()
    tp_mean = X_tp[feat].mean()
    delta   = fp_mean - tp_mean
    comparison.append({
        'feature': feat,
        'fp_mean': fp_mean,
        'tp_mean': tp_mean,
        'delta':   delta,
        'abs_delta': abs(delta),
    })

comp_df = pd.DataFrame(comparison).sort_values('abs_delta', ascending=False)

for _, row in comp_df.iterrows():
    direction = "FP higher" if row['delta'] > 0 else "TP higher"
    print(f"{row['feature']:<30}  {row['fp_mean']:>10.3f}  {row['tp_mean']:>10.3f}  "
          f"{row['delta']:>+10.3f}  {direction:>8}")

# ---------------------------------------------------------------------------
# Month and station breakdown of FPs
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("FALSE POSITIVES BY MONTH")
print("=" * 50)
fp_meta = meta_test[fp_mask].copy()
tp_meta = meta_test[tp_mask].copy()
fp_meta['month'] = pd.to_datetime(fp_meta['date']).dt.month
tp_meta['month'] = pd.to_datetime(tp_meta['date']).dt.month

month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
fp_by_month = fp_meta['month'].value_counts().sort_index()
tp_by_month = tp_meta['month'].value_counts().sort_index()
print(f"{'Month':<6}  {'FP':>5}  {'TP':>5}")
for m in sorted(set(list(fp_by_month.index) + list(tp_by_month.index))):
    print(f"{month_names.get(m,m):<6}  {fp_by_month.get(m,0):>5}  "
          f"{tp_by_month.get(m,0):>5}")

print("\n" + "=" * 50)
print("FALSE POSITIVES BY STATION (top 10)")
print("=" * 50)
fp_by_stn = fp_meta['station_name'].value_counts().head(10)
tp_by_stn = tp_meta['station_name'].value_counts()
print(f"{'Station':<10}  {'FP':>5}  {'TP':>5}")
for stn, fp_count in fp_by_stn.items():
    print(f"{stn:<10}  {fp_count:>5}  {tp_by_stn.get(stn,0):>5}")

# ---------------------------------------------------------------------------
# Key finding summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TOP 5 FEATURES WHERE FP AND TP DIFFER MOST")
print("=" * 70)
print("These are candidates for new engineered features:\n")
for _, row in comp_df.head(5).iterrows():
    direction = "lower" if row['delta'] < 0 else "higher"
    print(f"  {row['feature']:<30} FP is {direction} by {abs(row['delta']):.3f}")
    if row['delta'] < 0:
        print(f"    --> FPs have LOWER {row['feature']} than true blooms")
        print(f"    --> Threshold on this feature could filter FPs")
    else:
        print(f"    --> FPs have HIGHER {row['feature']} than true blooms")
        print(f"    --> FPs may be elevated-but-not-blooming stations")

# ---------------------------------------------------------------------------
# Figure: distribution of top discriminating features
# ---------------------------------------------------------------------------
top_feats = comp_df.head(6)['feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, feat in enumerate(top_feats):
    ax = axes[i]
    fp_vals = X_fp[feat].dropna()
    tp_vals = X_tp[feat].dropna()
    fn_vals = X_test[fn_mask][feat].dropna()

    bins = np.linspace(
        min(fp_vals.min(), tp_vals.min()),
        max(fp_vals.max(), tp_vals.max()), 25
    )
    ax.hist(fp_vals, bins=bins, alpha=0.5, color='red',   label=f'FP (n={len(fp_vals)})', density=True)
    ax.hist(tp_vals, bins=bins, alpha=0.5, color='green', label=f'TP (n={len(tp_vals)})', density=True)
    ax.hist(fn_vals, bins=bins, alpha=0.3, color='orange',label=f'FN (n={len(fn_vals)})', density=True)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Feature distributions: FP vs TP vs FN (test set, threshold=0.60)',
             fontsize=12)
plt.tight_layout()
plt.savefig("figures/model/fp_feature_distributions.png", dpi=150, bbox_inches='tight')
print("\nSaved figures/model/fp_feature_distributions.png")
print("\nOpen figures/model/fp_feature_distributions.png to see which features")
print("most clearly separate FPs from TPs -- those are your best candidates")
print("for a new engineered feature.")