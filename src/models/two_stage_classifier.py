"""
two_stage_classifier.py
-----------------------
Stage 1: LR at low threshold -- wide net, high recall
Stage 2: Second LR trained on Stage 1 positives from train set --
         filters false alarms from the candidate pool

The two-stage approach separates concerns:
  - Stage 1 asks: "could this be a bloom?" (recall-optimized)
  - Stage 2 asks: "of the flagged candidates, is this actually a bloom?" (precision-optimized)

Run from repo root:
    python src/models/two_stage_classifier.py
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
    X = rows[FEATURES].copy().reset_index(drop=True)
    y = rows['bloom_28d'].copy().reset_index(drop=True)
    return X, y

X_train, y_train = prepare(train)
X_val,   y_val   = prepare(val)
X_test,  y_test  = prepare(test)

MED = X_train.median()

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Train bloom rate: {y_train.mean()*100:.1f}%")
print(f"Val   bloom rate: {y_val.mean()*100:.1f}%")
print(f"Test  bloom rate: {y_test.mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# Stage 1: LR trained on full train set
# Low threshold -- maximize recall, accept false alarms
# ---------------------------------------------------------------------------
print("\nFitting Stage 1 (LR, recall-optimized)...")

scaler1 = StandardScaler()
X_tr_s1 = scaler1.fit_transform(X_train.fillna(MED))
X_v_s1  = scaler1.transform(X_val.fillna(MED))
X_te_s1 = scaler1.transform(X_test.fillna(MED))

lr1 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr1.fit(X_tr_s1, y_train)

p1_train = lr1.predict_proba(X_tr_s1)[:, 1]
p1_val   = lr1.predict_proba(X_v_s1)[:, 1]
p1_test  = lr1.predict_proba(X_te_s1)[:, 1]

# Find Stage 1 threshold on val: target recall >= 0.80
# i.e. catch at least 80% of blooms going into Stage 2
prec_v, rec_v, thresh_v = precision_recall_curve(y_val, p1_val)
# find lowest threshold where recall >= 0.80
candidates = thresh_v[rec_v[:-1] >= 0.80]
t1 = float(candidates.min()) if len(candidates) > 0 else 0.30
print(f"Stage 1 threshold (recall>=0.80 on val): {t1:.3f}")

# Stage 1 performance on val
preds1_val = (p1_val >= t1).astype(int)
r1_val = recall_score(y_val, preds1_val, zero_division=0)
p1_val_prec = precision_score(y_val, preds1_val, zero_division=0)
print(f"Stage 1 val -- precision: {p1_val_prec:.3f}  recall: {r1_val:.3f}  "
      f"candidates: {preds1_val.sum()} / {len(preds1_val)}")

# Stage 1 performance on test (for reference)
preds1_test = (p1_test >= t1).astype(int)
print(f"Stage 1 test -- precision: {precision_score(y_test, preds1_test, zero_division=0):.3f}  "
      f"recall: {recall_score(y_test, preds1_test, zero_division=0):.3f}  "
      f"candidates: {preds1_test.sum()} / {len(preds1_test)}")

# ---------------------------------------------------------------------------
# Stage 2: LR trained ONLY on Stage 1 positives from train set
# Goal: of the flagged candidates, which are actually blooms?
# ---------------------------------------------------------------------------
print("\nFitting Stage 2 (LR, precision-optimized on Stage 1 positives)...")

# Get Stage 1 positives from train
train_candidates_mask = p1_train >= t1
X_train_s2 = X_train[train_candidates_mask].copy().reset_index(drop=True)
y_train_s2 = y_train[train_candidates_mask].reset_index(drop=True)

print(f"Stage 2 train set: {len(X_train_s2):,} candidates "
      f"({y_train_s2.mean()*100:.1f}% bloom rate)")

if y_train_s2.sum() < 10:
    print("ERROR: too few bloom examples in Stage 2 train set -- lower Stage 1 threshold")
else:
    # Get Stage 1 positives from val for Stage 2 evaluation
    val_candidates_mask  = preds1_val == 1
    test_candidates_mask = preds1_test == 1

    X_val_s2  = X_val[val_candidates_mask].copy().reset_index(drop=True)
    y_val_s2  = y_val[val_candidates_mask].reset_index(drop=True)
    X_test_s2 = X_test[test_candidates_mask].copy().reset_index(drop=True)
    y_test_s2 = y_test[test_candidates_mask].reset_index(drop=True)

    print(f"Stage 2 val  set: {len(X_val_s2):,} candidates "
          f"({y_val_s2.mean()*100:.1f}% bloom rate)")
    print(f"Stage 2 test set: {len(X_test_s2):,} candidates "
          f"({y_test_s2.mean()*100:.1f}% bloom rate)")

    scaler2 = StandardScaler()
    X_tr_s2_scaled = scaler2.fit_transform(X_train_s2.fillna(MED))
    X_v_s2_scaled  = scaler2.transform(X_val_s2.fillna(MED))
    X_te_s2_scaled = scaler2.transform(X_test_s2.fillna(MED))

    # Stage 2 uses balanced weights -- bloom rate in candidates is higher
    # so the problem is more tractable
    lr2 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr2.fit(X_tr_s2_scaled, y_train_s2)

    p2_val  = lr2.predict_proba(X_v_s2_scaled)[:, 1]
    p2_test = lr2.predict_proba(X_te_s2_scaled)[:, 1]

    # Find Stage 2 threshold on val candidates: maximize F1
    if y_val_s2.sum() >= 3:
        prec_s2, rec_s2, thresh_s2 = precision_recall_curve(y_val_s2, p2_val)
        f1_s2 = 2 * prec_s2 * rec_s2 / (prec_s2 + rec_s2 + 1e-9)
        t2 = float(thresh_s2[f1_s2.argmax()]) if f1_s2.argmax() < len(thresh_s2) else 0.5
    else:
        t2 = 0.5
    print(f"Stage 2 threshold (best F1 on val candidates): {t2:.3f}")

    # ---------------------------------------------------------------------------
    # Final two-stage predictions on test
    # A sample is positive only if Stage 1 flags it AND Stage 2 confirms it
    # ---------------------------------------------------------------------------
    final_preds = np.zeros(len(y_test), dtype=int)
    stage1_pos_idx = np.where(test_candidates_mask)[0]

    preds2_test = (p2_test >= t2).astype(int)
    confirmed_idx = stage1_pos_idx[preds2_test == 1]
    final_preds[confirmed_idx] = 1

    print("\n" + "=" * 60)
    print("TEST SET RESULTS (2023-2025)")
    print("=" * 60)

    def report(name, y, preds, probs=None):
        print(f"\n{name}")
        if probs is not None:
            print(f"  AUC:       {roc_auc_score(y, probs):.4f}")
            print(f"  AP:        {average_precision_score(y, probs):.4f}")
        print(f"  Precision: {precision_score(y, preds, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y, preds, zero_division=0):.4f}")
        print(f"  F1:        {f1_score(y, preds, zero_division=0):.4f}")
        tp = int(((preds==1)&(np.array(y)==1)).sum())
        fp = int(((preds==1)&(np.array(y)==0)).sum())
        fn = int(((preds==0)&(np.array(y)==1)).sum())
        tn = int(((preds==0)&(np.array(y)==0)).sum())
        far = fp/(fp+tn) if (fp+tn)>0 else 0
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}  FAR={far:.3f}")

    # Single-stage LR at threshold 0.60 (current best)
    report("Single-stage LR (t=0.60)", y_test,
           (p1_test >= 0.60).astype(int), p1_test)

    # Two-stage
    report("Two-stage (Stage1 recall>=0.80, Stage2 best-F1)", y_test, final_preds)

    # Sweep Stage 2 thresholds to find best precision with F1 floor
    print("\n-- Stage 2 threshold sweep (test set) ----------------------")
    print(f"{'t2':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'Flagged':>8}")
    print("-" * 50)
    for t2_try in np.arange(0.10, 0.91, 0.10):
        fp_try = np.zeros(len(y_test), dtype=int)
        preds2_try = (p2_test >= t2_try).astype(int)
        confirmed_try = stage1_pos_idx[preds2_try == 1]
        fp_try[confirmed_try] = 1
        prec = precision_score(y_test, fp_try, zero_division=0)
        rec  = recall_score(y_test, fp_try, zero_division=0)
        f1   = f1_score(y_test, fp_try, zero_division=0)
        flagged = fp_try.sum()
        marker = "  <-- best F1" if abs(t2_try - t2) < 0.05 else ""
        print(f"{t2_try:>6.2f}  {prec:>10.3f}  {rec:>8.3f}  {f1:>6.3f}  "
              f"{flagged:>8}{marker}")

    # ---------------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Single stage PR curve
    prec_1, rec_1, _ = precision_recall_curve(y_test, p1_test)
    ap_1 = average_precision_score(y_test, p1_test)
    ax.plot(rec_1, prec_1, 'b-', lw=2, label=f'Single-stage LR AP={ap_1:.3f}')

    # Two-stage operating points at each t2
    ts2_points = []
    for t2_try in np.arange(0.10, 0.91, 0.05):
        fp_try = np.zeros(len(y_test), dtype=int)
        preds2_try = (p2_test >= t2_try).astype(int)
        confirmed_try = stage1_pos_idx[preds2_try == 1]
        fp_try[confirmed_try] = 1
        prec = precision_score(y_test, fp_try, zero_division=0)
        rec  = recall_score(y_test, fp_try, zero_division=0)
        ts2_points.append((rec, prec))

    recs2, precs2 = zip(*ts2_points)
    ax.plot(recs2, precs2, 'g-o', lw=2, markersize=4,
            label='Two-stage (varying t2)')

    # Mark current best single-stage
    ax.scatter(
        [recall_score(y_test, (p1_test>=0.60).astype(int), zero_division=0)],
        [precision_score(y_test, (p1_test>=0.60).astype(int), zero_division=0)],
        color='blue', s=100, zorder=5, label='Single-stage t=0.60'
    )
    # Mark two-stage best-F1
    ax.scatter(
        [recall_score(y_test, final_preds, zero_division=0)],
        [precision_score(y_test, final_preds, zero_division=0)],
        color='green', s=100, zorder=5, label=f'Two-stage best F1'
    )

    ax.axhline(y_test.mean(), color='gray', linestyle='--', alpha=0.6,
               label=f'No-skill ({y_test.mean()*100:.1f}%)')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Single-stage vs two-stage classifier\n(test set 2023-2025)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/model/two_stage_pr_curve.png", dpi=150, bbox_inches='tight')
    print("\nSaved figures/model/two_stage_pr_curve.png")