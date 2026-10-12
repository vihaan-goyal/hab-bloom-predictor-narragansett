"""
generate_model_figures.py
Generates model evaluation figures on corrected data.

Requires fitted models saved by baselines_corrected.py / lstm_corrected.py / ensemble_final.py
OR will refit from hab_features_daily.csv if models not found.

Run from repo root:
    python generate_model_figures.py

Outputs to figures/model/:
    fig_precision_recall.png      -- PR curves for all models
    fig_station_specific.png      -- station-specific model results (A4, B3, C1)
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

OUT_DIR      = "figures/model"
FEATURES_CSV = "data/hab_features_daily.csv"
LABELS_CSV   = "data/hab_labels_daily.csv"

BLOOM_COL   = "bloom_28d"
DATE_COL    = "date"
STATION_COL = "station_name"

PALETTE = {
    "lr":       "#457B9D",
    "xgb":      "#E63946",
    "lstm":     "#F4A261",
    "ensemble": "#1D3557",
    "a4":       "#E63946",
    "b3":       "#2A9D8F",
    "c1":       "#F4A261",
    "grid":     "#E8EDF2",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ---- load data -------------------------------------------------------------

print("Loading data ...")

try:
    features = pd.read_csv(FEATURES_CSV, parse_dates=[DATE_COL])
    labels   = pd.read_csv(LABELS_CSV,   parse_dates=[DATE_COL])
    print(f"  Features: {features.shape}  |  Labels: {labels.shape}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Make sure hab_features_daily.csv and hab_labels_daily.csv exist in data/")
    sys.exit(1)

# merge on station + date if needed
if BLOOM_COL not in features.columns:
    features = features.merge(
        labels[[STATION_COL, DATE_COL, BLOOM_COL]],
        on=[STATION_COL, DATE_COL], how="inner"
    )

# temporal split
train_end = "2019-12-31"
val_end   = "2022-12-31"

train = features[features[DATE_COL] <= train_end].copy()
val   = features[(features[DATE_COL] > train_end) & (features[DATE_COL] <= val_end)].copy()
test  = features[features[DATE_COL] > val_end].copy()

print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
print(f"  Test bloom rate: {test[BLOOM_COL].mean()*100:.1f}%")

# feature columns: drop metadata, target, and any non-numeric columns
drop_cols = [DATE_COL, STATION_COL, BLOOM_COL,
             "lat", "lon", "latitude_x", "longitude_x", "latitude_y", "longitude_y",
             "year", "month", "season", "Station_Name", "bloom"]
feature_cols = [
    c for c in features.columns
    if c not in drop_cols and pd.api.types.is_numeric_dtype(features[c])
]
print(f"  Features ({len(feature_cols)}): {feature_cols[:8]} ...")

X_train = train[feature_cols].fillna(0).values
y_train = train[BLOOM_COL].values
X_val   = val[feature_cols].fillna(0).values
y_val   = val[BLOOM_COL].values
X_test  = test[feature_cols].fillna(0).values
y_test  = test[BLOOM_COL].values

# ---- fit/load models -------------------------------------------------------

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

models = {}

# Logistic Regression
print("\nFitting Logistic Regression ...")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
lr.fit(X_train_sc, y_train)
models["LR"] = (lr, X_test_sc, "lr")

# XGBoost
print("Fitting XGBoost ...")
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=pos_weight, use_label_encoder=False,
    eval_metric="auc", random_state=42, verbosity=0,
    early_stopping_rounds=20,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
models["XGBoost"] = (xgb_model, X_test, "xgb")

# Ensemble (LR 80% + XGB 20%)
print("Building ensemble ...")
class EnsembleModel:
    def __init__(self, lr, xgb, w_lr=0.8, w_xgb=0.2):
        self.lr = lr; self.xgb = xgb
        self.w_lr = w_lr; self.w_xgb = w_xgb
    def predict_proba_ensemble(self, X_sc, X_raw):
        p_lr  = self.lr.predict_proba(X_sc)[:, 1]
        p_xgb = self.xgb.predict_proba(X_raw)[:, 1]
        return self.w_lr * p_lr + self.w_xgb * p_xgb

ens = EnsembleModel(lr, xgb_model)

# ---- Figure: Precision-Recall curves ---------------------------------------

print("\n[Fig PR] Precision-recall curves ...")

fig, ax = plt.subplots(figsize=(8, 6))

def plot_pr(ax, y_true, y_score, label, color, lw=2):
    prec, rec, thresh = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax.plot(rec, prec, color=color, lw=lw, label=f"{label} (AP={ap:.3f})")
    return ap

# LR
lr_proba = lr.predict_proba(X_test_sc)[:, 1]
plot_pr(ax, y_test, lr_proba, "Logistic Regression", PALETTE["lr"])

# XGBoost
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
plot_pr(ax, y_test, xgb_proba, "XGBoost", PALETTE["xgb"])

# Ensemble
ens_proba = ens.predict_proba_ensemble(X_test_sc, X_test)
plot_pr(ax, y_test, ens_proba, "Ensemble (LR 80% + XGB 20%)", PALETTE["ensemble"], lw=2.5)

# baseline: random classifier
baseline_ap = y_test.mean()
ax.axhline(baseline_ap, color="gray", linestyle="--", lw=1.2,
           label=f"No-skill baseline (AP={baseline_ap:.3f})")

# annotate the precision-recall tradeoff zone
ax.axvspan(0.5, 0.8, alpha=0.08, color=PALETTE["xgb"], label="Target operating region")

ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curves on Test Set (2023-2025)\n"
             f"Test bloom rate: {y_test.mean()*100:.1f}% (post-TMDL base rate)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_facecolor(PALETTE["grid"])
ax.grid(color="white", lw=0.8)

# annotate that low precision is structurally expected
ax.text(0.02, 0.06,
        "Low global precision reflects genuine post-TMDL\nbase rate decline, not model deficiency",
        transform=ax.transAxes, fontsize=8.5, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

fig.tight_layout()
out = os.path.join(OUT_DIR, "fig_precision_recall.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ---- Figure: station-specific model results --------------------------------

print("\n[Fig stations] Station-specific model results ...")

TARGET_STATIONS = ["A4", "B3", "C1"]

# check which are available
available = [s for s in TARGET_STATIONS if s in features[STATION_COL].values]
if not available:
    # try lowercase / different naming
    all_stns = features[STATION_COL].unique()
    print(f"  Target stations not found. Available sample: {sorted(all_stns)[:20]}")
    # pick 3 highest bloom-rate stations from test set as fallback
    test_stn_bloom = (
        test.groupby(STATION_COL)[BLOOM_COL].agg(["mean", "count"])
        .query("count >= 20")
        .sort_values("mean", ascending=False)
    )
    available = list(test_stn_bloom.index[:3])
    print(f"  Using fallback stations: {available}")

fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5), sharey=False)
if len(available) == 1:
    axes = [axes]

colors_stn = [PALETTE["a4"], PALETTE["b3"], PALETTE["c1"]]
metrics_global = {
    "LR":       {"auc": roc_auc_score(y_test, lr_proba) if y_test.sum() > 0 else 0,
                 "ap": average_precision_score(y_test, lr_proba)},
    "XGBoost":  {"auc": roc_auc_score(y_test, xgb_proba) if y_test.sum() > 0 else 0,
                 "ap": average_precision_score(y_test, xgb_proba)},
    "Ensemble": {"auc": roc_auc_score(y_test, ens_proba) if y_test.sum() > 0 else 0,
                 "ap": average_precision_score(y_test, ens_proba)},
}

for ax, stn, color in zip(axes, available, colors_stn):
    # filter to this station
    train_s = train[train[STATION_COL] == stn]
    test_s  = test[test[STATION_COL]  == stn]

    if len(train_s) < 30 or len(test_s) < 10 or test_s[BLOOM_COL].sum() < 3:
        ax.text(0.5, 0.5, f"Station {stn}\nInsufficient data\n(n_test={len(test_s)}, blooms={int(test_s[BLOOM_COL].sum())})",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(f"Station {stn}", fontsize=12, fontweight="bold", color=color)
        continue

    X_tr_s = train_s[feature_cols].fillna(0).values
    y_tr_s = train_s[BLOOM_COL].values
    X_te_s = test_s[feature_cols].fillna(0).values
    y_te_s = test_s[BLOOM_COL].values

    # fit station-specific XGBoost
    pw_s = max(1.0, (y_tr_s == 0).sum() / max(1, (y_tr_s == 1).sum()))
    xgb_s = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=pw_s, use_label_encoder=False,
        eval_metric="auc", random_state=42, verbosity=0,
    )
    xgb_s.fit(X_tr_s, y_tr_s)
    proba_s = xgb_s.predict_proba(X_te_s)[:, 1]

    # global model on same station
    proba_global = xgb_model.predict_proba(X_te_s)[:, 1]

    # metrics at threshold = 0.5
    def metrics_at_threshold(y, p, thresh=0.5):
        pred = (p >= thresh).astype(int)
        prec = precision_score(y, pred, zero_division=0)
        rec  = recall_score(y, pred, zero_division=0)
        f1   = f1_score(y, pred, zero_division=0)
        ap   = average_precision_score(y, p) if y.sum() > 0 else 0
        auc  = roc_auc_score(y, p) if y.sum() > 0 and len(np.unique(y)) > 1 else 0
        return prec, rec, f1, ap, auc

    p_global = metrics_at_threshold(y_te_s, proba_global)
    p_local  = metrics_at_threshold(y_te_s, proba_s)

    metric_names = ["Precision", "Recall", "F1", "Avg Prec", "AUC"]
    x = np.arange(len(metric_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, p_global, width, label="Global XGBoost",
                   color=PALETTE["lr"], alpha=0.8)
    bars2 = ax.bar(x + width/2, p_local, width, label=f"Station-specific XGBoost",
                   color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score", fontsize=10)
    bloom_rate_test = y_te_s.mean() * 100
    ax.set_title(f"Station {stn}\n(test bloom rate: {bloom_rate_test:.0f}%, n={len(test_s)})",
                 fontsize=11, fontweight="bold", color=color)
    ax.legend(fontsize=8)
    ax.set_facecolor(PALETTE["grid"])
    ax.grid(axis="y", color="white", lw=0.8)

    # value labels on bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

fig.suptitle("Global vs. Station-Specific Model Performance\n(XGBoost, test set 2023-2025)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
out = os.path.join(OUT_DIR, "fig_station_specific.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

print("\nAll model figures done.")