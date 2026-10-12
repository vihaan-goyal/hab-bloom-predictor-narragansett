import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv("data/hab_features_final.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df['bloom_7d_ahead'] = df.groupby('station_name')['bloom'].shift(-7)

FEATURES = [
    'latitude', 'longitude', 'month',
    'sea_water_temperature', 'sea_water_salinity',
    'oxygen_concentration_in_sea_water', 'pH',
    'chl_anomaly', 'chl_climatology',
    'chl_lag3', 'chl_lag7', 'chl_lag14', 'chl_lag21',
    'chl_roll7_mean', 'chl_roll7_std',
    'total_discharge_cfs', 'discharge_lag5',
    'discharge_lag10', 'discharge_lag15',
    'discharge_roll7_mean',
]

# Nitrogen-related features we will manipulate
NITROGEN_FEATURES = [
    'total_discharge_cfs', 'discharge_lag5',
    'discharge_lag10', 'discharge_lag15',
    'discharge_roll7_mean',
]

TARGET = 'bloom_7d_ahead'

# ------------------------------------------------------------------
# Train model on 1993-2019, evaluate on 2020-2022
# ------------------------------------------------------------------
train = df[df['date'].dt.year <= 2019]
val   = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)]
test  = df[df['date'].dt.year >= 2023]

def prepare(split, extra_cols=None):
    cols = FEATURES + [TARGET]
    if extra_cols:
        cols += extra_cols
    X = split[cols].dropna()
    y = X.pop(TARGET)
    return X, y

X_train, y_train = prepare(train)
X_val,   y_val   = prepare(val)

# Keep station and date for val set for analysis
val_meta = val[FEATURES + [TARGET, 'station_name', 'date']].dropna()
X_val_full = val_meta[FEATURES]
y_val_full = val_meta[TARGET]
station_val = val_meta['station_name']
date_val    = val_meta['date']

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_state=42,
    n_jobs=-1, eval_metric='logloss', verbosity=0
)
model.fit(X_train.values, y_train.values,
          eval_set=[(X_val.values, y_val.values)],
          verbose=False)

baseline_probs = model.predict_proba(X_val_full.values)[:,1]
baseline_auc   = roc_auc_score(y_val_full, baseline_probs)
print(f"Baseline Val AUC: {baseline_auc:.4f}")

THRESHOLD = 0.5
baseline_blooms = (baseline_probs >= THRESHOLD).sum()
print(f"Baseline predicted blooms: {baseline_blooms} / {len(baseline_probs)}")

# ------------------------------------------------------------------
# Part 1: Counterfactual Reduction Analysis
# How many predicted blooms are prevented if we reduce nitrogen
# (discharge) by X%?
# ------------------------------------------------------------------
print("\n" + "="*60)
print("PART 1: Counterfactual Nitrogen Reduction Analysis")
print("="*60)

reduction_levels = [0, 5, 10, 15, 20, 25, 30, 40, 50]
results_reduction = []

for pct in reduction_levels:
    X_cf = X_val_full.copy()
    factor = 1.0 - pct / 100.0
    X_cf[NITROGEN_FEATURES] = X_cf[NITROGEN_FEATURES] * factor

    cf_probs  = model.predict_proba(X_cf.values)[:,1]
    cf_blooms = (cf_probs >= THRESHOLD).sum()
    blooms_prevented = baseline_blooms - cf_blooms
    pct_prevented = blooms_prevented / baseline_blooms * 100 if baseline_blooms > 0 else 0

    results_reduction.append({
        'reduction_pct': pct,
        'predicted_blooms': cf_blooms,
        'blooms_prevented': blooms_prevented,
        'pct_prevented': pct_prevented,
    })
    print(f"  {pct:3d}% reduction -> {cf_blooms} blooms ({blooms_prevented} prevented, {pct_prevented:.1f}%)")

df_reduction = pd.DataFrame(results_reduction)
df_reduction.to_csv("data/nitrogen_reduction_results.csv", index=False)

# ------------------------------------------------------------------
# Part 2: Station Sensitivity Analysis
# Which stations show the biggest bloom probability reduction
# for a given nitrogen reduction?
# ------------------------------------------------------------------
print("\n" + "="*60)
print("PART 2: Station Sensitivity Analysis")
print("="*60)

REDUCTION_PCT = 30
station_results = []

for station in station_val.unique():
    mask = station_val == station
    X_station = X_val_full[mask].copy()

    if len(X_station) < 10:
        continue

    base_probs_s = model.predict_proba(X_station.values)[:,1]
    base_bloom_rate = (base_probs_s >= THRESHOLD).mean()

    X_cf = X_station.copy()
    X_cf[NITROGEN_FEATURES] = X_cf[NITROGEN_FEATURES] * (1 - REDUCTION_PCT / 100)
    cf_probs_s   = model.predict_proba(X_cf.values)[:,1]
    cf_bloom_rate = (cf_probs_s >= THRESHOLD).mean()

    sensitivity = base_bloom_rate - cf_bloom_rate
    lat = X_station['latitude'].iloc[0]
    lon = X_station['longitude'].iloc[0]

    station_results.append({
        'station': station,
        'latitude': lat,
        'longitude': lon,
        'baseline_bloom_rate': base_bloom_rate,
        'reduced_bloom_rate': cf_bloom_rate,
        'absolute_reduction': sensitivity,
        'relative_reduction_pct': sensitivity / base_bloom_rate * 100 if base_bloom_rate > 0 else 0,
        'n_samples': len(X_station),
    })

df_stations = pd.DataFrame(station_results).sort_values('absolute_reduction', ascending=False)
df_stations.to_csv("data/station_sensitivity_results.csv", index=False)

print(f"\nTop 10 most sensitive stations to {REDUCTION_PCT}% nitrogen reduction:")
print(df_stations[['station','latitude','longitude',
                    'baseline_bloom_rate','reduced_bloom_rate',
                    'absolute_reduction','relative_reduction_pct']].head(10).to_string(index=False))

# ------------------------------------------------------------------
# Part 3: Critical Threshold Analysis
# For each station, what % reduction is needed to push bloom
# probability below 50%?
# ------------------------------------------------------------------
print("\n" + "="*60)
print("PART 3: Critical Reduction Threshold Per Station")
print("="*60)

threshold_results = []

for station in station_val.unique():
    mask = station_val == station
    X_station = X_val_full[mask].copy()

    if len(X_station) < 10:
        continue

    base_probs_s = model.predict_proba(X_station.values)[:,1]
    high_risk = X_station[base_probs_s >= THRESHOLD]

    if len(high_risk) == 0:
        continue

    # Binary search for minimum reduction needed
    lo, hi = 0, 100
    for _ in range(20):
        mid = (lo + hi) / 2
        X_cf = high_risk.copy()
        X_cf[NITROGEN_FEATURES] = X_cf[NITROGEN_FEATURES] * (1 - mid / 100)
        cf_probs = model.predict_proba(X_cf.values)[:,1]
        if (cf_probs >= THRESHOLD).mean() < 0.5:
            hi = mid
        else:
            lo = mid

    threshold_results.append({
        'station': station,
        'latitude': high_risk['latitude'].iloc[0],
        'longitude': high_risk['longitude'].iloc[0],
        'min_reduction_needed_pct': round(hi, 1),
        'n_high_risk_days': len(high_risk),
    })

df_thresholds = pd.DataFrame(threshold_results).sort_values('min_reduction_needed_pct')
df_thresholds.to_csv("data/station_threshold_results.csv", index=False)

print("\nReduction needed to cut >50% of high-risk days per station:")
print(df_thresholds.to_string(index=False))

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

# Figure 1: Bloom prevention curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df_reduction['reduction_pct'], df_reduction['pct_prevented'],
        marker='o', color='steelblue', linewidth=2)
ax.fill_between(df_reduction['reduction_pct'], df_reduction['pct_prevented'],
                alpha=0.15, color='steelblue')
ax.axhline(y=50, linestyle='--', color='gray', alpha=0.7, label='50% prevention')
ax.set_xlabel("Nitrogen/Discharge Reduction (%)")
ax.set_ylabel("Predicted Blooms Prevented (%)")
ax.set_title("Bloom Prevention Curve: Impact of Nitrogen Reduction\n(Long Island Sound, 2020-2022 Validation)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/intervention/bloom_prevention_curve.png", dpi=150)
plt.close()
print("\nSaved figures/intervention/bloom_prevention_curve.png")

# Figure 2: Station sensitivity map
fig, ax = plt.subplots(figsize=(12, 6))
sc = ax.scatter(
    df_stations['longitude'],
    df_stations['latitude'],
    c=df_stations['absolute_reduction'],
    cmap='YlOrRd',
    s=df_stations['n_samples'] / df_stations['n_samples'].max() * 300 + 30,
    alpha=0.8,
    edgecolors='gray',
    linewidths=0.5
)
plt.colorbar(sc, ax=ax, label=f'Bloom Rate Reduction at {REDUCTION_PCT}% N Reduction')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Station Sensitivity to {REDUCTION_PCT}% Nitrogen Reduction\n(circle size = number of samples)")
for _, row in df_stations.head(5).iterrows():
    ax.annotate(row['station'],
                (row['longitude'], row['latitude']),
                textcoords="offset points", xytext=(5, 5), fontsize=7)
plt.tight_layout()
plt.savefig("figures/intervention/station_sensitivity_map.png", dpi=150)
plt.close()
print("Saved figures/intervention/station_sensitivity_map.png")

# Figure 3: Reduction needed per station (bar chart, top 15)
top15 = df_thresholds.head(15)
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x <= 20 else 'orange' if x <= 35 else 'red'
          for x in top15['min_reduction_needed_pct']]
ax.barh(top15['station'], top15['min_reduction_needed_pct'], color=colors)
ax.axvline(x=20, linestyle='--', color='green', alpha=0.7, label='20% (achievable)')
ax.axvline(x=35, linestyle='--', color='orange', alpha=0.7, label='35% (challenging)')
ax.set_xlabel("Minimum Nitrogen Reduction Needed (%)")
ax.set_title("Nitrogen Reduction Required to Prevent >50% of High-Risk Bloom Days\nby Station")
ax.legend()
plt.tight_layout()
plt.savefig("figures/intervention/station_reduction_needed.png", dpi=150)
plt.close()
print("Saved figures/intervention/station_reduction_needed.png")

print("\nPrevention analysis complete.")
print("Outputs saved to data/ and figures/")