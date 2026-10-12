import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/hab_features_daily.csv")
df['date'] = pd.to_datetime(df['date'])

# Recompute rolling features and bloom label for consistency
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
    idx = grp.index
    dates = grp['date'].values
    chl = grp['Chlorophyll'].values
    labels = np.zeros(len(grp), dtype=int)
    for i in range(len(grp)):
        mask = (dates > dates[i]) & (dates <= dates[i] + np.timedelta64(28, 'D'))
        if mask.any() and (chl[mask] > 10).any():
            labels[i] = 1
    df.loc[idx, 'bloom_28d'] = labels

# ── Features ──────────────────────────────────────────────────────────────────
features = [
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
features = [f for f in features if f in df.columns]

feature_labels = {
    'Chlorophyll':                       'CHL (current)',
    'chl_lag1':                          'CHL lag ~14d',
    'chl_lag2':                          'CHL lag ~28d',
    'chl_lag3':                          'CHL lag ~42d',
    'chl_lag4':                          'CHL lag ~56d',
    'chl_roll3_mean':                    'CHL rolling mean (6wk)',
    'chl_roll6_mean':                    'CHL rolling mean (3mo)',
    'chl_roll9_mean':                    'CHL rolling mean (4.5mo)',
    'chl_trend':                         'CHL trend slope',
    'chl_anomaly':                       'CHL anomaly',
    'chl_climatology':                   'CHL climatology',
    'do_lag1':                           'DO lag ~14d',
    'temp_lag1':                         'Temp lag ~14d',
    'sal_lag1':                          'Salinity lag ~14d',
    'sea_water_temperature':             'Temperature',
    'sea_water_salinity':                'Salinity',
    'oxygen_concentration_in_sea_water': 'Dissolved oxygen',
    'month':                             'Month',
    'latitude_x':                        'Latitude',
    'longitude_x':                       'Longitude',
    'nox_lag2':                          'NOX lag ~28d',
    'dip_lag2':                          'DIP lag ~28d',
    'dip_change':                        'DIP change',
    'dip_x_month':                       'DIP x month',
    'neighbor_chl3_mean':                'Neighbor CHL (3 nearest)',
    'neighbor_chl3_lag1':                'Neighbor CHL lag ~14d',
}

# ── Splits ─────────────────────────────────────────────────────────────────────
train = df[df['date'].dt.year <= 2019]
val   = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)]
test  = df[df['date'].dt.year >= 2023]

def get_xy(split):
    clean = split[features + ['bloom_28d']].dropna(subset=['bloom_28d'])
    return clean[features], clean['bloom_28d']

X_train, y_train = get_xy(train)
X_val,   y_val   = get_xy(val)
X_test,  y_test  = get_xy(test)

med = X_train.median()

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Bloom rates: train={y_train.mean():.1%} val={y_val.mean():.1%} test={y_test.mean():.1%}")

# ── XGBoost ────────────────────────────────────────────────────────────────────
print("\nTraining XGBoost (n=200, depth=6, lr=0.1)...")
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=pos_weight, eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42, verbosity=0,
)
xgb_model.fit(
    X_train.fillna(med), y_train,
    eval_set=[(X_val.fillna(med), y_val)],
    verbose=False,
)

xgb_val_auc  = roc_auc_score(y_val,  xgb_model.predict_proba(X_val.fillna(med))[:,1])
xgb_test_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test.fillna(med))[:,1])
print(f"XGBoost Val AUC:  {xgb_val_auc:.3f}")
print(f"XGBoost Test AUC: {xgb_test_auc:.3f}")

# ── Random Forest ─────────────────────────────────────────────────────────────
print("\nTraining Random Forest (n=200, balanced)...")
rf_model = RandomForestClassifier(
    n_estimators=200, class_weight='balanced',
    random_state=42, n_jobs=-1,
)
rf_model.fit(X_train.fillna(med), y_train)

rf_val_auc  = roc_auc_score(y_val,  rf_model.predict_proba(X_val.fillna(med))[:,1])
rf_test_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test.fillna(med))[:,1])
print(f"Random Forest Val AUC:  {rf_val_auc:.3f}")
print(f"Random Forest Test AUC: {rf_test_auc:.3f}")

# ── Figure 7: RF Feature Importances ─────────────────────────────────────────
print("\nGenerating fig7_rf_importances.png...")
importances = pd.Series(rf_model.feature_importances_, index=features)
importances.index = [feature_labels.get(f, f) for f in importances.index]
importances = importances.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(importances.index, importances.values, color='#2a6fa8', alpha=0.82)
ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=11)
ax.set_title(
    "Random Forest Feature Importances — Top 15\n"
    "(corrected model, 28-day forecast horizon)",
    fontsize=12,
)
ax.grid(True, alpha=0.3, axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("figures/model/fig7_rf_importances.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/model/fig7_rf_importances.png")

# ── Figure 8: SHAP Beeswarm (XGBoost) ────────────────────────────────────────
print("\nComputing SHAP values for XGBoost...")
explainer = shap.TreeExplainer(xgb_model)
X_shap = X_val.fillna(med)
shap_values = explainer.shap_values(X_shap)

X_display = X_shap.rename(columns=feature_labels)

print("Generating fig8_shap_beeswarm.png...")
shap.summary_plot(
    shap_values, X_display,
    plot_type="dot",
    max_display=15,
    show=False,
    color_bar_label="Feature value",
)
plt.title(
    "SHAP Feature Importance — XGBoost Bloom Predictor\n"
    "(corrected model, 28-day forecast horizon, val set 2020-2022)",
    fontsize=12, pad=15,
)
plt.xlabel("SHAP value (impact on log-odds of bloom)", fontsize=10)
plt.tight_layout()
plt.savefig("figures/model/fig8_shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/model/fig8_shap_beeswarm.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"XGBoost  Val AUC: {xgb_val_auc:.3f}  Test AUC: {xgb_test_auc:.3f}")
print(f"RF       Val AUC: {rf_val_auc:.3f}  Test AUC: {rf_test_auc:.3f}")

print("\nTop 10 SHAP values (mean |SHAP|):")
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=[feature_labels.get(f, f) for f in features],
).sort_values(ascending=False)
print(mean_shap.head(10).to_string())

print("\nTop 10 RF importances:")
print(importances.tail(10).iloc[::-1].to_string())
