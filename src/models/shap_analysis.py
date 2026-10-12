import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

print("Loading data...")
df = pd.read_csv("data/hab_features_final.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])
df['bloom_7d_ahead'] = df.groupby('station_name')['bloom'].shift(-7)

features = [
    'latitude', 'longitude', 'month',
    'sea_water_temperature', 'sea_water_salinity',
    'oxygen_concentration_in_sea_water', 'pH',
    'chl_anomaly', 'chl_climatology',
    'chl_lag3', 'chl_lag7', 'chl_lag14', 'chl_lag21',
    'chl_roll7_mean', 'chl_roll7_std',
]

target = 'bloom_7d_ahead'

train = df[df['date'].dt.year <= 2019]
val = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)]

X_train = train[features].dropna()
y_train = train.loc[X_train.index, target].dropna()
X_train = X_train.loc[y_train.index]

X_val = val[features].dropna()
y_val = val.loc[X_val.index, target].dropna()
X_val = X_val.loc[y_val.index]

# Retrain XGBoost
print("Training XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_state=42,
    n_jobs=-1, eval_metric='logloss', verbosity=0
)
xgb.fit(X_train, y_train)
print(f"Val AUC: {roc_auc_score(y_val, xgb.predict_proba(X_val)[:,1]):.4f}")

# SHAP analysis
print("Computing SHAP values (this takes a few minutes)...")
sample = X_val.sample(5000, random_state=42)
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(sample)

display_names = {
    'oxygen_concentration_in_sea_water': 'dissolved_oxygen',
    'sea_water_temperature': 'temperature',
    'sea_water_salinity': 'salinity',
}
sample_display = sample.rename(columns=display_names)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, sample_display, show=False)
plt.tight_layout()
plt.savefig('figures/model/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_summary.png")

# Bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, sample_display, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('figures/model/shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_importance.png")

print("Done.")