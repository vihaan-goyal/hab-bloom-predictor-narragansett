import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, 
                             average_precision_score, confusion_matrix)
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv("data/hab_features_final.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'])

# Features to use
features = [
    'latitude', 'longitude', 'month',
    'chl_anomaly', 'chl_climatology',
    'chl_roll7_mean', 'chl_roll7_std',
    'chl_lag3', 'chl_lag7', 'chl_lag14', 'chl_lag21',
]

# Predict bloom 7 days ahead
df['bloom_7d_ahead'] = df.groupby('station_name')['bloom'].shift(-7)
target = 'bloom_7d_ahead'

# Spatiotemporal split - train on 1993-2019, validate 2020-2022, test 2023-2025
train = df[df['date'].dt.year <= 2019]
val = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)]
test = df[df['date'].dt.year >= 2023]

print("\nMissing value rates in val set:")
for f in features:
    missing = val[f].isna().mean() * 100
    if missing > 0:
        print(f"  {f}: {missing:.1f}% missing")

print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
print(f"Train bloom rate: {train[target].mean()*100:.1f}%")
print(f"Val bloom rate: {val[target].mean()*100:.1f}%")
print(f"Test bloom rate: {test[target].mean()*100:.1f}%")

def prepare(df, features, target):
    subset = df[features + [target]].dropna()
    X = subset[features]
    y = subset[target]
    return X, y

X_train, y_train = prepare(train, features, target)
X_val, y_val = prepare(val, features, target)
X_test, y_test = prepare(test, features, target)

print(f"\nAfter dropping NaN - Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n{'='*40}")
    print(f"{name}")
    print(f"{'='*40}")
    print(classification_report(y_true, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.3f}")
    print(f"Avg Precision: {average_precision_score(y_true, y_prob):.3f}")

# Logistic Regression baseline
print("\nTraining Logistic Regression...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
evaluate("Logistic Regression - Validation", 
         y_val, lr.predict(X_val_scaled), lr.predict_proba(X_val_scaled)[:,1])

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
evaluate("Random Forest - Validation",
         y_val, rf.predict(X_val), rf.predict_proba(X_val)[:,1])

# Feature importance
importances = pd.Series(rf.feature_importances_, index=features)
importances = importances.sort_values(ascending=True)

print("\nTraining XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    verbosity=0
)
xgb.fit(X_train, y_train)
evaluate("XGBoost - Validation",
         y_val, xgb.predict(X_val), xgb.predict_proba(X_val)[:,1])

xgb.save_model("data/xgb_model.json")
print("Model saved to data/xgb_model.json")

fig, ax = plt.subplots(figsize=(10, 8))
importances.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Random Forest Feature Importances')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('figures/model/feature_importances.png', dpi=150)
plt.show()
print("\nDone.")