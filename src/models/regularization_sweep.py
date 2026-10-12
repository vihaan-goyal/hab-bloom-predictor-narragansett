r"""
regularization_sweep.py
------------------------
Two experiments on how the *recipe* shapes the weights (data still decides them):

  1. C sweep x {L2, L1}: for each regularization strength, refit and report
     validation AUC + the grouped chlorophyll coefficient share. Shows the
     tradeoff between weight concentration and performance.

  2. Ridge-trace figure: every feature's L2 coefficient plotted vs C on a log
     axis. Correlated chlorophyll features fan out and stabilize together;
     month/temperature stand alone. This is the figure to put in the paper.

Pipeline matches src/models/ablation_study.py so results line up with the lock.

Run from repo root:
    python src/models/regularization_sweep.py
Outputs:
    figures/model/regularization_path.png
"""

import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# -- Load + merge (mirrors ablation_study.py) ----------------------------------
hab = pd.read_csv('data/hab_features_tidal.csv')
daily = pd.read_csv('data/hab_features_daily.csv')[
    ['date', 'station_name', 'sal_lag2', 'sal_lag3', 'sal_lag4']]
hab['date'] = pd.to_datetime(hab['date'])
daily['date'] = pd.to_datetime(daily['date'])
for col in ['sal_lag2', 'sal_lag3', 'sal_lag4']:
    if col not in hab.columns:
        hab = hab.merge(daily[['date', 'station_name', col]],
                        on=['date', 'station_name'], how='left')


def load_percent_saturation():
    frames = []
    for f in sorted(glob.glob('data/raw/deep_wq_extra/deep_wq_S_*.csv')):
        frames.append(pd.read_csv(f, skiprows=[1],
                      usecols=['station_name', 'time', 'percent_saturation']))
    ps = pd.concat(frames, ignore_index=True)
    ps = ps[ps['station_name'].notna()].copy()
    ps['station_name'] = ps['station_name'].astype(str)
    ps['date'] = pd.to_datetime(ps['time'], utc=True).dt.tz_localize(None).dt.normalize()
    ps['percent_saturation'] = pd.to_numeric(ps['percent_saturation'], errors='coerce')
    return (ps.dropna(subset=['percent_saturation'])
              .groupby(['date', 'station_name'], as_index=False)['percent_saturation']
              .mean())


if 'percent_saturation' not in hab.columns:
    ps = load_percent_saturation()
    hab['station_name'] = hab['station_name'].astype(str)
    hab = hab.merge(ps, on=['date', 'station_name'], how='left')

df = hab

for n, min_p in [(3, 2), (6, 3), (9, 5), (14, 7), (21, 10)]:
    df[f'chl_roll{n}_mean'] = (df.groupby('station_name')['Chlorophyll']
        .transform(lambda x: x.rolling(n, min_periods=min_p).mean()))
df['chl_trend'] = (df.groupby('station_name')['Chlorophyll']
    .transform(lambda x: x.rolling(4, min_periods=3)
               .apply(lambda v: np.polyfit(range(len(v)), v, 1)[0])))

df['bloom_28d'] = 0
for station, grp in df.groupby('station_name'):
    dates, chl = grp['date'].values, grp['Chlorophyll'].values
    labels = np.zeros(len(grp), dtype=int)
    for i in range(len(grp)):
        mask = (dates > dates[i]) & (dates <= dates[i] + np.timedelta64(28, 'D'))
        if mask.any() and (chl[mask] > 10).any():
            labels[i] = 1
    df.loc[grp.index, 'bloom_28d'] = labels

BASE = [
    'Chlorophyll', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
    'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean', 'chl_trend',
    'chl_roll14_mean', 'chl_roll21_mean', 'chl_anomaly', 'chl_climatology',
    'do_lag1', 'temp_lag1', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4',
    'sea_water_temperature', 'sea_water_salinity',
    'oxygen_concentration_in_sea_water', 'month', 'latitude_x', 'longitude_x',
    'nox_lag2', 'dip_lag2', 'dip_change', 'dip_x_month',
    'neighbor_chl3_mean', 'neighbor_chl3_lag1', 'tidal_gt_anom',
    'tidal_msl_anom', 'percent_saturation',
]
BASE = [f for f in BASE if f in df.columns]

CHL_GROUP = ['Chlorophyll', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
             'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean', 'chl_trend',
             'chl_roll14_mean', 'chl_roll21_mean', 'chl_anomaly',
             'chl_climatology', 'neighbor_chl3_mean', 'neighbor_chl3_lag1']
CHL_IDX = [BASE.index(f) for f in CHL_GROUP if f in BASE]

# Features to trace individually in the path plot (the standalone/interesting ones).
TRACE = ['month', 'sea_water_temperature', 'chl_climatology', 'chl_roll3_mean',
         'chl_lag1', 'oxygen_concentration_in_sea_water', 'sea_water_salinity']

# -- Split + prep --------------------------------------------------------------
train = df[df['date'].dt.year <= 2019]
val = df[(df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)]


def prep(split):
    r = split[BASE + ['bloom_28d']].dropna(subset=['bloom_28d'])
    return r[BASE].copy(), r['bloom_28d'].copy()


X_tr, y_tr = prep(train)
X_va, y_va = prep(val)
MED = X_tr.median()
sc = StandardScaler().fit(X_tr.fillna(MED))
Xtr = sc.transform(X_tr.fillna(MED))
Xva = sc.transform(X_va.fillna(MED))

C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def fit(C, penalty):
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    lr = LogisticRegression(C=C, penalty=penalty, class_weight='balanced',
                            max_iter=5000, random_state=42, solver=solver)
    lr.fit(Xtr, y_tr)
    coef = lr.coef_[0]
    auc = roc_auc_score(y_va, lr.predict_proba(Xva)[:, 1])
    chl_share = 100 * np.abs(coef[CHL_IDX]).sum() / np.abs(coef).sum()
    n_nonzero = int((np.abs(coef) > 1e-6).sum())
    return coef, auc, chl_share, n_nonzero


# -- Experiment 1: sweep table -------------------------------------------------
print("=" * 70)
print("C SWEEP: validation AUC and grouped chlorophyll share")
print("  (locked model is C=0.05, L2)")
print("=" * 70)
print(f"{'penalty':>8}{'C':>9}{'val_AUC':>10}{'chl_share':>12}{'nonzero':>10}")
print("-" * 70)
l2_coefs = {}
for penalty in ['l2', 'l1']:
    for C in C_GRID:
        coef, auc, chl, nz = fit(C, penalty)
        if penalty == 'l2':
            l2_coefs[C] = coef
        star = '  <-- lock' if (penalty == 'l2' and C == 0.05) else ''
        print(f"{penalty:>8}{C:>9.3f}{auc:>10.4f}{chl:>11.1f}%{nz:>10}{star}")
    print("-" * 70)

# -- Experiment 2: ridge-trace figure ------------------------------------------
plt.figure(figsize=(8, 5))
Cs = np.array(C_GRID)
for f in BASE:
    idx = BASE.index(f)
    y = [l2_coefs[C][idx] for C in C_GRID]
    if f in TRACE:
        plt.plot(Cs, y, lw=2, label=f, zorder=3)
    else:
        color = '#c9a94f' if f in CHL_GROUP else '#cccccc'
        plt.plot(Cs, y, lw=0.8, color=color, alpha=0.6, zorder=1)

plt.xscale('log')
plt.axvline(0.05, color='k', ls='--', lw=0.8, alpha=0.5)
plt.text(0.052, plt.ylim()[1] * 0.92, 'locked C=0.05', fontsize=8, alpha=0.7)
plt.axhline(0, color='k', lw=0.5, alpha=0.4)
plt.xlabel('Regularization strength C (log scale; higher = less regularization)')
plt.ylabel('Standardized coefficient')
plt.title('L2 regularization path: chlorophyll features (gold) fan out together')
plt.legend(fontsize=7, loc='best', ncol=2)
plt.tight_layout()
plt.savefig('figures/model/regularization_path.png', dpi=150)
print("\nSaved figures/model/regularization_path.png")
print("Gold lines = chlorophyll group, gray = other features, bold = traced.")