# CLAUDE.md — HAB Bloom Predictor (Narragansett fork)

Fork of the Long Island Sound project retargeted at Narragansett Bay's
15-minute sonde network (RIDEM NBFSMN). The LIS pipeline and docs remain in
git history and in the parent repo `../hab-bloom-predictor`.

## Run instructions

Use the BASE conda env (`~/anaconda3/python.exe`), NOT the `hab` env — `hab`
has a broken LAPACK that crashes sklearn/np.linalg (exit 127). Run from repo
root.

```bash
# 1. Consolidate raw RIDEM corrected sonde files -> 15-min tidy CSV
python src/features/build_narragansett.py

# 2. Aggregate to station-days, build features + 7-day forward label
python src/features/build_narragansett_daily.py

# 3. Train (tier A/B x LR/GB), baselines, one test evaluation
python src/models/train_narragansett.py
```

## Data

| File | Contents |
|---|---|
| `data/raw/narragansett/nbfsmnYY.zip` | RIDEM annual archives, 2015–2023 (downloaded from datadem.ri.gov) |
| `data/narragansett_surface_15min.csv` | 2.35M readings, 15 stations, 2015–2023; station, datetime, temp_c, salinity_psu, do_pct, do_mgl, ph, chl_ugl |
| `data/narragansett_daily_features.csv` | 22,851 station-days (≥48 chl readings/day), lags/rolls/climatology + sonde-native features + `bloom_fwd` |
| `data/narragansett_model_results.csv` | Final metrics table (test 2023) |
| `data/narragansett_bloom_events.csv` | 380 bloom events 2021–23 with durations and ramp-up times (built in parent repo) |

All of `data/` is gitignored; regenerate with the three scripts above.
Known gaps: T-Wharf (F3) unparsed — it uses the NERRS SWMP export format,
not the RIDEM sheet layout. Sonde chl is fluorescence-derived, not extracted
chl-a. Most stations deploy May–Nov; B3w/B12w are the winter deployments.

## Conventions (inherited from LIS, unchanged)

- Label: any daily-mean chl > 10 µg/L within 7 days, right-censored → NaN
- LR spec: `LogisticRegression(C=0.05, class_weight='balanced')`, StandardScaler,
  train-median imputation
- Split: train ≤2020, val 2021–22, test 2023; threshold chosen on val only,
  exactly one test evaluation
- Report base rate and lift next to every precision number
- The honest task is **onset-only** (today ≤10): the all-days base rate is 0.55
  and persistence dominates there

## Headline numbers (test 2023)

Onset-only, GB, tier A: precision 0.718 @ POD 0.580, AUC 0.835, lift 2.07
(always-alert lift = 1.00; persistence cannot alert on onset days).
All-days GB AUC 0.909. Tier B (sonde-native features) does not beat tier A.

## LIS cross-reference

The parent repo's honest LIS numbers for the same recipe: AUC 0.875, precision
0.136 @ POD 0.875 (t*=0.35, h21). The fork exists to show the same recipe on
dense data: the precision jump (0.14 → 0.72) is the monitoring-cadence effect.

## Experiment scripts

One-off experiments that are not part of the main pipeline are archived in
`src/models/experiments/`. Do not import from them.
