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
| `data/narragansett_surface_15min.csv` | 4.52M readings, 18 stations, 2005–2023; station, datetime, temp_c, salinity_psu, do_pct, do_mgl, ph, chl_ugl |
| `data/narragansett_daily_features.csv` | 42,315 station-days (≥48 chl readings/day), lags/rolls/climatology + sonde-native features + `bloom_fwd` |
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
0.136 @ POD 0.875 (t*=0.35, h21). The fork tests why the same recipe gives
0.14 there and ~0.66 here. Answer (Part II): mostly event rarity, partly
cadence, partly sonde calibration — NOT a pure cadence effect.

## Part II scripts (2026-09-01)

| Script | Output |
|---|---|
| `src/models/rolling_origin_cv_nar.py` | `data/rolling_origin_cv_nar*.csv` — 9-fold CV, pooled CIs |
| `src/models/experiments/onset_rule_baselines.py`, `_cv.py` | rule-vs-model paired bootstrap |
| `src/models/experiments/horizon_sweep_nar.py` | `data/horizon_sweep_nar.csv`, fig5 |
| `src/models/experiments/cadence_thinning.py` | `data/cadence_thinning.csv` (thesis test) |
| `src/features/calibrate_sonde_chl.py` | `data/sonde_lab_calibration*.csv`, fig6 |
| parent `src/models/experiments/lis_buoy_recipe.py` | parent `data/lis_buoy_recipe.csv` |
| `src/transfer/transfer_eval.py`, `fetch_<site>.py`, `pooled_model_test.py`, `regime_models.py` | `data/transfer/*` — cross-site transfer (findings §19–22) |
| `predict_anywhere.py` + `release/narragansett_bloom_model.joblib` | frozen model for any site (findings §21) |
| `src/deploy/daily_inference_nar.py --date YYYY-MM-DD` | `data/narragansett_daily_predictions.csv` — per-date station probabilities (findings §18) |

Headline after Part II: pooled onset precision 0.656 [0.618, 0.692], lift 2.50.
Cadence thesis: FAILED its pre-registered test; rarity dominates (findings §11–12).

## Experiment scripts

One-off experiments that are not part of the main pipeline are archived in
`src/models/experiments/`. Do not import from them.
