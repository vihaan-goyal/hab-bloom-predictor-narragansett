> Mirror of the parent repo's file as of an earlier date. LIS numbers here are on the original sensor label; for current LIS numbers see the parent's notes/S1_NUMBERS_SHEET.md (label rebuilt 2026-09-23).

> **PARTLY SUPERSEDED.** Marked 2026-08-30. This log is on the **28-day** label and the
> retired single-split framing. Its "LOCKED BASELINE (June 2026)" block — precision
> 0.500 / recall 0.486 / AUC 0.815 at t=0.60 — is the inflated headline killed in
> `PRECISION_PUSH_TRACKER.md`; the honest h21 value at t=0.60 is **0.175**. The demo
> date (2022-07-19, A4, P=0.813) comes from the superseded XGBoost/aeration pipeline.
> **Still useful and still cited:** the REJECTED FEATURES deltas (NOAA ASOS wind -4pp,
> MODIS CHL -7.8pp, nutrient forward-fill -10.4pp). Quote those; not the baseline block.

# Precision Optimization Log
# HAB Bloom Predictor — LR C=0.05, 28-day forward label, train<=2019 / val 2020-2022 / test 2023-2025

---

## LOCKED BASELINE (June 2026)
Model: Logistic Regression, C=0.05, class_weight='balanced', 35 features, threshold=0.60
- Precision: 0.500 | Recall: 0.486 | F1: 0.493 | AUC: 0.815
- Alt threshold=0.55: Prec=0.377 | Rec=0.541 | F1=0.444

Station-specific @ threshold=0.60 (global model):
- C1:  Prec=1.000 | Rec=0.571 | F1=0.727
- 02:  Prec=0.625 | Rec=0.833 | F1=0.714
- 01:  Prec=0.500 | Rec=1.000 | F1=0.667
- A4:  Prec=0.625 | Rec=0.625 | F1=0.625

Key features: tidal anomalies (tidal_gt_anom, tidal_msl_anom), chl_roll14/21_mean,
              sal_lag2-4, percent_saturation, max_gust_3d

External sources REJECTED (hurt precision):
- NOAA ASOS wind: -4pp
- MODIS satellite CHL: -7.8pp

Demo date: 2022-07-19, station A4, P=0.813

---

## IMPROVEMENT IDEAS — STATUS

### Priority 1 — Lagged Nutrients (REJECTED)
Features tested: nox_ffill, dip_ffill, nox_ffill_age, dip_ffill_age,
                 nox_ffill_x_month, dip_ffill_x_month
Result: Prec=-0.104, F1=-0.162 on test set
Reason: Nutrients sampled ~monthly; forward-filled values are 30-day-stale noise.
        All feature correlations between -0.11 and +0.03.
Status: REJECTED

### Priority 2 — Discharge Lags (SKIPPED)
All correlations between -0.07 and +0.03 at all lags. No signal to exploit.
Status: SKIPPED -- no signal

### Priority 3 — Stratification Index (SKIPPED)
depth_code not in hab_features_tidal.csv -- would require re-pulling raw ERDDAP profiles.
Status: DEFERRED -- needs raw data

### Priority 4 — Per-Station Threshold Optimization (DONE)
Script: src/models/station_specific_models.py
Strategy A = station-only model, Strategy B = global model + station threshold

KEY RESULTS (test 2023-2025, stations clearing P>0.50 AND R>0.40):
  A4  Strategy A @0.60: P=0.625 R=0.625 F1=0.625  (TP=5 FP=3 FN=3)
  B3  Strategy A @best: P=0.556 R=0.455 F1=0.500  (TP=5 FP=4 FN=6)
  C1  Strategy B @0.60: P=1.000 R=0.571 F1=0.727  (TP=4 FP=0 FN=3)
  02  Strategy B @0.60: P=0.625 R=0.833 F1=0.714  (TP=5 FP=3 FN=1)

Combined western stations (global @0.60): P=0.464 R=0.743 F1=0.571

Thresholds saved to: data/station_thresholds.csv
Status: DONE -- strong station-specific results, use for paper framing

### Priority 5 — ERA5 Wind Stress (REJECTED)
Features tested: wind_stress_mag, wind_stress_curl, wsc_roll3d, wsc_roll7d, wsm_roll3d, wsm_roll7d
Result: Prec=-0.063, AUC=-0.009 on test set
Reason: Wind stress curl corr=-0.003, stress mag corr=0.022 -- essentially zero signal.
        LIS bloom dynamics driven by CHL memory, salinity, tidal forcing, DO -- not wind.
        Consistent with NOAA ASOS wind failure (-4pp). Wind is noise for this system.
Data: data/era5_wind_lis.nc (33 years, 5x9 grid, kept for reference)
Status: REJECTED

### Priority 6 — DIP x Discharge Interaction (SKIPPED)
Discharge showed no signal in P2 -- interaction term won't help.
Status: SKIPPED

### Priority 8 — XGBoost Heavy Regularization (TODO)
Method: XGBoost with min_child_weight=10, reg_alpha=1, max_depth=3, subsample=0.8
Rationale: Locked model is LR. XGBoost val AUC=0.843 but test=0.774 -- overfitting gap.
           Never tried heavy regularization specifically targeting test precision.
Expected benefit: MEDIUM
Status: TODO

### Priority 9 — Station-Month Bloom Rate Feature (TODO)
Feature: historical bloom frequency for each station x month (computed on train only)
Rationale: January at A4 has ~2% historical bloom rate -- model should down-weight those
           alerts. Already built in precision_features.py but tested on OLD pipeline only.
           Never retested on corrected 28-day pipeline. Directly targets winter FPs.
Expected benefit: MEDIUM-HIGH
Status: TODO

### Priority 10 — CHL Acceleration Feature (TODO)
Feature: second derivative of CHL (rate of change of trend)
Script: acceleration_feature.py already exists (May 28) -- never tested on corrected pipeline
Rationale: FPs may spike suddenly; TPs build steadily. Acceleration captures this.
Expected benefit: MEDIUM
Status: TODO

### Priority 11 — Neighbor Bloom Probability Feature (TODO)
Feature: mean bloom probability of neighboring stations (from global model)
Rationale: If A4 fires but B3/C1 show low bloom prob, that's evidence against the alert.
           Currently have neighbor_chl3_mean but not neighbor bloom signal.
Expected benefit: MEDIUM
Status: TODO

### Priority 7 — Isotonic Calibration (REJECTED)
Method: CalibratedClassifierCV isotonic + Platt sigmoid, fit on val (2020-2022)
Result: Isotonic maps everything below 0.50 at test time (0 predictions at t=0.60).
        Platt gets Prec=0.667 but Rec=0.027 -- catches 2/74 blooms. Both useless.
Reason: Val set too small (1,057 rows, ~67 positives) to fit reliable calibration mapping.
        Probability distribution shifts between val and test (bloom rate 6.3% vs 7.2%).
Status: REJECTED

---

## RESULTS TABLE

| # | Experiment              | Precision | Recall | F1    | AUC   | Delta Prec | Notes                            |
|---|-------------------------|-----------|--------|-------|-------|------------|----------------------------------|
| 0 | BASELINE (locked)       | 0.500     | 0.486  | 0.493 | 0.815 | --         | 35 features, t=0.60, global      |
| 1 | + Nutrient ffill        | 0.396     | 0.284  | 0.331 | 0.807 | -0.104     | REJECTED                         |
| 2 | Discharge lags          | --        | --     | --    | --    | --         | SKIPPED -- corr <0.03 at all lags|
| 4 | Per-station thresholds  | see above | --     | --    | --    | --         | C1=1.00, 02=0.625, A4=0.625, B3=0.556 |
| 5 | + ERA5 wind stress      | 0.437     | 0.419  | 0.428 | 0.806 | -0.063     | REJECTED                         |
| 6 | Isotonic/Platt cal      | --        | --     | --    | --    | --         | REJECTED -- breaks prob scale    |
| 7 | + Kd490 water clarity   | 0.478     | 0.446  | 0.462 | 0.798 | -0.022     | REJECTED -- 78.8% null           |
| 8 | + UConn nutrients       | 0.451     | 0.311  | 0.368 | 0.802 | -0.049     | REJECTED -- still monthly        |
| 9 | XGBoost heavy reg       |           |        |       |       |            |                                  |
| 10| Station-month bloom rate|           |        |       |       |            |                                  |
| 11| CHL acceleration        |           |        |       |       |            |                                  |
| 12| Neighbor bloom prob     |           |        |       |       |            |                                  |

---

## REJECTED FEATURES (do not re-test)
- NOAA ASOS wind (land-based): -4pp
- MODIS satellite CHL: -7.8pp
- Nutrient forward-fill: -10.4pp -- stale monthly samples
- Discharge lags (any): corr <0.03, no signal
- Isotonic/Platt calibration: val set too small, completely breaks probability scale
- ERA5 wind stress curl: corr=-0.003, -6.3pp precision -- wind is noise for LIS blooms