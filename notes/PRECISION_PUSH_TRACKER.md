# Precision Push Tracker  (refreshed)

> Mirror of the parent repo's file as of an earlier date. LIS numbers here are on the original sensor label; for current LIS numbers see the parent's notes/S1_NUMBERS_SHEET.md (label rebuilt 2026-09-23).

Status key: DONE / DEAD / OVERTAKEN / OPEN

Powered backbone for all inference: rolling-origin CV, 156 pooled positives (h21)
or 235 (h28), 11 folds. Files: data/cv_pred_orig_h21.csv, data/cv_pred_orig_h28.csv.

============================================================
HEADLINE CONCLUSION (what this whole thread established)
============================================================
The precision ceiling is REAL but its cause is monitoring cadence + base rate,
NOT a post-TMDL nitrogen mechanism. Across every test, things that improve AUC do
NOT improve precision; precision tracks base rate. That AUC-vs-precision split is
the quantitative signature of a cadence/base-rate-limited problem and is the
paper's real scientific finding. Confirmed THREE independent ways: longer horizon,
original-vs-sustained label, and label leniency all show more-positives -> higher
raw precision -> LOWER skill (lift). The skillful target is the rare sustained bloom.

Key facts, all verified and reproducible from current files:
- Honest precision at t=0.60: 0.175 (h21) / 0.261 (h28). The old 0.500 headline was
  test-set threshold-selection inflation. DEAD.
- PR-AUC: 0.221 (h21), 0.264 (h28). Clean monotone PR tradeoff, no plateau.
- 99% of 21-day forward windows hold <=1 chlorophyll reading (48% hold zero). The
  "ground truth" is itself mostly unobserved.
- No buoy chlorophyll exists for LIS (NERACOOS/UConn buoys measure temp/sal/DO/
  turbidity only). The ~21-day ship survey is the ONLY in-situ chl record. This
  makes cadence a fundamental observing-system limit, and turns the future sensor
  device into a documented gap-filler.
- A4/B3 mechanistic attribution does NOT hold: errors are Sound-wide, and A4/B3 are
  simply the sparsest-sampled stations (20-21d gaps vs 15d network median).

============================================================
DONE
============================================================
- #1 bootstrap CI + PR curve: plateau is noise; base-rate-limited; monotone PR.
- #3 move up PR curve: high-precision mode is t~0.80 (h21) or top-5% slice.
- #2 FP label audit: ran. RESULT NOT AS EXPECTED. Of 343 FPs at t=0.60:
  near-miss 17%, unobserved 17%, "genuine_low" 66% BUT 100% of genuine_low rest on
  a single reading -> unverifiable, not confirmed model error. Optimistic ceiling
  if near-misses relabeled: precision 0.175 -> 0.315.
- VERIFY (21d vs 28d): paired AUC +0.037, CI [0.016, 0.059]. Holds. Per-year check
  clears the sparse-year confound (2019=0.870, 2024=0.871). Reproducible.
- VERIFY (sustained vs original): paired AUC +0.059 h28 [0.017,0.105], +0.055 h21
  [0.024,0.089]. Holds on AUC. Script: src/models/label_compare.py.

============================================================
DEAD (tested and rejected, or ruled out by findings)
============================================================
- #4 selective prediction / station abstention: NO gain, CI [-0.024, +0.037]
  straddles zero. Station reliability not stable across years. DEAD.
- #5 USGS river discharge: a daily covariate cannot fix an unobserved window;
  improves ranking at best, and discharge lags already logged null. Ruled out.
- #6 phycocyanin / buoy chlorophyll: confirmed NO LIS buoy chl exists. DEAD as a
  data source, but became a paper finding (the observational gap).
- #7 STRATIFICATION feature: DEAD, and HURTS. Was buildable (11,233 station-dates,
  98% coverage, surface-bottom temp+salinity from hab_labels_final.csv). Adding
  strat_temp/strat_dens/depth_span to the C=0.05 LR: AUC 0.852->0.838, paired diff
  -0.014 [-0.021,-0.007] P(>0)=0.000; lift 5.72->4.86. Real physical mechanism but
  no marginal signal over the chlorophyll features; spends regularization on noise.
  Drop it. Last physically-motivated feature on the list.
- LABEL REFINEMENT (near-miss FPs): DEAD, and worse than null. Paired lift bootstrap:
  grace dlift=-1.60 [-2.46,-0.90], tol -2.25 [-3.57,-1.21], both -2.91 [-4.28,-1.74],
  all P(>0)=0.000. The audit's 0.315 was pure base-rate illusion: loosening the label
  raises raw precision (0.175->0.380) while lift FALLS (5.72->2.91). Lenient labels
  manufacture less-predictable positives. Do NOT report a refinement gain.
  This is now the THIRD independent confirmation (horizon, sustained, leniency) that
  precision is base-rate-governed and the skillful target is the rare sustained bloom.

============================================================
OPEN - genuinely live (not yet run)
============================================================
LOW VALUE / EXHAUSTIVE-ONLY
- #11 TabPFN v2, #12 EBM: cheap bake-off vs locked LR, ~an afternoon. Low odds of
  beating LR (bottleneck is data/cadence, not model capacity) but quick.
- #13 LightGBM + monotonic constraints, #14 temporal importance weighting: medium
  effort, low odds.
- #8 PAR/photoperiod, #9 two-stage model, #10 decoupling feature: low value now
  given the cadence finding; would mostly chase AUC, not precision.

============================================================
DEAD / DO NOT REVISIT (prior)
============================================================
More nutrient concentration data (oracle +1.5pp). XGBoost (-33pp). Satellite (4km
too coarse). Deep tabular nets. More RNNs. SVM. Plain stacking. The 12 prior
rejected feature experiments.

============================================================
PIPELINE HYGIENE FIXED THIS THREAD
============================================================
- --clean-labels no-op was a STALE-FILE artifact, not a code bug. Label logic sound.
- Added --horizon flag to rolling_origin_cv.py.
- build_dataset strips baked bloom_28d/is_sustained/is_exceedance before labeling;
  stripped baked bloom_28d from hab_features_tidal.csv.
- New scripts (in outputs): precision_bootstrap_ci.py, fp_audit.py,
  selective_prediction.py, cadence_check.py, lisicos_buoy_pull.py, label_compare.py.

============================================================
WRITING TODO (deferred, but flagged)
============================================================
- Rewrite PRECISION_CEILING_INVESTIGATION.md + A4/B3 attribution to match findings
  (cadence/base-rate, not nitrogen).
- Reframe headline: 21d = ranking-optimal horizon; top-5% slice / 28d = precision
  product. Do NOT claim 21d wins on everything; it loses to 28d on precision.
- Sustained label = label-quality/ranking improvement, NOT a precision gain (state
  explicitly; AUPRC is a tie).
- Cadence limit + no-buoy-chl as the headline limitation; sensor device as the fix.