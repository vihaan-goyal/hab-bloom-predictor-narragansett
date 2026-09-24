# Precision Ceiling Investigation: Results

> Mirror of the parent repo's file as of an earlier date. LIS numbers here are on the original sensor label; for current LIS numbers see the parent's notes/S1_NUMBERS_SHEET.md (label rebuilt 2026-09-23).

Canonical record of the powered, leakage-free investigation into the HAB precision
ceiling. Supersedes the original planning version of this document. Every hypothesis
below was tested under rolling-origin expanding-window cross-validation (11 folds,
test years 2015-2025, ~4,040 pooled out-of-sample rows), with clustered bootstrap
CIs (cluster = station-year, 2000 resamples). Paired comparisons resample identical
rows for both arms, which is far more sensitive than comparing two overlapping
marginal CIs.

## Locked configuration

- Model: Logistic Regression, C=0.05, class_weight='balanced', 35 features.
- Evaluation: rolling-origin CV, train <= T-2, test = T, pooled across 2015-2025.
- Primary horizon: 21 days (see Finding 4).
- Operating point: fixed threshold 0.60 (threshold strategy is not a lever, Finding 1).
- Headline metric: AUC (threshold-free, powered, stable). F1/precision reported as a
  secondary operating point only.

## Headline numbers

- Pooled AUC at 21 days: 0.852, 95% CI [0.823, 0.878].
- Pooled AUC at 28 days (prior default): 0.815, 95% CI [0.784, 0.844].
- Honest operating-point precision at t=0.60: **0.175 at h21**, 0.261 at h28 (NOT 0.500).
  *(Corrected 2026-08-29. This line previously read "~0.26" with no horizon, which
  conflicted with PRECISION_PUSH_TRACKER.md. Recomputed directly from
  `data/cv_pred_orig_h21.csv` / `_h28.csv` at t=0.60: h21 gives 0.175 over 416 alerts
  at a 0.039 base rate; h28 gives 0.261 over 318 alerts at 0.058. The old "~0.26" was
  the h28 figure carried into an h21 context. This document standardises on h21, so
  quote 0.175.)*
- High-confidence alert precision (top 5% of predictions): ~0.22 to 0.28.

---

## Finding 1: the original plateau was not real

Claim tested: precision plateaus at ~0.500 (a mechanistic ceiling).

Result: the single-split 0.500 was inflated by two artifacts.
- The threshold 0.60 was selected by maximizing F1 on the test set itself (17
  thresholds swept on test). That is selection on the reported quantity.
- It came from one favorable 2023-2025 block with only 74 positives. Wilson CI on
  that precision was already [0.387, 0.613].

Under powered CV, pooled precision is ~0.26 and the F1 CI is wide (clustered width
~0.26 on the single split, ~0.10 pooled). The 12 previously "rejected" experiments
moved metrics by 0.02 to 0.05, i.e. 3 to 6x smaller than the noise band. They were
never distinguishable from baseline.

Verdict: the plateau was sampling noise plus a test-tuned threshold, not a
mechanistic ceiling. No threshold strategy is a lever: fixed 0.60 F1=0.300,
walk-forward F1=0.250, per-year-val F1=0.275, all CIs overlapping.

## Finding 2: label confirmability is a real, validated limit (REAL GAIN)

Claim tested: a large share of bloom labels are single-sample noise that no model
can rank.

Result (fixed-model comparison, same rows, same probabilities, only the scored
label changes):
- AUC vs original (any-exceedance) label:   0.815
- AUC vs sustained-only label:              0.882
- Paired difference: +0.067, 95% CI [0.027, 0.110], positive in 100% of resamples.

The same ranking separates sustained blooms significantly better than all
exceedances. Single-sample exceedances are provably the ranking-degrading cases.

Confound identified: cleaning the label confounds bloom persistence with sampling
density. Median sampling gap is 16-22 days every year (fine), but the p90 gap is
huge in exactly the years that go 100% single-sample: 2020 (99d), 2021 (124d),
2023 (70d), 2025 (119d), vs 2019/2024 (35-37d). So the single-sample class is a
mixture of true transient spikes and real blooms whose confirmation sample was
missed.

Verdict: do NOT retarget the model to sustained-only (it confounds persistence with
sampling and deletes whole years). Report this as a label-sensitivity / data-quality
finding: single-sample exceedances impose a ceiling that better-resolved monitoring,
not better modeling, would lift. This is a policy-relevant water-management result.

## Finding 3: sampling cadence drives the worst years (data, not model)

The years with the lowest per-year AUC and the highest single-sample rates are the
sparse-tail years (large p90 gaps). The constraint is monitoring temporal
resolution at the affected stations, not the algorithm.

## Finding 4: 21-day horizon is the sweet spot (REAL GAIN)

Claim tested: the 28-day target mixes easy near-term and hard far-term cases.

Per-horizon pooled AUC / AUPRC-lift-over-base-rate:
- 7d:  degenerate (2 positives at this cadence; not measurable). Note in paper.
- 14d: AUC 0.836, AUPRC lift 6.70.
- 21d: AUC 0.852, AUPRC lift 5.72.
- 28d: AUC 0.815, AUPRC lift 4.53.

Paired 21d minus 28d AUC: +0.037, 95% CI [0.016, 0.059], positive in 100% of
resamples. Per-year confound check: well-sampled anchor years hold high AUC at 21d
(2019 = 0.870, 2024 = 0.871), so the gain is real signal, not a sparse-year artifact.

Verdict: lock 21 days as the primary horizon. Headline AUC rises from 0.815 to 0.852
purely by forecasting at the horizon where the signal lives. The 14-21 day range is
also the operationally useful window for aeration response.

## Finding 5: nonlinearity does not help (lever closed)

Claim tested: splines + interactions inside LR add signal trees overfit on.

Result (horizon 21): baseline AUC 0.852, augmented (spline bases on Chlorophyll /
chl_roll14 / chl_roll21 + two interactions) AUC 0.855. Paired difference +0.003,
95% CI [-0.014, +0.020], P(>0)=0.61.

Verdict: not distinguishable. The regularized linear model already captures the
signal. Combined with XGBoost underperforming (overfit), model complexity is
provably not the bottleneck. Keep plain LR.

## Finding 6: A4 regime feature does not help (lever closed)

Claim tested: a trailing CHL-to-bloom decoupling ratio detects the post-TMDL regime
at A4 without nutrient data. Feature embargoed: trailing window ends at t-horizon so
it cannot peek at the label's forward window.

Result (horizon 21):
- Network-wide paired difference: ~0.000 (null).
- A4-only: baseline AUC already 0.883, augmented 0.884, paired difference +0.001,
  95% CI [-0.002, +0.005] (also null at lookback 180/min-count 2).

Key insight: baseline AUC on A4 (0.883) already exceeds the network average (0.852).
A4 was never a ranking problem. The A4 false positives are a precision-at-threshold
artifact, not a missing ranking signal, which is why every A4-targeted feature
failed.

Verdict: data-driven regime detection at A4, with proper temporal embargo, does not
improve discrimination. Lever closed.

## Partial pooling (Section 4 of original plan): partially addressed

Threshold side: tested. Walk-forward and per-year thresholds both lost to a single
fixed 0.60 (Finding 1). Threshold instability under low base rates is real but a
fixed operating point handles it.

Model side: station fixed effects / regime-cluster intercepts in the LR were NOT
separately tested under powered CV. Deprioritized because the regime-feature and
spline nulls indicate station-level and nonlinear structure are not ranking levers.
This is the one remaining untested item if completeness is required; expected value
is low.

---

## Summary of the investigation

Two real gains, both from better evaluation, not a bigger model:
1. Label confirmability: sustained-bloom AUC 0.882 vs 0.815, difference CI excludes 0.
2. Horizon selection: 21-day AUC 0.852 vs 28-day 0.815, paired difference CI excludes 0.

Everything model-side returned null under powered testing: threshold strategies,
splines/interactions, the A4 regime feature; trees overfit. The binding constraints
are label confirmability and monitoring cadence, both data-collection limits.

The original "mechanistic nitrogen precision ceiling" is superseded. It remains true
that A4 has a post-TMDL false-positive cluster, but the broader, better-supported
ceiling is label confirmability under current sampling, and the model's ranking
(AUC) was never the limiting factor.

## Scripts (all in src/models/)

- rolling_origin_cv.py : powered walk-forward CV harness (--threshold-mode, --clean-labels)
- bootstrap_ci.py      : model-agnostic bootstrap CIs on saved predictions
- label_audit.py       : single-sample vs sustained audit + sampling cadence table
- label_auc_compare.py : fixed-model label AUC comparison (Finding 2)
- horizon_decomp.py    : per-horizon AUC/AUPRC + paired horizon diff + per-year (Finding 4)
- spline_test.py       : nonlinearity test (Finding 5)
- regime_test.py       : embargoed A4 decoupling-ratio test (Finding 6)
- label_utils.py       : shared label builder (single source of truth)