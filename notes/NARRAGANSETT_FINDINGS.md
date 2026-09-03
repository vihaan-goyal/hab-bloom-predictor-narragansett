# Narragansett findings — replication of the LIS analyses

Written 2026-09-01. Source scripts: `src/models/narragansett_lis_analyses.py`,
`src/models/bootstrap_narragansett.py`, `src/models/train_narragansett.py`.
Data: 22,851 station-days (2015–2023, 15 stations, daily aggregates of
15-minute sondes). Label: daily-mean chl > 10 µg/L within 7 days.

Every analysis from the LIS chapter now has a Narragansett counterpart.
Two replicate, one diverges, one is newly possible.

## 1. DO conditioning replicates (cross-system result)

*Figure: figures/nar_fig1_do_temp_conditioning.png*

At elevated-but-subbloom chlorophyll (5–10 µg/L, n=7,047), the bloom
probability by dissolved-oxygen tercile:

| DO tercile | range (mg/L) | P(bloom ≤7 d) |
|---|---|---|
| low | 2.4–7.2 | **0.63** |
| mid | 7.2–8.1 | 0.45 |
| high | 8.1–14.5 | 0.43 |

Same direction as LIS (0.30 low vs 0.19 mid/high at its 21-d horizon).
Low DO marks bloom-prone water in both estuaries. Correlational, not causal:
low DO and blooms share causes (stratification, nutrient load, decay of
prior biomass).

## 2. Temperature does NOT replicate (honest divergence)

LIS: bloom rate nearly flat in temperature (22.4% below 15 °C vs 25.5%
above). Narragansett: strongly temperature-dependent — **0.276 below 15 °C
vs 0.611 above** (n=7,751 / 14,982); warm tercile 0.60 vs cold 0.38 within
the elevated-chl band. Consistent with the different phenologies: LIS bloom
frequency peaks Feb–Mar (cold-water diatoms); Narragansett onsets are
bimodal summer — June (77 of 419) and September (74).

Caveat: most NBFSMN stations deploy May–Nov; cold-season coverage rests on
the two winter stations (B3w, B12w), so the <15 °C sample is thinner and
station-biased.

## 3. The bloom run-up is ~3 days (newly measurable)

*Figure: figures/nar_fig2_epoch_composite.png*

Superposed-epoch composite over 419 clean onsets (≥5 quiet days before
first >10 day):

| day vs onset | chl (µg/L) | DO (mg/L) |
|---|---|---|
| −21 | 10.4 | 8.16 |
| −7 | 7.6 | 7.85 |
| −3 | **6.5** | 7.67 |
| 0 | **11.9** | 8.15 |
| +7 | 11.5 | 8.09 |

Chlorophyll sits ~6.5 three days out, then nearly doubles by onset day:
**the actual ramp is ~3 days**, with a small DO dip in the preceding week.
This is the mechanism behind the LIS precision ceiling stated as a
measurement: a 3-day ramp cannot be forecast by a 21-day sampling interval.
(Full curve: `data/narragansett_epoch_composite.csv`. The elevated chl at
day −21 reflects prior bloom waves — blooms recur in trains.)

## 4. No point of no return (replicates)

*Figure: figures/nar_fig3_escape_probability.png*

P(reach >10 µg/L within 7 d) as a function of today's daily-mean chl:

| chl bin | n | P |
|---|---|---|
| (0, 2] | 1,265 | 0.02 |
| (2, 4] | 4,642 | 0.07 |
| (4, 6] | 3,700 | 0.19 |
| (6, 8] | 2,903 | 0.43 |
| (8, 9] | 1,334 | 0.61 |
| (9, 10] | 1,166 | **0.71** |

A smooth monotone climb with no discontinuity — same conclusion as the LIS
box/ball/empirical PONR analysis: risk accumulates gradually; there is no
threshold past which a bloom is committed. Even at chl 9–10, 29% of days do
not proceed to bloom. Two systems, two sampling regimes, same shape.

## 5. Model results with clustered CIs (context)

*Figure: figures/nar_fig4_precision_comparison.png*

Station-clustered bootstrap (13 test clusters, n_boot=2000, seed=42),
test 2023:

- GB onset-only: precision 0.718 [0.643, 0.792], lift 2.07 [1.57, 2.69],
  POD 0.580 [0.444, 0.694]; **lift lower bound > 1.5** — decisively beats
  always-alert, which the LIS basin alert never did (its CI touched zero).
- GB all-days AUC 0.909 [0.880, 0.926].
- Tree model beats LR here (LIS: LR won) — dense data supplies the event
  count tree models need. Sonde-native features still add nothing over the
  LIS-analog set in either model.

## Presentation framing

Chapter 1 (LIS): sparse monitoring caps precision at 0.14; nothing fixes it.
Chapter 2 (Narragansett): same recipe, dense data → 0.72 onset precision;
the ramp the model exploits is 3 days long — shorter than one LIS revisit
gap. Replications (DO, no-PONR) show the biology transfers; the temperature
divergence shows the two bays are genuinely different systems, not copies.


## 6. Full-data update (2026-09-01, all 21 available years attempted)

Ingested every RIDEM archive 2003-2023 (2003-04 remain unparsed -- third
format, 34 files skipped): **4.52M readings, 18 stations, 2005-2023,
42,315 station-days**, bottom-sonde coverage 22.8%. Train grew 14.6k -> 34.1k
station-days; test 2023 unchanged. Results (GB, onset-only, clustered CIs):
precision 0.696 [0.622, 0.778], POD 0.600 [0.452, 0.719], lift 2.00
[1.50, 2.68], AUC 0.839 [0.780, 0.878] -- statistically identical to the
9-year build. **Doubling the training data did not move the model.**

Stratification (surface-minus-bottom density/temp/salinity + bottom DO,
tier C) adds nothing: removing it costs 0.4pp precision; tier C == tier A
within noise. pH removal actually *helps* (+1.3pp). The feature story is
now tested against everything the network measures: chlorophyll history,
salinity, and DO carry the signal; everything else is decoration.

Group weights (GB onset, precision lost on removal): chl history -3.2pp,
salinity -3.1pp, DO -2.3pp, month -1.7pp, current chl -1.4pp,
climatology -1.0pp, chl rate/accel -1.0pp, temperature -0.7pp,
stratification -0.4pp, pH +1.3pp (harmful). Full table:
`data/narragansett_ablation.csv`.


---

# Part II — Closing the gaps (2026-09-01, six parallel experiments)

Pre-registered in the plan file before any run. Scripts under
`src/models/experiments/`, `src/models/rolling_origin_cv_nar.py`,
`src/features/calibrate_sonde_chl.py`; parent repo:
`src/models/experiments/lis_buoy_recipe.py`.

## 7. Multi-year rolling-origin CV: the single-year number holds

Folds T=2015..2023, train <= T-2, val = T-1, test = T (LIS convention). Pooled
out-of-fold onset rows n=15,118 (3,964 positives, base 0.262), station-year
clustered bootstrap n=2000:

| model | onset precision | POD | lift | AUC |
|---|---|---|---|---|
| GB tier A | 0.656 [0.618, 0.692] | 0.653 | 2.50 [2.21, 2.83] | 0.878 |
| LR | 0.634 [0.593, 0.674] | 0.646 | 2.42 [2.17, 2.74] | 0.868 |

Per-year onset precision 0.55–0.77 (GB); lift 1.95–3.75, highest in the
rarest years (2019, 2020). The 2023 fold reproduces the single split
(0.724 vs 0.696, AUC 0.840 vs 0.839). Files: `data/rolling_origin_cv_nar*.csv`.

## 8. Trivial-rule baselines: the model wins, modestly

Onset task, rules tuned on val only. Single year 2023 (13 clusters): GB beats
chl>6, chl>7, roll3, rate, all-rows climatology and always-alert with CIs
excluding 0 — but an onset-day station×season lookup (lift 1.74 vs 2.00) was
not separable. Pooled over the 9 CV folds (116 clusters) it is:

| rule | pooled lift | GB − rule | 95% CI |
|---|---|---|---|
| GB tier A | 2.50 | — | — |
| chl > c (val POD≥0.6; c≈7) | 2.29 | +0.21 | [+0.10, +0.34] |
| onset station×DOY climatology | 2.14 | +0.36 | [+0.22, +0.50] |
| chl > c (val F1; c≈6) | 2.10 | +0.40 | [+0.27, +0.57] |

Honest framing: **the model reliably but modestly beats a one-line
chlorophyll threshold** (≈ +0.06 precision at matched recall). Files:
`data/onset_rule_baselines*.csv`.

## 9. Skill vs lead time: lift decays smoothly, no cliff

GB tier A, test 2023, onset-only. AUC and raw precision are misleading
across horizons because the base rate climbs from 0.08 (1 d) to 0.64 (21 d);
lift is the honest axis:

| horizon (d) | 1 | 2 | 3 | 5 | 7 | 10 | 14 | 21 |
|---|---|---|---|---|---|---|---|---|
| onset base rate | 0.08 | 0.14 | 0.19 | 0.27 | 0.35 | 0.44 | 0.53 | 0.64 |
| onset lift | 6.6 | 4.1 | 3.2 | 2.3 | 2.0 | 1.8 | 1.5 | 1.4 |

At 14–21 days the model is barely above climatology — the regime LIS
operates in. Figure `figures/nar_fig5_skill_vs_horizon.png`;
`data/horizon_sweep_nar.csv`.

## 10. Sonde fluorescence vs lab chlorophyll: the two "10 µg/L" differ

734 same-station same-day pairs, 2006–2022, 14 stations (lab grab samples
found in the RIDEM archives). Sonde ranks chlorophyll well (Spearman 0.75)
but reads high and noisy: lab ≈ 0.76 + 0.72·sonde (r² 0.23; log-log r² 0.55).
**Lab 10 µg/L ≈ sonde 12.8 [11.4, 15.0] (OLS) to ~16 (robust).** Station-days
above 10 (sonde): 34.9%; above the calibrated 12.8: 24.3%. The Narragansett
label at sonde-10 is therefore looser than LIS's lab-10, and even at the
calibrated threshold Narragansett exceeds ~5× more often than LIS (~5%).
Figure `figures/nar_fig6_calibration.png`; `data/sonde_lab_calibration*.csv`.

## 11. Cadence thinning within Narragansett: thesis FAILS its pre-registered test

Subsample each station to one observation every k days (5 phases), rebuild
lags as prior *samples*, relabel, retrain. Pre-registered criterion: onset
precision at k=21 < 0.30 confirms the cadence thesis; > 0.5 refutes it.

| k (days) | 1 | 3 | 7 | 14 | 21 |
|---|---|---|---|---|---|
| LIS-realistic label, h21: precision | 0.86 | 0.84 | 0.70 | 0.45 | **0.52** |
| same: AUC | 0.87 | 0.85 | 0.81 | 0.75 | 0.81 |
| same: base rate | 0.64 | 0.57 | 0.45 | 0.27 | 0.25 |

Thinning to LIS cadence degrades skill (AUC 0.87→0.81, precision 0.86→0.52)
but stops far above LIS's 0.14. **Cadence is real but secondary.** The
larger confound is event rarity: LIS's base rate is 0.046 vs 0.25–0.64 here.
(A matched-rarity rerun — re-thresholding Narragansett to a 5% base rate,
then thinning — is in §12.) `data/cadence_thinning.csv`.

## 12. LIS buoys at 15-minute cadence: still boat-level skill

UConn LISICOS ECO-FL fluorometers at WLIS and EXRX (2019–2026, night-only
de-quenched daily means, heavy QC: a dead-sensor year dropped, gain drifts
7× between years). Bloom defined in fluorescence space at the train-period
95th percentile (onset base rate 0.08 realized). Train ≤2022, val 2023–24,
test 2025–26 (78 onset positives):

| model | onset precision | lift | AUC |
|---|---|---|---|
| LR | 0.156 | 1.90 | 0.805 |
| HistGB | 0.180 | 2.18 | 0.748 |
| value > 58% of threshold (rule) | 0.354 | 4.30 | 0.791 |

Dense sampling inside LIS does **not** produce Narragansett-like precision;
it reproduces the boat network's (0.14 / 2.7×). Caveats are severe
(uncalibrated sensor, two buoys, two test years), but the direction agrees
with §11. Parent repo: `data/lis_buoy_recipe.csv`. The parent note
`PRECISION_PUSH_TRACKER.md` claimed "no buoy chlorophyll exists" — corrected.


## 13. The decisive test: Narragansett at LIS rarity

Re-threshold the Narragansett label so the train-period all-days positive
rate at h21 matches LIS (5%): that requires **chl > 52.5 µg/L** (10% → 39.0).
Same features, same models, daily (k=1) sampling, test 2023 onset-only:

| threshold | base rate (2023) | GB precision | GB lift | AUC | n_pos |
|---|---|---|---|---|---|
| 10 µg/L (original) | 0.64 | 0.86 | 1.3 | 0.87 | ~1,000 |
| 39.0 (10% rarity) | 0.073 | 0.48 | 6.6 | 0.85 | 177 |
| **52.5 (LIS 5% rarity)** | 0.009 | **0.139** | 16.3 | 0.97 | 21 |
| LIS boat network (reference) | 0.046 | 0.136 | 2.7 | 0.875 | 48 |

**With daily data, at LIS-level rarity, precision is 0.139 — identical to
LIS's 0.136.** Rarity alone moves precision 0.86 → 0.14. Thinning to 21 days
at matched rarity is not estimable (the 2023 test has < 1 expected positive),
so cadence cannot be isolated at that rarity; from §11, its effect at the
original threshold is 0.86 → 0.52.

Two honest notes. (1) Lift at matched rarity is far higher with daily data
(16× vs 2.7×) — dense sampling may buy ranking skill even where precision
stays flat — but with 21 positives that number is unstable and the 2023
base rate (0.009) undershoots the 0.05 target because 2023 was a quiet year.
(2) The threshold that reproduces LIS rarity in sonde units (52.5) is ~3–4×
the calibrated lab-10 equivalent (§10): Narragansett is a genuinely bloomier
system, not merely a miscalibrated one. `data/cadence_thinning_matched.csv`.

### Revised thesis, final form

**Precision is set by rarity.** The same recipe gives 0.14 precision in LIS
and 0.14 in Narragansett once the label is made equally rare. Everything
else — cadence (−0.3 at boat spacing), calibration (sonde ≈ 1.3–1.6× lab),
model class, feature engineering — is second-order. The quantity that
transfers between systems is lift over climatology, and a dense network
appears to raise lift substantially even when precision cannot move.


## 14. Pre-registered tuning search: nothing beats the reference at a fixed bloom definition

360 configurations (horizon 2–10 d × threshold 10/12.8/20 µg/L × tier A/B ×
12 model settings), scored on validation only, one test evaluation, 30-shuffle
permutation null. Fixed rule: maximise val onset lift subject to POD ≥ 0.6.

- The rule selected the **rarest** label in the grid (h=2, T=20, base 3–5%):
  val lift 9.8, test lift 7.8 [5.2, 14.3] but test **precision 0.405** and
  POD 0.53 — 100 false alarms per 68 hits. Real, not chance (100th percentile
  of the null) — and useless as an "improvement": lift = precision/base rate,
  so a lift-maximising search simply hunts rarity. This is the base-rate trap
  in its purest form and is the reason lift must always be quoted with its
  base rate.
- **At the fixed definition (h=7, T=10) the pre-registered grid's best is the
  existing reference model** (GB depth 3 / leaf 50, tier A). Tier B and
  hyperparameters change nothing material at any horizon.
- The useful output is the landscape: at fixed POD≈0.6, lead time trades
  against precision — h2 0.32 / h5 0.36 / h7 0.41 / h10 0.45 at T=20; at
  T=10, h7 precision 0.70 is the best precision-per-lead-time cell.
- A 2-day, >20 µg/L alert (AUC 0.93, precision ~0.40) is a legitimate
  *different product* (imminent-dense-bloom warning), not a tuned version of
  the 7-day one.

Files: `data/tuning_search_nar_{grid,selected,null}.csv`,
`src/models/experiments/tuning_search_nar.py`.


## 15. Why LIS is bloom-rare: the 2014 cliff

Share of station-days with chlorophyll > 10 µg/L (LIS: lab bottle samples;
Narragansett: sonde daily mean, with the lab-calibrated 12.8 equivalent):

| period | LIS | Narragansett (sonde >10) | Narragansett (>12.8) |
|---|---|---|---|
| 2005–2013 | 0.14–0.59 (mean ~0.38) | 0.27–0.43 | 0.18–0.31 |
| **2014** | **0.09** | 0.36 | 0.26 |
| 2015–2023 | **0.06** (0.03–0.11) | 0.34 | 0.23 |

Before 2014 LIS was as bloomy as Narragansett — in 2009–2013 bloomier
(0.42–0.59). It fell to 0.09 in 2014, the year the Clean Water Act nitrogen
TMDL for the Sound was met, and has stayed at 0.03–0.11 since. Narragansett,
with no comparable intervention, held flat throughout: it is effectively the
control for LIS's cleanup. Today's gap is 5.6× (raw) / 3.8× (calibrated).

This is the origin of the rarity that caps LIS precision (§13): the forecast
is hard because pollution control worked. Same 10 µg/L threshold, same
recipe — the event became rare. Script: inline comparison of
`data/hab_features_tidal.csv` (parent) and `data/narragansett_daily_features.csv`.


**Caveat added 2026-09-03 (nitrogen).** Narragansett Bay's own nitrogen load
fell by more than half after 2006 as Rhode Island treatment plants added
nitrogen removal (Wikipedia, citing Oviatt et al.), yet the sonde exceedance
rate in this dataset is flat at 0.27–0.43 from 2005 to 2023. A comparable
nitrogen cut next door produced no cliff in the same kind of data. That does
not disprove the TMDL reading of the LIS 2014 step, but it means the step is
either a much sharper ecological response or partly a measurement change. The
question is in the draft email to CT DEEP / UConn (parent
`notes/EMAIL_DRAFT_2014_CLIFF.md`).

## 16. Lift at LIS rarity, nine test years: dense sampling triples the lift

Confirms §13's single-year lead with pooled rolling-origin CV (2015–2023),
GB tier A, onset-only, t* per fold on val (POD ≥ 0.6), station-year clustered
bootstrap. Thresholds chosen to bracket LIS's 0.046 base rate at h21:

| threshold | h | base rate | precision | lift [95% CI] | top-decile lift | AUC | n_pos |
|---|---|---|---|---|---|---|---|
| 39 µg/L | 21 | 0.069 | 0.479 | **6.9 [5.3, 9.3]** | 6.5 | 0.909 | 1,541 |
| 52.5 µg/L | 21 | 0.034 | 0.289 | **8.5 [6.0, 12.8]** | 7.9 | 0.956 | 768 |
| LIS boat network | 21 | 0.046 | 0.136 | 2.7 | — | 0.875 | 48 |

The single-year 16× (§13) was the high tail; the nine-year value is **~7–8×,
lower CI bound 5.3–6.0, versus 2.7× for the LIS boat network**. The
threshold-free top-decile lift agrees, so it is not a t* artefact. Precision
at matched rarity is 0.29–0.48 here vs 0.14 in LIS (the single-year "0.139"
match in §13 was 2023-specific); with pooled years, daily sampling does raise
precision ~2–3× at LIS rarity as well.

**What this changes in the thesis:** rarity still sets the *ceiling* on
precision (no config at 3–7% base rate exceeds ~0.5), but dense sampling
buys a large gain in *ranking* — roughly 3× the lift — at any rarity. The
management translation: continuous sensors in LIS would not make alerts
"mostly right," but would send the sampling boat to the right station about
three times more efficiently than the current 21-day schedule.

Caveats: base rate varies 0.009–0.064 across test years; per-fold POD swings
because t* transfers poorly year to year; 768 positives at 52.5.
`data/lift_at_rarity_cv.csv`, `src/models/experiments/lift_at_rarity_cv.py`.


## 17. Model probability 21 days before each bloom, both bays

*Figure: figures/nar_fig8_prob_21d_before.png.* Out-of-sample: LIS locked LR
(h21, train ≤2019, onsets 2020–25); Narragansett GB tier A trained on a
21-day label (train ≤2020, onsets 2021–23). Null = non-bloom days at the
same stations/years.

| | LIS (nearest visit ~21 d before, n=91) | Narragansett (day −21, n=157) |
|---|---|---|
| median p before a bloom | **0.46** [0.30, 0.61] | **0.85** [0.53, 0.97] |
| median p, all null days | 0.20 | 0.08 |
| median p, station+month-matched null | 0.40 | 0.54 |
| onsets above t* | 68% (0.35) | 76% (0.50) |
| AUC onset-vs-null at −21 | 0.77 | 0.89 |

Narragansett approach curve (median p21 at −21/−14/−7/−3/−1): 0.85 / 0.86 /
0.83 / 0.81 / 0.89 for onsets, ~0.07 for null — **flat**. At a 21-day
horizon the model reports a persistent high-risk *state*, not an approaching
event; the approach signal is in the 7-day model (§ case study, fig 7).

Why the pre-bloom probabilities are not in the 90s: (1) a calibrated
probability equals the observed hit rate — in LIS a 21-day alert is right
~1 time in 7 at its threshold, so a well-calibrated model *should* sit near
0.3–0.6 before real blooms, not 0.95; a model saying 0.95 there would be
overconfident. (2) Class-balanced training pushes outputs toward 0.5 in the
ambiguous middle. (3) Most of the 21-day-ahead signal is seasonal: against
station+month-matched nulls the gap narrows to 0.46 vs 0.40 (LIS) and
0.85 vs 0.54 (Narragansett). (4) Irreducible weather: whether an elevated
day tips into a bloom depends on wind, light and rain that have not happened
yet. (5) Narragansett's 0.85 is inflated by 40% of onsets having a fading
prior bloom on day −21; open-bay F7 onsets score ~0.1.

Files: `data/prob_before_onset_nar.csv` (+ `_null`, `_trajectory`),
`src/models/experiments/prob_before_onset.py`, `fig_prob_21d_before.py`;
parent `src/models/experiments/prob_before_onset_lis.py`,
`data/prob_before_onset_lis.csv`.


### 17b. Are the high-probability days real? (calibration check, LIS)

Rolling-origin CV predictions, LIS, 2015–2025 (`parent data/cv_predictions.csv`):

| model said ≥ | days | bloomed within 21 d |
|---|---|---|
| 0.90 | 12 | 58% |
| 0.80 | 41 | 32% |
| 0.70 | 103 | 21% |
| 0.60 | 208 | 14% |

Plain answer to "did the 90%+ days actually bloom?": mostly no — 7 of 12
did (58%); far above the ~5% background, far below the printed 90%.

Ranking is monotonic (higher score → more blooms) but the model is
**overconfident at the top**: a "0.9" is empirically ~0.6. Cause: balanced
class weights inflate scores for a rare event; the isotonic/Platt fix was
tried and rejected (too few positives). The ten most confident LIS days are
all western-Narrows stations (A4, B3, 15) in February — the winter diatom
regime — and 7 of 10 bloomed. Report lift and ranking skill, not raw
probabilities, and say so.

## 18. Per-date scorer and two worked examples (2026-09-03)

`src/deploy/daily_inference_nar.py --date D` is the Narragansett analogue of
the LIS `daily_inference.py`: it trains HistGB tier A on every station-day whose
7-day label window closed on or before D (walk-forward, no leakage), scores each
station's latest day, alerts at the val-chosen t* = 0.50, and writes
`data/narragansett_daily_predictions.csv` with a `status` column
("blooming" = chl > 10 today, a persistence question; "onset risk" = the real
forecast). Six stations (B12w, B3W, B3a, B5, F3, UB2015) are winter or one-off
deployments and show as stale on summer dates.

**Pick the demo date carefully.** 2023-07-19 is a bay-wide bloom: 9 of 11 live
stations are already above 10 µg/L (B13 39.9, B4 38.6, B3 30.7) and the model
returns 0.99 for all of them, which is persistence, not forecasting.

**Best case, 2023-09-08: every alert was followed by a bloom.** Nine stations
were onset-risk (chl ≤ 10); the model alerted on six; all six exceeded 10 µg/L
within 7 days. Of the three cleared stations, B14 (p = 0.49, just under t*)
bloomed; B7 and F7 did not.

| Station | chl today | P(bloom ≤7 d) | Alert | Bloomed |
|---|---|---|---|---|
| B4 | 9.7 | 0.879 | yes | yes |
| B3 | 8.4 | 0.717 | yes | yes |
| B2 | 8.9 | 0.628 | yes | yes |
| B13 | 7.1 | 0.605 | yes | yes |
| B6 | 9.8 | 0.579 | yes | yes |
| B12 | 5.8 | 0.550 | yes | yes |
| B14 | 7.2 | 0.493 | no | yes (miss) |
| B7 | 4.5 | 0.174 | no | no |
| F7 | 3.6 | 0.028 | no | no |

Precision 6/6, POD 6/7. Caveat for the talk: the base rate that day was 7/9,
so the lift is only 1.3×; the model's real contribution is the ranking, which
put the two non-bloomers at the bottom. 2023-06-24 is similar (5 of 5 onset
alerts bloomed, B6 missed at 0.48).

**Representative case, 2023-06-05.** All 12 live stations below 10; six
alerts; four bloomed (F5, F4, B12, B3), two false alarms (B14, B2), one miss
(B4 at 0.30). Precision 0.67 against a base rate of 5/11 = 0.45, lift 1.5×, in
line with the 2.0× [1.5, 2.7] test-year estimate (§3). F4 alerting at chl 1.7
and blooming is the "history is everything" result in one row: the 21-day
history and station climatology carry the signal, not today's value.

## 19. Cross-site transfer: does the Narragansett model work elsewhere? (2026-09-03, running)

Harness: `src/transfer/transfer_eval.py`. For each new site: build the same
tier-A station-day features from that site's sub-daily sondes; label = bloom
within 7 d, two definitions (p75 = the station's own 75th percentile of daily
chl, the fair one because fluorometers are not inter-comparable; abs10 = 10
µg/L where units allow); onset-only rows (today ≤ threshold). Three tests:
**zero-shot raw** (Narragansett GB applied as-is), **zero-shot rescaled**
(target chl columns quantile-mapped onto the Narragansett chl distribution
first), **refit** (GB/LR retrained on the target, rolling-origin CV). Baselines:
always-alert, station-DOY climatology, chl>c rule with c chosen on the val
year. Station-year clustered bootstrap CIs. Fetch scripts:
`src/transfer/fetch_<site>.py`; data under `data/transfer/` (gitignored).

**Onset-only, p75 label. Lift = precision / base rate.**

| Site | Data | Zero-shot rescaled: prec / lift [CI] / AUC | chl>c rule: lift [CI] / AUC | Refit GB: lift [CI] / AUC |
|---|---|---|---|---|
| Western Lake Erie (NOAA GLERL, freshwater cyanobacteria) | 4 buoys, 2014–18, May–Oct, 15-min, 1,605 onset rows | 0.70 / **1.96 [1.56, 2.42]** / 0.67 | 1.45 [1.23, 1.70] / 0.63 | 1.07 [1.00, 1.22] / 0.55 (2018 fold only) |
| Celtic Sea shelf (Cefas SmartBuoy, UK) | 2 moorings, 2014–15, 30-min, 663 onset rows (below harness floor; pilot) | 0.45 / 2.8 [1.6, 5.9] / **0.88** | 6.4 [4.3, 9.8] / 0.73 (2015 only, 4 clusters) | not testable (2 years) |

Readings so far:

1. **Raw zero-shot fails everywhere; rescaled zero-shot works.** Applied to
   raw values the Narragansett model never alerts (Lake Erie median chl 3,
   Celtic Sea ~1 a.u., vs Narragansett ~8 µg/L). After quantile-mapping the
   chlorophyll scale it ranks bloom risk at AUC 0.67 (lake) and 0.88 (shelf
   sea) with no retraining. What transfers is the *shape* of the chlorophyll
   trajectory before a bloom, not the units.
2. **Lake Erie: the transferred model beats the best simple rule** with
   non-overlapping CIs (1.96 vs 1.45). The refit is weaker because only 2018
   qualifies as a test year.
3. **Celtic Sea: the transferred model ranks well but a threshold rule is more
   precise** (0.86 vs 0.45), because that regime has one large spring bloom a
   year and persistence is strong. Two stations, two years, so a pilot only;
   the multi-year Liverpool Bay / Gabbard archive needs a free WaveNet login.
4. Caveats: Lake Erie salinity is a constant 0 (five dead features), buoys are
   summer-only, and fluorescence is total phytoplankton, not the toxin-forming
   cyanobacteria. Cefas chlorophyll is bead-standardised fluorescence, never
   calibrated to extracted chl-a, so abs10 is void there.

(Chesapeake Bay, NERRS reserves, IMOS Australia, San Francisco Bay: pending.)

## Revised thesis (supersedes the "Presentation framing" above)

1. LIS forecasting is capped near precision 0.14 and 13 fixes failed (Ch. 1).
2. The same recipe on Narragansett sondes reaches 0.66 [0.62, 0.69] onset
   precision across nine test years and beats every trivial rule (§7–8).
3. **Why the gap? Mostly rarity, partly cadence, partly calibration.**
   Blooms (at any comparable threshold) are ~5× more frequent in Narragansett
   (§10); thinning Narragansett to boat cadence costs ~0.3 precision but not
   0.7 (§11); LIS buoys sampled every 15 min stay at boat-level skill (§12).
   Precision is a base-rate quantity; lift is the fair comparison, and on
   lift the two bays are within a factor of ~1.5 of each other (2.0–2.5 vs
   2.7–3.0).
4. What transfers across bays: DO conditioning, the smooth no-PONR risk
   curve, and the 3-day run-up; what does not: temperature dependence.
5. What a manager should take away: in a bloom-rare system like LIS, no
   sampling cadence or model class produces high-precision alerts, because
   precision is bounded by rarity; the actionable quantity is lift over
   climatology, which is ~2–3× in both bays.
