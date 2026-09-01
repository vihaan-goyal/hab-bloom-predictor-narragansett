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
