# Narragansett findings — replication of the LIS analyses

Written 2026-09-01. Source scripts: `src/models/narragansett_lis_analyses.py`,
`src/models/bootstrap_narragansett.py`, `src/models/train_narragansett.py`.
Data: 22,851 station-days (2015–2023, 15 stations, daily aggregates of
15-minute sondes). Label: daily-mean chl > 10 µg/L within 7 days.

Every analysis from the LIS chapter now has a Narragansett counterpart.
Two replicate, one diverges, one is newly possible.

## 1. DO conditioning replicates (cross-system result)

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
