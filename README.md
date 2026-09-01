# HAB Bloom Predictor — Narragansett Bay

**What dense monitoring buys: the same bloom-forecasting recipe, moved from a
21-day boat network to a 15-minute sonde network**

Vihaan Goyal, Westhill High School, Stamford, Connecticut

Fork of [hab-bloom-predictor](../hab-bloom-predictor) (Long Island Sound).
The LIS chapter's full README, findings, and 13-attempt rejection ledger are in
the parent repo and in this repo's git history (branch point `8ae2e2a`).

---

## The two-chapter thesis

**Chapter 1 (LIS, parent repo).** A regularized logistic regression forecasts
chlorophyll exceedances (>10 µg/L within 21 d) in Long Island Sound with good
ranking skill (AUC 0.875) but alert precision capped near 0.14, and cannot be
tuned to clearly beat trivial baselines at basin level. Thirteen improvement
attempts failed. The diagnosis: the median gap between samples (21 days)
equals the forecast horizon — the monitoring cadence, not the model, is the
binding constraint.

**Chapter 2 (this repo).** Test the diagnosis by moving the same recipe to a
bay monitored 2,000× more frequently. Narragansett Bay's fixed-site network
(RIDEM NBFSMN) samples every 15 minutes. If the LIS ceiling is really a
cadence effect, the same model class on dense data should forecast bloom
onsets with usable precision. It does.

## Data

RIDEM Narragansett Bay Fixed-Site Monitoring Network, corrected (post-
calibration) sonde files, 2015–2023: **2.35M readings, 15 stations**,
temp / salinity / DO / DO% / pH / chlorophyll fluorescence, aggregated to
**22,851 station-days** (days with ≥48 chlorophyll readings). Most stations
deploy May–November; two winter stations (B3w, B12w) cover the cold season.
T-Wharf (F3) is not yet parsed (different NERRS export format).

Measured bloom dynamics (2021–23 subset, 380 events, daily-mean chl >10 µg/L):

- **Median bloom duration: 4 days (IQR 2–8).** The median bloom starts and
  ends inside a single LIS revisit gap — the direct, measured justification
  for Chapter 1's cadence claim.
- Ramp-up from <5 µg/L to >10 µg/L: median 14 d; 36% of blooms ramp in ≤7 d.
- 31% of station-days exceed 10 µg/L (LIS test-era station-day rate: ~5%).

## Model and protocol (inherited from LIS unchanged)

Label: any daily-mean chl >10 µg/L within 7 days, right-censored → NaN.
Models: LogisticRegression (locked LIS spec, C=0.05, balanced) and
HistGradientBoosting. Feature tiers: **A** = LIS-analog features only
(chl lags/rolls/anomaly/climatology, temp, sal, DO, month); **B** = A +
sonde-native features (diel DO swing, night DO minimum, within-day chl
max/std, day-over-day chl rate and acceleration, pH, DO%, temp range).
Split: train ≤2020, val 2021–22, test 2023. Threshold chosen on val
(max F1); exactly one test evaluation. Base rate and lift reported beside
every precision.

## Results (test 2023)

| Model | Features | AUC | POD | Precision | base rate | Lift |
|---|---|---|---|---|---|---|
| GB, all days | A | 0.909 | 0.825 | 0.859 | 0.551 | 1.56 |
| LR, all days | A | 0.889 | 0.875 | 0.803 | 0.551 | 1.46 |
| persistence (chl>10 today) | — | 0.887 | 0.589 | 0.931 | 0.551 | 1.69 |
| **GB, onset-only** (today ≤10) | A | 0.835 | 0.580 | **0.718** | 0.347 | **2.07** |
| LR, onset-only | A | 0.801 | 0.700 | 0.628 | 0.347 | 1.81 |
| always-alert, onset | — | — | 1.000 | 0.347 | 0.347 | 1.00 |

Three honest readings:

1. **The all-days task is inflated.** With a 0.55 base rate, persistence gets
   0.93 precision by restating that blooms persist. This is the base-rate trap
   from the LIS basin alert, reappearing on schedule.
2. **The onset-only task is the real forecast** — days not currently blooming,
   where persistence cannot alert at all. There the model predicts new blooms
   7 days ahead at **72% precision** (LIS: 14%). Same recipe, dense data:
   the ~5× precision gain is the measured value of monitoring cadence.
3. **Sonde-native features add nothing** (tier B ≈ tier A everywhere). The
   LIS result that feature engineering doesn't move this system replicates in
   a second bay. The information is in the chlorophyll history; what varies
   is how often you sample it.

## Reproduce

```bash
# BASE conda env (not `hab` — broken LAPACK), from repo root
python src/features/build_narragansett.py          # raw xlsx -> 15-min CSV
python src/features/build_narragansett_daily.py    # station-days + label
python src/models/train_narragansett.py            # models + baselines -> results CSV
```

Raw zips: `https://datadem.ri.gov/documents/bart/nbfsmnYY.zip` (2003–2023
available; 2015–2023 used here). `data/` is gitignored throughout.

## Findings

The full replication of the LIS analysis suite (DO/temp conditioning,
superposed-epoch composites, point-of-no-return, seasonality) is written up
with tables in [`notes/NARRAGANSETT_FINDINGS.md`](notes/NARRAGANSETT_FINDINGS.md).
Headlines: DO conditioning and "no point of no return" replicate across both
bays; temperature dependence does not (LIS flat, Narragansett strong); and
the measured bloom run-up is **~3 days** — the direct mechanism behind the
LIS precision ceiling.

## Known limitations / open items

- Sonde chlorophyll is fluorescence-derived, not extracted chl-a (the LIS
  method); absolute values are not directly comparable across chapters.
- No clustered-bootstrap CIs on the fork's numbers yet (LIS convention:
  station-year clusters, n=2000) — run before quoting differences as findings.
- T-Wharf station unparsed; winter coverage limited to two stations.
- Onset-only POD (0.58) means ~2 of 5 new blooms are still missed at the
  chosen threshold; the threshold sweep trades this against precision.
