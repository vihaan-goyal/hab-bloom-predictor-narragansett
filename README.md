# HAB Bloom Predictor — Narragansett Bay

**Same recipe, two bays: what actually limits bloom forecasting — sampling
cadence, or how rare blooms are?**

Vihaan Goyal, Westhill High School, Stamford, Connecticut

Fork of [hab-bloom-predictor](../hab-bloom-predictor) (Long Island Sound).
The LIS chapter's full README, findings, and 13-attempt rejection ledger are in
the parent repo and in this repo's git history (branch point `8ae2e2a`).

---

## The two-chapter thesis (revised 2026-09-01 after six pre-registered tests)

**Chapter 1 (LIS, parent repo).** A regularized logistic regression forecasts
chlorophyll exceedances (>10 µg/L within 21 d) in Long Island Sound with good
ranking skill (AUC 0.875) but alert precision capped near 0.14. Thirteen
improvement attempts failed.

**Chapter 2 (this repo).** The same recipe on Narragansett Bay's 15-minute
sonde network reaches **0.66 [0.62, 0.69] onset precision across nine test
years** (rolling-origin CV) and beats every trivial rule with CIs excluding
zero. It also measures what LIS could not: a ~3-day bloom run-up, and a
smooth risk curve with no point of no return.

**Why the gap — tested, not assumed.** The original hypothesis was that
sampling cadence explains it. Three controlled tests say cadence is real but
secondary:
- Thinning Narragansett to one sample every 21 days cuts onset precision
  0.86 → 0.52, not to 0.14 (pre-registered criterion failed).
- LIS buoy fluorometers sampled every 15 minutes still give boat-level skill
  (onset precision 0.16–0.18).
- Sonde chlorophyll reads ~1.3–1.6× above lab chlorophyll (n=734 pairs), and
  even at the calibrated threshold Narragansett blooms ~5× more often than LIS.
- **Decisive:** re-threshold Narragansett to LIS's 5% rarity and keep daily
  sampling — onset precision falls to **0.139, identical to LIS's 0.136**.
  Rarity alone reproduces the LIS ceiling (findings §13).

**But dense sampling does buy ranking skill.** At LIS-level rarity, nine years
of daily data give lift **7–8× [5.3–6.0 lower bound]** vs the boat network's
2.7× (findings §16): sensors would not make LIS alerts mostly right, but
would target sampling ~3× more efficiently.

**Precision is a base-rate quantity.** On the fair axis, lift over
climatology, the two bays are within ~1.5× of each other (2.0–2.5 vs
2.7–3.0). The defensible conclusion: in a bloom-rare system like LIS no
cadence or model class produces high-precision alerts; the actionable
quantity everywhere is a 2–3× lift over climatology. Full write-up with
tables: [`notes/NARRAGANSETT_FINDINGS.md`](notes/NARRAGANSETT_FINDINGS.md).

## Data

RIDEM Narragansett Bay Fixed-Site Monitoring Network, corrected (post-
calibration) sonde files, 2005–2023: **4.52M readings, 18 stations**,
temp / salinity / DO / DO% / pH / chlorophyll fluorescence, aggregated to
**42,315 station-days** (days with ≥48 chlorophyll readings; bottom-sonde
coverage 23%). Most stations
deploy May–November; two winter stations (B3w, B12w) cover the cold season.
T-Wharf (F3) is not yet parsed (different NERRS export format).

Measured bloom dynamics (2021–23 subset, 380 events, daily-mean chl >10 µg/L):

- **Median bloom duration: 4 days (IQR 2–8).** The median bloom starts and
  ends inside a single LIS revisit gap. (Cadence matters — §11 shows it costs
  ~0.3 precision — but it is not the main reason LIS precision is low.)
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

## Results (single split, test 2023; pooled 9-year CV in the findings note §7)

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
   7 days ahead at **72% precision** (LIS: 14%). The gap is mostly a base-rate
   effect (blooms ~5× more frequent here), not a cadence effect — see the
   thesis section and findings §10–12.
3. **Sonde-native features add nothing** (tier B ≈ tier A everywhere). The
   LIS result that feature engineering doesn't move this system replicates in
   a second bay. The information is in the chlorophyll history; what varies
   is how often you sample it.

## Use the model on your own water body

Two files, no training, no Narragansett data: `predict_anywhere.py` and
`release/narragansett_bloom_model.joblib` (122 kB). Needs pandas, numpy,
scikit-learn, joblib.

```bash
python predict_anywhere.py readings.csv                      # any cadence
python predict_anywhere.py readings.csv --date 2018-08-15    # report one day
python predict_anywhere.py daily_means.csv --min-readings 1  # daily data
```

`readings.csv` columns: `station, datetime, chl` (required), `temp, sal, do`
(optional; lakes use sal = 0). Chlorophyll can be in any fluorometer units:
the script quantile-rescales it onto the training scale, which is what makes
transfer work (findings §19). Output: `bloom_predictions.csv` with a
probability and alert per station-day, plus a table for the latest day.

Tested on six other systems (Chesapeake, NERRS reserves, UK shelf, Australia,
Lake Erie, SF Bay): the exported model matches a locally trained one at most
sites, lift 1.3–2.5× over always-alert, AUC 0.60–0.86 (findings §19–20).

### Where it works (coverage conclusion, 2026-09-04)

Tested across seven sonde networks on three continents plus a freshwater
lake, and against four satellite chlorophyll products (findings 19-23):

| Input available at a site | What to run | Expected skill |
|---|---|---|
| Sub-daily chlorophyll sonde (any units), any coast or lake | `predict_anywhere.py` with the exported model | onset lift 1.3-2.5x over always-alert, AUC 0.6-0.86; matches a locally trained model at most sites |
| Same, plus 3+ years of local history | refit locally (`src/transfer/transfer_eval.py` recipe) | +0.3-0.5 lift in fresh/estuarine water; no gain on open shelf |
| Satellite chlorophyll only (300 m to 4 km) | nowcast / screening only | 7-day onset lift 1.07-1.26, below climatology: **not a forecast** (findings 23) |

Satellites failed the pre-registered test because the run-up is visible only
20-40% of days, satellite and sonde chlorophyll agree weakly inside estuaries
(Spearman 0.1-0.2), and a satellite-only model mostly predicts its own next
value (lift 1.75 against itself, 1.18 against the water). A water-type
("regime") model library was also tested and rejected (findings 22). The
catalog of every public sub-daily chlorophyll sonde the model can run on is
being built in `data/registry/` (findings 24).

## Reproduce

```bash
# BASE conda env (not `hab` — broken LAPACK), from repo root
python src/features/build_narragansett.py          # raw xlsx -> 15-min CSV
python src/features/build_narragansett_daily.py    # station-days + label
python src/models/train_narragansett.py            # models + baselines -> results CSV
```

Raw zips: `https://datadem.ri.gov/documents/bart/nbfsmnYY.zip` (2003–2023
available; 2005–2023 parsed, 2003–04 format unsupported). `data/` is gitignored throughout.

## Findings

The full replication of the LIS analysis suite (DO/temp conditioning,
superposed-epoch composites, point-of-no-return, seasonality) is written up
with tables in [`notes/NARRAGANSETT_FINDINGS.md`](notes/NARRAGANSETT_FINDINGS.md).
Headlines: DO conditioning and "no point of no return" replicate across both
bays; temperature dependence does not (LIS flat, Narragansett strong); and
the measured bloom run-up is **~3 days**, shorter than one LIS revisit gap.

## Known limitations / open items

- Sonde chlorophyll is fluorescence-derived, not extracted chl-a (the LIS
  method); absolute values are not directly comparable across chapters.
- Sonde chlorophyll ≈ 1.3–1.6× lab chlorophyll; the sonde-10 label is looser
  than LIS's lab-10 (findings §10).
- T-Wharf station unparsed; winter coverage limited to two stations.
- Onset-only POD (0.58) means ~2 of 5 new blooms are still missed at the
  chosen threshold; the threshold sweep trades this against precision.
