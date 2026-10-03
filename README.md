# HAB Bloom Predictor — Narragansett Bay

> **LIS cross-reference update (2026-09-28).** LIS reference values quoted in this file (precision 0.117, lift 2.59×, AUC 0.825, base 0.045) predate the parent repo's 2026-09-28 audit (leak fixes, unobserved windows dropped, single validation threshold t*=0.20). Current LIS 21-day station-day values: precision 0.125, lift 1.63×, base 0.077 (560 rows, 43 events); walk-forward AUC 0.736 (2023-25) and 0.693 (2016-25). The comparisons' direction is unchanged: LIS stays far below Narragansett at matched rarity. Parent: notes/S1_NUMBERS_SHEET.md. **Update 2026-10-02:** LIS 2024 is now on DEEP's lab-corrected scale, which removed about 40 raw-sensor bloom events from the 28-day test (49 → 9 in 2024; walk-forward pooled events 121 → 94). LIS walk-forward AUC is now 0.602 (2023-25, 16 events) and 0.660 (2016-25); the 2023-25 precision and lift above are not yet recomputed. The direction is unchanged, and the gap to Narragansett is wider.


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
ranking skill (AUC 0.825) but alert precision near 0.12 (0.117 at the 21-day
operating point; was AUC 0.875 / precision 0.14 on the original sensor label,
before the parent rebuilt the LIS label on the lab scale on 2026-09-23).
Thirteen improvement attempts failed (sensor label; not re-run).

**Chapter 2 (this repo).** The same recipe on Narragansett Bay's 15-minute
sonde network reaches **0.66 [0.62, 0.69] onset precision across nine test
years** (rolling-origin CV) and beats every trivial rule with CIs excluding
zero. It also measures what LIS could not: a ~3-day bloom run-up, and a
smooth risk curve with no point of no return.

**Why the gap — tested, not assumed.** The original hypothesis was that
sampling cadence explains it. Three controlled tests say cadence is real but
secondary:
- Thinning Narragansett to one sample every 21 days cuts onset precision
  0.86 → 0.52, not to the Sound's ~0.12 (pre-registered criterion failed).
- LIS buoy fluorometers sampled every 15 minutes still give boat-level skill
  (onset precision 0.16–0.18).
- Sonde chlorophyll reads ~1.3–1.6× above lab chlorophyll (n=734 pairs), and
  even at the calibrated threshold Narragansett blooms ~5× more often than LIS.
- **Decisive:** re-threshold Narragansett to LIS's 5% rarity and keep daily
  sampling — onset precision falls to **0.09–0.14, and the Sound's 0.117 sits
  inside that range**. Rarity reproduces most of the LIS ceiling, but not
  all of it: on the rebuilt label the Sound also separates the classes less
  well (overlap 0.62 vs 0.52; findings §13, §28).

**But dense sampling does buy ranking skill.** At LIS-level rarity, nine years
of daily data give lift **6.9–8.5× [5.3–6.0 lower bound]** vs the boat network's
2.59× (findings §16): sensors would not make LIS alerts mostly right, but
would target sampling ~3× more efficiently.

**Precision is mostly a base-rate quantity.** On the fair axis, lift, the
two bays are close (Narragansett 2.0–2.5 vs LIS 2.59; was 2.7–3.0 on the
original sensor label). The defensible conclusion: in a bloom-rare system like LIS no
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
| GB, all days | A | 0.907 | 0.823 | 0.845 | 0.554 | 1.52 |
| LR, all days | A | 0.894 | 0.862 | 0.813 | 0.554 | 1.47 |
| persistence (chl>10 today) | — | 0.886 | 0.589 | 0.932 | 0.554 | 1.68 |
| **GB, onset-only** (today ≤10) | A | 0.829 | 0.572 | **0.682** | 0.351 | **1.94** |
| LR, onset-only | A | 0.806 | 0.667 | 0.638 | 0.351 | 1.82 |
| always-alert, onset | — | — | 1.000 | 0.351 | 0.351 | 1.00 |

(2026-09-28: re-run after the leak fix: prior-years chl climatology, and a negative label needs at
least 4 observed days in the window. The table previously showed pre-2026-09-01 numbers, e.g. GB
onset 0.718 / lift 2.07.)

Three honest readings:

1. **The all-days task is inflated.** With a 0.55 base rate, persistence gets
   0.93 precision by restating that blooms persist. This is the base-rate trap
   from the LIS basin alert, reappearing on schedule.
2. **The onset-only task is the real forecast** — days not currently blooming,
   where persistence cannot alert at all. There the model predicts new blooms
   7 days ahead at **68% precision** (LIS: ~11%). The gap is mostly a base-rate
   effect (blooms ~5× more frequent here), not a cadence effect — see the
   thesis section and findings §10–12.
3. **Sonde-native features add nothing** (tier B ≈ tier A everywhere). The
   LIS result that feature engineering doesn't move this system replicates in
   a second bay. The information is in the chlorophyll history; what varies
   is how often you sample it.

## Use the model on your own water body

Two files, no training, no Narragansett data: `predict_anywhere.py` and
`release/narragansett_bloom_model_v2.joblib` (124 kB; v2 of 2026-09-28 uses the leak-free
prior-years climatology and a threshold of 0.45 taken from the rolling CV. The v1 file stays frozen
for the prospective test). Needs pandas, numpy,
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
Lake Erie, SF Bay): the exported model ranks as well as a locally trained one at most
sites, AUC 0.61–0.85, lift about 1.3–1.9× over always-alert (findings §19, leak-free re-run of
2026-09-28; lift depends on the threshold a site picks).

### Where it works (coverage conclusion, 2026-09-04)

Tested across seven sonde networks on three continents plus a freshwater
lake, and against four satellite chlorophyll products (findings 19-23):

| Input available at a site | What to run | Expected skill |
|---|---|---|
| Sub-daily chlorophyll sonde (any units), any coast or lake | `predict_anywhere.py` with the exported model | onset lift ~1.3-2x over always-alert (ERDDAP median 1.51), AUC 0.6-0.85; ranks as well as a locally trained model at most sites (2026-09-28 leak-free figures) |
| Same, plus 3+ years of local history | refit locally (`src/transfer/transfer_eval.py` recipe) | AUC gain only in Chesapeake (+0.09) and SF Bay (+0.05); lift gain +0.2 (Chesapeake, NERRS) to +0.5 (SF Bay), none in Australia; at the best ERDDAP sites a small lift gain (median +0.05, 10 of 16 sites) with no AUC gain (findings 19 and 24 addendum, leak-free re-runs of 2026-09-28; the earlier "+0.3-0.5" and "-0.06" figures used leaky scaling and cross-fold thresholds) |
| Satellite chlorophyll only (300 m to 4 km) | nowcast / screening only | 7-day onset lift 1.07-1.26, below climatology: **not a forecast** (findings 23) |

Satellites failed the pre-registered test because the run-up is visible only
20-40% of days, satellite and sonde chlorophyll agree weakly inside estuaries
(Spearman 0.1-0.2), and a satellite-only model mostly predicts its own next
value (lift 1.75 against itself, 1.18 against the water). A water-type
("regime") model library was also tested and rejected (findings 22). The exported
model has since been run on 87 further public sonde sites found by crawling
48 ERDDAP servers: median onset lift 1.51, 65 of 74 scored sites with a
confidence interval above 1.0, none below, median AUC 0.74 (findings 24, 2026-09-28 leak-free
re-score with the v2 model: rescaling fit on calibration years only; the first scoring, 1.58 and
67 of 74, fit it on the whole record. The 74 sites are partly correlated, e.g. 9 Indian River
Lagoon stations; the median over 50 dataset families is 1.54.
`data/registry/site_skill_causal_v2.csv`).

## Reproduce

```bash
# BASE conda env (not `hab` — broken LAPACK), from repo root
python src/features/build_narragansett.py          # raw xlsx -> 15-min CSV
python src/features/build_narragansett_daily.py    # station-days + label
python src/models/train_narragansett.py            # models + baselines -> results CSV
```

The neural-network experiments (findings §26–27, `src/nn/`) need PyTorch, which the
base env deliberately does not carry. They run in a second env built from
`environment-nn.yml` (same pins plus a CPU torch wheel):

```bash
conda env create -f environment-nn.yml && conda activate hab-nn
python -m src.nn.build_windows                     # 15-min window cache -> data/nn/
python -m src.nn.seq_vs_daily --cells gb,mlp,cnn,hyb --score   # findings 26, fig 12
python -m src.nn.pooled_site_nn                    # findings 27, fig 13
```

Raw zips: `https://datadem.ri.gov/documents/bart/nbfsmnYY.zip` (2003–2023
available; 2005–2023 parsed, 2003–04 format unsupported). `data/` is gitignored throughout.

Every numbered section of the findings note names the script, inputs and
output that produced it: see the **Reproducibility map** at the top of
[`notes/NARRAGANSETT_FINDINGS.md`](notes/NARRAGANSETT_FINDINGS.md). Two
sections (§12, §17) also need the parent LIS repo checked out beside this one
as `../hab-bloom-predictor`. One data source (Cefas SmartBuoy) requires a
manual portal export; §19 says exactly what to request.

**Clean-machine check (2026-09-05).** The committed `environment.yml` had never
been tested and did not describe the environment that produced the results
(it pinned Python 3.11, pandas 3.0 and scikit-learn 1.8; every number here was
run under Python 3.13, pandas 2.3.3, scikit-learn 1.7.2, and the released
`.joblib` unpickles with version warnings under 1.8). It now pins the versions
actually used. `conda env create -f environment.yml` was then run from scratch
into a fresh env on Windows 11 / conda 26.1: it built in a few minutes, and in
it `np.linalg.lstsq`, a HistGradientBoosting fit, `daily_inference_nar.py`,
`bloom_rate_by_period.py` and a warning-free load of
`release/narragansett_bloom_model.joblib` all ran. The old "`hab` env is
broken" note described one damaged local env, not the recipe.

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
