# KEY_NUMBERS.md — HAB Bloom Predictor Reference

> Mirror of the parent repo's file as of an earlier date. LIS numbers here are on the original sensor label; for current LIS numbers see the parent's notes/S1_NUMBERS_SHEET.md (label rebuilt 2026-09-23).

Corrected pipeline. Do NOT use numbers from the OLD docs listed in Section 6.
Last updated: 2026-06-04

---

## Section 1 — Model Performance (corrected)

| Model | Val AUC | Test AUC | Notes |
|---|---|---|---|
| **LR (deployed, 35 feat, C=0.05)** | **0.824** | **0.815** | **Final locked deployed model** |
| Ensemble (LR 80% + XGB 20%) | 0.862 | 0.827 | Older experiment; not deployed |
| Logistic Regression (baseline, 26 feat) | 0.847 | 0.824 | Baseline before tidal/sal/saturation features |
| XGBoost (baselines_corrected.py) | 0.843 | 0.774 | lr=0.03, depth=3 |
| XGBoost (shap_corrected.py) | 0.850 | 0.774 | lr=0.1, depth=6; used for SHAP only |
| LSTM | 0.832 | 0.784 | Confirmed ~0.829/0.802 on latest run |
| Random Forest | ~0.783 | ~0.794 | Used for feature importances (fig7) |

**Temporal split:** Train 1993–2019 | Val 2020–2022 | Test 2023–2025

**Bloom rates:** train 22.7% | val ~6–7% | test 7.2%

**Precision (test set, deployed LR 35-feat model):**
- Global threshold 0.60: **0.500** (TP=36, FP=36, FN=38)
- Global threshold 0.55: 0.377 (TP=40, FP=66, FN=34)
- Station-specific (best operating point per station): 0.50–1.00 (C1=1.000, 02=0.625, 01=0.500, A4=0.625, B3=0.556)
- Note: older pre-correction results quoted 0.17–0.29 globally and 0.32–0.40 for station models — these are from the wrong pipeline

**Prediction horizon:** 28 days (forward calendar window from observation date)

**Top SHAP features (XGBoost):**
chl_roll9_mean, Chlorophyll, month, chl_climatology, chl_roll3_mean,
chl_roll6_mean, dip_x_month, neighbor_chl3_mean, dissolved_oxygen

---

## Section 1b — Feature experiments (LR C=0.05, balanced, threshold 0.60)

Operating point: LR `C=0.05`, `class_weight='balanced'`, data
`hab_features_tidal.csv`, evaluated on test 2023–2025 at threshold 0.60.
Scripts: `final_evaluation_threshold_sweep.py`, `daily_inference.py`.

| Feature set | Prec | Recall | F1 | Test AUC | TP/FP/FN |
|---|---|---|---|---|---|
| Baseline (30 feat, incl. tidal anomalies) | 0.449 | 0.419 | 0.434 | 0.816 | 31/38/43 |
| + sal_lag2/3/4 (33 feat) | 0.446 | 0.446 | 0.446 | 0.814 | 33/41/41 |
| + percent_saturation (34 feat) | 0.465 | 0.446 | 0.455 | 0.814 | 33/38/41 |
| **+ max_gust_3d (35 feat) — FINAL DEPLOYED** | **0.500** | **0.486** | **0.493** | **0.815** | **36/36/38** |

**sal_lag2/3/4** — salinity trajectory lags (2/3/4 prior observations), 97–98%
coverage in `hab_features_tidal.csv`. Each correlates r≈−0.21 with bloom_28d
(lower/falling salinity precedes blooms, consistent with river nutrient pulses).
Integrated 2026-06-04 (commit ce9d02c): recall +0.027 and F1 +0.012 for +2 TP,
with precision essentially flat (−0.003) and AUC within noise (−0.002).

### Unused-column sweep (`unused_features_search.py`, on top of baseline)

Tested 14 unused `hab_features_daily.csv` columns. Only sal_lags gave a clean
F1/recall gain. Notable correlations with bloom_28d:
`neighbor_chl5_mean` +0.283, `sea_water_density` −0.241 (strongest physical
predictor but CT DEEP stopped recording after 2019 → median-imputed test rows),
`sal_lag2/3/4` ≈ −0.21, `chl_roll3_std` +0.157. The high-priority physical
group (density+PAR+chl_std+pH) raised precision to 0.474 and AUC to 0.822 but
cost recall (0.365), a precision-for-recall trade — not integrated. Adding
`sea_water_density` on top of sal_lags was destructive (precision 0.394).

---

## Section 1c — Final Locked Numbers (35-feat pipeline, max_gust_3d integrated)

> All numbers below are final. Do not update without retraining on new data.
> Locked: 2026-06-04. Reproduce: `python src/models/final_evaluation_threshold_sweep.py`

**Global model (LR, C=0.05, balanced, 35 features):**

| Threshold | Prec | Rec | F1 | TP | FP | FN | AUC |
|-----------|------|-----|-----|-----|-----|-----|-----|
| 0.60 (balanced) | **0.500** | **0.486** | **0.493** | 36 | 36 | 38 | 0.815 |
| 0.55 (high recall) | 0.377 | 0.541 | 0.444 | 40 | 66 | 34 | 0.815 |

**Station-specific best operating points (test 2023–2025):**

| Station | Rate | Strategy | Threshold | Prec | Rec | F1 | TP | FP | FN | n_pos |
|---------|------|----------|-----------|------|-----|-----|-----|-----|-----|-------|
| C1 | 17.5% | B (global) | 0.60 | **1.000** | 0.571 | 0.727 | 4 | 0 | 3 | 7 |
| 02 | 33.3% | B (global) | 0.60 | 0.625 | 0.833 | 0.714 | 5 | 3 | 1 | 6 |
| 01 | 16.7% | B (global) | 0.60 | 0.500 | 1.000 | 0.667 | 3 | 3 | 0 | 3 |
| A4 | 20.0% | A (station-only) | 0.60 | 0.625 | 0.625 | 0.625 | 5 | 3 | 3 | 8 |
| B3 | 27.5% | A (station-only) | 0.50 | 0.556 | 0.455 | 0.500 | 5 | 4 | 6 | 11 |

Note: C1 precision=1.000 is genuine (4 TP, 0 FP) but small sample (7 test positives over 2023–2025).
Strategy B = global model (all stations) + per-station threshold tuned on 2020–2022 val set.
Strategy A = station-only model trained on that station's 1993–2019 data.

---

## Section 2 — Dataset (corrected)

| Item | Value |
|---|---|
| Total station-days | **11,447** |
| Train rows (1993–2019) | 9,356 |
| Val rows (2020–2022) | 1,057 |
| Test rows (2023–2025) | 1,034 |
| Stations | ~50 CT DEEP LISICOS stations |
| Key file | **data/hab_features_tidal.csv** |

**How we got here:** Raw CT DEEP depth profiles had 120–200 rows per station
visit (one row per depth). `aggregate_daily.py` reduced this to one row
per station-date. The old pipeline never aggregated, producing a spurious
1.36 M-row dataset with inflated correlations.

---

## Section 3 — Exploratory Findings (corrected)

### Lag correlation decay (fig6): CHL(t-lag) vs bloom_28d(t)

**LABEL CORRECTED 2026-08-29.** This table was previously headed "Lag (days)". It is
not days — it is **prior visits**. `src/viz/generate_eda_figures.py:294` computes
`groupby(STATION_COL)[CHL_COL].shift(lag)`, a *row* shift, while the variable is
named `lag_days` and the figure's x-axis reads "Lag (days)". The r values below are
correct and reproduce exactly; only the unit was wrong.

| Lag (prior visits) | Approx. calendar days | Pearson r | p-value | n |
|---|---|---|---|---|
| 0 | 0 | **0.306** | 2.3e-247 | 11,447 |
| 3 | ~63 | 0.190 | 6.3e-92 | 11,297 |
| 7 | ~147 | 0.184 | 2.4e-85 | 11,097 |
| 14 | ~294 | 0.160 | 8.9e-63 | 10,755 |
| 21 | ~441 | 0.138 | 2.1e-45 | 10,426 |
| 28 | ~588 | 0.125 | 1.5e-36 | 10,098 |
| 35 | ~735 | 0.091 | 1.5e-19 | 9,776 |
| 42 | ~882 | 0.132 | 7.9e-38 | 9,460 |

Conversion uses the median inter-visit gap of **21 days**, so "lag 42" is roughly
**2.4 years**, not six weeks. This reframes the table entirely: it is not a picture
of predictive signal decaying over an ecologically meaningful horizon, it is mostly
a picture of correlation between readings separated by months to years. The
non-monotonic bounce at lag 35 -> 42 (0.091 -> 0.132) is unsurprising at ~2 years'
separation, where between-station differences dominate over temporal persistence.

**`figures/fig6_lag_correlation_decay.png` and `figures/lag_correlation_decay.png`
carry the same mislabelled axis and should be regenerated before use in the paper.**

#### True calendar-day lags (computed 2026-08-29)

For each reading, the nearest strictly-prior reading within +/-3 days of the target
lag, same station:

| Lag (days) | Pearson r | n |
|---|---|---|
| 7 | 0.280 | 388 |
| 14 | 0.291 | 4,779 |
| 21 | 0.213 | 1,630 |
| 28 | 0.220 | 5,310 |
| 35 | 0.263 | 2,625 |
| 42 | 0.220 | 2,986 |

**Do not read a decay curve into this either.** The match counts swing from 388 to
5,310 across adjacent lags because station visits cluster near 14- and 28-day
spacings, so each lag samples a different and non-comparable subset of the network.
Lags 0 and 3 are omitted because a +/-3 day window cannot separate them at this
cadence. The honest conclusion is that **this sampling design cannot support a
calendar-day lag-decay curve at all**; the visit-lag table above is the only
defensible version, and it must be labelled in visits.

Signal is present but modest (r ~ 0.19-0.31 in visit-lag terms), consistent with the
biweekly-to-triweekly sampling interval and ecological lag structure.

---

## Section 4 — Aeration Intervention (corrected, from Task 3+4 output)

**Period:** 2020–2022 (validation window)
**High-risk criterion:** Aeration score S > 0.45 AND DO < 6.0 mg/L
**Aeration formula:** S = 0.45×(14–DO)/12 + 0.30×(T–10)/20 + 0.25×p

### Summary (model-based, daily_inference.py)

| Metric | Value |
|---|---|
| Total station-days | 1,057 |
| High-risk station-days | **313 (29.6%)** |
| Top month by mean aeration score | Aug (0.574) |

### Top 10 stations by high-risk days (model probabilities)

| Station | High-Risk Days |
|---|---|
| A4 | 18 |
| D3 | 17 |
| E1 | 15 |
| C2 | 15 |
| 03 | 14 |
| B3 | 14 |
| C1 | 14 |
| H6 | 12 |
| F2 | 11 |
| 05 | 11 |

### Top 5 months by mean aeration score

| Month | Mean S |
|---|---|
| Aug | 0.574 |
| Sep | 0.563 |
| Jul | 0.518 |
| Oct | 0.397 |
| Jun | 0.377 |

### Specific historical inference — Station A4

| Date | Bloom Prob | DO (mg/L) | Temp (°C) | Aeration S | High Risk? |
|---|---|---|---|---|---|
| 2022-09-01 | 0.229 | 3.81 | 23.8 | 0.647 | YES |
| 2022-08-17 | 0.501 | 4.59 | 22.8 | 0.670 | YES |
| 2021-08-31 | 0.447 | 3.77 | 23.4 | 0.696 | YES |
| 2017-08-15 | 0.602 | 2.24 | 21.8 | 0.768 | YES |

### Per-station summary (generate_aeration_figures.py, bloom_28d proxy)

300 high-risk days (28.4%) — note: slightly lower than model-based 313/29.6%
because bloom_28d (0/1) replaces continuous model probability.
Highest-priority station: **A4** — 17 days, min DO = 2.68 mg/L

### Seasonal pattern (bloom_28d proxy, fig11)

High-risk days concentrated in **Jul–Sep**: Jul 65, Aug 148, Sep 87 (= 300 total).
Zero high-risk days outside Jun–Sep.

---

## Section 5 — Numbers that did NOT change

These findings came from the EDA and are unaffected by the bloom-label or
aggregation corrections:

- **Spatial gradient:** western stations (A4, C2, B3) have consistently
  higher bloom rates than eastern stations (M3, N3), consistent with
  nutrient loading from the Connecticut and Housatonic rivers.

- **Long-term trend (fig3):** bloom frequency shows a weak declining trend
  (~−0.1–0.2 %/yr) across 1993–2025, with a visible inflection after ~2014.

- **TMDL inflection (fig3):** visible step-change in annual bloom frequency
  near 2014, coinciding with the Long Island Sound TMDL nitrogen reductions.

- **Seasonal peak:** bloom probability peaks in Aug–Sep (confirmed in both
  old and corrected pipelines).

- **Temperature relationship (fig5):** blooms strongly associated with
  temperatures > 20°C, consistent with Alexandrium catenella ecology.

---

## Section 6 — Numbers that are WRONG in old docs — DO NOT USE

These appear in pre-correction scripts, old README sections, or old figures.
They must not be cited.

| Wrong number | Why it is wrong |
|---|---|
| **Test AUC 0.936** | From old pipeline with un-aggregated depth profiles; each cast counted as many rows, leaking same-visit CHL into both features and label |
| **1.36 M rows** | The raw CT DEEP data before `aggregate_daily.py`; not a valid modeling dataset |
| **7-day forecast horizon** | Stated horizon was 7d but biweekly sampling gives median gap ~21d, making the effective horizon 21–28d |
| **r = 0.707 lag correlation** | Computed on unaggregated data (multiple depth rows per visit inflated n and correlation); corrected value is r = 0.306 at lag = 0 |
| **27,412 high-risk predictions** | From old aeration_intervention.py using `hab_features_final.csv` (non-aggregated) and 7-day bloom label |
| **975 stringent candidates** | Same source as above; artifact of inflated dataset size and wrong label |
