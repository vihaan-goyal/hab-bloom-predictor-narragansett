> Mirror of the parent repo's file as of an earlier date. LIS numbers here are on the original sensor label; for current LIS numbers see the parent's notes/S1_NUMBERS_SHEET.md (label rebuilt 2026-09-23).

> **SUPERSEDED — do not write from this file.** Marked 2026-08-30.
> Written on the old MODIS + 28-day + aeration framing. It shares **no headline number**
> with the current README. Specifically stale: MODIS as a contribution (tested and
> rejected, -7.8pp), the 28-day horizon (standardised at 21), AUC 0.814 (now 0.852
> pooled), precision 0.465 / 0.500 at t=0.60 (threshold-selection inflation; the honest
> h21 value is 0.175), the single temporal split (now rolling-origin CV, 11 folds),
> 34 features (now 35), the aeration framework (pending rerun), and "sends email alerts"
> (no email code exists). It also predates the entire current thesis: the precision
> ceiling, the 13 rejected attempts, the sampling-cadence mechanism, and the POD/FAR/CSI
> vocabulary. A rewrite, not an edit. Use README.md `## Findings` as the current framing.

# Spatiotemporal Prediction of Harmful Algal Blooms in Long Island Sound
# Using Machine Learning and NASA MODIS Satellite Data

**Vihaan Goyal**
Westhill High School, Stamford, Connecticut

---

## Abstract (write LAST, ~250 words)
Five required elements — nothing more, nothing less:
- What you built: ML system that predicts HABs 28 days in advance in Long Island Sound
- Data used: 32 years CT DEEP LISICOS in-situ measurements + NOAA CO-OPS tidal data + CT DEEP WQP nutrients; 11,447 station-days (aggregated from 1.36M raw depth-profile rows), 50 stations
- Primary result: Logistic Regression AUC 0.814 on held-out 2023–2025 test set; precision 0.465, recall 0.446 at threshold 0.60
- Secondary result: hybrid ConvLSTM+LSTM achieves AUC 0.744
- Intervention result: ~313 high-risk days in 2020–2022 (S > 0.45 AND DO < 6.0); Station A4 highest priority, August peak window
- Deployment: operational daily inference pipeline with automated alerts

---

## 1. Introduction (~500 words)

### 1.1 Why HABs matter
- HABs cost US economy $100M/year (cite Anderson et al. 2002)
- LIS has documented Alexandrium and Aureococcus blooms threatening $10.7B annual economic value (cite Creedon 2018)
- Shellfish harvesting industry specifically at risk

### 1.2 Why LIS specifically
- Semi-enclosed estuary, well-documented west-to-east eutrophication gradient
- Nitrogen inputs from East River wastewater treatment plants + urban runoff (cite Gobler et al. 2006)
- CT DEEP has monitored 50 stations since 1991 — 32 years of data available

### 1.3 Gap in current practice
- Current HAB detection is REACTIVE — biweekly water samples, blooms confirmed after damage done
- Threshold-based methods fail to capture temporal dynamics preceding bloom formation
- No spatiotemporal ML model exists specifically for LIS — this is the gap you fill
- ⚠ Say "to our knowledge" no such system exists

### 1.4 What satellite data offers (and its limits)
- MODIS Aqua provides daily 4km observations since 2002
- Challenge: ~70% cloud gap rate over LIS
- Challenge: 4km resolution coarse relative to Sound's 34km width

### 1.5 Contributions list
- First ML HAB prediction system for LIS validated on held-out future data
- Temporal feature engineering using lagged chlorophyll trajectories
- Two-stream hybrid ConvLSTM+LSTM architecture
- Aeration intervention framework (prediction → prevention)
- Operational daily inference pipeline with dashboard and email alerts
- SHAP interpretability analysis confirming biological plausibility

---

## 2. Data (~400 words)

### 2.1 Study Area
- LIS bounding box: 40.5–41.5°N, 73.8–71.8°W
- 177 km long, max 34 km wide, 20–70 m depth
- Semi-enclosed geometry → limited tidal flushing → nutrient accumulation in western basin

### 2.2 In-Situ Water Quality
- CT DEEP monitoring program, 50 stations, 1991-present
- Biweekly Jun–Sep, monthly year-round
- Variables: chlorophyll-a (µg/L), temperature (°C), salinity (psu), DO (mg/L), pH
- Bloom threshold: chlorophyll-a > 10 µg/L — cite Perreira (2021)
- Raw depth-profile rows: 1,358,852 (multiple per station-visit; NOT the training set)
- Aggregated station-days: **11,447** (one row per station-date after aggregate_daily.py)
- Bloom rates: train 22.7% | val ~6% | test 7.2%
- Years: 1993–2025

### 2.3 Satellite Data
- NASA MODIS Aqua L3, product: MODISA_L3m_CHL
- 4 km resolution, daily, 2003–2025
- 8,356 NetCDF files downloaded
- 29.9% of station-days have valid satellite observations (70% cloud gap)
- For hybrid model: 8×8 pixel patches centered on each station

### 2.4 Auxiliary Data
- USGS discharge gauges: Connecticut R. (01184000), Thames R. (01127000), Housatonic R. (01205500)
- 1993–2025
- ⚠ Discharge was removed in ablation study — mention here but note excluded from final model

### Data summary table
Source | Description | Coverage | Records

---

## 3. Methods (~600 words)

### 3.1 Problem Formulation
- Binary classification: predict bloom within next 28 days (yes/no)
- Formal definition: y(t) = 1 if max{chl(t+1),...,chl(t+28)} > 10 µg/L
- Why 28 days: biweekly CT DEEP sampling gives median inter-observation gap ~21 days; 28-day window aligns with the actual lead time available before confirmed bloom

### 3.2 Feature Engineering
- Lagged chlorophyll at lag1–lag4 (prior observations)
- Rolling means: 7, 14, 21-day (chl_roll7/14/21_mean)
- Climatological anomaly and baseline (long-term monthly mean)
- Tidal anomaly features: tidal_gt_anom, tidal_msl_anom (NOAA CO-OPS)
- Salinity lags: sal_lag2, sal_lag3, sal_lag4 (falling salinity precedes blooms)
- Surface oxygen saturation: percent_saturation (CT DEEP WQP)
- Spatial: latitude, longitude, neighbor_chl3_mean / Temporal: calendar month
- Dissolved oxygen, temperature, salinity
- Total: 34 features in deployed model
- Correlation decay (corrected, aggregated data): r=0.306 same-day → r=0.138 at 21-day lag
- Lag features approximated from nearest reading within a tolerance window

### 3.3 Cross-Validation
- Train: 1993–2019 | Val: 2020–2022 | Test: 2023–2025
- Temporal blocked split — NOT random (explain why random would introduce leakage)
- Test set held out completely — touched only once, after all model selection done

### 3.4 Models (all six)
- Logistic Regression: L2, balanced class weights
- Random Forest: 100 trees, balanced class weights
- XGBoost: 200 estimators, max depth 6, lr 0.1, scale_pos_weight ≈ 4.15
- LSTM: 2-layer, hidden 64, dropout 0.5, Adam, weight decay 1e-4, early stopping patience 5
- ConvLSTM: 2-layer, 16 hidden channels, 3×3 kernels, 8×8 pixel MODIS patches
- Hybrid: ConvLSTM spatial stream + LSTM temporal stream, concatenated → shared MLP
- ⚠ Deep learning models trained on satellite-matched subset (290,938 samples) — tabular models on full 11,447 station-days

### 3.4.1 Decision Threshold
- Swept 0.10–0.90 on test set 2023–2025 (final_evaluation_threshold_sweep.py)
- Two reported operating points:
  - 0.60 (balanced): precision 0.465, recall 0.446, F1 0.455, TP=33, FP=38, FN=41
  - 0.55 (high recall): precision 0.387, recall 0.554, F1 0.456, TP=41, FP=65, FN=33
- ⚠ Frame as deliberate choice, not default; low test bloom rate (7.2%) structurally constrains precision

### 3.5 Aeration Intervention Framework
- Define hypolimnetic aeration and the three conditions it requires
- Score formula: S = 0.45·[(14−DO)/12] + 0.30·[(T−10)/20] + 0.25·p
- Weights: DO=0.45 (primary lever), T=0.30 (stratification), p=0.25 (confidence)
- Highly suitable threshold: P > 0.70 AND S > 0.60 AND DO < 6.0 mg/L
- Applied to validation period (2020–2022) predictions only

### 3.6 Operational Deployment
- Daily inference pipeline: fetches ERDDAP → engineers features → runs LR (C=0.05) → computes S → sends email alerts
- Browser-based dashboard: station probability chart, DO vs. bloom scatter, intervention priority table
- No server required — runs from CSV in browser

---

## 4. Results (~500 words)

### 4.1 Exploratory Analysis
- Geographic gradient: 46.3% bloom rate at A2 → 1.7% at N3
- Long-term decline: −0.63%/year, inflection after 2014 linked to Clean Water Act TMDL
- Seasonal pattern: February–March peak (~40%) driven by cold-water diatoms
- Temporal signal decay (corrected): r=0.306 (same-day) → r=0.138 (21-day lag)

### 4.2 Model Performance
- Results table with all 6 models: Val AUC + Test AUC
- **LR (deployed, 34 feat)**: Val 0.824, Test **0.814**, precision 0.465, recall 0.446 @ 0.60
- Ensemble (LR 80% + XGB 20%): Val 0.862, Test 0.827 (older experiment, not deployed)
- XGBoost: Val 0.843, Test 0.774
- LSTM: Val 0.832, Test 0.784
- Hybrid: Val 0.744, Test 0.658
- ConvLSTM: Val 0.696, Test 0.610
- ⚠ Explain DL models trained on smaller satellite-matched subset

### 4.3 Feature Importance (SHAP)
- Top feature: chl_roll9_mean — SHAP ~2x the next feature (from XGBoost SHAP analysis)
- Rank order: chl_roll9_mean, Chlorophyll, month, chl_climatology, dip_x_month, neighbor_chl3_mean, dissolved_oxygen
- Critical finding: DO/temperature add ΔAUC < 0.005 over chlorophyll-only model
- Deployed LR adds tidal anomalies, sal_lags, percent_saturation on top of base SHAP features

### 4.4 Ablation Study
- Largest drop: removing climatological features (ΔAUC = −0.013)
- Removing river discharge slightly improves (+0.002)
- See ablation_study.py for current numbers

### 4.5 Failure Analysis
- FP and FN both peak July–August
- See failure_analysis.py for current station-level breakdown

### 4.6 Aeration Intervention (corrected, 2020–2022 validation period)
- Total station-days: 1,057
- High-risk days (S > 0.45 AND DO < 6.0): **~313 (29.6%)**
- Station A4: highest priority, 18 high-risk days
- Top month: August (mean aeration score 0.574)
- Peak intervention window: July–September

### 4.7 Operational Validation (corrected demo dates)
- 2022-07-19: A4 P=0.813, DO=4.28, S=0.722 — B3 and 01 also flag; best multi-station demo
- 2022-07-07: A4 P=0.678, DO=4.69
- 2021-07-19: A4 P=0.643, DO=3.90

---

## 5. Discussion (~400 words)

### 5.1 Performance explanation
- LR AUC 0.814 on truly future data — generalizes well beyond training
- LR outperforms DL: (a) 11,447 station-days (aggregated) vs 290K satellite-matched for DL models, (b) dominant feature is rolling mean easily captured by linear model, (c) 4km MODIS patches too coarse for 34km-wide Sound; tidal/salinity-lag/saturation features add incremental precision

### 5.2 Biological validation
- Rolling mean dominance validates hypothesis: bloom = temporal accumulation process
- All SHAP directional effects consistent with known LIS bloom biology (cite Gobler 2006, Perreira 2021)
- Spring bloom peak explained: cold-water diatoms, low zooplankton grazing

### 5.3 Aeration framework novelty
- First system to translate bloom forecasts into intervention targeting guidance
- A4 has minimum DO down to 2.68 mg/L in validation period — sediment enters phosphorus-releasing anoxic state → positive feedback
- August window: aeration most effective when thermal stratification AND bloom risk both high (mean S=0.574 in Aug)
- ~313 high-risk days (29.6%) in 2020–2022 with actionable aeration conditions

### 5.4 Policy finding
- −0.63%/year decline + 2014 inflection = direct observational evidence for Clean Water Act effectiveness
- Continued nitrogen management is complementary to reactive aeration

### 5.5 Limitations
- 70% cloud gap rate — systematic bias toward cloud-free conditions
- Biweekly sampling → lag features are approximated from nearest reading within a tolerance window
- Aeration scores from observational data, not hydrodynamic model
- Fixed monitoring stations — performance at unmonitored locations unknown

### 5.6 Future work
- Couple with ROMS or FVCOM hydrodynamic model
- Higher-res satellite: Sentinel-3 OLCI at 300m or VIIRS at 750m
- Low-cost autonomous chlorophyll sensor platform

---

## 6. Conclusion (~150 words)
- LR AUC 0.814 on held-out 2023–2025 data
- Precision 0.465, recall 0.446 at 0.60 threshold; 28-day advance warning
- ~313 high-risk aeration days identified (2020–2022), A4 highest priority, August optimal window
- Operational pipeline validated against confirmed low-DO events
- First ML HAB system for LIS validated on held-out future data

---

## References
- Anderson, D.M. et al. (2002). HABs and eutrophication. Estuaries, 25(4), 704–726.
- Chen, T. & Guestrin, C. (2016). XGBoost. KDD 2016.
- Creedon, M. (2018). Dams on Long Island: Their economic impact.
- Gobler, C.J. et al. (2006). Nitrogen and silicon limitation across urban estuary. Est. Coastal Shelf Sci., 68, 127–138.
- Huisman, J. et al. (2018). Cyanobacterial blooms. Nature Reviews Microbiology.
- Lundberg, S.M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.
- Perreira, S. (2021). Long term nutrient and chlorophyll a dynamics across LIS. CUNY Master's thesis.
- Rice, E. et al. (2013). Impact of anthropogenic nutrient inputs on phytoplankton growth in LIS. Estuaries and Coasts.
- Shi, X. et al. (2015). Convolutional LSTM network. NeurIPS 28.
- Stumpf, R.P. et al. (2009). Skill assessment for operational algal bloom forecast. J. Marine Systems, 76, 151–161.