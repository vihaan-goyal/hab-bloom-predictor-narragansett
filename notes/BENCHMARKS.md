# Benchmarks: LIS EWS vs Reference HAB Forecast Systems

Task 3 of the August 2026 plan. Status: draft. C-HARM numbers still TBD
(need Anderson et al. 2016 full text; see Open Items).

## Why this comparison is framed in POD / FAR / CSI

These are the standard categorical verification metrics for operational
hazard forecasts (NOAA HAB-OFS, NWS severe weather). Using them makes the
LIS system directly comparable to the reference systems below and avoids
inventing a private vocabulary. Definitions: POD = TP/(TP+FN),
FAR = FP/(TP+FP), CSI = TP/(TP+FP+FN).

## Key precedent: assessable-only verification is standard practice

The NOAA Gulf of Mexico HAB Operational Forecast System skill assessment
for Texas (Kavanaugh et al. 2015, NOAA Technical Report NOS CO-OPS 080)
could assess only a small fraction of issued forecasts, because
validation required field observations that often did not exist along
sparsely monitored coastline:

- Transport direction: 9 of 187 issued forecasts assessable (4.8%)
- Transport distance: 5 of 208 assessable (2.4%)
- Respiratory irritation: 0.9-8.9% assessable depending on bloom year

Forecasts without observational evidence were recorded as "unconfirmed"
and EXCLUDED from skill statistics, not counted as wrong. The Florida
assessment (Kavanaugh et al. 2013) similarly ranged 10-54% assessable.

This is the same convention as our verifiable-window FAR (ver_far):
score alerts only where a verification observation exists within the
horizon. In our test era, 41% of station-day windows contain zero
observations; the cadence-thinning experiment shows all-window FAR
degrades with sampling sparsity while verifiable-window FAR is flat
(~0.8), decomposing alert error into a cadence-driven component and a
cadence-invariant (biological) residual.

## Benchmark table

Base rate and lift columns added 2026-08-31. Lift = precision / base rate — the only
metric here measured against a reference rather than rewarding a convenient base rate.
**None of the external systems publish their event base rates**, so their lift cannot
be computed (n/r), which means none of the published POD/FAR pairs below can be
compared against that system's own climatology. Reporting both is a contribution of
this work, not standard practice. Reference-forecast rows come from
`src/models/reference_baselines.py` (`data/reference_baselines.csv`).

| System | Target | Lead | POD | FAR | base rate | lift | n (assessable) | Notes |
|---|---|---|---|---|---|---|---|---|
| LIS basin alert (this work) | chl-a > 10 ug/L, western LIS | 21 d | 1.00 | 0.60 | 0.293 | 1.37 | 41 basin-days, 12 events (2023-2025) | Initiation forecast; threshold pre-registered on 2020-2022 val; POD exact CI [0.735, 1.000]. **Advantage over always-alert has a CI touching zero** (+0.390 [+0.000, +0.909]); see reference rows |
| LIS station-day (this work) | same | 21 d | 0.875 | 0.875 | 0.046 | 2.7 | 956 station-days (2023-2025) | ver_far ~0.83; empty-window fraction 41%. Beats always-alert decisively: paired lift diff +1.651 [+1.197, +2.228] |
| — always-alert, basin (reference) | same | — | 1.00 | 0.707 | 0.293 | 1.00 | same 41 basin-days | Requires no data; the floor |
| — persistence, basin (reference) | same | — | 0.75 | 0.471 | 0.293 | 1.81 | same 41 basin-days | "Was the last reading above 10?" — numerically outscores the basin model's 1.37 (paired CI includes 0) |
| NOAA GoM HAB-OFS (TX), transport direction | K. brevis bloom movement | days | 1.00 | 0.200 | n/r | n/r | 6 (of 146 issued) | BY2011-2012; tracking an already-identified bloom |
| NOAA GoM HAB-OFS (TX), high respiratory irritation | aerosol impact | 3-4 d | 1.00 | 0.043 | n/r | n/r | 22 | BY2011-2012, single large bloom year |
| NOAA Lake Erie seasonal forecast (Stumpf et al. 2012) | seasonal cyanobacteria severity index | months | n/a | n/a | n/r | n/r | 10 years | Continuous product judged on RMSE: 0.55 CI (discharge-only), 0.37 CI (blended). Benchmark for our planned seasonal severity product, not the EWS |
| C-HARM Pseudo-nitzschia model, all CA stations (Anderson et al. 2016) | P-n bloom > 10^4 cells/L | nowcast | 0.67 | 0.67 | n/r | n/r | 2014-2015, weekly pier sampling | Optimized prediction point 0.52 (POD/FAR crossover); total accuracy 43%; ROC near 1:1 line |
| C-HARM particulate DA model, Santa Cruz Wharf | pDA > 500 ng/L | nowcast | 0.68 | -- | n/r | n/r | same | Their best result: AUC 0.77 at prediction point 0.6; P-n model at same site AUC 0.33 (below random); pDA at Stearns Wharf AUC 0.04 (very few events) |

## Related ML systems that are NOT comparable (checked 2026-09-01)

Both surfaced by a Google search for "ML model that predicts harmful algal
blooms". Neither forecasts, neither reports base rates or trivial baselines,
so neither can go in the table above.

- **CyFi — Cyanobacteria Finder** (NASA / DrivenData, open-source Python).
  Sentinel-2 imagery → cyanobacteria density *now* at a point, for small
  inland lakes/rivers; 8,979 training samples. A **nowcast** of a different
  organism (cyanobacteria, WHO severity levels) in a different setting; the
  authors state it is not applicable to coastal/marine water. No forecast
  lead time; no quantitative skill published in the blog post.
  https://www.earthdata.nasa.gov/news/blog/applying-machine-learning-harmful-algal-blooms
- **Mermer, Zhang & Demir 2024 (EarthArXiv 7979)** — "Predicting HABs Using
  Ensemble ML and Explainable AI", Lake Erie 2013–2020. Regresses chl-a
  concentration from co-sampled water quality (R² ≈ 0.85 for XGBoost/Deep
  Forest; SHAP highlights particulate N/C, total P). This is a **same-time
  regression** — chl-a from nutrients measured in the same sample — not a
  forward forecast; no lead time, no event definition, no baseline.
  https://eartharxiv.org/repository/view/7979/

### Positioning table (2026-09-01) — "different question, stricter evaluation"

| | CyFi (NASA) | Mermer et al. 2024 | NOAA HAB-OFS / C-HARM | This work |
|---|---|---|---|---|
| What it does | detects blooms *now* from satellite | explains chl-a from same-sample nutrients | forecasts / tracks a known bloom | **forecasts new blooms days ahead** |
| Lead time | 0 | 0 | days | **7 d (Narragansett) / 21 d (LIS)** |
| Water type | small inland lakes | Lake Erie (freshwater) | Gulf of Mexico, California coast | two temperate estuaries |
| Skill reported | qualitative | R² 0.85 | POD / FAR | AUC 0.88; precision 0.66 [0.62, 0.69]; POD 0.65 |
| Base rate stated | no | no | no | **yes** |
| Beats a trivial baseline? | not tested | not tested | HAB-OFS: vs random chance only (Heidke skill score, CO-OPS 080; see correction below); C-HARM: not tested | **yes — CIs exclude 0** |
| Multi-year held-out test | not stated | not stated | small n | **9 years, station-year clustered CIs** |
| Pre-registered negatives | no | no | no | **14 rejected attempts + one refuted hypothesis** |

Read across: ahead on evaluation rigor (base rate, trivial baselines,
clustered CIs, pre-registration); behind on deployment scope (CyFi is a
shipped tool, NOAA is operational); not comparable on raw numbers (an R²
from same-sample regression is not a forecast skill). Phrase it as
"stricter," never "better."

Why this matters for framing: the general literature answers "is there ML for
HABs?" with nowcasts and same-sample regressions. Our task — an exceedance
*forecast* days ahead with a stated base rate, lift, and CIs — is the
question those systems don't attempt, which is why the honest comparators
remain the operational forecast systems (NOAA HAB-OFS, C-HARM) above.

## Operational products comparison (2026-09-06)

Every cell below was checked against the linked primary source on 2026-09-06
(access date in the last column). "Not stated" means the source does not say
it, not that it is false. Products are ordered from the closest analogue
(bloom-transport bulletins) to the furthest (satellite nowcasts). Same house
framing as the positioning table: different question, stricter evaluation;
never "better".

| Product | Operator | Target (what is predicted) | Lead time | Inputs | Spatial coverage | Water types | Retraining / site-specific calibration required | Skill metric reported and value | Base rate stated | Evaluated against a trivial baseline | Source URL (accessed 2026-09-06) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NOAA Lake Erie HAB Forecast (seasonal forecast + bulletins) | NOAA NCCOS ("The Lake Erie Forecast is operated by the National Centers for Coastal Ocean Science"); bulletins became official NOAA forecasts via CO-OPS | Bulletins: location, extent, transport, surface scum vs mixed, intensity of a *Microcystis* bloom already detected by satellite; seasonal: severity index (satellite biomass over the peak 30 days) | Bulletins: "no more than three to four days in advance" (FAQ); "minimum of 96 hours" (main page); "next 3 days" (2014 CO-OPS article). Seasonal: issued end of June for July-Oct | Sentinel-3 OLCI + true-colour imagery, predicted winds, LEOFS 3-D circulation model, field samples/toxin; seasonal: Maumee River spring discharge + total-phosphorus load | Western Lake Erie basin (Pointe Mouillee, MI to Magee Marsh, OH); bulletins Mon + Thu during blooms | Freshwater (Great Lake) | Yes: seasonal regression refit on Lake Erie years only (NOAA 2025 release: models now trained on 2013-present); bulletin hydrodynamics are lake-specific | Bulletins: none stated on product pages. Seasonal (Stumpf et al. 2012, 2002-2011): blended discharge + TP model RMS error 0.37 CI vs 0.55 CI for discharge alone; exponential discharge model r2 0.97. 2025: predicted SI 2-4, observed 2.4 | No (continuous severity product; bulletins track a bloom already present) | Not stated (no climatology or persistence comparison on any page checked) | https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/lake-erie/ ; https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/lake-erie/faqs/ ; https://tidesandcurrents.noaa.gov/news/lake_erie_hab_article.html ; https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0042444 ; https://coastalscience.noaa.gov/news/2025-lake-erie-harmful-algal-bloom-seasonal-assessment/ |
| NOAA Gulf of Mexico HAB-OFS (*Karenia brevis*) | NOAA NCCOS (bulletins); transitioned to operations by NOAA CO-OPS, Florida Oct 2004, Texas 1 Oct 2010 | Bloom transport direction/distance, intensification, and level of respiratory irritation (very low / low / moderate / high) by coastal region; beach-level respiratory risk every 3 h | Region-level: "forecast by region for the next 3-4 days" (CO-OPS 080, Table 4); beach-level: "every 3 hours out to 30 hours"; bulletins twice weekly (Mon + Thu) during blooms | Ocean-colour satellite imagery, *K. brevis* cell counts (FWC FWRI, HABScope), NDFD wind speed/direction, transport models, buoy data, public-health reports | Florida Gulf coast, Texas coast, Florida east coast; Texas forecast regions "approximately 30-60 km in length"; number of beaches not stated | Marine coastal (Gulf of Mexico) | Species-specific; each forecast is initialised from that week's local cell counts and imagery (not a trained statistical model transferred between sites) | Texas assessment BY2010-2014 (n assessable): "high" respiratory irritation POD 1.00, FAR 0.043, proportion correct 96.7%, n = 22 (BY2011-2012 only); transport direction POD 1.00, FAR 0.200, n = 6. Only 0.9-8.9% (respiratory) and 2.4-4.8% (transport) of issued forecasts were assessable | No (2x2 contingency counts are given; the event rate is not reported as a number) | **Yes, vs random chance**: Heidke skill score, "91.9% improvement over chance" for "high" respiratory (BY2011-2012), 0.00 in BY2013-2014 (n = 2). Not vs always-alert or persistence | https://tidesandcurrents.noaa.gov/publications/NOAA_Technical_Report_NOS_COOPS_080.pdf ; https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/gulf-coast/faqs/ ; https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/gulf-coast/ |
| NOAA / NCCOS C-HARM (California Harmful Algae Risk Mapping), v3 | SCCOOS and CeNCOOS (IOOS regional associations) with UC Santa Cruz, NCCOS, NASA Applied Sciences; v3 driven by NOAA WCOFS | Probability of a *Pseudo-nitzschia* bloom (> 10^4 cells/L), particulate domoic acid > 500 ng/L, cellular domoic acid | Daily nowcast + 72-hour (3-day) forecast (Anderson et al. 2016); the 2023 CoastWatch page shows nowcast, 1- and 2-day maps with a 3-day option | 3-km California ROMS (now WCOFS) SST, salinity, currents; satellite ocean colour gap-filled with DINEOF; empirical habitat (logistic) models | California coast to southern Oregon (v3); original ROMS domain Crescent City to Ensenada, ~1000 km offshore, 3.3 km grid; validated at 9 weekly pier stations San Diego-Humboldt | Marine shelf / upwelling coast | Statistical models trained on California pier data; pixel-level skill "constrained by the lowest resolution input product"; transfer outside California not stated | Nowcast vs pier obs 2014-2015 (Anderson et al. 2016): *P-n* all stations POD = FAR = 0.67 at prediction point 0.52, total accuracy 43%; Santa Cruz Wharf pDA AUC 0.77 (POD 0.68 at 0.6); *P-n* at Santa Cruz AUC 0.33; pDA at Stearns Wharf AUC 0.04. Forecast (not nowcast) skill not assessed | No | Partly: ROC/AUC is judged against the 0.5 "random chance" line; no climatology, persistence or always-alert comparison | https://repository.library.noaa.gov/view/noaa/33076 ; https://ioos.noaa.gov/models/california-harmful-algae-risk-mapping-c-harm/ ; https://coastwatch.noaa.gov/cwn/news/2023-03-14/c-harm-predicting-harmful-algal-blooms-satellite-data.html ; https://sccoos.org/c-harm-v3/ ; https://sccoos.org/california-hab-bulletin/harmful-algal-bloom/ |
| EPA / NASA CyAN (Cyanobacteria Assessment Network) | EPA with NASA, NOAA, USGS (USACE joined 2023) | Cyanobacteria Index (cyanobacteria abundance, cells/mL-equivalent) *now*; weekly maximum composite; EPA page also lists "an experimental cyanoHAB forecasting model" with "weekly forecasts for over 2,000 lakes" (July 2024) | 0 (nowcast; "typically three days" from satellite pass to app). Lead time of the experimental forecast: not stated | Sentinel-3 OLCI at 300 m (Sentinel-2 MSI added 2024; MERIS for the 2002-2012 archive) | Continental US: 2,370 resolvable lakes and reservoirs (weekly product); NASA release: "over 2,300 lakes" in CONUS plus "more than 5,000 in Alaska" | Freshwater lakes and reservoirs (EPA also lists estuaries) | No per-lake training: one CI algorithm; alert thresholds set by the user from state guidelines | CI field validation: MAPE 28.6%, R2 0.95 over 10^4 to > 10^6 cells/mL (Schaeffer et al. 2018, citing prior validation). No skill stated for the experimental forecast | No | Not stated | https://www.epa.gov/water-research/cyanobacteria-assessment-network-cyan ; https://pmc.ncbi.nlm.nih.gov/articles/PMC6781247/ ; https://phys.org/news/2021-10-nasa-dataset-cyanobacteria-lakes.html |
| Chesapeake Bay HAB forecasts (what actually exists): (a) NOAA MERHAB empirical habitat system 2005-2010; (b) VIMS CBEFS experimental HAB forecast (2026 upgrade); (c) *Margalefidinium* / *A. monilatum* system in development 2023-2028 | (a) UMCES / MD DNR / NOAA CSDL, "running operationally at the National Weather Service National Centers for Environmental Prediction" during the project; (b) VIMS with NOAA, U. Maryland, FlowWest; (c) ODU with VIMS, NCCOS, MARACOOS. NOAA's own East Coast HAB page lists **no** operational Chesapeake HAB forecast today, only the new study (c) | (a) Probability of bloom of *Karlodinium veneficum*, *Prorocentrum minimum*, *Microcystis aeruginosa*; (b) *P. minimum* (bloom = > 1,000 cells/mL) plus six other HAB groups, "experimental"; also non-HAB forecasts (Vibrio, hypoxia/DO, sea nettles, marine heat waves) | (a) "Daily nowcasts and 3-day forecasts"; (b) today / tomorrow / 5-day ("expanded from 2 to 5 days"); (c) not stated | (a) SST, salinity, month, from satellite, in-situ and ChesROMS 3-D model; (b) water temperature, salinity, pH, solar irradiance, nitrogen from VIMS models and observing systems; (c) daily-to-sub-daily fixed-site sampling, weekly DataFlow cruises, PlanktoScope imaging, satellite | Chesapeake Bay and major tributaries; habitat model built on 37 CBP stations, ~3,600 observations 1984-2020; (c) lower Bay, York and James Rivers | Temperate estuary (brackish to mesohaline) | Yes: Bay-specific habitat models fit to Chesapeake monitoring data; regional/seasonal accuracy varies 30-90% | *P. minimum* habitat model (Frontiers 2023): overall accuracy GLM 78.7 +/- 2.4%, GAM 82.7 +/- 2.5%, "30-90% depending on region and season"; no AUC / POD / FAR reported. No skill stated on the CBEFS site or the 2026 NOAA announcement beyond "more accurate than ever before" | **Yes** (Frontiers 2023: bloom class "ratio < 10%", SMOTE applied) | Not stated (no climatology or trivial-classifier comparison; accuracy on a < 10% class is not interpretable without one) | https://coastalscience.noaa.gov/project/development-and-implementation-of-an-operational-harmful-algal-bloom-prediction-system-for-chesapeake-bay/ ; https://www.vims.edu/cbefs/ ; https://oceanservice.noaa.gov/news/july26/transforming-chesapeake.html ; https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1127649/full ; https://oceanservice.noaa.gov/hazards/hab/east-coast.html ; https://coastalscience.noaa.gov/project/developing-a-chesapeake-bay-hab-monitoring-and-forecast-system-for-margalefidinium-and-alexandrium-blooms/ |
| UK: SAMS HABreports (Scotland). Cefas itself runs monitoring only (weekly FSA/FSS biotoxin and phytoplankton results, no forecast product); ShellEye (PML/Cefas/SAMS, 2015-2019) was a satellite-bulletin pilot with no lead time or skill published | Scottish Association for Marine Science (SAMS) | Weekly traffic-light risk of *Dinophysis* (DSP), *Pseudo-nitzschia* (ASP), *Alexandrium* (PSP), *Karenia mikimotoi* (fish kills) and shellfish biotoxin exceedance at aquaculture sites; 5-day trajectory of a bloom already observed | Weekly bulletin; "5-day early warning simulations" (particle tracking) | Weekly regulatory phytoplankton counts and biotoxin data, CMEMS satellite chlorophyll, WeStCOMS-FVCOM (west coast) and Mercator-IBI36 (Shetland) hydrodynamics, wind and temperature forecasts | Scottish coast incl. Shetland (~75% of Scottish shellfish production); number of sites not stated | Coastal marine, sea lochs | Yes: expert-judgement risk index per region; satellite LDA classifier trained on historical bloom images with manual masking; hydrodynamic models are region-specific | 2017-2019 "mean success rate for predicting incidences of the three major shellfish toxin syndromes ... was 74%": ASP 97%, DSP 72%, PSP 65% correct. Drifter separation for Mercator-IBI36 5.7 +/- 2.9 km and 7.3 +/- 3.8 km | No | Not stated | https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.631732/full ; https://www.cefas.co.uk/data-and-publications/habs/ ; https://pml.ac.uk/projects/shelleye/ |
| Australia: CSIRO short-term blue-green algae forecasting tools (Lake Hume, River Murray, Melbourne Water lagoons). eReefs (GBR) is a near-real-time chlorophyll *hindcast/nowcast* on 4 km and 1 km grids, "could also be used for short term forecasts" but no operational bloom forecast; AquaWatch is "progressing towards" an operational system after the 2025 South Australian bloom | CSIRO for Murray-Darling Basin Authority, WaterNSW, Goulburn-Murray Water, Melbourne Water; eReefs: CSIRO, BoM, AIMS, GBRF, Qld Government | "short-term prediction of e.g. blue green algal development" (cyanobacteria) | "7 to 14 days ahead" | "weather forecasts and flow conditions" driving hydrodynamic models | Lake Hume and River Murray sub-catchments; one Melbourne Water treatment lagoon; eReefs: whole Great Barrier Reef | Freshwater reservoir / river; eReefs: tropical shelf | Yes: waterbody-specific hydrodynamic model per site | Not stated (no skill values on the CSIRO pages; eReefs cites RMSE / Willmott skill for physics, no chlorophyll-forecast skill) | No | Not stated | https://www.csiro.au/en/research/natural-environment/ecosystems/Blue-green-algae/Our-research ; https://research.csiro.au/ereefs/summary/ ; https://ereefs.aims.gov.au/about.html ; https://www.csiro.au/en/news/All/Articles/2026/June/Algal-blooms-explained |
| **This work** (Narragansett 7-day onset model + LIS 21-day basin model) | This project (student, two public repos); no agency operates it | Onset of a chlorophyll exceedance at the *same sonde*: per-site p75 label within 7 d (Narragansett / ERDDAP sites); chl-a > 10 ug/L within 21 d (western LIS). No species, no toxin | 7 d (any sub-daily sonde); 21 d (LIS boat network) | Sonde chlorophyll only (plus temperature / salinity / DO where the site has them); no satellite, no model fields | Narragansett Bay + 87 further public ERDDAP sonde sites on three continents (74 scorable, 598 station-years; Gulf of Mexico, Florida, Carolinas, California, Great Lakes, Alaska, British Columbia, Pacific islands) + 50 CT DEEP LISICOS stations 1993-2025 | Estuaries, shelf piers, Great Lakes, Pacific-island coasts, one freshwater lake | **None**: one frozen exported model, chlorophyll quantile-rescaled per site; refitting on the 11 best new sites changed median lift by -0.06 (findings 24 addendum) | Onset lift over always-alert 1.3-2.5x at most sites, median 1.58 (IQR 1.31-2.05) over 74 sites, 67 of 74 with bootstrap CI above 1.0 and none below; median AUC 0.74; LIS station-day 21-d: POD 0.875, FAR 0.875, lift 2.7 (paired diff vs always-alert +1.65 [+1.20, +2.23]) | **Yes, every site** (median 0.28; LIS 0.046 station-day / 0.293 basin-day) | **Yes**: always-alert and persistence, station-year clustered bootstrap CIs; the LIS basin-day CI touches zero and is reported as such | Fork README "Where it works" and `notes/NARRAGANSETT_FINDINGS.md` section 24 (2026-09-05); `data/registry/site_skill.csv`; benchmark table above |

Correction to the positioning table above (2026-09-06): its "Beats a trivial
baseline? not tested" cell for NOAA HAB-OFS was wrong. The Texas assessment
(CO-OPS 080, section 3.4) reports Heidke skill scores, which reference
random chance; it still does not compare against always-alert or
persistence, and it states no base rate. The C-HARM half of that cell stands.

**What is new here.** Three things no row above does together. (1) An
*onset* lead time at any chlorophyll sonde: every operational product either
tracks a bloom already found (Lake Erie, Gulf of Mexico, HABreports) or
predicts a probability field for one species in one region (C-HARM,
Chesapeake); the exported model gives 7 days of warning before a site's own
exceedance at 74 sites it never saw. (2) No retraining: every other row is
region-specific by construction (lake-specific regressions, California-trained
habitat models, Bay-specific GLMs, per-waterbody hydrodynamics); here one
frozen model runs on a new site with a quantile rescale, and local refits
did not help. (3) Base rate and trivial-baseline comparison always shown: one
row states a base rate (Chesapeake, < 10%), one compares against chance
(HAB-OFS, Heidke), none shows both, and none uses always-alert or persistence
as the reference. Here both appear on every site with clustered CIs, including
the LIS basin-day result whose CI touches zero.

**What is not.** No toxin, no cell counts, no species: the label is a
chlorophyll exceedance, which is a biomass proxy, not a health outcome
(HAB-OFS, C-HARM, HABreports forecast the harmful organism or toxin; CyAN
counts cyanobacteria). No satellite coverage: the model needs a sonde in the
water, so it cannot map a bloom's extent the way Lake Erie, CyAN or C-HARM
do. Not operational: no agency runs it, no bulletin is issued, and the
prospective test (findings 25) is frozen but not started. No agency
validation: every number in the last row is self-reported from a held-out
test; every other row has been assessed or run by NOAA, EPA, SAMS or CSIRO.
Different question, stricter evaluation; never "better".

## Comparability caveats (must appear in the paper alongside the table)

1. Task difficulty differs. GoM transport and respiratory forecasts
   TRACK a bloom that has already been identified by sampling; our
   system forecasts INITIATION 21 days ahead. Tracking supports much
   lower FAR; the numbers are context, not a leaderboard.
2. Lead time differs by an order of magnitude (days vs 21 days).
3. Targets differ: chlorophyll biomass exceedance (ours) vs toxic
   species / toxin impact (GoM, C-HARM). See the biomass-vs-species
   framing issue (Vaudrey, Getchis).
4. Small n on both sides: our basin test has 12 events; the GoM
   transport assessment has n=6. Quote CIs, not point values, wherever
   possible.
5. Verification density differs: California pier monitoring is far
   denser than LIS cruise sampling, which is the subject of our
   cadence-thinning result. Notably, Anderson et al. attribute much of
   C-HARM's weak pixel-level skill to spatial/temporal mismatch between
   3-km model output and weekly pier sampling: the third reference
   system (after both NOAA GoM assessments) whose reported skill is
   limited by verification density rather than model quality.
6. Comparison context for AUC: C-HARM nowcasts score AUC 0.33-0.77
   against pier observations; the LIS system scores 0.815 (test) /
   0.852 (rolling-origin CV) at a 21-day horizon. Different target and
   region, but the LIS system operates within the skill range of the
   pre-operational state of the art while forecasting 21 days ahead
   rather than nowcasting.

## Open items

- [x] Pull C-HARM POD/FAR tables from Anderson et al. 2016 (Harmful
      Algae 59). DONE via NOAA IR open-access manuscript
      (repository.library.noaa.gov/view/noaa/33076). Values in table.
- [ ] Check the Florida HAB-OFS assessment (Kavanaugh et al. 2013,
      CO-OPS 073) for additional POD/FAR values at higher n.
- [ ] Decide table placement in paper (Discussion vs Results).

## Sources

- Kavanaugh, K.E., Derner, K., Davis, E., Urizar, C. (2015). Assessment
  of the Western Gulf of Mexico Harmful Algal Bloom Operational Forecast
  System (GOMX HAB-OFS). NOAA Technical Report NOS CO-OPS 080.
  https://tidesandcurrents.noaa.gov/publications/NOAA_Technical_Report_NOS_COOPS_080.pdf
- Stumpf, R.P., Wynne, T.T., Baker, D.B., Fahnenstiel, G.L. (2012).
  Interannual Variability of Cyanobacterial Blooms in Lake Erie.
  PLOS ONE 7(8): e42444.
- Stumpf, R.P., Johnson, L.T., Wynne, T.T., Baker, D.B. (2016).
  Forecasting annual cyanobacterial bloom biomass to inform management
  decisions in Lake Erie. J. Great Lakes Res.
- Anderson, C.R., et al. (2016). Initial skill assessment of the
  California Harmful Algae Risk Mapping (C-HARM) system. Harmful Algae.
  https://pubmed.ncbi.nlm.nih.gov/28073500/
- Kavanaugh, K.E., et al. (2013). Assessment of the Eastern Gulf of
  Mexico HAB-OFS (Florida). NOAA Technical Report NOS CO-OPS 073.