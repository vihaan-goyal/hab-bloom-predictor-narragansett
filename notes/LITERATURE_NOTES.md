# Literature Notes — HAB Bloom Predictor Project

## Primary Reference: Perreira (2021)
**Full citation:** Perreira, S. (2021). Long Term Nutrient and Chlorophyll a Dynamics across Long Island Sound and Impacts on Dissolved Oxygen Conditions within the Western Sound (1991-2019). CUNY Academic Works. https://academicworks.cuny.edu/cc_etds_theses/961

---

### Key Findings That Validate Your Results

**1. West-to-East Chlorophyll Gradient**
- Perreira confirms a well-documented west-to-east decreasing gradient in both CHLA and nutrients across LIS
- Average maximum spring CHLA: Narrows = 16.5 µg/L, WLIS = 9.5, CLIS = 7.4, ELIS = 9.1
- Your model independently identified the same gradient (station A2 at 46% bloom rate vs. station N3 at 1.7%)
- Gobler (2006) calls this a "eutrophication gradient" -- cite both

**2. Bloom Threshold: 10 µg/L**
- Perreira explicitly uses >10 µg/L as the strong bloom threshold
- ≥20 µg/L defined as "poor" status per National Coastal Condition Report standards
- Your 10 µg/L threshold is scientifically justified

**3. Post-2014 Decline in Bloom Frequency**
- Perreira documents nutrient reductions under Clean Water Act phases:
  - Phase III (2001-2016): 58.5% nitrogen reduction target
  - PIV (2017-2019): TMDL goal achieved, NOx down 74% relative to 2001-2016
- The step in the monitoring record after 2014 is mostly a sensor scale change, not the TMDL: the old label came from DEEP's CTD fluorometer, which read 1.8-3.2x the lab in 2009-2013 (SeaBird to YSI EXO2 around 2009/2010) and 0.8-1.05x from 2016 on (DEEP, 2026-09-23). The lab record has no 2014 step, only a real, temporary low in 2012-2017.
- Use the nitrogen-reduction history as background, not as the explanation of the step in the bloom trend figure

**4. Spring (Feb-Mar) Bloom Dominance**
- Perreira confirms seasonal trend: larger bloom in late winter/early spring (Feb-Mar), smaller bloom in late summer, another in early fall
- Spring bloom dominated by diatoms; summer bloom dominated by dinoflagellates (George et al. 2015)
- Cold temperatures reduce zooplankton grazing, allowing diatom blooms to develop unchecked
- This explains your counterintuitive Feb-Mar peak in bloom frequency

**5. The 2001-2016 CHLA Rebound (Explains Your 2002 Spike)**
- Despite nitrogen reductions, CHLA increased 77-297% across LIS in PIII (2001-2016) vs PII (1995-2000)
- Attributed to phytoplankton community shift: smaller species outcompete larger diatoms under lower nitrogen (Rice et al. 2013, Suter et al. 2014)
- Increasing temperatures also contributed (warmer decades favor certain species)
- Your 2002 spike and elevated bloom rates through 2012 are part of this documented rebound

**6. CT DEEP Station Coverage Limitation (Critical for Your Limitations Section)**
- The standard CTDEEP spring bloom estimate uses only B3, D3, F3 stations
- Station A4 (westernmost, highest CHLA) is EXCLUDED from the standard estimate
- This means CT DEEP UNDERESTIMATES Narrows bloom intensity by ~36.7%
- Your training data uses all 50 stations but still has this geographic bias
- Acknowledge this as a limitation: western-most bloom dynamics may be underrepresented
## Cloud Coverage Analysis
- 8,356 satellite files covering 2003-2025
- 1,185,288 CT DEEP observations matched to satellite dates
- 354,685 records have both in-situ and satellite chlorophyll (29.9% coverage)
- Cloud cover is the primary data gap -- not a file availability issue
- Coverage consistent with Perreira (2021) who used the same CT DEEP dataset

**7. Multiple Factors Drive Blooms (Not Just Nutrients)**
- Simple linear regressions between CHLA and nutrients showed poor correlations (r² < 0.3)
- Only multiple regression combining 8 variables achieved r² = 0.6
- Key factors: temperature, density stratification, spring CHLA trajectory, precipitation, discharge
- This justifies your multi-feature LSTM approach over simple threshold rules

---

### Additional References to Find and Read

| Paper | Why It Matters |
|-------|---------------|
| George et al. 2015 | Winter-spring phytoplankton bloom dynamics in LIS, temperature-nutrient-grazing interactions |
| Rice et al. 2013 | Interdecadal chlorophyll trends in central LIS basin |
| Gobler et al. 2006 | Nitrogen-silicon limitation across the East River-LIS system |
| Anderson & Gordon 2001 | Nutrient pulses, plankton blooms, seasonal hypoxia in western LIS |
| Suter et al. 2014 | Phytoplankton assemblage changes during nitrogen load decreases in LIS |
| Lee et al. 2008 | Bottom dissolved oxygen characteristics in LIS |
| Wilson et al. 2008 | Long-term variations in hypoxic conditions in western LIS |

---

## Paper Outline Updates Based on Perreira (2021)

### Introduction additions:
- Cite Perreira for LIS eutrophication context and historical bloom dynamics
- Mention Clean Water Act phases as regulatory context
- Note that despite nutrient reductions, blooms persisted through 2016 due to community shifts -- existing monitoring is reactive, not predictive

### Methods additions (Section 2.2):
- "Bloom conditions were defined as chlorophyll-a concentrations exceeding 10 µg/L, consistent with standards established for Long Island Sound water quality assessment (Perreira, 2021)"
- Note CT DEEP monitoring program details (47 stations sampled bi-weekly in summer, 17 monthly year-round)
- Acknowledge A4 station coverage gap as data limitation

### Results additions (Section 4.1):
- Reference Perreira's west-east gradient when presenting your station bloom rate map
- Do not attribute the post-2014 step to CWA Phase III/TMDL achievement: it is mostly a CTD sensor scale change (DEEP, 2026-09-23); the lab record shows only a temporary 2012-2017 low

### Discussion additions (Section 5):
- Spring bloom peak: explain via diatom dominance in cold water, reduced grazing (cite George et al. 2015 via Perreira)
- 2002 spike: attribute to documented PIII CHLA rebound (cite Perreira, Rice et al. 2013)
- Post-2014 step: mostly a sensor scale change in the monitoring record, not the TMDL (DEEP, 2026-09-23); the nitrogen reductions themselves are real background (cite Perreira)
- Limitation: CT DEEP station A4 coverage gap means westernmost bloom dynamics may be underrepresented
- Limitation: poor individual correlations between nutrients and blooms (r² < 0.3) justify multi-feature ML approach

---

## Key Quotes for Paper (Paraphrased Per Copyright Rules)

- Western LIS experiences the highest bloom frequency due to proximity to East River nitrogen inputs and sewage effluent from NYC wastewater treatment plants (Perreira 2021, Gobler et al. 2006)
- Spring blooms in LIS are dominated by diatoms that thrive in cold, well-mixed water; as temperatures rise in summer, dinoflagellates replace diatoms as the dominant group (Patten et al. 2010 via Perreira 2021)
- Despite significant nitrogen reductions under Clean Water Act Phase III, bloom intensity rebounded from 2001-2016, attributed to shifts in phytoplankton community composition toward smaller species with more efficient nutrient uptake (Rice et al. 2013, Suter et al. 2014 via Perreira 2021)

---

# Cold-Water Blooms and the Freshwater/Marine Boundary

Added 2026-08-29. Covers the cyanobacteria question — whether WHO alert levels or
any cyanobacteria-based threshold can apply to this system — and what does and does
not support the Feb–Mar winter bloom regime.

## Reinl et al. 2023, "Blooms also like it cold"

*Limnol. Oceanogr. Lett.* 8: 546–564. doi:10.1002/lol2.10316. Open access.
28 authors, Global Lake Ecological Observatory Network.

**SCOPE LIMIT — read before citing.** This paper is **freshwater lakes only**.
Verified by text search of the full PDF: zero occurrences of "salinity", "saline",
or "brackish"; both "marine" hits sit in the reference list; the one "estuar" hit is
an institutional affiliation (Lake Superior National Estuarine Research Reserve).
Every documented bloom is a lake. **Do not cite it as evidence for a marine or
estuarine cold bloom** — it will not survive review.

What it does establish, and can be cited for:

- Cold-water bloom is defined as **< 15 °C**, including ice-covered conditions.
  This is the citable threshold for "cold blooms are a recognised regime".
- ~40 documented cold-water bloom observations, **19 under ice cover**.
- Dominant taxa are *Dolichospermum* and *Aphanizomenon* (Nostocales); also
  *Microcystis*, *Planktothrix*, *Limnothrix*, *Raphidiopsis*, *Synechococcus*.
- Three-type typology: (1) initiated and persisted in cold water; (2) metalimnetic
  blooms brought to the surface by mixing, seiches, overturn, or upwelling;
  (3) initiated in warm water and persisted into cold.
- **Transferable mechanism:** cold blooms follow mixing-induced nutrient pulses,
  with large diatoms growing first and cyanobacteria proliferating as the pulse is
  depleted (Salmaso and Cerasino 2012). They report *Planktothrix agardhii* and
  *L. redekei* co-occurring with the diatom *Stephanodiscus minutulus* at **2 °C** —
  the same temperature as our winter regime, though in freshwater.

## Applying the 15 °C threshold to our data

Computed on `data/hab_features_tidal.csv`; exceedances = Chlorophyll > 10 µg/L with
a paired temperature reading (n = 2,787):

| Metric | Value |
|---|---|
| Exceedances below 15 °C (Reinl definition) | 883 (31.7 %) |
| Exceedances below 10 °C | 699 (25.1 %) |
| Exceedances below 5 °C | 559 (20.1 %) |
| Median temperature of all exceedances | 19.3 °C |
| Bloom rate when < 15 °C | 22.4 % |
| Bloom rate when ≥ 15 °C | 25.5 % |
| Overall bloom rate | 24.4 % |

Two points for the paper:

1. **A third of our blooms are "cold" by Reinl's own threshold**, so the concept
   transfers even though the paper's evidence base does not.
2. **Bloom rate is essentially flat across 15 °C** (22.4 % vs 25.5 %) — temperature
   is close to uninformative for bloom probability in LIS. Worth stating explicitly,
   because it cuts against the warming-drives-blooms framing that dominates the HAB
   literature.

For citing the winter regime itself, use the marine LIS sources already in this file
(George et al. 2015; Patten et al. 2010 via Perreira 2021) — diatom spring blooms in
cold, well-mixed water with reduced grazing. Reinl supports the framing only, not
the system.

## Why cyanobacteria and the WHO alert levels do not apply here

WHO *Guidelines on Recreational Water Quality* Vol 1 (2021) Alert Level Framework
(Vigilance 1–12 µg/L chl-a; Alert Level 1 12–24; Alert Level 2 scum / Secchi < 0.5–1 m)
is scoped to **freshwater and brackish cyanobacteria**. WHO publishes no chl-a alert
values for marine HABs.

Salinity rules cyanobacteria out of this system quantitatively:

| Organism | Documented salinity limit | Source |
|---|---|---|
| *Microcystis*: growth, MC production, cell quota | unaffected to **10 PSU**, cell lysis above | Verspagen 2006; Tonk 2007; Lewitus 2008, via Wallace et al. 2025 |
| **Microcystin toxin** stability | unaffected to **20 PSU** | Mazur & Plinski 2001, via Wallace et al. 2025 |
| *Microcystis* in 35 PSU seawater | ~48 h, then dies | Wallace et al. 2025 |
| *Dolichospermum circinale* | not above **7.5 PSU** | FW-to-marine review, PMC11539047 |
| SF Estuary "brackish" stations holding cyanobacteria | **1–6 ppt**, and *Microcystis* still predominantly in the freshwater zone | Kurobe et al. 2018, PLOS ONE 13(9):e0203953 |

(An earlier draft of this table listed 18 PSU for *Microcystis*, from a secondary
review. The primary literature compiled in Wallace et al. 2025 gives **10 PSU** with
lysis above. Use 10.)

**Our record: minimum salinity ever recorded 22.96 PSU, zero readings below 18 PSU
(n = 11,381), median 27.3.** Our freshest observation in 32 years sits above the
survival ceiling of the most salt-tolerant bloom-forming cyanobacterium on record.
All 2,796 exceedances occurred at ≥ 22.96 PSU.

Regulatory confirmation of the same boundary: NYSDEC NYHABS covers "freshwater
(non-marine) HABs" only and routes marine HABs to a separate shellfish-biotoxins
programme. Gobler / Stony Brook 2024 reports cyanobacteria in "more than two dozen
lakes and ponds" while the bays and estuaries got rust tide (*Cochlodinium*). NY Sea
Grant assigns cyanobacteria to the Great Lakes and Long Island ponds, and brown /
rust / mahogany tides to the marine bays, and notes that only ~40 of ~8,000
cyanobacteria species produce toxins.

**Consequence for framing:** chl-a > 10 µg/L is a **biomass** threshold, not a harm
threshold, and it is stricter than every published marine comparator (NJDEP > 20 µg/L
or 2× long-term mean; Florida 20 µg/L adverse-effects, 40 µg/L bloom-present). The
LIS taxa that actually cause harm — *Alexandrium* (PSP), *Pseudo-nitzschia* (domoic
acid), *Margalefidinium/Cochlodinium polykrikoides* (rust tide), *Aureococcus
anophagefferens* (brown tide) — are not distinguishable from chl-a. "Phytoplankton
biomass exceedance" is what this model predicts; state that rather than letting a
reviewer find it.

## The one legitimate cyanobacteria angle (future work, not this paper)

Wallace, M. K., Kudela, R. M., Gobler, C. J. (2025). "Microcystin contamination of
shellfish along the freshwater-to-marine continuum within US mid-Atlantic and
Northeast estuaries." *Harmful Algae* 145: 102860. doi:10.1016/j.hal.2025.102860.
Open access (CC BY 4.0). **Full text read.**

Time series 2017–2021 in wild and cultured bivalves across three large US East Coast
estuaries: Chesapeake Bay, the Hudson River Estuary, and Long Island Sound, all of
which host microcystin-producing cyanobacterial HABs **within their watersheds**.
Discrete temperature / DO / salinity by YSI Professional Plus; chl-a and phycocyanin
on a bbe Moldaenke FluoroProbe; tissue MDL 0.15 ng g⁻¹ wet weight.

### The salinity numbers — these supersede the estimates above

This paper is the primary source and it is **stricter than the companion review**:

| Quantity | Value | Source cited |
|---|---|---|
| *Microcystis* tolerates | up to **10 PSU** | Tonk et al. 2007; Miller et al. 2010; Preece et al. 2017 |
| *Microcystis* growth, MC production, cell quota unaffected | up to **10 PSU**; **cell lysis above** | Verspagen 2006; Tonk 2007; Lewitus 2008 |
| *Microcystis* survival in 35 PSU seawater | **48 h** only | as above |
| **Microcystin toxin** stability unaffected | up to **20 PSU** | Mazur & Plinski 2001 |

**Correction to this note's earlier draft:** I first recorded 18 PSU as the
*Microcystis* ceiling, taken from a companion review. The primary literature cited
here puts it at **10 PSU**, with lysis above that. Use 10 PSU. Also, the "often
exceeding 10 ng g⁻¹" figure in an earlier draft was from a search summary and is
**not** in the paper — the real Conscience Bay figure is > 80 ng g⁻¹ (below).

The cells/toxin split is the crux: **cells lyse above ~10 PSU, but the toxin stays
stable to ~20 PSU.** Both ceilings sit below our minimum of 22.96 PSU.

### Where microcystin was and was not found

- **Rarely detected across Chesapeake Bay.** Commonly quantified in the Hudson
  estuary and in **two LIS sub-estuaries: Stony Brook Harbor and Conscience Bay** —
  described in the paper as small, with only minor freshwater contributions. Neither
  is open Sound; neither is among our 50 stations.
- **The Hudson gradient is the cleanest natural experiment in the paper.** Piermont
  Pier / Tappan Zee / Dobbs Ferry sites run **0–10 PSU** and microcystin was detected;
  Pier 26 downriver runs **10–25 PSU** and levels were very low or undetectable. The
  authors state *Microcystis* transported downriver "would not survive the higher
  salinities of New York Harbor."
- **Conscience Bay:** its southern extreme has "salinities permissive of *Microcystis*
  (**< 10 PSU**)". Setauket Mill Pond upstream hit **> 700 µg cyano-chl-a L⁻¹** in
  summer 2016. Conscience Bay oysters reached **> 80 ng g⁻¹** in Sept–Oct 2021 while
  Stony Brook oysters were undetectable in the same months.
- Detected in clams (*Mercenaria mercenaria*, *Corbicula fluminea*), Eastern oysters
  (*Crassostrea virginica*), mussels (*Mytilus edulis*, *Geukensia demissa*).
- **Eastern oysters carried significantly more microcystin (p < 0.05)** and were often
  positive when co-sampled species were not — proposed vector for hepatotoxic
  shellfish poisoning. Oysters exceeded the California action level even when
  cyanobacteria and microcystin were low or undetectable in *both* the freshwater
  source and the estuary.
- **Cold-season persistence:** microcystin remained in oysters into autumn after
  water-column biomass and toxin had fallen off, implying depuration slows in colder
  months. Ties to the cold-water theme above — the toxin signal outlives the bloom.

### Guideline values (the regulatory gap, quantified)

- WHO tolerable daily intake, lifetime exposure: **0.04 µg microcystin per kg body
  weight per day** (Fawell et al. 1999).
- Preece et al. 2024 convert that to a seafood guideline of **24 ng g⁻¹ ww**
  (100 g day⁻¹ consumption, 60 kg body weight).
- California OEHHA action level derives from a separate rat study (Heinze 1999),
  reference dose 0.0064 µg kg⁻¹ day⁻¹ via EPA BMD software (Butler et al. 2012).

Note this is a WHO value that **does** apply to a marine context — via the shellfish
consumption pathway rather than a water-column chl-a threshold. That is the correct
way to invoke WHO for this system, and it is a different mechanism from the ALF.

### What this means for us

The toxin-transport angle **also does not reach our stations**. Detections cluster at
< 10 PSU at the freshwater-influenced heads of small embayments; the toxin itself is
only documented stable to 20 PSU. Our minimum is 22.96 PSU and our median is 27.3.
Both the organism and its toxin drop out below where our record begins.

If the continuum project is ever pursued, the target sites are named and specific:
Conscience Bay (southern end), Setauket Mill Pond, Stony Brook Harbor — plus the
Hudson gradient as a template. Data available from the authors on request.

The mechanism is the point: **cells do not survive the salinity transition, toxins
do.** Microcystin production continues in saline water, and dissolved toxin is
bioconcentrated by shellfish to >100× ambient. Ramos et al. 2021 (*Harmful Algae*
103:102004) is an independent example — saxitoxins from the freshwater cyanobacterium
*Raphidiopsis raciborskii* contaminating marine mussels.

Open regulatory gap worth aiming at: no guideline values currently exist for
microcystins in commercially harvested shellfish. California uses 0.8 µg/L as a
recreational-water Caution Action Trigger and 10 µg/kg for fish tissue.

This reframes ponds as **source** and LIS as **receiver** rather than as extra rows,
and would make our existing river-discharge features mechanistic (discharge as the
transport vector) rather than merely correlational. Different target variable
(microcystin in shellfish, not chl-a), so it is a separate project.

## Why we cannot simply add the ponds for more data

1. Features are marine: salinity in a 23–31 band, `tidal_gt_anom`, `tidal_msl_anom`,
   neighbour-station chlorophyll, LIS river discharge. Ponds have no tides, no
   salinity gradient, and no neighbour network — most features are undefined there.
2. The label cannot be built. Ours needs a continuous monitoring series ("any chl-a
   > 10 within 21 d at this station"). NYHABS is a bloom *report* system:
   presence-only and observer-driven, so "no bloom" is indistinguishable from "nobody
   looked". No trustworthy negatives means no classifier.
3. **We already tested this transfer and it failed.** `src/models/seasonal_severity.py`
   found Stumpf-style Lake Erie cyanobacteria severity does not transfer to LIS
   (r = 0.23, LOYO at chance) — freshwater↔marine transfer failing in our own results.
4. It does not address the real bottleneck, which is temporal: 21-day median sampling
   gap, 48 % of gaps exceeding the forecast horizon. More waterbodies do not shrink
   that gap.

Also checked and negative: the raw ERDDAP export `deep_wq.csv` holds 106 real
stations and 11,508 station-days with usable chlorophyll; the model already uses
11,447. The 56 unused stations contribute **60 station-days in total**, about one
visit each. The 50-station set is ~99.5 % of everything available — there is no spare
data in this source.

## Hattenrath-Lehmann & Gobler 2016 — Suffolk County HAB synthesis

"Historical Occurrence and Current Status of Harmful Algal Blooms in Suffolk County,
NY, USA." December 2016, 121 pp. Free PDF via NY Sea Grant (HABActionPlan-Synthesis-092617.pdf)
and suffolkcountyny.gov. **Read.**

Adopts the HABHRCA (2016) HAB definition: a small subset of algal species — diatom,
dinoflagellate, and cyanobacterial — that produce toxins or grow excessively, harming
humans, animals, or the environment. Explicitly notes **a HAB need not reach visible
proportions** (e.g. *Alexandrium*) nor produce a toxin to cause harm (e.g. *Aureococcus*).

Suffolk County hosts at least five HAB types annually, "a distinction potentially
unmatched in the US", and the report confirms the freshwater/marine split we rely on:
freshwater toxic cyanobacteria in **more than two dozen lakes**, while brown tide,
rust tide, PSP and DSP are marine.

### Organism-specific bloom thresholds — the marine equivalents we lacked

This closes a gap flagged earlier in this file, where we noted that no marine
taxon-specific thresholds were on hand:

| Organism / HAB | Threshold | Notes |
|---|---|---|
| *Alexandrium fundyense* (PSP) | **> 1,000 cells L⁻¹** = site of concern | 19 such sites on Long Island; **6 in the Northport Bay–Huntington Harbor complex**, i.e. LIS north shore |
| *Alexandrium*, chronic hotspot | **> 3,000 cells L⁻¹** annual max | Northport Bay, every year since monitoring began |
| *Alexandrium*, largest recorded | **1,000,000 cells L⁻¹** (2008) | closed > 7,000 acres of shellfish beds; 1,400 µg STX eq. 100 g⁻¹ |
| *Alexandrium*, terrapin mortality 2015 | **> 10⁴ cells L⁻¹** | Flanders Bay; 540 µg STX eq. 100 g⁻¹ |
| *Cochlodinium polykrikoides* (rust tide) | dense patches **10⁵ cells mL⁻¹**, **chl-a > 100 µg L⁻¹**; background **10³–10⁴ cells mL⁻¹** | mid/late summer (Jul–Aug) through October |
| *Aureococcus anophagefferens* (brown tide) | **> 10⁵ cells mL⁻¹** | Quantuck Bay peak; blooms in 29 of past 31 years |
| *Prorocentrum minimum* (mahogany tide) | **> 3,000 cells mL⁻¹** (Maryland DNR "threshold above which living resources are impacted"); fish kills **> 10⁴ cells mL⁻¹** | Tango et al. 2005 |

Note the one directly comparable number: **rust tide dense patches exceed 100 µg/L
chl-a — ten times our 10 µg/L threshold.**

### Rust tide was reported in LIS, and our data cannot see it

The report states that in **2012 and 2016**, warm summers produced "anecdotal reports
of *Cochlodinium* bloom patches" in Long Island Sound **from CTDEEP** — our own data
provider — and an extensive bloom in Port Jefferson Harbor in 2016. It suggests other
north-shore harbours may be vulnerable in warm years.

Checked against `data/hab_features_tidal.csv`, Jul–Oct chlorophyll:

| Year | n | mean | z vs all Jul–Oct | max |
|---|---|---|---|---|
| 2012 | 204 | 11.18 | +0.39 | 42.5 |
| 2016 | 237 | 4.21 | **−0.60** | 14.8 |

**2016 is below average in our record despite a documented LIS rust tide that year.**
Across all 11,447 station-days, exactly **one** reading exceeds 100 µg/L (record max
103.2 µg/L), so the rust-tide dense-patch level is essentially absent from our data.

This is a concrete, citable limitation: *Cochlodinium* forms dense, highly localised
surface aggregations, and a routine station-day mean over multiple casts dilutes them
away. **Our monitoring design cannot detect a rust tide even when one is reported in
our own water body by our own data provider.** It is the strongest available evidence
for the framing point above — chl-a > 10 µg/L measures phytoplankton biomass, not
harm, and the two come apart in both directions: benign winter diatom blooms are
counted, genuine HABs are missed.

Incidental confirmation of the TMDL regime shift already in this file: Jul–Oct mean
chlorophyll drops from 11–13 µg/L (2009–2013) to 3.5–5.7 µg/L (2014–2020).
Superseded 2026-09-23: the LIS label was rebuilt on the lab scale (the CTD fluorometer read 2-3x high before 2016); current LIS numbers in the parent's notes/S1_NUMBERS_SHEET.md, rationale in the parent's notes/LABEL_REBUILD_PREREG.md.

## Sources to obtain

- [x] Wallace et al. 2025, *Harmful Algae* 145:102860 — obtained and read in full (CC BY 4.0)
- [x] Hattenrath-Lehmann & Gobler 2016, Suffolk County HAB synthesis — obtained and read
- [x] Reinl et al. 2023 — obtained and read
- [x] Kurobe et al. 2018, PLOS ONE — open access, salinity-gradient evidence

## Related unpublished work

Martin, L. (2026). "Cold Weather Behavior: Phycocyanin and Cyanobacteria in Lagoons
of the Long Island Sound." BA thesis, Purchase College SUNY. Sponsor R. W. Taylor.

Four sites in Rye, Westchester (Playland Lake, Kirby Pond, Manursing Lake, Port
Chester Harbor), weekly Sept 2025 – Mar 2026, YSI sonde. Reports phycocyanin rising
as temperature fell, peaking just before full freeze (p = 0.01237, R² = 0.82), then
the relationship collapsing after the freeze (p = 0.5442, R² = 0.048).

Caveats before citing: (a) the sites are impounded coastal ponds, not the Sound
proper — "lagoons of the Long Island Sound" describes location, not salinity; (b) the
sonde recorded salinity but it is never reported or analysed anywhere in the
document, so overlap with our system is undetermined; (c) the headline correlation
regresses two variables that both trend monotonically with season across ~13 weekly
points, so the p-value overstates the evidence; (d) the phycocyanin signal is
attributed to "cyanobacteria and/or dinoflagellates", but dinoflagellates use
peridinin and chl-c and do not contain phycocyanin.