# Prospective forecast protocol (pre-registration)

**protocol_version 1.1** | 1.0 frozen_at 2026-09-06T00:37:39Z (UTC), written 2026-09-05 | 1.1 amendment 2026-09-06, re-frozen 2026-09-06T19:13:45Z (see Amendments at the end)

> **FIRST ISSUANCE NOT YET MADE; formal start after ISEF Form 1A is signed.**
> Everything below was fixed *before* any dated forecast was issued. `data/prospective/ledger.csv`
> does not exist as of this writing. Dry runs (`--dry-run`) write nothing to the ledger.

## 1. What is being tested

The frozen Narragansett-trained bloom model (`release/narragansett_bloom_model.joblib`,
HistGradientBoosting, 23 tier-A features) is applied every week, in real time, to 20
pre-registered sonde stations it has never been trained on. Each issuance records
P(daily-mean chlorophyll exceeds the station's own 75th percentile within 7 days).
The rows are verified 9+ days later against the readings that actually followed.

| item | value |
|---|---|
| model sha256 (full) | `5c0f7a1701599d2adf2c973699ae36872fa7cff0593a59258631c2d76fc04c7c` |
| model_version (ledger stamp) | `5c0f7a1701599d2a` |
| code sha at freeze | `5b699c6` (working tree dirty: the prospective scripts were uncommitted at freeze time) |
| alert threshold | 0.50 (the model's shipped `pack["threshold"]`) |
| horizon | 7 days |
| window pulled per issuance | 35 days (`--window-days`), features built on the FULL site history |
| stale rule | last station-day more than 3 days before issue date -> `status=stale` |
| warm-up rule | fewer than 21 station-days in the 35-day window, or `chl_roll21_mean` NaN -> `status=warmup` |
| feed_down rule | fetch failed/empty AND no station-day in the window -> `status=feed_down` (row still written, numerics NaN) |

**Rescaling rule.** Every site's chlorophyll columns are quantile-mapped onto the
Narragansett training distribution (`predict_anywhere.rescale_chl`) using the site's
*full record to date* as the source distribution. This is the same rule the retrospective
transfer tests used; it is recomputed on each issuance as history grows (it is a
per-site scale, not a label).

## 2. Sites (20 stations, 4 groups)

Selection rule for `erddap_top`, verbatim from the design (implemented in
`src/deploy/prospective_sites.py::erddap_top_from_catalog`):

> the 10 rows of data/registry/site_skill.csv with lift_lo > 1, n_onset >= 500,
> server IOOS-Sensors, catalog end >= 2026-07-01, sorted by lift desc; metadata
> (url, chl_var, sal_var, do_var, temp_var, station_var, cadence_min) from
> insitu_catalog.csv; min_readings = 48 if cadence_min <= 20 else 12; carry
> t_star_site from site_skill.csv. Exclude PacIOOS.

The rule yields exactly the expected ten (order below is lift-descending).
Narragansett Bay NERR (the training bay's neighbours) was sought on the IOOS Sensors
ERDDAP on 2026-09-05 and is absent; only Kachemak Bay (kac_ss, kac_h3) carries ChlFluor there.

| group | site_id | feed | min_readings | chl units |
|---|---|---|---|---|
| lis_buoy | WLIS_ECO_FL, EXRX_ECO_FL | UConn merlin ERDDAP, `Avg_FL`, UTC night hours 21-09 only | 12 | ECO-FL fluorescence, night-only, uncalibrated |
| nerrs | kac_ss, kac_h3 | IOOS Sensors ERDDAP `nerrs_kac{ss,h3}wq`, QARTOD agg in {1,2} or NaN (1.1 wording; same code as 1.0) | 48 | ug/L ChlFluor |
| chesapeake | MSC, AWS, AES, MAB, RIV, SPS | MD DNR Eyes on the Bay `JustDownload.cfm` | 48 | ug/L TChlPreCal |
| erddap_top | edu_ucsc_scwharf1, scripps-pier-automated-shore-sta-1, oa2-mbari-buoy, newport-pier-automated-shore-sta, mlml_mlml_sea, tiburon-water-tibc1, edu_calpoly_marine_morro, edu_humboldt_humboldt (48); indian-river-lagoon-banana-river, indian-river-lagoon-vero-beach-i (12) | IOOS Sensors ERDDAP; from 1.1: `<chl_var>_qc_agg` flags 4 and 9 dropped; scripps uses `_eco` (see Amendments) | 48 / 12 | site fluorometer units |

## 3. Frozen thresholds (output of `python -m src.deploy.prospective_freeze --force`, protocol 1.1, 2026-09-06T19:13:45Z)

`chl_p75_site` = 75th percentile of daily-mean chl over the station's cached record
(LIS: night-only, stuck-sensor days with >50% identical readings removed). Stored in
`data/prospective/site_p75.csv`; the script refuses to overwrite it without `--force`.
The 1.0 values (frozen 2026-09-06T00:37Z) are in the `p75 1.0` column; the only designed change
is Scripps (channel switched, record restarts 2024-12-04). The other 19 stations differ from 1.0
only by the station-days the 2026-09-05/06 dry-run pulls appended to their histories (largest:
WLIS -2.08, EXRX +1.18, MSC +0.26, RIV -0.33; the rest within +/-0.12).

| site_group | site_id | chl_p75_site | p75 1.0 | n_station_days | history_first | history_last | min_readings | t_star_site |
|---|---|---|---|---|---|---|---|---|
| lis_buoy | WLIS_ECO_FL | 247.50 | 249.58 | 1692 | 2021-03-05 | 2026-09-06 | 12 | |
| lis_buoy | EXRX_ECO_FL | 385.86 | 384.68 | 1834 | 2019-05-15 | 2026-09-06 | 12 | |
| nerrs | kac_ss | 3.11 | 3.11 | 3571 | 2011-01-27 | 2025-12-31 | 48 | |
| nerrs | kac_h3 | 2.77 | 2.77 | 2021 | 2012-06-28 | 2026-05-11 | 48 | |
| chesapeake | MSC | 23.67 | 23.40 | 3883 | 2013-03-28 | 2026-09-06 | 48 | |
| chesapeake | AWS | 25.88 | 25.84 | 3443 | 2016-05-26 | 2026-09-02 | 48 | |
| chesapeake | AES | 25.28 | 25.37 | 3293 | 2016-05-25 | 2026-09-02 | 48 | |
| chesapeake | MAB | 10.95 | 10.86 | 1691 | 2018-04-06 | 2026-09-06 | 48 | |
| chesapeake | RIV | 79.74 | 80.07 | 2232 | 2014-04-10 | 2026-08-25 | 48 | |
| chesapeake | SPS | 18.77 | 18.66 | 2433 | 2010-04-15 | 2026-08-25 | 48 | |
| erddap_top | edu_ucsc_scwharf1 | 27.27 | 27.27 | 2327 | 2013-02-14 | 2026-08-17 | 48 | 0.90 |
| erddap_top | scripps-pier-automated-shore-sta-1 (`_eco`, fresh_site) | 1.33 | 17.78 (`_ctd`) | 563 | 2024-12-04 | 2026-09-06 | 48 | (withdrawn) |
| erddap_top | oa2-mbari-buoy | 10.35 | 10.30 | 2771 | 2015-05-14 | 2026-09-05 | 48 | 0.85 |
| erddap_top | newport-pier-automated-shore-sta | 13.32 | 13.32 | 4502 | 2013-01-24 | 2026-08-11 | 48 | 0.80 |
| erddap_top | mlml_mlml_sea | 3.78 | 3.78 | 5309 | 2010-09-03 | 2026-09-06 | 48 | 0.60 |
| erddap_top | tiburon-water-tibc1 | 3.82 | 3.82 | 4808 | 2008-08-08 | 2026-09-06 | 48 | 0.70 |
| erddap_top | edu_calpoly_marine_morro | 2.86 | 2.86 | 3319 | 2016-01-02 | 2026-09-06 | 48 | 0.65 |
| erddap_top | edu_humboldt_humboldt | 4.03 | 4.03 | 2949 | 2013-02-15 | 2026-09-06 | 48 | 0.35 |
| erddap_top | indian-river-lagoon-banana-river | 4.39 | 4.39 | 1618 | 2022-01-01 | 2026-07-01 | 12 | 0.60 |
| erddap_top | indian-river-lagoon-vero-beach-i | 3.87 | 3.87 | 3939 | 2014-12-17 | 2026-09-06 | 12 | 0.75 |

## 4. Outcome rule (`src/deploy/score_ledger.py`)

For a ledger row with `last_obs_date = L`, once `L + 7 + 2 (latency) <= as_of`:
pull the readings after L, rebuild station-days with the same `min_readings`, and take
`fwd = station-days with L < date <= L + 7`.

* `outcome = 1` if `max(fwd.chl) > chl_p75_site`
* `outcome = 0` if no exceedance and `n_fwd_days >= 4`
* otherwise NaN: `pending` while `as_of <= L + 21`, then `unverifiable`

Outcomes once written (0 or 1) are never changed; `scored.csv` is rewritten atomically.
Only the highest `revision` per (issue_date, site_id, station) is scored.

## 5. Analysis plan

**Primary.** Rows with `status in {ok, stale}` and `onset_row = True` (chl today <= p75,
so persistence cannot alert), alert = `bloom_prob >= 0.50`, outcome as above. Report
precision, POD, lift = precision / base rate, AUC; pooled and per site_group.

**Secondaries.** (a) all rows (persistence included); (b) `alert_site_t = bloom_prob >=
t_star_site` for erddap_top (the threshold each site's retrospective test chose; a site with
`fresh_site=True` has no t* and is left out of this stratum, see 1.1 (a));
(c) the `warmup` stratum separately.

**Uncertainty.** Clustered bootstrap, n = 2000, seed 42, cluster = station-week
(`site_id|station` x `issue_date`), via `transfer_eval.boot_ci` (precision, lift, AUC)
and a matching POD bootstrap.

**Reporting gate.** A stratum is reported only when n >= 30 verified rows and >= 5
positives; otherwise the digest says "below pre-registered minimum". First planned
formal read: after 12 issuances or pooled onset n >= 100, whichever is later.

**Pre-registered expectation bands (lift over always-alert, onset rows).**

| group | band | source |
|---|---|---|
| lis_buoy | 2-3x | LIS boat lift 2.7x; buoy recipe |
| nerrs, chesapeake | 1.3-2.0x | retrospective transfer tests (findings 19) |
| erddap_top | 1.5-2.5x | site_skill.csv lift range 2.0-6.7 shrunk for selection |

Below 1.0x in a gated stratum = the model does not transfer to that group; that result is reported as-is.

## 6. Change policy

Any change to the site list, thresholds, rescaling rule, outcome rule, horizon, model
file or analysis plan bumps `protocol_version` and is recorded here with the date;
rows already in the ledger are never edited or deleted (re-issues append a higher
`revision`). Every ledger row carries `model_version`, `code_version`, `protocol_version`.

## 7. Operations

```
python -m src.deploy.prospective_freeze          # 1.0: 2026-09-06T00:37Z; 1.1 re-freeze (--force): 2026-09-06T19:13Z
python -m src.deploy.score_ledger                # verify due rows, write scored.csv + skill_summary.csv
python -m src.deploy.prospective_forecast        # issue today's rows (exit 2 if date already issued)
scripts\run_prospective.cmd                      # both steps, appending to data\prospective\logs\last_run.log
```

Tracked: `data/prospective/{site_p75.csv, ledger.csv, scored.csv, skill_summary.csv, issued/, outbox/}`,
`notes/prospective/digest_<D>.md`. Ignored caches: `data/prospective/{raw,history,logs}/`.
No scheduled task is registered yet.

## Amendments

### 1.1, 2026-09-06 (before the first issuance; ledger still empty)

Trigger: the 2026-09-06 pre-start health check (live probes cached under
`data/prospective/raw/probe_2026-09-06/`). Code at amendment: `8d010e3` (working tree dirty).

**(a) scripps-pier-automated-shore-sta-1: chlorophyll variable changed to `..._eco`.**
The catalog `chl_var` `mass_concentration_of_chlorophyll_in_sea_water_ctd` has read a constant
0.0 since 2026-04-09 (last non-zero value -0.86 on 2026-04-09T16:36Z), every such reading flagged
QARTOD aggregate 4 (fail, flat-line) by the provider, and NaN after 2026-09-01T21:28Z. The 1.0 dry
run therefore scored `chl_today = 0.00` against a p75 of 17.78, a row that could only ever verify
as 0. The dataset's second chlorophyll channel `mass_concentration_of_chlorophyll_in_sea_water_eco`
(ECO fluorometer, same units) runs 2024-12-04 to present with flag 1 on nearly all readings
(e.g. 2026-08-23..09-06: 254-360 readings/day, daily means 0.54-1.14 ug/L). Changes:
`chl_var` overridden in `prospective_sites.SITE_OVERRIDES`; history re-seeded from `_eco` only
(live pull `time>=2024-12-01`, QARTOD {1,2,NaN} applied, raw cached under
`data/prospective/raw/freeze_1.1/`, old `_ctd` history discarded); `chl_p75_site` re-frozen on that
record (about 21 months instead of 13 years); `t_star_site` withdrawn (it was chosen on `_ctd`) and
`fresh_site=True`, so `alert_site_t` is NaN for this site and it is excluded from the
`erddap_top@t_site` stratum. It is reported inside erddap_top with the caveat that its threshold
rests on a short record.

**(b) newport-pier-automated-shore-sta: not changed.** Its `_ctd` chl went NaN after
2026-08-11T15:24Z; an `_eco` channel appeared 2026-08-13 (daily means 0.26-0.67 ug/L) but three
weeks is too short to freeze a p75. It stays on `_ctd` and will sit in `feed_down`/`warmup` until
the CTD returns. A dated amendment may switch it to `_eco` once 90 days of `_eco` exist.

**(c) erddap_top QARTOD filter (new rule).** `fetch_ioos` looks up `info/<id>/index.json` once
per run; if `<chl_var>_qc_agg` exists it is requested and chl is dropped where the flag is 4 (fail)
or 9 (missing); flags 1, 2, 3 and NaN are kept. (First drafted as {1,2,NaN} like nerrs; narrowed the
same day, before any issuance, because flag 3 "suspect" removes 73% of mlml_mlml_sea and would
have made its frozen p75 incomparable. Flag 4 alone excludes the Scripps flat-line zeros.) All ten
erddap_top datasets carry the variable. Histories seeded
before 1.1 (the `data/registry/sites` caches, which have no qc column, plus the 2026-09-05/06
dry-run pulls) are left as they are; the filter applies to every live pull from this amendment on.
The Scripps zeros would have been excluded by this rule (flag 4). Observed effect in the 1.1 dry
run (35-day window to 2026-09-06) under the final {1,2,3,NaN} rule: mlml_mlml_sea drops its
9,214 flag-4 readings of 20,616 (45%) and keeps flag 3; the other nine sites lose under 2%.
mlml remains the site to watch for warmup/stale drift under this rule.

**(d) nerrs NaN-flag behaviour documented.** `parse_nerrs_frame` keeps readings whose QARTOD
aggregate flag is NaN/absent (`qc in {1,2} or qc is NaN`); this was the 1.0 code behaviour and the
1.0 p75 values used it. The 1.0 text "QARTOD agg in {1,2}" is corrected above. Not a data change.

**Also in 1.1 (diagnostic, no rule change).** The ledger `note` for an empty fetch now says which of
three things happened: `dataset ends <T> (before window)` (ERDDAP 404 outside actual_range),
`<N> rows in window, 0 usable chl readings (NaN or outside [0, 1000))`, or the old
`feed returned no readings in window` (bare header). Status logic is unchanged.

**Health check that triggered this (last 10 days to 2026-09-06).** Current: WLIS, EXRX, MSC, MAB,
mlml, tiburon, morro, humboldt, vero-beach (last obs 2026-09-06), oa2 (09-05). Not current:
kac_ss (dataset ends 2026-07-31; chl NaN all of 2026), kac_h3 (dataset ends 2026-08-26; chl NaN
since 2026-05-12, all flags 2), banana-river (every variable NaN since 2026-07-01, dataset still
updating hourly), scwharf1 (dataset ends 2026-08-19), newport (b), scripps (a), RIV and SPS
(Eyes on the Bay last rows 2026-08-25), AWS and AES (last rows 2026-09-02).

**p75 before/after (1.0 -> 1.1).** Only Scripps changes by design; the other 19 stations were
re-frozen on the same seeds plus the readings appended by the dry runs, and their values are
listed in section 3 (any drift there comes from those appended days, not from a rule change).

### 1.2, 2026-09-06 (expectation band only; no data, site, threshold or rule change)

The lis_buoy expected onset-lift band was set at 2-3x from section 12 (recipe retrained on the
buoys at a p95 label). A retrospective zero-shot run of the frozen model on the same buoys at the
protocol's own p75 label (`src/transfer/eval_lis_buoys.py`, findings 25.1) gives EXRX 1.9-2.5x
[CI 1.3-4.6] and WLIS 1.1-1.5x [1.0-1.7] at thresholds 0.35-0.50. The band is revised to
**1.1-2.5x**, and WLIS and EXRX are to be reported as separate strata once each passes the
n >= 30 / 5-positive gate. Made before any issuance; `data/prospective/ledger.csv` does not exist.

