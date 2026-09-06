# Prospective forecast protocol (pre-registration)

**protocol_version 1.0** | frozen_at 2026-09-06T00:37:39Z (UTC) | written 2026-09-05

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
| nerrs | kac_ss, kac_h3 | IOOS Sensors ERDDAP `nerrs_kac{ss,h3}wq`, QARTOD agg in {1,2} | 48 | ug/L ChlFluor |
| chesapeake | MSC, AWS, AES, MAB, RIV, SPS | MD DNR Eyes on the Bay `JustDownload.cfm` | 48 | ug/L TChlPreCal |
| erddap_top | edu_ucsc_scwharf1, scripps-pier-automated-shore-sta-1, oa2-mbari-buoy, newport-pier-automated-shore-sta, mlml_mlml_sea, tiburon-water-tibc1, edu_calpoly_marine_morro, edu_humboldt_humboldt (48); indian-river-lagoon-banana-river, indian-river-lagoon-vero-beach-i (12) | IOOS Sensors ERDDAP | 48 / 12 | site fluorometer units |

## 3. Frozen thresholds (output of `python -m src.deploy.prospective_freeze`, 2026-09-06T00:37Z)

`chl_p75_site` = 75th percentile of daily-mean chl over the station's cached record
(LIS: night-only, stuck-sensor days with >50% identical readings removed). Stored in
`data/prospective/site_p75.csv`; the script refuses to overwrite it without `--force`.

| site_group | site_id | chl_p75_site | n_station_days | history_first | history_last | min_readings | t_star_site |
|---|---|---|---|---|---|---|---|
| lis_buoy | WLIS_ECO_FL | 249.58 | 1656 | 2021-03-05 | 2026-07-09 | 12 | |
| lis_buoy | EXRX_ECO_FL | 384.68 | 1798 | 2019-05-15 | 2026-07-09 | 12 | |
| nerrs | kac_ss | 3.11 | 3571 | 2011-01-27 | 2025-12-31 | 48 | |
| nerrs | kac_h3 | 2.77 | 2021 | 2012-06-28 | 2026-05-11 | 48 | |
| chesapeake | MSC | 23.40 | 3843 | 2013-03-28 | 2025-12-31 | 48 | |
| chesapeake | AWS | 25.84 | 3411 | 2016-05-26 | 2025-12-31 | 48 | |
| chesapeake | AES | 25.37 | 3261 | 2016-05-25 | 2025-12-31 | 48 | |
| chesapeake | MAB | 10.86 | 1681 | 2018-04-06 | 2024-12-16 | 48 | |
| chesapeake | RIV | 80.07 | 2208 | 2014-04-10 | 2025-11-10 | 48 | |
| chesapeake | SPS | 18.66 | 2416 | 2010-04-15 | 2025-11-06 | 48 | |
| erddap_top | edu_ucsc_scwharf1 | 27.27 | 2327 | 2013-02-14 | 2026-08-17 | 48 | 0.90 |
| erddap_top | scripps-pier-automated-shore-sta-1 | 17.78 | 4769 | 2013-01-22 | 2026-09-01 | 48 | 0.85 |
| erddap_top | oa2-mbari-buoy | 10.30 | 2770 | 2015-05-14 | 2026-09-04 | 48 | 0.85 |
| erddap_top | newport-pier-automated-shore-sta | 13.32 | 4502 | 2013-01-24 | 2026-08-11 | 48 | 0.80 |
| erddap_top | mlml_mlml_sea | 3.78 | 5308 | 2010-09-03 | 2026-09-05 | 48 | 0.60 |
| erddap_top | tiburon-water-tibc1 | 3.82 | 4806 | 2008-08-08 | 2026-09-04 | 48 | 0.70 |
| erddap_top | edu_calpoly_marine_morro | 2.86 | 3318 | 2016-01-02 | 2026-09-05 | 48 | 0.65 |
| erddap_top | edu_humboldt_humboldt | 4.03 | 2947 | 2013-02-15 | 2026-09-03 | 48 | 0.35 |
| erddap_top | indian-river-lagoon-banana-river | 4.39 | 1618 | 2022-01-01 | 2026-07-01 | 12 | 0.60 |
| erddap_top | indian-river-lagoon-vero-beach-i | 3.87 | 3937 | 2014-12-17 | 2026-09-04 | 12 | 0.75 |

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
t_star_site` for erddap_top (the threshold each site's retrospective test chose);
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
python -m src.deploy.prospective_freeze          # done once (2026-09-06T00:37Z); --force to redo
python -m src.deploy.score_ledger                # verify due rows, write scored.csv + skill_summary.csv
python -m src.deploy.prospective_forecast        # issue today's rows (exit 2 if date already issued)
scripts\run_prospective.cmd                      # both steps, appending to data\prospective\logs\last_run.log
```

Tracked: `data/prospective/{site_p75.csv, ledger.csv, scored.csv, skill_summary.csv, issued/, outbox/}`,
`notes/prospective/digest_<D>.md`. Ignored caches: `data/prospective/{raw,history,logs}/`.
No scheduled task is registered yet.
