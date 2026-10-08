# Pre-registration: harbor early-warning performance (Narragansett v2 GB)

Written 2026-10-08, before any of the event-based numbers below were computed.

## Question
At harbor scale the device cannot treat the water; its product is the warning. Does the
Narragansett forecast give people (shellfish growers, hatcheries, harbormasters) usable
days of notice before a bloom starts, at a false-alarm rate they would tolerate?

## Data and model
- `data/narragansett_daily_features.csv`, tier-A features, GB settings of `export_model.py`.
- Rolling-origin folds exactly as `src/models/rolling_origin_cv_nar.py`: test year T in
  2015-2023, train <= T-2, validation T-1, threshold t* = the fold's validation-F1 choice
  (leak-free). Secondary only: the frozen device threshold 0.45 (it is the median of these
  same fold t* values, so it carries mild look-ahead; reported, not used for the pass rule).

## Definitions (fixed now)
- **Bloom onset:** station-day d with daily-mean chl > 10 ug/L whose previous 5 calendar
  days are all observed with chl <= 10.
- **Warned:** an alert (p >= t*) on at least one of days d-7 .. d-1. **Lead time** =
  d minus the earliest alert day in that window (1-7 days). Day-0 alerts do not count.
- **Alert episode:** a run of alert days at one station with no observed non-alert day
  between them. **False alarm:** an episode that starts on a day with chl <= 10 and is not
  followed by chl > 10 at that station from its start through its last day + 7.
- **Season:** May-October. False alarms are counted per station-season (station-year with
  >= 60 observed season days).
- **Baseline:** "chlorophyll rule" alert = today's chl >= X, X chosen on the validation
  year by the same F1 rule, same events and episodes.

## Pass rule (all three, pooled over 2015-2023 test years, fold t*)
1. >= 60% of onsets warned at least 1 day ahead.
2. Median false-alarm episodes per station-season <= 3.
3. Share of onsets warned >= the chlorophyll-rule baseline's share.

Uncertainty: station-year clustered bootstrap, 2,000 resamples, seed 42, for the warned
share and median lead time. Script: `src/deploy/harbor_warning_eval.py`.
Whatever the result, it is reported as is.

## Result (run 2026-10-08 19:05, after this file was hashed: sha256 d7226e2c...)
Model 85.4% warned [81.1, 89.4], median 1 false alarm per station-season; chlorophyll rule 95.2%, median 3.
Rules 1 and 2 pass, rule 3 fails: **FAIL**. Details and the exploratory matched-false-alarm comparison:
`notes/NARRAGANSETT_FINDINGS.md` §31.
