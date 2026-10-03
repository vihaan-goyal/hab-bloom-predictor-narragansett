# Pre-registration: does the Narragansett model beat the calendar on bloom starts? (written 2026-09-28, before running)

**Question.** On onset days (today's daily-mean chl ≤ 10 µg/L), does the Narragansett model rank
"bloom within 7 days" better than a calendar forecast built only from past years? Persistence
cannot alert on onset days, so the calendar is the natural baseline.

**Data.**
- The walk-forward out-of-fold predictions written by `src/models/rolling_origin_cv_nar.py`, after
  the 2026-09-28 leak and label fixes (causal climatology; label NaN when the 7-day window has
  fewer than 4 observed days).
- Model: GB tier A, as that script fits it.
- Each fold: train on years < T with the script's own buffer, test on year T.
- Only onset rows are scored.

**Calendar baseline (fixed now).**
- For each fold year T and test row (station s, date d), the forecast is the bloom-within-7-days
  rate at station s in d's 15-day day-of-year bin.
- It uses only rows dated before T minus 7 days (every such label has resolved).
- Same label as the model, onset rows only.
- Fallbacks, each needing at least 5 rows: station × bin → all stations × bin → the overall past
  onset rate.

**Primary test.** Pooled onset AUC of the model minus the calendar, with a paired
station-year clustered bootstrap (2,000 resamples, seed 42). **The model beats the calendar if the
one-sided p = P(difference ≤ 0) is < 0.05.**

**Secondary.**
- The most recent fold (2023) alone.
- The per-fold difference, and the number of folds the model wins.
- Lift at the top 10% of each fold's onset rows, for both.

**Whatever the result, it is reported as is.** No change to the model, label or baseline after
seeing it.

## Result (run 2026-09-28, after the rule above was written and after the fork audit fixes; `src/models/calendar_baseline_nar.py`, `data/calendar_baseline_nar.log`)
- **Primary: PASS.** Pooled onset rows 2015-2023, 14,703 rows, 3,964 events: model AUC **0.877** vs calendar **0.817**, difference **+0.060 [+0.047, +0.074]**, one-sided p < 0.0001.
- **Per fold:** the model wins **9 of 9** years (+0.009 to +0.083).
- **Most recent fold, 2023:** 0.835 vs 0.768, **+0.066 [+0.018, +0.128]**, p = 0.0005.
- **Lift at the top 10% of each fold:** model 3.05 vs calendar 2.61 (2023: 2.40 vs 1.83).

**Reading.** With daily sensor data, the model forecasts bloom starts clearly better than the season alone, every year, including the most recent. Contrast with Long Island Sound, whose 3-weekly boat data is no better than the calendar in 2023-25 (parent `notes/CALENDAR_BASELINE_PREREG.md`; after the 2026-10-02 lab-scale fix the calendar is ahead there, not significantly).
