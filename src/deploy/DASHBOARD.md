# HAB Early Warning: Daily Inference

**21-day chlorophyll exceedance warnings for Long Island Sound, locked pipeline**
Vihaan Goyal, Westhill High School, Stamford, Connecticut

This replaces the earlier XGBoost/7-day dashboard. All numbers below come from
the locked logistic regression evaluated out-of-sample; the previous version's
figures (0.936 AUC, 7-day horizon, aeration scores) are superseded.

---

## What the system does

For a target date, `daily_inference.py`:

1. Loads the canonical dataset through `src/models/locked_pipeline.py`, the
   single source of truth shared with the evaluation harness. No feature
   engineering is duplicated in the deploy layer.
2. Trains the locked model (LogisticRegression, C=0.05, balanced, 35 features)
   on every station visit whose 21-day label window closed on or before the
   target date. Walk-forward: no future information.
3. Scores the most recent visit at each station. Visits older than 45 days are
   reported as stale and not scored.
4. Issues an alert where P(exceedance within 21 days) >= t* = 0.35.
5. Writes `data/daily_predictions.csv`.

An alert means: **prioritize a water sample at this station within the next
three weeks.**

## Operating characteristics

The alert threshold t* = 0.35 was frozen by a pre-registered rule (highest
threshold reaching POD >= 0.8 on out-of-sample 2020 to 2022 predictions) and
evaluated once on out-of-sample 2023 to 2025:

| POD | FAR | Precision | CSI |
|---|---|---|---|
| 0.875 [0.750, 0.962] | 0.875 [0.828, 0.923] | 0.125 [0.077, 0.172] | 0.122 |

Roughly 1 in 8 alerts precedes a verified exceedance, a 2.7x lift over the
4.6% base rate, while 42 of 48 test-period events were flagged three weeks
ahead. The high false alarm ratio is a deliberate cost-asymmetry choice: a
missed bloom carries ecological and shellfish-industry costs, while a false
alarm prompts a sample at a station where, about half the time, no sample
would otherwise occur in the window.

## Running it

```
conda activate hab
python src/deploy/daily_inference.py                    # today
python src/deploy/daily_inference.py --date 2025-06-01  # backtest a date
```

Output columns: `station_name`, `date` (latest visit used), `days_old`,
`bloom_prob`, `alert`.

## Honest limitations, visible by design

- **Coverage is bounded by monitoring cadence.** The system can only score
  stations with a recent visit; on a typical date a large fraction of the 50
  stations are stale. This is the same sampling limitation that caps alert
  precision (median inter-sample gap equals the 21-day horizon).
- **Biomass, not toxins.** The target is chlorophyll-a exceedance, a
  eutrophication signal that skews to western LIS. State-monitored
  toxin-producing HAB species concentrate in the east and are not predicted
  by this system.
- **Aeration scoring is omitted** pending the intervention framework rerun on
  corrected data. The previous dashboard's intervention flags should not be
  used.

## Provenance

Model spec, feature list, data loading, and label definition live in
`src/models/locked_pipeline.py`. Threshold selection: `scripts/warning/warning_operating_point.py`
(sweep in `data/warning_operating_point_locked.csv`). Robustness and CIs:
`scripts/warning/warning_robustness.py`. Headline reproduction:
`python src/models/rolling_origin_cv.py --horizon 21` (pooled AUC 0.852).