"""
harbor_warning_eval.py -- event-based early-warning performance of the Narragansett forecast
------------------------------------------------------------------------------------------
Pre-registered in notes/HARBOR_WARNING_PREREG.md (definitions and pass rule fixed before running).

At harbor scale the product is the warning, so this scores EVENTS, not days:
  - share of bloom onsets warned 1-7 days ahead, and the lead time
  - false-alarm episodes per station-season (May-Oct)
  - the same for a chlorophyll-rule baseline (alert when today's chl >= X, X chosen on val)

Folds as src/models/rolling_origin_cv_nar.py: test T in 2015-2023, train <= T-2, val T-1,
GB tier A, t* = val-F1 choice. Secondary: frozen device threshold 0.45.

Run from fork root, BASE env:  python src/deploy/harbor_warning_eval.py
Writes data/harbor_warning_events.csv, data/harbor_warning_summary.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BLOOM = 10.0
HORIZON = 7
QUIET_DAYS = 5
TEST_YEARS = range(2015, 2024)
THRESHOLDS = np.round(np.arange(0.10, 0.9001, 0.05), 2)
CHL_RULE_GRID = np.round(np.arange(2.0, 15.01, 0.5), 1)
DEVICE_T = 0.45
SEASON = range(5, 11)
MIN_SEASON_DAYS = 60
N_BOOT, SEED = 2000, 42

# Tier-A feature list and GB settings, copied from src/deploy/export_model.py
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']
GB_KW = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50,
             l2_regularization=1.0, random_state=42, class_weight="balanced")


def f1_at(y, alert):
    tp = int((alert & (y == 1)).sum()); fp = int((alert & (y == 0)).sum()); fn = int((~alert & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def best_t(y, score, grid):
    f1s = [f1_at(y, score >= t) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def station_events(s, alert_col):
    """s: one station's test-year rows sorted by date (date, chl, alert col).
    Returns (onsets, episodes) as lists of dicts."""
    s = s.set_index("date")
    chl = s.chl
    alert = s[alert_col]
    onsets = []
    for d, c in chl.items():
        if not (c > BLOOM):
            continue
        prev = pd.date_range(d - pd.Timedelta(days=QUIET_DAYS), d - pd.Timedelta(days=1))
        pc = chl.reindex(prev)
        if pc.isna().any() or (pc > BLOOM).any():
            continue
        win = pd.date_range(d - pd.Timedelta(days=HORIZON), d - pd.Timedelta(days=1))
        wa = alert.reindex(win)
        hit = wa[wa == True]  # noqa: E712
        lead = int((d - hit.index.min()).days) if len(hit) else 0
        onsets.append(dict(date=d, warned=len(hit) > 0, lead=lead))
    runs, run = [], []
    for d, a in alert.items():          # observed days only: gaps do not break a run
        if a:
            run.append(d)
        elif run:
            runs.append(run); run = []
    if run:
        runs.append(run)
    eps = []
    for r in runs:
        start, end = r[0], r[-1]
        if chl.loc[start] > BLOOM:       # alert during an ongoing bloom: neither true nor false
            continue
        after = chl.loc[start:end + pd.Timedelta(days=HORIZON)]
        eps.append(dict(start=start, false=not (after > BLOOM).any()))
    return onsets, eps


def score_system(test, alert_col, label):
    on_rows, season_rows = [], []
    for st, s in test.groupby("station"):
        s = s.sort_values("date")
        onsets, eps = station_events(s[["date", "chl", alert_col]], alert_col)
        yr = int(s.date.dt.year.iloc[0])
        for o in onsets:
            on_rows.append(dict(system=label, station=st, year=yr, cluster=f"{st}_{yr}", **o))
        n_season = int(s.date.dt.month.isin(SEASON).sum())
        if n_season >= MIN_SEASON_DAYS:
            nf = sum(1 for e in eps if e["false"] and e["start"].month in SEASON)
            season_rows.append(dict(system=label, station=st, year=yr, false_eps=nf))
    return on_rows, season_rows


def boot_ci(df, fn):
    rng = np.random.default_rng(SEED)
    groups = {k: g for k, g in df.groupby("cluster")}
    keys = np.array(list(groups))
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(keys, len(keys), replace=True)
        vals.append(fn(pd.concat([groups[k] for k in pick])))
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
    df["year"] = df.date.dt.year
    lab = df.dropna(subset=["bloom_fwd"]).copy()
    lab["bloom_fwd"] = lab.bloom_fwd.astype(int)

    onsets, seasons = [], []
    for T in TEST_YEARS:
        train, val = lab[lab.year <= T - 2], lab[lab.year == T - 1]
        test = df[(df.year == T) & df.chl.notna()].copy()
        med = train[TIER_A].median(numeric_only=True)
        m = HistGradientBoostingClassifier(**GB_KW).fit(train[TIER_A].fillna(med).values, train.bloom_fwd.values)
        pv = m.predict_proba(val[TIER_A].fillna(med).values)[:, 1]
        t_star = best_t(val.bloom_fwd.values, pv, THRESHOLDS)
        x_rule = best_t(val.bloom_fwd.values, val.chl.values, CHL_RULE_GRID)
        test["p"] = m.predict_proba(test[TIER_A].fillna(med).values)[:, 1]
        test["a_model"] = test.p >= t_star
        test["a_device"] = test.p >= DEVICE_T
        test["a_rule"] = test.chl >= x_rule
        print(f"T={T}: t*={t_star:.2f}  chl rule X={x_rule:.1f}  test days={len(test)}")
        for col, label in (("a_model", "model_tstar"), ("a_device", "model_0.45"), ("a_rule", "chl_rule")):
            o, s = score_system(test, col, label)
            for r in o:
                r["test_year"] = T
            onsets += o; seasons += s

    ev = pd.DataFrame(onsets)
    se = pd.DataFrame(seasons)
    ev.to_csv("data/harbor_warning_events.csv", index=False)
    rows = []
    for label in ("model_tstar", "model_0.45", "chl_rule"):
        e = ev[ev.system == label]
        f = se[se.system == label]
        lead = e.loc[e.warned, "lead"]
        w_ci = boot_ci(e, lambda d: d.warned.mean())
        l_ci = boot_ci(e, lambda d: d.loc[d.warned, "lead"].median() if d.warned.any() else np.nan)
        rows.append(dict(system=label, n_onsets=len(e), warned_share=e.warned.mean(),
                         warned_lo=w_ci[0], warned_hi=w_ci[1],
                         lead_median=lead.median(), lead_lo=l_ci[0], lead_hi=l_ci[1],
                         lead_ge3_share=(e.lead >= 3).mean(),
                         n_station_seasons=len(f), false_eps_median=f.false_eps.median(),
                         false_eps_mean=f.false_eps.mean()))
    summ = pd.DataFrame(rows)
    summ.to_csv("data/harbor_warning_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))

    m_ = summ.set_index("system")
    c1 = m_.loc["model_tstar", "warned_share"] >= 0.60
    c2 = m_.loc["model_tstar", "false_eps_median"] <= 3
    c3 = m_.loc["model_tstar", "warned_share"] >= m_.loc["chl_rule", "warned_share"]
    print(f"\nPASS RULE: (1) warned >= 60%: {c1}  (2) median false eps/season <= 3: {c2}  "
          f"(3) >= chl rule: {c3}  ->  {'PASS' if (c1 and c2 and c3) else 'FAIL'}")


if __name__ == "__main__":
    main()
