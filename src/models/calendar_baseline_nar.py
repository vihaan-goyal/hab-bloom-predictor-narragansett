"""
calendar_baseline_nar.py -- does the Narragansett model beat the calendar on bloom starts? (pre-registered)
===========================================================================================================
Implements notes/CALENDAR_BASELINE_PREREG_NAR.md. Rebuilds the GB tier-A walk-forward out-of-fold
predictions exactly as src/models/rolling_origin_cv_nar.py does (train <= T-2, val T-1, test T,
same spec and skip rule), keeps onset rows (chl <= 10), and compares them with a calendar forecast:
the station x 15-day-DOY-bin bloom-within-7-days rate from onset rows dated before Jan 1 of T minus
7 days (fallbacks: all stations x bin, then overall; each needs >= 5 rows).

Run from fork root, BASE env:  python src/models/calendar_baseline_nar.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BLOOM, HORIZON, MIN_ROWS, MIN_VAL_POS = 10.0, 7, 5, 5
TEST_YEARS = range(2015, 2024)
N_BOOT, SEED, TOP_FRAC = 2000, 42, 0.10
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']
GB_KW = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50,
             l2_regularization=1.0, random_state=42, class_weight="balanced")


def calendar(test, past):
    past = past.assign(bin=(past.date.dt.dayofyear - 1) // 15)
    sb = past.groupby(["station", "bin"]).bloom_fwd.agg(["mean", "size"])
    bb = past.groupby("bin").bloom_fwd.agg(["mean", "size"])
    overall = past.bloom_fwd.mean()
    out = []
    for s, b in zip(test.station, (test.date.dt.dayofyear - 1) // 15):
        if (s, b) in sb.index and sb.loc[(s, b), "size"] >= MIN_ROWS:
            out.append(sb.loc[(s, b), "mean"])
        elif b in bb.index and bb.loc[b, "size"] >= MIN_ROWS:
            out.append(bb.loc[b, "mean"])
        else:
            out.append(overall)
    return np.array(out, dtype=float)


def paired(y, a, b, keys, rng):
    groups = [np.where(keys == k)[0] for k in pd.unique(keys)]
    d = []
    for _ in range(N_BOOT):
        s = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if 0 < y[s].sum() < len(s):
            d.append(roc_auc_score(y[s], a[s]) - roc_auc_score(y[s], b[s]))
    d = np.array(d)
    return np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()


def top_lift(r, col):
    hits = alerts = 0
    for _, g in r.groupby("fold"):
        k = max(1, int(round(TOP_FRAC * len(g))))
        top = g.sort_values(col, ascending=False, kind="stable").head(k)
        hits += top.y.sum(); alerts += k
    return (hits / alerts) / r.y.mean()


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
    df["year"] = df.date.dt.year
    lab = df.dropna(subset=["bloom_fwd"]).copy()
    lab["bloom_fwd"] = lab.bloom_fwd.astype(int)
    lab["onset"] = lab.chl <= BLOOM
    rows = []
    for T in TEST_YEARS:
        train, val, test = lab[lab.year <= T - 2], lab[lab.year == T - 1], lab[lab.year == T]
        if len(train) == 0 or test.bloom_fwd.sum() == 0 or val.bloom_fwd.sum() < MIN_VAL_POS:
            continue
        med = train[TIER_A].median(numeric_only=True)
        m = HistGradientBoostingClassifier(**GB_KW).fit(train[TIER_A].fillna(med).values,
                                                        train.bloom_fwd.values)
        te = test[test.onset]
        past = lab[lab.onset & (lab.date < pd.Timestamp(f"{T}-01-01") - pd.Timedelta(days=HORIZON))]
        rows.append(pd.DataFrame({"fold": T, "station": te.station.values, "date": te.date.values,
                                  "y": te.bloom_fwd.values,
                                  "model": m.predict_proba(te[TIER_A].fillna(med).values)[:, 1],
                                  "calendar": calendar(te, past)}))
    r = pd.concat(rows, ignore_index=True)
    r["key"] = r.station.astype(str) + "_" + pd.to_datetime(r.date).dt.year.astype(str)
    r.to_csv("data/calendar_baseline_nar.csv", index=False)

    def report(tag, d):
        y = d.y.values
        am, ac = roc_auc_score(y, d.model), roc_auc_score(y, d.calendar)
        lo, hi, p = paired(y, d.model.values, d.calendar.values, d.key.values, rng)
        print(f"{tag}: onset rows {len(d)}, events {y.sum()} | model AUC {am:.3f}, calendar {ac:.3f}, "
              f"difference {am - ac:+.3f} [{lo:+.3f}, {hi:+.3f}], one-sided p = {p:.4f}")
        print(f"   lift at top {TOP_FRAC:.0%} per fold: model {top_lift(d, 'model'):.2f}, "
              f"calendar {top_lift(d, 'calendar'):.2f}")

    print("PRIMARY (all folds):")
    report("  pooled", r)
    print("Per fold (AUC model - calendar):")
    wins = 0
    for T, g in r.groupby("fold"):
        if 0 < g.y.sum() < len(g):
            dlt = roc_auc_score(g.y, g.model) - roc_auc_score(g.y, g.calendar)
            wins += dlt > 0
            print(f"  {T}: onset events {int(g.y.sum()):>4}  {dlt:+.3f}")
    print(f"  model wins {wins} of {r.fold.nunique()} folds")
    print("SECONDARY (most recent fold, 2023):")
    report("  2023", r[r.fold == 2023].reset_index(drop=True))


if __name__ == "__main__":
    main()
