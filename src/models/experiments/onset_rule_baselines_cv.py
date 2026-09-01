"""Extend rule baselines to rolling-CV folds (W2 follow-up).

Resolves the e2 ambiguity from onset_rule_baselines.py (single test year 2023:
onset-climatology rule's CI vs GB included 0) by pooling nine test years.

Folds (identical to src/models/rolling_origin_cv_nar.py, not imported):
  test year T in 2015..2023; train = year <= T-2; val = year == T-1; test = T.
  Fold skipped if test has no positives or val has < 5 positives.

Per fold:
  GB tier A (locked spec, train-median impute), alert at t* = argmax val F1
    over 0.10..0.90 step 0.05 on ALL val days (W4's pooled-onset convention).
  Rule a   chl > c, c in 2..10 step 0.25, argmax val-onset F1
  Rule a2  chl > c, c with val-onset POD >= 0.6 and max val-onset precision
  Rule e2  station x 15-day-DOY bloom rate from that fold's TRAIN ONSET rows
           (chl <= 10), alert if rate >= 0.5; unseen cell -> train-onset mean
Test onset rows (chl <= 10) scored, pooled across folds.

Pooled metrics per method + paired station-YEAR clustered bootstrap
(cluster = station + "_" + year, n_boot=2000, seed=42) of lift(GB)-lift(rule).

Output: data/onset_rule_baselines_cv.csv
Run from repo root with the BASE anaconda python (not the hab env).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BLOOM = 10.0
TEST_YEARS = range(2015, 2024)
THRESHOLDS = np.round(np.arange(0.10, 0.9001, 0.05), 2)
N_BOOT = 2000
SEED = 42
MIN_VAL_POS = 5
C_GRID = np.arange(2.0, 10.0 + 1e-9, 0.25)

TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']


def metrics(y, alert):
    y = np.asarray(y).astype(int); alert = np.asarray(alert).astype(int)
    tp = int(((alert == 1) & (y == 1)).sum()); fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum()); n = len(y)
    prec = tp / (tp + fp) if tp + fp else np.nan
    pod = tp / (tp + fn) if tp + fn else np.nan
    base = (tp + fn) / n if n else np.nan
    lift = prec / base if base and not np.isnan(prec) else np.nan
    return dict(precision=prec, pod=pod, base_rate=base, lift=lift,
                tp=tp, fp=fp, fn=fn, n=n, n_pos=tp + fn)


def f1_of(m):
    p, r = m["precision"], m["pod"]
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def clim_alert(train_frame, frame):
    rate = (train_frame.assign(bin=(train_frame.date.dt.dayofyear - 1) // 15)
            .groupby(["station", "bin"]).bloom_fwd.mean())
    tb = frame.assign(bin=(frame.date.dt.dayofyear - 1) // 15)
    pr = tb.set_index(["station", "bin"]).index.map(rate).values
    pr = np.where(pd.isna(pr), float(train_frame.bloom_fwd.mean()), pr.astype(float))
    return pr >= 0.5


# ------------------------------------------------------------------ data
df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
df["year"] = df.date.dt.year
lab = df.dropna(subset=["bloom_fwd"]).copy()
lab["bloom_fwd"] = lab.bloom_fwd.astype(int)
lab["cluster"] = lab.station.astype(str) + "_" + lab.year.astype(str)

oof, fold_log = [], []
for T in TEST_YEARS:
    train = lab[lab.year <= T - 2]
    val = lab[lab.year == T - 1]
    test = lab[lab.year == T]
    if len(train) == 0 or test.bloom_fwd.sum() == 0 or val.bloom_fwd.sum() < MIN_VAL_POS:
        print(f"skip T={T}: train={len(train)} val_pos={val.bloom_fwd.sum()} "
              f"test_pos={test.bloom_fwd.sum()}")
        continue
    val_on = val[val.chl <= BLOOM]
    test_on = test[test.chl <= BLOOM]
    yv_on = val_on.bloom_fwd.values
    yt_on = test_on.bloom_fwd.values

    # GB, W4 convention: t* on all val days
    med = train[TIER_A].median(numeric_only=True)
    prep = lambda d: d[TIER_A].fillna(med).values
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                        max_iter=300, min_samples_leaf=50,
                                        l2_regularization=1.0, random_state=42,
                                        class_weight="balanced").fit(prep(train), train.bloom_fwd.values)
    pv = gb.predict_proba(prep(val))[:, 1]
    f1s = [f1_of(metrics(val.bloom_fwd.values, pv >= t)) for t in THRESHOLDS]
    t_star = float(THRESHOLDS[int(np.argmax(f1s))])
    pt_on = gb.predict_proba(prep(test_on))[:, 1]
    gb_alert = (pt_on >= t_star).astype(int)

    # rule a: argmax val-onset F1
    f1s = [f1_of(metrics(yv_on, val_on.chl.values > c)) for c in C_GRID]
    c_a = float(C_GRID[int(np.argmax(f1s))])
    # rule a2: val-onset POD >= 0.6, max precision
    best = None
    for c in C_GRID:
        m = metrics(yv_on, val_on.chl.values > c)
        if not np.isnan(m["pod"]) and m["pod"] >= 0.6 and not np.isnan(m["precision"]):
            if best is None or m["precision"] > best[1]:
                best = (float(c), m["precision"])
    c_a2 = best[0] if best else np.nan
    a2_alert = (test_on.chl.values > c_a2) if best else np.zeros(len(test_on), bool)
    # rule e2: onset climatology from train onset rows
    e2_alert = clim_alert(train[train.chl <= BLOOM], test_on)

    fold_log.append(dict(T=T, n_test_onset=len(test_on), test_onset_pos=int(yt_on.sum()),
                         t_star=t_star, c_a=c_a, c_a2=c_a2))
    oof.append(pd.DataFrame(dict(
        T=T, cluster=test_on.cluster.values, y=yt_on,
        GB_tierA=gb_alert,
        a_chl_gt_c_valF1=(test_on.chl.values > c_a).astype(int),
        a2_chl_gt_c_valPOD0p6=a2_alert.astype(int),
        e2_clim_station_doy15_onset=e2_alert.astype(int))))

oof = pd.concat(oof, ignore_index=True)
folds = pd.DataFrame(fold_log)
print("\nPer-fold params:")
print(folds.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# --------------------------------------------------------- pooled + boot
methods = ["GB_tierA", "a_chl_gt_c_valF1", "a2_chl_gt_c_valPOD0p6",
           "e2_clim_station_doy15_onset"]
y = oof.y.values
clusters = oof.cluster.values
uniq = np.unique(clusters)
idx_by_cl = {c: np.flatnonzero(clusters == c) for c in uniq}
rng = np.random.default_rng(SEED)
boot_idx = [np.concatenate([idx_by_cl[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
            for _ in range(N_BOOT)]

gb_al = oof["GB_tierA"].values
rows = []
for mth in methods:
    al = oof[mth].values
    m = metrics(y, al)
    if mth == "GB_tierA":
        diff, lo, hi, nn = 0.0, 0.0, 0.0, 0
        lifts = np.array([metrics(y[ix], al[ix])["lift"] for ix in boot_idx])
        l_lo, l_hi = np.nanpercentile(lifts, [2.5, 97.5])
    else:
        d = np.array([metrics(y[ix], gb_al[ix])["lift"] - metrics(y[ix], al[ix])["lift"]
                      for ix in boot_idx])
        ok = ~np.isnan(d)
        lo, hi = np.percentile(d[ok], [2.5, 97.5]); nn = int((~ok).sum())
        diff = metrics(y, gb_al)["lift"] - m["lift"]
        lifts = np.array([metrics(y[ix], al[ix])["lift"] for ix in boot_idx])
        l_lo, l_hi = np.nanpercentile(lifts, [2.5, 97.5])
    rows.append(dict(rule=mth, n=m["n"], n_pos=m["n_pos"], n_folds=oof["T"].nunique(),
                     n_clusters=len(uniq), precision=m["precision"], pod=m["pod"],
                     base_rate=m["base_rate"], lift=m["lift"], lift_ci_lo=l_lo, lift_ci_hi=l_hi,
                     tp=m["tp"], fp=m["fp"], fn=m["fn"],
                     lift_diff_vs_gb=diff, ci_lo=lo, ci_hi=hi, n_boot_nan=nn))
res = pd.DataFrame(rows)
res.to_csv("data/onset_rule_baselines_cv.csv", index=False)
pd.set_option("display.width", 220)
print("\nPooled onset rows across folds (paired station-year clustered bootstrap):")
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nVERDICT:")
for _, r in res.iloc[1:].iterrows():
    inc0 = r.ci_lo <= 0.0 <= r.ci_hi
    print(f"  {r.rule:<30s} lift={r.lift:.2f} diff={r.lift_diff_vs_gb:+.2f} "
          f"CI=[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]  "
          f"{'CI INCLUDES 0' if inc0 else 'GB reliably better'}")
