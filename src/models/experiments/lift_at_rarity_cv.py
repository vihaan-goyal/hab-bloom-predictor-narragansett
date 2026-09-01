"""W8 lift-at-rarity pooled CV: does daily (Narragansett sonde) sampling give
much higher LIFT than the LIS boat network at LIS-level bloom rarity?

LIS reference (h=21 forward label, boat network, base 0.046): precision 0.136,
lift 2.7, AUC 0.875.  A single-year test (2023, 21 positives) put Narragansett
daily-sampling lift at ~16x with matched precision; this script re-asks with
nine rolling-origin test years and station-year clustered bootstrap CIs.

Design
  Bloom thresholds T in {10, 20, 39, 52.5} ug/L (52.5 ~ LIS 5% train-period
  h21 positive rate; 39 ~ 10%); horizons h in {7, 21}.  For each (T, h) the
  forward label is rebuilt with the standard routine: 1 if any daily-mean chl
  > T on a day in (d, d+h], 0 if d+h <= station's last date, else NaN.
  Rolling-origin folds, test year Y in 2015..2023: train year <= Y-2,
  val = Y-1, test = Y.  Models: GB (HistGradientBoosting, locked spec) and LR
  (C=0.05, balanced, StandardScaler); train-median imputation.
  t* per fold on VAL ONSET rows (today's chl <= T): maximise lift subject to
  POD >= 0.6; if infeasible, maximise F1.  Folds with < 5 val onset positives
  are skipped.  Out-of-fold TEST ONSET predictions are pooled per (T, h,
  model) with each fold's own t*.  Station-year clustered bootstrap (n=2000,
  seed 42) gives 95% CIs for precision, lift, AUC and the top-decile metrics.
  Top-decile alert = within each fold's test onset rows, alert on the 10%
  highest probabilities (threshold-free; per-fold so probability scales are
  not mixed across folds).

Output: data/lift_at_rarity_cv.csv (one row per T, h, model).
Run from repo root with the BASE anaconda python (not the hab env):
  ~/anaconda3/python.exe src/models/experiments/lift_at_rarity_cv.py
Conventions copied from src/models/rolling_origin_cv_nar.py and
src/models/experiments/cadence_thinning_matched.py (no imports from them).
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BLOOM_THRESHOLDS = (10.0, 20.0, 39.0, 52.5)
HORIZONS = (7, 21)
TEST_YEARS = range(2015, 2024)
PROB_THRESHOLDS = np.round(np.arange(0.05, 0.9501, 0.05), 2)
MIN_VAL_POS = 5
POD_FLOOR = 0.6
TOP_FRAC = 0.10
N_BOOT = 2000
SEED = 42
OUT = "data/lift_at_rarity_cv.csv"
LIS_REF = dict(precision=0.136, lift=2.7, auc=0.875, base_rate=0.046)

# Tier-A feature list, copied verbatim from src/models/train_narragansett.py
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']


def forward_label(df, horizon, bloom):
    """Standard forward label: any chl > bloom in (d, d+h]; 0 if d+h <= last
    station date; NaN otherwise (right-censored)."""
    lab = np.full(len(df), np.nan)
    for st, g in df.groupby("station"):
        ix = np.flatnonzero((df["station"] == st).values)
        d = g["date"].values.astype("datetime64[D]")          # sorted per station
        cum = np.concatenate([[0], np.cumsum(g["chl"].values > bloom)])
        end = d + np.timedelta64(horizon, "D")
        lo = np.searchsorted(d, d, side="right")
        hi = np.searchsorted(d, end, side="right")
        pos = (cum[hi] - cum[lo]) > 0
        lab[ix] = np.where(pos, 1.0, np.where(end <= d.max(), 0.0, np.nan))
    return lab


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


def f1_from(m):
    p, r = m["precision"], m["pod"]
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def choose_threshold(yv, pv):
    """Max lift s.t. POD >= POD_FLOOR on val onset rows; fallback max F1."""
    ms = [metrics(yv, pv >= t) for t in PROB_THRESHOLDS]
    feas = [(m["lift"], t) for m, t in zip(ms, PROB_THRESHOLDS)
            if not np.isnan(m["pod"]) and m["pod"] >= POD_FLOOR and not np.isnan(m["lift"])]
    if feas:
        lift, t = max(feas)
        return float(t), "max_lift|POD>=0.6", float(lift)
    f1s = [f1_from(m) for m in ms]
    return float(PROB_THRESHOLDS[int(np.argmax(f1s))]), "max_F1(fallback)", float(max(f1s))


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def fit_model(name, Xtr, ytr):
    if name == "LR":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000,
                               random_state=SEED).fit(sc.transform(Xtr), ytr)
        return lambda X: m.predict_proba(sc.transform(X))[:, 1]
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                       min_samples_leaf=50, l2_regularization=1.0,
                                       random_state=SEED, class_weight="balanced").fit(Xtr, ytr)
    return lambda X: m.predict_proba(X)[:, 1]


def top_decile_alert(p):
    """Alert on the ceil(10%) highest probabilities (ties broken by rank)."""
    k = int(np.ceil(TOP_FRAC * len(p)))
    alert = np.zeros(len(p), dtype=int)
    if k > 0:
        alert[np.argsort(-p, kind="stable")[:k]] = 1
    return alert


def clustered_bootstrap(s, rng):
    clusters = s.cluster.values
    uniq = np.unique(clusters)
    idx_by_cl = {c: np.flatnonzero(clusters == c) for c in uniq}
    y, p, a, a10 = s.y.values, s.p.values, s.alert.values, s.alert_top10.values
    boots = {k: [] for k in ("precision", "lift", "auc", "top10_precision", "top10_lift")}
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by_cl[c] for c in pick])
        bm = metrics(y[ix], a[ix]); b10 = metrics(y[ix], a10[ix])
        boots["precision"].append(bm["precision"]); boots["lift"].append(bm["lift"])
        boots["auc"].append(safe_auc(y[ix], p[ix]))
        boots["top10_precision"].append(b10["precision"]); boots["top10_lift"].append(b10["lift"])
    out = {}
    for k, v in boots.items():
        v = np.asarray(v, dtype=float)
        out[f"{k}_ci_lo"] = np.nanpercentile(v, 2.5); out[f"{k}_ci_hi"] = np.nanpercentile(v, 97.5)
    out["n_clusters"] = len(uniq)
    return out


def main():
    t0 = time.time()
    df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    df["year"] = df.date.dt.year
    df["cluster"] = df.station.astype(str) + "_" + df.year.astype(str)

    fold_rows, pooled_rows = [], []
    rng = np.random.default_rng(SEED)
    for T in BLOOM_THRESHOLDS:
        for h in HORIZONS:
            df["y"] = forward_label(df, h, T)
            lab = df.dropna(subset=["y"]).copy()
            lab["y"] = lab.y.astype(int)
            lab["onset"] = lab.chl <= T
            train_rate = lab[lab.year <= 2020].y.mean()
            oof = {"LR": [], "GB": []}
            for Y in TEST_YEARS:
                train = lab[lab.year <= Y - 2]; val = lab[lab.year == Y - 1]; test = lab[lab.year == Y]
                ov, ot = val.onset.values, test.onset.values
                val_on_pos = int(val.y.values[ov].sum())
                if len(train) == 0 or train.y.nunique() < 2 or val_on_pos < MIN_VAL_POS or len(test) == 0:
                    print(f"T={T} h={h} skip Y={Y}: n_train={len(train)} val_onset_pos={val_on_pos} "
                          f"n_test={len(test)}", flush=True)
                    continue
                med = train[TIER_A].median(numeric_only=True)
                prep = lambda d: d[TIER_A].fillna(med).values
                ytr, yv, yt = train.y.values, val.y.values, test.y.values
                for mn in ("LR", "GB"):
                    predict = fit_model(mn, prep(train), ytr)
                    pv, pt = predict(prep(val)), predict(prep(test))
                    t_star, rule, val_obj = choose_threshold(yv[ov], pv[ov])
                    mv = metrics(yv[ov], pv[ov] >= t_star)
                    mt = metrics(yt[ot], pt[ot] >= t_star)
                    a10 = top_decile_alert(pt[ot]); m10 = metrics(yt[ot], a10)
                    fold_rows.append(dict(
                        bloom_T=T, horizon=h, model=mn, test_year=Y, t_star=t_star, rule=rule,
                        val_onset_pos=val_on_pos, val_precision=mv["precision"], val_pod=mv["pod"],
                        val_lift=mv["lift"], test_onset_n=mt["n"], test_onset_pos=mt["n_pos"],
                        test_precision=mt["precision"], test_pod=mt["pod"], test_base=mt["base_rate"],
                        test_lift=mt["lift"], test_auc=safe_auc(yt[ot], pt[ot]),
                        top10_precision=m10["precision"], top10_lift=m10["lift"]))
                    oof[mn].append(pd.DataFrame(dict(
                        test_year=Y, cluster=test.cluster.values[ot], y=yt[ot], p=pt[ot],
                        alert=(pt[ot] >= t_star).astype(int), alert_top10=a10)))
            for mn in ("LR", "GB"):
                if not oof[mn]:
                    continue
                s = pd.concat(oof[mn], ignore_index=True)
                pm = metrics(s.y.values, s.alert.values); p10 = metrics(s.y.values, s.alert_top10.values)
                row = dict(bloom_T=T, horizon=h, model=mn, n_folds=s.test_year.nunique(),
                           train_rate_le2020_alldays=train_rate,
                           n_onset=pm["n"], n_pos=pm["n_pos"], base_rate=pm["base_rate"],
                           precision=pm["precision"], pod=pm["pod"], lift=pm["lift"],
                           auc=safe_auc(s.y.values, s.p.values), tp=pm["tp"], fp=pm["fp"], fn=pm["fn"],
                           top10_precision=p10["precision"], top10_pod=p10["pod"], top10_lift=p10["lift"],
                           top10_n_alert=int(s.alert_top10.sum()))
                row.update(clustered_bootstrap(s, rng))
                pooled_rows.append(row)
                print(f"T={T} h={h} {mn}: n_onset={pm['n']} n_pos={pm['n_pos']} base={pm['base_rate']:.4f} "
                      f"prec={pm['precision']:.3f} POD={pm['pod']:.3f} lift={pm['lift']:.2f} "
                      f"[{row['lift_ci_lo']:.2f},{row['lift_ci_hi']:.2f}] AUC={row['auc']:.3f} "
                      f"top10 prec={p10['precision']:.3f} lift={p10['lift']:.2f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT, index=False)
    folds = pd.DataFrame(fold_rows)

    pd.set_option("display.width", 250)
    ff = lambda x: f"{x:.3f}"
    print("\n=== Per-fold test ONSET metrics (t* from val onset rows: max lift s.t. POD>=0.6, else max F1) ===")
    print(folds.to_string(index=False, float_format=ff))

    print("\n=== Pooled out-of-fold test ONSET rows, one row per (T, h, model) ===")
    cols = ["bloom_T", "horizon", "model", "n_folds", "n_onset", "n_pos", "base_rate", "precision",
            "precision_ci_lo", "precision_ci_hi", "pod", "lift", "lift_ci_lo", "lift_ci_hi",
            "auc", "auc_ci_lo", "auc_ci_hi", "top10_precision", "top10_precision_ci_lo",
            "top10_precision_ci_hi", "top10_lift", "top10_lift_ci_lo", "top10_lift_ci_hi", "top10_pod"]
    print(pooled[cols].to_string(index=False, float_format=ff))

    print(f"\n=== Compare to LIS boat network (h21): precision {LIS_REF['precision']}, lift "
          f"{LIS_REF['lift']}, AUC {LIS_REF['auc']}, base {LIS_REF['base_rate']} ===")
    for T in (52.5, 39.0):
        g = pooled[(pooled.bloom_T == T) & (pooled.horizon == 21) & (pooled.model == "GB")]
        if not len(g):
            print(f"T={T} h=21 GB: no pooled row"); continue
        r = g.iloc[0]
        print(f"T={T:>5} h=21 GB: base={r.base_rate:.3f} (LIS 0.046) | precision={r.precision:.3f} "
              f"[{r.precision_ci_lo:.3f},{r.precision_ci_hi:.3f}] (LIS 0.136) | lift={r.lift:.2f} "
              f"[{r.lift_ci_lo:.2f},{r.lift_ci_hi:.2f}] (LIS 2.7) | AUC={r.auc:.3f} "
              f"[{r.auc_ci_lo:.3f},{r.auc_ci_hi:.3f}] (LIS 0.875) | POD={r.pod:.3f} | "
              f"top-decile precision={r.top10_precision:.3f} lift={r.top10_lift:.2f} "
              f"[{r.top10_lift_ci_lo:.2f},{r.top10_lift_ci_hi:.2f}] | n_onset={int(r.n_onset)} "
              f"n_pos={int(r.n_pos)} folds={int(r.n_folds)} | lift CI lower bound > 2.7: "
              f"{r.lift_ci_lo > LIS_REF['lift']}")
    print(f"\nWrote {OUT} ({len(pooled)} rows). Runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
