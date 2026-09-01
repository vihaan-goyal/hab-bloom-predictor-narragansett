"""Rerun thinning at LIS-matched base rate.

Same design as cadence_thinning.py, but the bloom threshold is no longer
10 ug/L: it is the chl level T whose all-days, train-period (<= 2020), h=21
positive rate on the k=1 daily series is ~TARGET (0.05, i.e. LIS rarity;
and 0.10 as a second setting).  T is searched over 10..60 ug/L (step 0.5).
Features are unchanged; only the label (and the onset definition chl <= T)
uses T.  Onset-only test metrics for k in {1,3,7,14,21}, both label variants,
h in {7,21}, LR and GB, up to 5 phases.

Output: data/cadence_thinning_matched.csv
Run from repo root with the BASE anaconda python (not the `hab` env).
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TRAIN_MAX = 2020
VAL_YEARS = (2021, 2022)
TEST_YEARS = (2023,)
KS = (1, 3, 7, 14, 21)
N_PHASES = 5
HORIZONS = (7, 21)
TARGETS = (0.05, 0.10)
SEED = 42
OUT = "data/cadence_thinning_matched.csv"
LIS_REF = dict(precision=0.136, auc=0.875, base_rate=0.046, lift=0.136 / 0.046)

FEATURES = ["chl", "chl_lag1", "chl_lag2", "chl_lag3", "chl_lag4",
            "chl_roll3_mean", "chl_roll6_mean", "chl_roll9_mean", "chl_trend",
            "chl_climatology", "chl_anomaly",
            "do", "do_lag1", "temp", "temp_lag1",
            "sal", "sal_lag1", "sal_lag2", "sal_lag3", "sal_lag4", "month"]


def thin(full, k, phase):
    day_idx = (full["date"] - full.groupby("station")["date"].transform("min")).dt.days
    return full[(day_idx % k) == phase].copy()


def build_features(obs):
    obs = obs.sort_values(["station", "date"]).reset_index(drop=True)
    g = obs.groupby("station")
    for j in (1, 2, 3, 4):
        obs[f"chl_lag{j}"] = g["chl"].shift(j)
        obs[f"sal_lag{j}"] = g["sal"].shift(j)
    obs["do_lag1"] = g["do"].shift(1)
    obs["temp_lag1"] = g["temp"].shift(1)
    for w in (3, 6, 9):
        obs[f"chl_roll{w}_mean"] = g["chl"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean())
    obs["chl_trend"] = obs["chl"] - obs["chl_roll6_mean"]
    obs["month"] = obs["date"].dt.month
    obs["year"] = obs["date"].dt.year
    obs["doy_bin"] = ((obs["date"].dt.dayofyear - 1) // 15).astype("int64")
    clim = obs[obs.year <= TRAIN_MAX].groupby(["station", "doy_bin"])["chl"].mean()
    obs["chl_climatology"] = pd.MultiIndex.from_frame(obs[["station", "doy_bin"]]).map(clim)
    obs["chl_climatology"] = obs["chl_climatology"].astype(float)
    obs["chl_anomaly"] = obs["chl"] - obs["chl_climatology"]
    return obs


def forward_label(obs, ref, horizon, bloom, require_obs_in_window):
    lab = np.full(len(obs), np.nan)
    ref = ref.sort_values(["station", "date"])
    for st, rg in ref.groupby("station"):
        m = (obs["station"] == st).values
        if not m.any():
            continue
        rd = rg["date"].values.astype("datetime64[D]")
        bloom_cum = np.concatenate([[0], np.cumsum((rg["chl"].values > bloom))])
        t = obs.loc[m, "date"].values.astype("datetime64[D]")
        end = t + np.timedelta64(horizon, "D")
        lo = np.searchsorted(rd, t, side="right")
        hi = np.searchsorted(rd, end, side="right")
        pos = (bloom_cum[hi] - bloom_cum[lo]) > 0
        out = np.where(pos, 1.0, np.where(end <= rd.max(), 0.0, np.nan))
        if require_obs_in_window:
            out = np.where(((hi - lo) == 0) & ~pos, np.nan, out)
        lab[m] = out
    return lab


def metrics(y, alert):
    tp = int(((alert == 1) & (y == 1)).sum()); fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum()); tn = int(((alert == 0) & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    n = tp + fp + fn + tn
    base = (tp + fn) / n if n else np.nan
    return dict(precision=prec, pod=pod, base_rate=base,
                lift=prec / base if base else np.nan, tp=tp, fp=fp, fn=fn, n_pos=tp + fn)


EMPTY = dict(auc=np.nan, precision=np.nan, pod=np.nan, base_rate=np.nan, lift=np.nan,
             n_test=0, n_pos=0, t_star=np.nan, tp=0, fp=0, fn=0)


def fit_eval(train, val, test, model_name, bloom):
    med = train[FEATURES].median(numeric_only=True)

    def prep(d):
        return d[FEATURES].fillna(med).values

    ytr = train.y.values; yv = val.y.values; yt = test.y.values
    if model_name == "LR":
        sc = StandardScaler().fit(prep(train))
        m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000,
                               random_state=SEED).fit(sc.transform(prep(train)), ytr)
        pv = m.predict_proba(sc.transform(prep(val)))[:, 1]
        pt = m.predict_proba(sc.transform(prep(test)))[:, 1]
    else:
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                           min_samples_leaf=50, l2_regularization=1.0,
                                           random_state=SEED, class_weight="balanced")
        m.fit(prep(train), ytr)
        pv = m.predict_proba(prep(val))[:, 1]
        pt = m.predict_proba(prep(test))[:, 1]
    ts = np.arange(0.05, 0.96, 0.05)
    f1s = []
    for t in ts:
        mm = metrics(yv, (pv >= t).astype(int))
        p, r = mm["precision"], mm["pod"]
        good = p and r and not np.isnan(p) and not np.isnan(r)
        f1s.append(2 * p * r / (p + r) if good else 0)
    t_star = float(ts[int(np.argmax(f1s))])
    om = (test.chl <= bloom).values
    r = metrics(yt[om], (pt[om] >= t_star).astype(int))
    r["auc"] = roc_auc_score(yt[om], pt[om]) if len(np.unique(yt[om])) == 2 else np.nan
    r["n_test"] = int(om.sum()); r["t_star"] = t_star
    return r


def find_threshold(full, target):
    """Smallest-|error| T in 10..60 whose train-period all-days h21 positive rate ~ target."""
    yr = full.date.dt.year.values
    best = None
    for T in np.arange(10, 60.01, 0.5):
        lab = forward_label(full, full, 21, T, False)
        rate = np.nanmean(lab[(yr <= TRAIN_MAX) & ~np.isnan(lab)])
        if best is None or abs(rate - target) < abs(best[1] - target):
            best = (float(T), float(rate))
    return best


def main():
    t0 = time.time()
    full = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
    full = full[["station", "date", "chl", "temp", "sal", "do"]].sort_values(
        ["station", "date"]).reset_index(drop=True)
    thresholds = {}
    for tgt in TARGETS:
        T, rate = find_threshold(full, tgt)
        thresholds[tgt] = T
        print(f"target={tgt:.2f}: bloom threshold T={T:.1f} ug/L "
              f"(train all-days h21 positive rate={rate:.4f})", flush=True)

    rows = []
    for tgt in TARGETS:
        bloom = thresholds[tgt]
        rng = np.random.default_rng(SEED)
        for k in KS:
            phases = sorted(rng.choice(k, size=min(N_PHASES, k), replace=False).tolist())
            for phase in phases:
                obs = build_features(thin(full, k, phase))
                for variant in ("full_truth", "thinned_truth"):
                    ref = full if variant == "full_truth" else obs
                    for h in HORIZONS:
                        obs["y"] = forward_label(obs, ref, h, bloom,
                                                 require_obs_in_window=(variant == "thinned_truth"))
                        lab = obs.dropna(subset=["y"]).copy()
                        lab["y"] = lab.y.astype(int)
                        train = lab[lab.year <= TRAIN_MAX]
                        val = lab[lab.year.isin(VAL_YEARS)]
                        test = lab[lab.year.isin(TEST_YEARS)]
                        ok = (len(train) > 0 and len(val) > 0 and len(test) > 0
                              and train.y.nunique() == 2 and val.y.nunique() == 2)
                        for mn in ("LR", "GB"):
                            r = fit_eval(train, val, test, mn, bloom) if ok else dict(EMPTY)
                            rows.append(dict(target_rate=tgt, bloom_threshold=bloom, k=k,
                                             phase=phase, label_variant=variant, horizon=h,
                                             model=mn, auc=r["auc"], precision=r["precision"],
                                             pod=r["pod"], base_rate=r["base_rate"],
                                             lift=r["lift"], n_test=r["n_test"],
                                             n_pos=r["n_pos"], t_star=r["t_star"],
                                             tp=r["tp"], fp=r["fp"], fn=r["fn"]))
                print(f"target={tgt:.2f} k={k:2d} phase={phase:2d} elapsed={time.time()-t0:.0f}s",
                      flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("\n=== Summary over phases (onset-only test rows, today's chl <= T) ===")
    for tgt in TARGETS:
        for variant in ("full_truth", "thinned_truth"):
            for h in HORIZONS:
                for mn in ("LR", "GB"):
                    sub = out[(out.target_rate == tgt) & (out.label_variant == variant)
                              & (out.horizon == h) & (out.model == mn)]
                    print(f"\n-- target={tgt:.2f} (T={thresholds[tgt]:.1f}) {variant} h={h} {mn} --")
                    print(f"{'k':>3} {'nph':>3} {'AUC':>15} {'precision':>15} {'POD':>15} "
                          f"{'base_rate':>15} {'lift':>15} {'n_test':>7} {'n_pos':>6}")
                    for k, g in sub.groupby("k"):
                        def ms(c):
                            v = g[c].dropna()
                            return (f"{v.mean():.3f} +/- {v.std(ddof=0):.3f}" if len(v)
                                    else "n/a")
                        print(f"{k:>3} {len(g):>3} {ms('auc'):>15} {ms('precision'):>15} "
                              f"{ms('pod'):>15} {ms('base_rate'):>15} {ms('lift'):>15} "
                              f"{g.n_test.mean():>7.0f} {g.n_pos.mean():>6.1f}")

    print("\n=== Headline comparisons vs LIS "
          f"(precision {LIS_REF['precision']:.3f}, AUC {LIS_REF['auc']:.3f}, "
          f"base {LIS_REF['base_rate']:.3f}, lift {LIS_REF['lift']:.2f}) ===")
    for tgt in TARGETS:
        for (k, variant, tag) in ((1, "full_truth", "(i) daily sampling at matched rarity"),
                                  (21, "thinned_truth", "(ii) 21-day cadence, thinned truth")):
            for mn in ("LR", "GB"):
                g = out[(out.target_rate == tgt) & (out.k == k) & (out.label_variant == variant)
                        & (out.horizon == 21) & (out.model == mn)]
                print(f"target={tgt:.2f} T={thresholds[tgt]:.1f} k={k:2d} {variant:13s} h21 {mn}: "
                      f"precision={g.precision.mean():.3f}+/-{g.precision.std(ddof=0):.3f} "
                      f"AUC={g.auc.mean():.3f} POD={g.pod.mean():.3f} "
                      f"base={g.base_rate.mean():.3f} lift={g.lift.mean():.2f} "
                      f"n_test={g.n_test.mean():.0f} n_pos={g.n_pos.mean():.1f}  {tag}")
    print(f"\nWrote {OUT} ({len(out)} rows). Runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
