"""W1 cadence-thinning experiment.

Controlled test of the cadence thesis within one bay: take the Narragansett
daily sonde series, thin it to every k-th day (k = 1..28, several random
phases), rebuild the LIS-analog (tier-A) features on the THINNED series with
lags defined in prior SAMPLES (as in LIS), rebuild the forward bloom label two
ways, retrain the locked-spec models, and evaluate onset-only test rows.

Label variants
  full_truth    : any FULL daily-series chl > 10 within `horizon` days after the
                  observed date (right-censored NaN past the station's last day)
  thinned_truth : same, but using only the thinned series' future rows
                  (LIS-realistic; NaN if no observed row falls in the window)

Split by year: train <= 2020, val 2021-2022, test 2023.  Threshold t* = argmax
val F1 over 0.05..0.95 (step 0.05), chosen on all val rows (as in
train_narragansett.py).  Test metrics on onset-only rows (observed chl <= 10).

Output: data/cadence_thinning.csv
Run from repo root with the BASE anaconda python (not the `hab` env).
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

BLOOM = 10.0
TRAIN_MAX = 2020
VAL_YEARS = (2021, 2022)
TEST_YEARS = (2023,)
KS = (1, 2, 3, 5, 7, 10, 14, 21, 28)
N_PHASES = 5
HORIZONS = (7, 21)
SEED = 42
OUT = "data/cadence_thinning.csv"
LIS_REF = dict(precision=0.136, auc=0.875, base_rate=0.046)

FEATURES = ["chl", "chl_lag1", "chl_lag2", "chl_lag3", "chl_lag4",
            "chl_roll3_mean", "chl_roll6_mean", "chl_roll9_mean", "chl_trend",
            "chl_climatology", "chl_anomaly",
            "do", "do_lag1", "temp", "temp_lag1",
            "sal", "sal_lag1", "sal_lag2", "sal_lag3", "sal_lag4", "month"]


def thin(full, k, phase):
    """Keep rows whose day index from the station's first date is == phase mod k."""
    day_idx = (full["date"] - full.groupby("station")["date"].transform("min")).dt.days
    return full[(day_idx % k) == phase].copy()


def build_features(obs):
    """Tier-A-equivalent features on the observed (thinned) series; lags = prior samples."""
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
    # station x 15-day DOY-bin climatology on TRAIN years only
    clim = obs[obs.year <= TRAIN_MAX].groupby(["station", "doy_bin"])["chl"].mean()
    obs["chl_climatology"] = pd.MultiIndex.from_frame(obs[["station", "doy_bin"]]).map(clim)
    obs["chl_climatology"] = obs["chl_climatology"].astype(float)
    obs["chl_anomaly"] = obs["chl"] - obs["chl_climatology"]
    return obs


def forward_label(obs, ref, horizon, require_obs_in_window):
    """bloom_fwd for each observed row using `ref` (full or thinned) as truth.

    Window is (t, t+horizon] calendar days.  1 if any ref chl > BLOOM in window;
    NaN if window runs past the station's last ref date (right-censored) or, when
    require_obs_in_window, if no ref row falls in the window; else 0.
    """
    lab = np.full(len(obs), np.nan)
    ref = ref.sort_values(["station", "date"])
    for st, rg in ref.groupby("station"):
        m = (obs["station"] == st).values
        if not m.any():
            continue
        rd = rg["date"].values.astype("datetime64[D]")
        bloom_cum = np.concatenate([[0], np.cumsum((rg["chl"].values > BLOOM))])
        t = obs.loc[m, "date"].values.astype("datetime64[D]")
        end = t + np.timedelta64(horizon, "D")
        lo = np.searchsorted(rd, t, side="right")
        hi = np.searchsorted(rd, end, side="right")
        n_in = hi - lo
        pos = (bloom_cum[hi] - bloom_cum[lo]) > 0
        out = np.where(pos, 1.0, np.where(end <= rd.max(), 0.0, np.nan))
        if require_obs_in_window:
            out = np.where((n_in == 0) & ~pos, np.nan, out)
        lab[m] = out
    return lab


def metrics(y, alert):
    tp = int(((alert == 1) & (y == 1)).sum()); fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum()); tn = int(((alert == 0) & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    base = (tp + fn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else np.nan
    return dict(precision=prec, pod=pod, base_rate=base,
                lift=prec / base if base else np.nan, tp=tp, fp=fp, fn=fn)


def fit_eval(train, val, test, model_name):
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
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                           max_iter=300, min_samples_leaf=50,
                                           l2_regularization=1.0, random_state=SEED,
                                           class_weight="balanced").fit(prep(train), ytr)
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
    om = (test.chl <= BLOOM).values
    r = metrics(yt[om], (pt[om] >= t_star).astype(int))
    r["auc"] = roc_auc_score(yt[om], pt[om]) if len(np.unique(yt[om])) == 2 else np.nan
    r["n_test"] = int(om.sum()); r["t_star"] = t_star
    return r


EMPTY = dict(auc=np.nan, precision=np.nan, pod=np.nan, base_rate=np.nan,
             lift=np.nan, n_test=0, t_star=np.nan, tp=0, fp=0, fn=0)


def main():
    t0 = time.time()
    full = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
    full = full[["station", "date", "chl", "temp", "sal", "do"]].sort_values(
        ["station", "date"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for k in KS:
        phases = sorted(rng.choice(k, size=min(N_PHASES, k), replace=False).tolist())
        for phase in phases:
            obs = thin(full, k, phase)
            gap = obs.groupby("station")["date"].diff().dt.days.dropna()
            median_gap = float(gap.median())
            obs = build_features(obs)
            for variant in ("full_truth", "thinned_truth"):
                ref = full if variant == "full_truth" else obs
                for h in HORIZONS:
                    obs["y"] = forward_label(obs, ref, h,
                                             require_obs_in_window=(variant == "thinned_truth"))
                    lab = obs.dropna(subset=["y"]).copy()
                    lab["y"] = lab.y.astype(int)
                    train = lab[lab.year <= TRAIN_MAX]
                    val = lab[lab.year.isin(VAL_YEARS)]
                    test = lab[lab.year.isin(TEST_YEARS)]
                    ok = (len(train) > 0 and len(val) > 0 and len(test) > 0
                          and train.y.nunique() == 2 and val.y.nunique() == 2)
                    for mn in ("LR", "GB"):
                        r = fit_eval(train, val, test, mn) if ok else dict(EMPTY)
                        rows.append(dict(k=k, phase=phase, label_variant=variant, horizon=h,
                                         model=mn, auc=r["auc"], precision=r["precision"],
                                         pod=r["pod"], base_rate=r["base_rate"], lift=r["lift"],
                                         n_test=r["n_test"], median_gap_days=median_gap,
                                         t_star=r["t_star"], tp=r["tp"], fp=r["fp"], fn=r["fn"],
                                         n_train=len(train), n_val=len(val)))
            print(f"k={k:2d} phase={phase:2d} obs={len(obs):6d} median_gap={median_gap:.0f}d "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    cols = ["k", "phase", "label_variant", "horizon", "model", "auc", "precision", "pod",
            "base_rate", "lift", "n_test", "median_gap_days", "t_star", "tp", "fp", "fn",
            "n_train", "n_val"]
    out = out[cols]
    out.to_csv(OUT, index=False)

    def fmt(x):
        return f"{x:.3f}"

    print("\n=== Summary over phases: mean +/- sd (onset-only test rows) ===")
    for variant in ("full_truth", "thinned_truth"):
        for h in HORIZONS:
            for mn in ("LR", "GB"):
                sub = out[(out.label_variant == variant) & (out.horizon == h) & (out.model == mn)]
                print(f"\n-- {variant} h={h} {mn} --")
                print(f"{'k':>3} {'nph':>3} {'gap':>4} {'AUC':>15} {'precision':>15} "
                      f"{'POD':>15} {'base_rate':>15} {'lift':>15} {'n_test':>7}")
                for k, g in sub.groupby("k"):
                    def ms(c):
                        v = g[c].dropna()
                        if not len(v):
                            return "n/a"
                        return f"{v.mean():.3f} +/- {v.std(ddof=0):.3f}"
                    print(f"{k:>3} {len(g):>3} {g.median_gap_days.mean():>4.0f} {ms('auc'):>15} "
                          f"{ms('precision'):>15} {ms('pod'):>15} {ms('base_rate'):>15} "
                          f"{ms('lift'):>15} {g.n_test.mean():>7.0f}")

    print("\n=== k=21, GB, h=21 vs LIS reference ===")
    k21 = out[(out.k == 21) & (out.model == "GB") & (out.horizon == 21)]
    print(k21[["label_variant", "phase", "auc", "precision", "pod", "base_rate", "lift",
               "n_test", "t_star"]].to_string(index=False, float_format=fmt))
    print(f"LIS reference (tier-A, onset): precision={LIS_REF['precision']:.3f} "
          f"AUC={LIS_REF['auc']:.3f} base_rate={LIS_REF['base_rate']:.3f}")
    for variant in ("full_truth", "thinned_truth"):
        p = k21[k21.label_variant == variant].precision.mean()
        if p < 0.30:
            verdict = "cadence thesis CONFIRMED within one bay (precision < 0.30)"
        elif p > 0.5:
            verdict = "thesis WRONG (precision > 0.5)"
        else:
            verdict = "INDETERMINATE (0.30 <= precision <= 0.50)"
        print(f"k=21 GB h21 {variant}: mean onset precision={p:.3f} -> {verdict}")

    print("\n=== Sanity check: k=1, GB, full_truth, h=7 (expect precision ~0.70, AUC ~0.84) ===")
    s = out[(out.k == 1) & (out.model == "GB") & (out.label_variant == "full_truth")
            & (out.horizon == 7)].iloc[0]
    print(f"precision={s.precision:.3f} AUC={s.auc:.3f} POD={s.pod:.3f} "
          f"base_rate={s.base_rate:.3f} n_test={s.n_test} t*={s.t_star}")
    ok = abs(s.precision - 0.70) <= 0.05 and abs(s.auc - 0.84) <= 0.03
    print("SANITY CHECK:", "PASS" if ok else "FAIL")
    print(f"\nWrote {OUT} ({len(out)} rows). Runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
