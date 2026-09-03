"""
transfer_eval.py -- does the Narragansett bloom model transfer to another bay?
----------------------------------------------------------------------------
Shared harness for the cross-site transfer tests. One tidy input, one
evaluation protocol, one results table per source, so the sources compare.

INPUT  data/transfer/<source>_15min.csv   (any sub-daily cadence)
       columns: station, datetime, chl_ugl, temp_c, salinity_psu, do_mgl
       optional: do_pct, ph.  Freshwater: fill salinity_psu with 0.
OUTPUT data/transfer/<source>_daily.csv    station-days + tier-A features
       data/transfer/<source>_results.csv  one row per (eval, model, threshold)

Bloom definitions (both always run):
  p75   = station's own 75th percentile of daily-mean chl (fluorometers are
          not inter-comparable, so this is the fair cross-site definition)
  abs10 = daily-mean chl > 10 ug/L (the LIS / Narragansett definition)
Label = bloom within HORIZON=7 days, right-censored NaN. Onset-only rows =
today's chl <= threshold (persistence cannot alert there).

Evaluations:
  zeroshot_raw : GB trained on ALL Narragansett (label abs10), applied to the
                 target's raw features, alert at t*=0.50.
  zeroshot_qm  : same model, target chl columns quantile-mapped onto the
                 Narragansett chl distribution first (removes sensor scale).
  refit        : GB + LR refit on the target with rolling-origin CV
                 (train <= T-2, val T-1 for t*, test T), pooled out-of-fold.
  baselines    : always-alert, station-DOY climatology, chl>c rule (c on val).
CIs: station-year clustered bootstrap, n=2000, seed=42.

Usage (fork root, BASE env):
    python -m src.transfer.transfer_eval --source chesapeake --min-readings 48
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HORIZON = 7
NAR_PATH = "data/narragansett_daily_features.csv"
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']
GB_KW = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50,
             l2_regularization=1.0, random_state=42, class_weight="balanced")


# ----------------------------------------------------------------- features
def build_daily(df15, min_readings=48):
    """Sub-daily tidy table -> station-day table with tier-A features."""
    df = df15.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "chl_ugl"])
    df["date"] = df["datetime"].dt.normalize()
    for c in ("temp_c", "salinity_psu", "do_mgl"):
        if c not in df: df[c] = np.nan
    day = df.groupby(["station", "date"]).agg(
        chl=("chl_ugl", "mean"), temp=("temp_c", "mean"),
        sal=("salinity_psu", "mean"), do=("do_mgl", "mean"),
        n=("chl_ugl", "count")).reset_index()
    day = day[day.n >= min_readings].sort_values(["station", "date"]).reset_index(drop=True)
    g = day.groupby("station")
    for k in (1, 2, 3, 4):
        day[f"chl_lag{k}"] = g["chl"].shift(k)
        day[f"sal_lag{k}"] = g["sal"].shift(k)
    day["do_lag1"] = g["do"].shift(1)
    day["temp_lag1"] = g["temp"].shift(1)
    for w in (3, 6, 9, 14, 21):
        day[f"chl_roll{w}_mean"] = g["chl"].transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean())
    day["chl_trend"] = day["chl"] - day["chl_roll6_mean"]
    day["month"] = day["date"].dt.month
    day["doy_bin"] = (day["date"].dt.dayofyear - 1) // 15
    clim = day.groupby(["station", "doy_bin"])["chl"].transform("mean")
    day["chl_climatology"] = clim
    day["chl_anomaly"] = day["chl"] - clim
    day["year"] = day["date"].dt.year
    return day.drop(columns=["doy_bin"])


def add_label(day, thr_col):
    """bloom_fwd: any chl > threshold within HORIZON days; right-censored NaN.
    thr_col is a per-row threshold column (constant or per-station)."""
    day = day.copy()
    lab = np.full(len(day), np.nan)
    for st, grp in day.groupby("station"):
        dates = grp["date"].values; chl = grp["chl"].values
        thr = grp[thr_col].values; last = dates.max()
        for j, i in enumerate(grp.index):
            end = dates[j] + np.timedelta64(HORIZON, "D")
            m = (dates > dates[j]) & (dates <= end)
            if m.any() and (chl[m] > thr[m]).any():
                lab[i] = 1
            elif end <= last:
                lab[i] = 0
    day["bloom_fwd"] = lab
    return day


# ------------------------------------------------------------------ metrics
def metrics(y, alert):
    y = np.asarray(y).astype(int); a = np.asarray(alert).astype(int)
    tp = int(((a == 1) & (y == 1)).sum()); fp = int(((a == 1) & (y == 0)).sum())
    fn = int(((a == 0) & (y == 1)).sum()); n = len(y)
    prec = tp / (tp + fp) if tp + fp else np.nan
    pod = tp / (tp + fn) if tp + fn else np.nan
    base = (tp + fn) / n if n else np.nan
    return dict(n_test=n, tp=tp, fp=fp, fn=fn, precision=prec, pod=pod,
                base_rate=base, lift=prec / base if base else np.nan)


def boot_ci(df, pcol, ycol, t, n_boot=2000, seed=42):
    """Station-year clustered bootstrap of precision, lift, AUC."""
    rng = np.random.default_rng(seed)
    cl = (df.station.astype(str) + "_" + df.year.astype(str)).values
    groups = {c: np.where(cl == c)[0] for c in np.unique(cl)}
    keys = list(groups); out = {"precision": [], "lift": [], "auc": []}
    p = df[pcol].values; y = df[ycol].values.astype(int)
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        m = metrics(y[idx], p[idx] >= t)
        out["precision"].append(m["precision"]); out["lift"].append(m["lift"])
        out["auc"].append(roc_auc_score(y[idx], p[idx]) if 0 < y[idx].mean() < 1 else np.nan)
    lo = {f"{k}_lo": np.nanpercentile(v, 2.5) for k, v in out.items()}
    hi = {f"{k}_hi": np.nanpercentile(v, 97.5) for k, v in out.items()}
    return {**lo, **hi}


def pick_t(y, p):
    ts = np.arange(0.05, 0.96, 0.05); best, bt = -1, 0.5
    for t in ts:
        m = metrics(y, p >= t)
        if np.isnan(m["precision"]) or np.isnan(m["pod"]): continue
        f1 = 2 * m["precision"] * m["pod"] / (m["precision"] + m["pod"] + 1e-12)
        if f1 > best: best, bt = f1, float(t)
    return bt


def summarise(df, pcol, t, source, evalname, model, thr_name):
    on = df[df.chl <= df.thr]           # onset-only rows
    rows = []
    for scope, d in (("all", df), ("onset", on)):
        if len(d) < 30 or d.bloom_fwd.nunique() < 2: continue
        r = metrics(d.bloom_fwd, d[pcol] >= t)
        r.update(source=source, eval=evalname, model=model, threshold=thr_name,
                 scope=scope, t_star=t, auc=roc_auc_score(d.bloom_fwd, d[pcol]),
                 n_pos=int(d.bloom_fwd.sum()), years=f"{d.year.min()}-{d.year.max()}")
        r.update(boot_ci(d, pcol, "bloom_fwd", t))
        rows.append(r)
    return rows


# --------------------------------------------------------------- reference
def train_reference():
    nar = pd.read_csv(NAR_PATH, parse_dates=["date"]).dropna(subset=["bloom_fwd"])
    med = nar[TIER_A].median(numeric_only=True)
    gb = HistGradientBoostingClassifier(**GB_KW).fit(
        nar[TIER_A].fillna(med).values, nar.bloom_fwd.astype(int).values)
    return gb, med, nar


def quantile_map(target, ref_vals):
    """Map target chl-scale columns onto the Narragansett chl distribution."""
    t = target.copy()
    ref = np.sort(ref_vals[~np.isnan(ref_vals)])
    src = np.sort(t["chl"].dropna().values)
    def qm(x):
        q = np.searchsorted(src, x, side="right") / len(src)
        return np.interp(np.clip(q, 0, 1), np.linspace(0, 1, len(ref)), ref)
    for c in ("chl", "chl_lag1", "chl_lag2", "chl_lag3", "chl_lag4",
              "chl_roll3_mean", "chl_roll6_mean", "chl_roll9_mean",
              "chl_roll14_mean", "chl_roll21_mean", "chl_climatology"):
        v = t[c].values.astype(float); ok = ~np.isnan(v); v2 = v.copy(); v2[ok] = qm(v[ok]); t[c] = v2
    t["chl_trend"] = t["chl"] - t["chl_roll6_mean"]
    t["chl_anomaly"] = t["chl"] - t["chl_climatology"]
    return t


# ------------------------------------------------------------------- refit
def rolling_refit(day, model_name):
    """Pooled out-of-fold predictions; returns df with column p and t used."""
    lab = day.dropna(subset=["bloom_fwd"]); years = sorted(lab.year.unique())
    parts = []; ts = []
    for T in years:
        tr = lab[lab.year <= T - 2]; va = lab[lab.year == T - 1]; te = lab[lab.year == T]
        if len(tr) < 500 or len(va) < 100 or te.bloom_fwd.nunique() < 2 or tr.bloom_fwd.nunique() < 2:
            continue
        med = tr[TIER_A].median(numeric_only=True)
        prep = lambda d: d[TIER_A].fillna(med).values
        if model_name == "GB":
            m = HistGradientBoostingClassifier(**GB_KW).fit(prep(tr), tr.bloom_fwd.astype(int))
            pv, pt = m.predict_proba(prep(va))[:, 1], m.predict_proba(prep(te))[:, 1]
        else:
            sc = StandardScaler().fit(prep(tr))
            m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000,
                                   random_state=42).fit(sc.transform(prep(tr)), tr.bloom_fwd.astype(int))
            pv, pt = m.predict_proba(sc.transform(prep(va)))[:, 1], m.predict_proba(sc.transform(prep(te)))[:, 1]
        t = pick_t(va.bloom_fwd.values, pv); ts.append(t)
        parts.append(te.assign(p=pt, t_fold=t))
    if not parts: return None, None
    return pd.concat(parts), float(np.median(ts))


def climatology_baseline(day):
    lab = day.dropna(subset=["bloom_fwd"]); parts = []
    for T in sorted(lab.year.unique()):
        tr = lab[lab.year <= T - 2]; te = lab[lab.year == T]
        if len(tr) < 500 or te.bloom_fwd.nunique() < 2: continue
        rate = tr.assign(b=(tr.date.dt.dayofyear - 1) // 15).groupby(["station", "b"]).bloom_fwd.mean()
        b = (te.date.dt.dayofyear - 1) // 15
        p = pd.Series(list(zip(te.station, b))).map(rate).values.astype(float)
        p = np.where(np.isnan(p), tr.bloom_fwd.mean(), p)
        parts.append(te.assign(p=p))
    return pd.concat(parts) if parts else None


def rule_baseline(day):
    """alert if chl_today > c, c chosen on val year (max F1)."""
    lab = day.dropna(subset=["bloom_fwd"]); parts = []
    for T in sorted(lab.year.unique()):
        va = lab[lab.year == T - 1]; te = lab[lab.year == T]
        if len(va) < 100 or te.bloom_fwd.nunique() < 2: continue
        cs = np.quantile(va.chl.dropna(), np.linspace(0.3, 0.95, 14)); best, bc = -1, cs[0]
        for c in cs:
            m = metrics(va.bloom_fwd, va.chl > c)
            if np.isnan(m["precision"]) or np.isnan(m["pod"]): continue
            f1 = 2 * m["precision"] * m["pod"] / (m["precision"] + m["pod"] + 1e-12)
            if f1 > best: best, bc = f1, c
        parts.append(te.assign(p=(te.chl > bc).astype(float)))
    return pd.concat(parts) if parts else None


# -------------------------------------------------------------------- main
def run(source, min_readings):
    os.makedirs("data/transfer", exist_ok=True)
    df15 = pd.read_csv(f"data/transfer/{source}_15min.csv")
    day = build_daily(df15, min_readings)
    print(f"[{source}] station-days={len(day):,} stations={day.station.nunique()} "
          f"years={day.year.min()}-{day.year.max()} chl median={day.chl.median():.2f}")
    if len(day) < 1000:
        raise SystemExit("fewer than 1000 station-days -- not enough to evaluate")
    day["thr_p75"] = day.groupby("station")["chl"].transform(lambda s: s.quantile(0.75))
    day["thr_abs10"] = 10.0
    day.to_csv(f"data/transfer/{source}_daily.csv", index=False)

    gb, med, nar = train_reference()
    rows = []
    for thr_name in ("p75", "abs10"):
        d = add_label(day, f"thr_{thr_name}").rename(columns={f"thr_{thr_name}": "thr"})
        lab = d.dropna(subset=["bloom_fwd"]).copy()
        if lab.bloom_fwd.nunique() < 2 or lab.bloom_fwd.sum() < 30:
            print(f"  [{thr_name}] too few positives ({int(lab.bloom_fwd.sum())}) -- skipped"); continue
        print(f"  [{thr_name}] labeled={len(lab):,} pos={lab.bloom_fwd.mean():.3f} "
              f"onset rows={(lab.chl <= lab.thr).sum():,}")
        lab["p"] = gb.predict_proba(lab[TIER_A].fillna(med).values)[:, 1]
        rows += summarise(lab, "p", 0.5, source, "zeroshot_raw", "GB_nar", thr_name)
        qm = quantile_map(lab, nar.chl.values)
        lab["p_qm"] = gb.predict_proba(qm[TIER_A].fillna(med).values)[:, 1]
        rows += summarise(lab, "p_qm", 0.5, source, "zeroshot_qm", "GB_nar", thr_name)
        for mn in ("GB", "LR"):
            oof, t = rolling_refit(d, mn)
            if oof is None: print(f"  [{thr_name}] refit {mn}: not enough years"); continue
            rows += summarise(oof, "p", t, source, "refit_cv", mn, thr_name)
        oof = climatology_baseline(d)
        if oof is not None: rows += summarise(oof, "p", 0.5, source, "baseline", "climatology", thr_name)
        oof = rule_baseline(d)
        if oof is not None: rows += summarise(oof, "p", 0.5, source, "baseline", "chl_rule", thr_name)
        rows += summarise(lab.assign(p=1.0), "p", 0.5, source, "baseline", "always_alert", thr_name)

    res = pd.DataFrame(rows)
    res.to_csv(f"data/transfer/{source}_results.csv", index=False)
    cols = ["eval", "model", "threshold", "scope", "n_test", "n_pos", "base_rate",
            "precision", "precision_lo", "precision_hi", "pod", "lift", "lift_lo", "lift_hi", "auc"]
    print(res[cols].round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--min-readings", type=int, default=48,
                    help="min sub-daily chl readings per station-day (48 for 15-min, 12 for hourly)")
    a = ap.parse_args()
    run(a.source, a.min_readings)
