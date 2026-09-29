"""
tank_feature_check.py -- how much forecast skill survives with only the features a bench tank can measure
======================================================================================================
The bench tanks (parent repo, notes/mitigation/00_CONTROL_LOOP.md) run the Narragansett model on
their own sensor readings: chlorophyll (DIY fluorometer) and temperature (DS18B20) automatically,
DO / salinity by hand kit. A tank has no multi-year history, so no site climatology.

On real Narragansett data (same split and onset-only task as train_narragansett.py: train <= 2020,
val 2021-22, test 2023, GB tier-A spec; the daily file's chl climatology is prior-years only since
2026-09-28):
  1. RETRAIN on reduced feature sets (what the model could learn from tank-style inputs).
  2. DEPLOY: the GB trained on the full tier A, applied the way predict_anywhere.py would run on a
     tank. Each test station's 2023 readings are fed to predict_anywhere.build_daily as a
     one-season record (so rolling features warm up from zero and there is no prior-year
     climatology), with only the tank's sensors (and optionally its kits), no rescaling (the data are
     already on the training scale), and a threshold chosen the same way on 2021-22 run as
     one-season records. This replaces the first 2026-09-28 version, whose "deploy" row only
     median-filled columns of the full-history table.

Metrics: onset-only test AUC (today's chl <= 10) with a station-clustered bootstrap CI; POD,
precision and lift at the validation-F1 threshold (the fork's convention).

Run from fork root, BASE env:  python src/models/tank_feature_check.py
Writes data/tank_feature_check.csv.
"""
import importlib.util

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BLOOM, TRAIN_MAX, VAL_YEARS, TEST_YEARS = 10.0, 2020, (2021, 2022), (2023,)
MIN_READINGS = 48      # same daily-coverage rule as build_narragansett_daily.py
N_BOOT, SEED = 2000, 42
GB_KW = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=50,
             l2_regularization=1.0, random_state=42, class_weight="balanced")

CHL = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4', 'chl_roll3_mean', 'chl_roll6_mean',
       'chl_roll9_mean', 'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend']
CLIM = ['chl_anomaly', 'chl_climatology']
TEMP = ['temp', 'temp_lag1']
DO = ['do', 'do_lag1']
SAL = ['sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4']
TIER_A = CHL + CLIM + DO + TEMP + SAL + ['month']
FULL = "A full (prior-years climatology, as built)"
SETS = {
    FULL: TIER_A,
    "retrain: tank + kits (chl, temp, DO, salinity, month)": CHL + TEMP + DO + SAL + ['month'],
    "retrain: tank sensors (chl, temp, month)": CHL + TEMP + ['month'],
    "retrain: chlorophyll only": CHL,
}


def load_pa():
    spec = importlib.util.spec_from_file_location("pa", "predict_anywhere.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def metrics(y, alert):
    tp = int(((alert == 1) & (y == 1)).sum()); fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    return dict(pod=tp / (tp + fn) if tp + fn else np.nan, precision=prec,
                lift=prec / y.mean() if y.mean() else np.nan, tp=tp, fp=fp, fn=fn)


def val_f1_threshold(yv, pv):
    ts = np.arange(0.05, 0.96, 0.05); best, bt = -1, 0.5
    for t in ts:
        m = metrics(yv, (pv >= t).astype(int))
        f1 = (2 * m["precision"] * m["pod"] / (m["precision"] + m["pod"])
              if m["precision"] and m["pod"] and not np.isnan(m["precision"]) else 0)
        if f1 > best:
            best, bt = f1, t
    return float(bt)


def boot_auc(y, p, groups, rng):
    idx = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    keys, out = list(idx), []
    for _ in range(N_BOOT):
        b = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        if 0 < y[b].sum() < len(b):
            out.append(roc_auc_score(y[b], p[b]))
    return np.percentile(out, [2.5, 97.5])


def report(rows, rng, name, n_feat, t, te, pt):
    on = (te.chl <= BLOOM).values
    yt = te.bloom_fwd.astype(int).values
    y_on, p_on = yt[on], pt[on]
    lo, hi = boot_auc(y_on, p_on, te.station.values[on], rng)
    r = dict(setup=name, n_features=n_feat, t_star=t, onset_auc=roc_auc_score(y_on, p_on),
             auc_lo=lo, auc_hi=hi, onset_n=int(on.sum()), onset_base=float(y_on.mean()),
             **metrics(y_on, (p_on >= t).astype(int)))
    rows.append(r)
    print(f"{name:<62} AUC {r['onset_auc']:.3f} [{lo:.3f}, {hi:.3f}]  t={t:.2f}  "
          f"POD {r['pod']:.3f}  prec {r['precision']:.3f}  lift {r['lift']:.2f}  n={r['onset_n']}")


def season_records(pa, raw, years, keep_cols):
    """Each station-year of raw readings fed to predict_anywhere.build_daily as its own record."""
    parts = []
    for y in years:
        r = raw[raw.datetime.dt.year == y]
        for _, g in r.groupby("station"):
            d = pa.build_daily(g[["station", "datetime"] + keep_cols], MIN_READINGS)
            if len(d):
                parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main():
    df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
    df["year"] = df.date.dt.year
    rng = np.random.default_rng(SEED)
    rows = []
    lab = df.dropna(subset=["bloom_fwd"])
    tr = lab[lab.year <= TRAIN_MAX]
    va = lab[lab.year.isin(VAL_YEARS)]
    te = lab[lab.year.isin(TEST_YEARS)]
    print(f"test 2023 onset rows: {(te.chl <= BLOOM).sum():,}, base "
          f"{te.loc[te.chl <= BLOOM, 'bloom_fwd'].mean():.3f}\n")

    models = {}
    for name, feats in SETS.items():
        med = tr[feats].median(numeric_only=True)
        m = HistGradientBoostingClassifier(**GB_KW).fit(tr[feats].fillna(med).values,
                                                        tr.bloom_fwd.astype(int).values)
        models[name] = (m, med, feats)
        pv = m.predict_proba(va[feats].fillna(med).values)[:, 1]
        pt = m.predict_proba(te[feats].fillna(med).values)[:, 1]
        t = val_f1_threshold(va.bloom_fwd.astype(int).values, pv)
        report(rows, rng, name, len(feats), t, te, pt)

    # DEPLOY: the full tier-A model, fed one-season tank-style records through predict_anywhere
    pa = load_pa()
    m, med, feats = models[FULL]
    raw = pd.read_csv("data/narragansett_surface_15min.csv",
                      usecols=["station", "datetime", "chl_ugl", "temp_c", "salinity_psu", "do_mgl"])
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce", format="mixed")
    raw = raw.dropna(subset=["datetime"]).rename(
        columns={"chl_ugl": "chl", "temp_c": "temp", "salinity_psu": "sal", "do_mgl": "do"})
    raw = raw[raw.datetime.dt.year.isin(list(VAL_YEARS) + list(TEST_YEARS))]
    labels = lab[["station", "date", "bloom_fwd"]]
    for name, cols in (("DEPLOY: tank sensors (chl, temp), one-season record", ["chl", "temp"]),
                       ("DEPLOY: tank + kits (chl, temp, DO, sal), one-season record",
                        ["chl", "temp", "sal", "do"])):
        out = {}
        for part, years in (("val", VAL_YEARS), ("test", TEST_YEARS)):
            d = season_records(pa, raw, years, cols).merge(labels, on=["station", "date"], how="inner")
            d["p"] = m.predict_proba(d[feats].fillna(med).values)[:, 1]
            out[part] = d
        t = val_f1_threshold(out["val"].bloom_fwd.astype(int).values, out["val"].p.values)
        report(rows, rng, name, len(feats), t, out["test"], out["test"].p.values)
    pd.DataFrame(rows).to_csv("data/tank_feature_check.csv", index=False)
    print("\nwrote data/tank_feature_check.csv")


if __name__ == "__main__":
    main()
