"""
regime_models.py -- do water-type ("regime") models beat one Narragansett model?
-------------------------------------------------------------------------------
Pre-registered test (plan 2026-09-03). Every station in the seven-site pool is
assigned a regime from its own median salinity:
    fresh      sal < 0.5 PSU
    estuarine  0.5 <= sal < 30
    marine     sal >= 30
    lake       source == lake_erie (report only; no other lakes to train on)
Narragansett stations with no salinity sensor -> estuarine (bay membership).
Label: own-station p75 within 7 d; onset-only rows (raw chl <= p75); all chl
columns quantile-mapped to the Narragansett scale before pooling.

Leave-one-SITE-out (site = source x regime). For each held-out site, four
models scored on its onset rows in its later years (threshold chosen on its
first year, identically for every model):
    regime        GB trained on the other sites of the same regime
    narragansett  frozen release model (in-sample when the held-out site is
                  Narragansett itself; flagged, excluded from pooling)
    all_sites     GB trained on every other site regardless of regime
    local_refit   rolling-origin CV on the site itself (upper bound)

Verdict rule (fixed before running): regime "wins" if pooled regime lift beats
pooled narragansett lift with non-overlapping 95% CIs AND beats all_sites.

Outputs: data/transfer/regime_loso.csv, data/transfer/regime_loso_pooled.csv,
release/regime_models.joblib (regime models fit on all their sites).
Run from fork root, BASE env:  python -m src.transfer.regime_models
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.transfer.transfer_eval import (GB_KW, TIER_A, add_label, boot_ci,
                                        metrics, pick_t, quantile_map,
                                        rolling_refit)

SOURCES = ["chesapeake", "nerrs", "cefas_full", "imos", "lake_erie", "sfbay", "narragansett"]
NAR_MODEL = "release/narragansett_bloom_model.joblib"
OUT = "data/transfer/regime_loso.csv"
OUT_POOL = "data/transfer/regime_loso_pooled.csv"
OUT_MODELS = "release/regime_models.joblib"
MIN_ONSET = 1000


def assign_regime(d, source):
    med = d.groupby("station")["sal"].median()
    reg = pd.Series(np.select([med < 0.5, med < 30], ["fresh", "estuarine"], "marine"), index=med.index)
    reg[med.isna()] = "estuarine" if source == "narragansett" else "unknown"
    if source == "lake_erie":
        reg[:] = "lake"
    return d["station"].map(reg)


def load(source, nar_chl):
    if source == "narragansett":
        d = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
        d["year"] = d.date.dt.year
        d["thr_p75"] = d.groupby("station")["chl"].transform(lambda s: s.quantile(0.75))
    else:
        p = f"data/transfer/{source}_daily.csv"
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p, parse_dates=["date"])
    d = add_label(d, "thr_p75").rename(columns={"thr_p75": "thr"}).dropna(subset=["bloom_fwd"])
    d["regime"] = assign_regime(d, source)
    d = d[d.regime != "unknown"].copy()
    d["chl_raw"] = d["chl"]
    d = quantile_map(d, nar_chl)
    d["source"] = source
    d["station"] = source + "_" + d["station"].astype(str)
    d["site"] = source + "-" + d["regime"]
    return d


def fit(df):
    med = df[TIER_A].median(numeric_only=True)
    m = HistGradientBoostingClassifier(**GB_KW).fit(
        df[TIER_A].fillna(med).fillna(0.0).values, df.bloom_fwd.astype(int).values)
    return m, med


def score(model, med, d):
    return model.predict_proba(d[TIER_A].fillna(pd.Series(med)).fillna(0.0).values)[:, 1]


def row(d, p, t, **tags):
    d = d.assign(p=p, alert=(p >= t).astype(float))
    r = metrics(d.bloom_fwd, d.alert)
    r.update(t_star=t, auc=roc_auc_score(d.bloom_fwd, d.p), n_pos=int(d.bloom_fwd.sum()), **tags)
    r.update(boot_ci(d, "p", "bloom_fwd", t))
    return r, d


def main():
    nar_chl = pd.read_csv("data/narragansett_daily_features.csv").chl.values
    pack = joblib.load(NAR_MODEL)
    sites = [s for s in (load(src, nar_chl) for src in SOURCES) if s is not None]
    pool = pd.concat(sites, ignore_index=True)
    print("regime assignment (stations / station-days):")
    print(pool.groupby(["regime", "source"]).agg(stations=("station", "nunique"),
                                                 days=("bloom_fwd", "size")).to_string())

    rows, held = [], []
    for site, d in pool.groupby("site"):
        regime = d.regime.iloc[0]; source = d.source.iloc[0]
        onset_all = d[d.chl_raw <= d.thr]
        years = sorted(d.year.unique())
        if len(onset_all) < MIN_ONSET or len(years) < 2:
            print(f"skip {site}: onset rows={len(onset_all)} years={len(years)}"); continue
        y0 = years[0]
        cal = onset_all[onset_all.year <= y0]; test = onset_all[onset_all.year > y0]
        if (len(cal) < 50 or cal.bloom_fwd.nunique() < 2) and len(years) > 2:
            y0 = years[1]
            cal = onset_all[onset_all.year <= y0]; test = onset_all[onset_all.year > y0]
        if test.bloom_fwd.nunique() < 2 or len(test) < 200 or cal.bloom_fwd.nunique() < 2:
            print(f"skip {site}: too little calibration/test data"); continue
        print(f"\n[{site}] regime={regime} cal<= {y0} test rows={len(test):,} pos={test.bloom_fwd.mean():.3f}")

        same = pool[(pool.regime == regime) & (pool.site != site)]
        others = pool[pool.site != site]
        models = {}
        if len(same) >= 2000:
            models["regime"] = fit(same) + (len(same),)
        models["all_sites"] = fit(others) + (len(others),)
        models["narragansett"] = (pack["model"], pack["medians"], 42207)
        for name, (m, med, ntr) in models.items():
            t = pick_t(cal.bloom_fwd.values, score(m, med, cal))
            ins = (name == "narragansett" and source == "narragansett")
            r, dd = row(test, score(m, med, test), t, site=site, regime=regime, model=name,
                        in_sample=ins, n_train=ntr)
            rows.append(r); held.append(dd.assign(site=site, regime=regime, model=name, in_sample=ins))
            print(f"  {name:13s} t={t:.2f} prec={r['precision']:.3f} lift={r['lift']:.2f} "
                  f"[{r['lift_lo']:.2f},{r['lift_hi']:.2f}] auc={r['auc']:.3f}"
                  + ("  (IN-SAMPLE)" if ins else ""))
        oof, t = rolling_refit(d, "GB")
        if oof is not None:
            oo = oof[oof.chl_raw <= oof.thr].copy()
            if oo.bloom_fwd.nunique() == 2 and len(oo) >= 200:
                oo["alert"] = (oo.p >= oo.t_fold).astype(float)
                r = metrics(oo.bloom_fwd, oo.alert)
                r.update(t_star=t, auc=roc_auc_score(oo.bloom_fwd, oo.p), n_pos=int(oo.bloom_fwd.sum()),
                         site=site, regime=regime, model="local_refit", in_sample=False, n_train=len(d))
                r.update(boot_ci(oo, "alert", "bloom_fwd", 0.5))
                rows.append(r); held.append(oo.assign(site=site, regime=regime, model="local_refit", in_sample=False))
                print(f"  {'local_refit':13s} t={t:.2f} prec={r['precision']:.3f} lift={r['lift']:.2f} "
                      f"[{r['lift_lo']:.2f},{r['lift_hi']:.2f}] auc={r['auc']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    H = pd.concat(held, ignore_index=True)
    H = H[~H.in_sample]
    H[["site", "regime", "model", "station", "year", "bloom_fwd", "p", "alert"]].to_csv(
        "data/transfer/regime_loso_predictions.csv", index=False)
    prow = []
    groups = [("all", "all", H)] + [("regime", r, g) for r, g in H.groupby("regime")]
    for scope, regime, g in groups:
        for model, gm in g.groupby("model"):
            if gm.bloom_fwd.nunique() < 2:
                continue
            m = metrics(gm.bloom_fwd, gm.alert)
            m.update(scope=scope, regime=regime, model=model, n_sites=gm.site.nunique(),
                     n_pos=int(gm.bloom_fwd.sum()), auc=roc_auc_score(gm.bloom_fwd, gm.p))
            m.update(boot_ci(gm, "alert", "bloom_fwd", 0.5))
            prow.append(m)
    P = pd.DataFrame(prow)
    P.to_csv(OUT_POOL, index=False)
    cols = ["scope", "regime", "model", "n_sites", "n_test", "n_pos", "base_rate", "precision",
            "pod", "lift", "lift_lo", "lift_hi", "auc"]
    print("\nPOOLED over held-out sites (in-sample Narragansett rows excluded; each site's own threshold):")
    print(P[cols].round(3).to_string(index=False))

    A = P[P.scope == "all"].set_index("model")
    if {"regime", "narragansett", "all_sites"} <= set(A.index):
        reg, nar, alls = A.loc["regime"], A.loc["narragansett"], A.loc["all_sites"]
        win = (reg.lift_lo > nar.lift_hi) and (reg.lift > alls.lift)
        print(f"\nVERDICT: regime lift {reg.lift:.2f} [{reg.lift_lo:.2f},{reg.lift_hi:.2f}] vs "
              f"narragansett {nar.lift:.2f} [{nar.lift_lo:.2f},{nar.lift_hi:.2f}] vs all_sites "
              f"{alls.lift:.2f} [{alls.lift_lo:.2f},{alls.lift_hi:.2f}] -> "
              f"{'REGIME WINS' if win else 'NO WIN (pre-registered criterion not met)'}")

    lib = {}
    for regime, g in pool[pool.regime.isin(["fresh", "estuarine", "marine"])].groupby("regime"):
        m, med = fit(g)
        lib[regime] = dict(model=m, medians=med.to_dict(), n_rows=len(g), sites=sorted(g.site.unique()))
    joblib.dump(dict(regimes=lib, rule="fresh: sal<0.5; estuarine: 0.5-30; marine: >=30 PSU",
                     features=TIER_A, chl_quantiles=pack["chl_quantiles"]), OUT_MODELS, compress=3)
    print(f"\nwrote {OUT}, {OUT_POOL}, {OUT_MODELS}")


if __name__ == "__main__":
    main()
