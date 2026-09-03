"""
pooled_model_test.py -- train on every OTHER site, test blind on Narragansett
----------------------------------------------------------------------------
Reverse of the transfer test in transfer_eval.py. One GB model is trained on
the pooled station-days of all foreign sites (Chesapeake, NERRS, Cefas UK,
IMOS Australia, Lake Erie, SF Bay if present), each site's chlorophyll columns
quantile-mapped onto the Narragansett scale and labelled with its own
station p75 within 7 d. The model never sees a Narragansett row. It is then
scored on Narragansett exactly like the reference model (train_narragansett.py):
test year 2023, onset-only, label chl > 10 ug/L within 7 d, t* chosen on the
2021-22 val years (threshold calibration only; no refitting).

Reference to beat (Narragansett-trained GB, onset, test 2023):
  precision 0.696, POD 0.600, base 0.347, lift 2.00, AUC 0.839.

Output: data/transfer/pooled_to_narragansett.csv + printed table.
Run from fork root, BASE env:
    python -m src.transfer.pooled_model_test
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.transfer.transfer_eval import (GB_KW, TIER_A, add_label, boot_ci,
                                        metrics, pick_t, quantile_map)

SITES = ["chesapeake", "nerrs", "cefas_full", "imos", "lake_erie", "sfbay"]
NAR = "data/narragansett_daily_features.csv"
OUT = "data/transfer/pooled_to_narragansett.csv"


def load_site(name, nar_chl):
    p = f"data/transfer/{name}_daily.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, parse_dates=["date"])
    d = add_label(d, "thr_p75").rename(columns={"thr_p75": "thr"})
    d = d.dropna(subset=["bloom_fwd"])
    d = quantile_map(d, nar_chl)              # put chl columns on Narragansett scale
    d["station"] = name + "_" + d["station"].astype(str)
    d["site"] = name
    return d


def evaluate(model, med, test, label, scope_mask, t, tag, train_set):
    d = test[scope_mask].copy()
    d["p"] = model.predict_proba(d[TIER_A].fillna(med).fillna(0.0).values)[:, 1]
    r = metrics(d[label], d.p >= t)
    r.update(train_set=train_set, label=label, scope=tag, t_star=t,
             auc=roc_auc_score(d[label], d.p), n_pos=int(d[label].sum()))
    d["bloom_fwd"] = d[label]
    r.update(boot_ci(d, "p", "bloom_fwd", t))
    return r


def main():
    nar = pd.read_csv(NAR, parse_dates=["date"])
    nar["year"] = nar.date.dt.year
    nar["thr_p75"] = nar.groupby("station")["chl"].transform(lambda s: s.quantile(0.75))
    nar["bloom_p75"] = add_label(nar, "thr_p75")["bloom_fwd"]
    nar = nar.dropna(subset=["bloom_fwd"]).copy()
    nar["bloom_fwd"] = nar.bloom_fwd.astype(int)
    nar["bloom_p75"] = nar.bloom_p75.fillna(0).astype(int)

    sites = [s for s in (load_site(n, nar.chl.values) for n in SITES) if s is not None]
    pool = pd.concat(sites, ignore_index=True)
    print("pooled training set:")
    print(pool.groupby("site").agg(rows=("bloom_fwd", "size"), pos=("bloom_fwd", "mean")).round(3).to_string())
    med = pool[TIER_A].median(numeric_only=True)
    model = HistGradientBoostingClassifier(**GB_KW).fit(
        pool[TIER_A].fillna(med).fillna(0.0).values, pool.bloom_fwd.astype(int).values)

    val = nar[nar.year.isin((2021, 2022))]
    test = nar[nar.year == 2023]
    rows = []
    for label in ("bloom_fwd", "bloom_p75"):
        pv = model.predict_proba(val[TIER_A].fillna(med).fillna(0.0).values)[:, 1]
        t = pick_t(val[label].values, pv)
        thr = 10.0 if label == "bloom_fwd" else test.thr_p75
        onset = (test.chl <= thr).values
        rows.append(evaluate(model, med, test, label, np.ones(len(test), bool), t, "all", "pooled_foreign"))
        rows.append(evaluate(model, med, test, label, onset, t, "onset", "pooled_foreign"))
        rows.append(evaluate(model, med, test, label, onset, 0.5, "onset_t0.5", "pooled_foreign"))

    # sensitivity: drop the biggest and worst-transferring site
    if "chesapeake" in pool.site.unique():
        sub = pool[pool.site != "chesapeake"]
        m2 = HistGradientBoostingClassifier(**GB_KW).fit(
            sub[TIER_A].fillna(med).fillna(0.0).values, sub.bloom_fwd.astype(int).values)
        pv = m2.predict_proba(val[TIER_A].fillna(med).fillna(0.0).values)[:, 1]
        t = pick_t(val.bloom_fwd.values, pv)
        rows.append(evaluate(m2, med, test, "bloom_fwd", (test.chl <= 10).values, t,
                             "onset", "pooled_minus_chesapeake"))

    on = test[test.chl <= 10].copy()
    rows.append(dict(train_set="always_alert", label="bloom_fwd", scope="onset", t_star=np.nan,
                     auc=0.5, n_pos=int(on.bloom_fwd.sum()), **metrics(on.bloom_fwd, np.ones(len(on)))))
    rows.append(dict(train_set="reference_narragansett_GB", label="bloom_fwd", scope="onset",
                     t_star=0.50, auc=0.839, n_pos=int(on.bloom_fwd.sum()), n_test=len(on),
                     precision=0.696, pod=0.600, base_rate=0.347, lift=2.00))

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    cols = ["train_set", "label", "scope", "t_star", "n_test", "n_pos", "base_rate",
            "precision", "precision_lo", "precision_hi", "pod", "lift", "lift_lo", "lift_hi", "auc"]
    print("\nTest on Narragansett 2023 (model never saw Narragansett):")
    print(res.reindex(columns=cols).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
