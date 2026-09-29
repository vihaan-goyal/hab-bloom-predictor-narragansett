"""
refit_top_sites.py -- would training on a site's own data beat the exported model there?
------------------------------------------------------------------------------------
For the best-scoring new ERDDAP sites (top by AUC and by precision, >= 300 onset rows),
compare on IDENTICAL test rows:
  exported   the exported model's leak-free probabilities (data/registry/predictions_causal_<model>/:
             rescaling fit on calibration years, prior-years climatology), threshold chosen per fold
             on year T-1 only, test years restricted to after the calibration period (2026-09-28;
             before, one threshold was tuned on every test year but one, i.e. on other folds' tests)
  refit      HistGB trained on the site's own earlier years, rolling-origin CV
             (train <= T-2, val T-1 for t*, test T), same onset rows
Label: own-station p75 within 7 d. Station-year bootstrap CIs.
Output: data/registry/refit_top_sites.csv + printed table.
Run from fork root, BASE env:  python -m src.registry.refit_top_sites [--n 12]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.transfer.transfer_eval import add_label, boot_ci, build_daily, metrics, pick_t, rolling_refit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--model", choices=["v1", "v2"], default="v2"); a = ap.parse_args()
    sk = pd.read_csv(f"data/registry/site_skill_causal_{a.model}.csv").dropna(subset=["lift"])
    sk = sk[sk.n_onset >= 300]
    cat = pd.read_csv("data/registry/insitu_catalog.csv").drop_duplicates(subset=["dataset_id"]).set_index("dataset_id")
    ids = list(dict.fromkeys(list(sk.sort_values("auc", ascending=False).dataset_id.head(a.n))
                             + list(sk.sort_values("precision", ascending=False).dataset_id.head(a.n))))
    rows = []
    for did in ids:
        site = pd.read_csv(f"data/registry/sites/{did}.csv")
        site = site.rename(columns={"chl": "chl_ugl", "temp": "temp_c", "sal": "salinity_psu", "do": "do_mgl"})
        cad = float(cat.cadence_min.get(did, 15))
        day = build_daily(site, 48 if cad <= 20 else 12)
        day["thr"] = day.groupby("station")["chl"].transform(lambda s: s.quantile(0.75))
        day = add_label(day, "thr").dropna(subset=["bloom_fwd"]).reset_index(drop=True)
        pred = pd.read_csv(f"data/registry/predictions_causal_{a.model}/{did}.csv",
                           parse_dates=["date"])[["station", "date", "bloom_prob", "ycal"]]
        day = day.merge(pred, on=["station", "date"], how="left")
        oof, t_refit = rolling_refit(day, "GB")
        if oof is None:
            print(f"{did}: not enough years for a refit", flush=True); continue
        ycal = pred.ycal.dropna().max() if pred.ycal.notna().any() else -1
        test = oof[(oof.chl <= oof.thr) & oof.bloom_prob.notna() & (oof.year > ycal)].copy()
        if len(test) < 100 or test.bloom_fwd.nunique() < 2:
            print(f"{did}: too few onset test rows", flush=True); continue
        # exported model's threshold: per fold, chosen on that fold's validation year T-1 only
        t_by_year = {}
        for T in sorted(test.year.unique()):
            va = day[(day.year == T - 1) & (day.chl <= day.thr) & day.bloom_prob.notna()]
            t_by_year[T] = (pick_t(va.bloom_fwd.values, va.bloom_prob.values)
                            if len(va) >= 50 and va.bloom_fwd.nunique() == 2 else 0.5)
        test["t_zs"] = test.year.map(t_by_year)
        test["alert_zs"] = (test.bloom_prob >= test.t_zs).astype(float)
        t_zs = float(np.median(list(t_by_year.values())))
        test["alert_refit"] = (test.p >= test.t_fold).astype(float)
        zs = metrics(test.bloom_fwd, test.alert_zs); zs_ci = boot_ci(test, "alert_zs", "bloom_fwd", 0.5)
        rf = metrics(test.bloom_fwd, test.alert_refit); rf_ci = boot_ci(test.assign(alert=test.alert_refit), "alert", "bloom_fwd", 0.5)
        r = dict(dataset_id=did, server=cat.server.get(did, "?"), n_test=len(test), test_years=f"{test.year.min()}-{test.year.max()}",
                 base_rate=zs["base_rate"], zs_precision=zs["precision"], zs_pod=zs["pod"], zs_lift=zs["lift"],
                 zs_lift_lo=zs_ci["lift_lo"], zs_lift_hi=zs_ci["lift_hi"], zs_auc=roc_auc_score(test.bloom_fwd, test.bloom_prob),
                 refit_precision=rf["precision"], refit_pod=rf["pod"], refit_lift=rf["lift"],
                 refit_lift_lo=rf_ci["lift_lo"], refit_lift_hi=rf_ci["lift_hi"], refit_auc=roc_auc_score(test.bloom_fwd, test.p))
        r["delta_lift"] = r["refit_lift"] - r["zs_lift"]; r["delta_auc"] = r["refit_auc"] - r["zs_auc"]
        rows.append(r)
        print(f"{did:36s} n={len(test):5d}  exported lift {r['zs_lift']:.2f} [{r['zs_lift_lo']:.2f},{r['zs_lift_hi']:.2f}] auc {r['zs_auc']:.2f} | "
              f"refit lift {r['refit_lift']:.2f} [{r['refit_lift_lo']:.2f},{r['refit_lift_hi']:.2f}] auc {r['refit_auc']:.2f} | "
              f"d_lift {r['delta_lift']:+.2f} d_auc {r['delta_auc']:+.2f}", flush=True)
    out = pd.DataFrame(rows); out.to_csv(f"data/registry/refit_top_sites_causal_{a.model}.csv", index=False)
    if len(out):
        print(f"\n{len(out)} sites | median delta lift {out.delta_lift.median():+.2f} | median delta AUC {out.delta_auc.median():+.3f} | "
              f"refit better on lift at {(out.delta_lift > 0).sum()}, exported better at {(out.delta_lift < 0).sum()}")


if __name__ == "__main__":
    main()
