"""
refit_top_sites.py -- would training on a site's own data beat the exported model there?
------------------------------------------------------------------------------------
For the best-scoring new ERDDAP sites (top by AUC and by precision, >= 300 onset rows),
compare on IDENTICAL test rows:
  exported   the frozen Narragansett model (bloom_prob from data/registry/predictions/),
             threshold chosen on the refit's validation years
  refit      HistGB trained on the site's own earlier years, rolling-origin CV
             (train <= T-2, val T-1 for t*, test T), same onset rows
Label: own-station p75 within 7 d. Station-year bootstrap CIs.
Output: data/registry/refit_top_sites.csv + printed table.
Run from fork root, BASE env:  python -m src.registry.refit_top_sites [--n 12]
"""
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.transfer.transfer_eval import add_label, boot_ci, build_daily, metrics, pick_t, rolling_refit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12); a = ap.parse_args()
    sk = pd.read_csv("data/registry/site_skill.csv").dropna(subset=["lift"])
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
        pred = pd.read_csv(f"data/registry/predictions/{did}.csv", parse_dates=["date"])[["station", "date", "bloom_prob"]]
        day = day.merge(pred, on=["station", "date"], how="left")
        oof, t_refit = rolling_refit(day, "GB")
        if oof is None:
            print(f"{did}: not enough years for a refit", flush=True); continue
        test = oof[(oof.chl <= oof.thr) & oof.bloom_prob.notna()].copy()
        if len(test) < 100 or test.bloom_fwd.nunique() < 2:
            print(f"{did}: too few onset test rows", flush=True); continue
        val_years = [y - 1 for y in sorted(oof.year.unique())]
        val = day[day.year.isin(val_years) & (day.chl <= day.thr) & day.bloom_prob.notna()]
        t_zs = pick_t(val.bloom_fwd.values, val.bloom_prob.values) if len(val) >= 50 and val.bloom_fwd.nunique() == 2 else 0.5
        test["alert_refit"] = (test.p >= test.t_fold).astype(float)
        zs = metrics(test.bloom_fwd, test.bloom_prob >= t_zs); zs_ci = boot_ci(test, "bloom_prob", "bloom_fwd", t_zs)
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
    out = pd.DataFrame(rows); out.to_csv("data/registry/refit_top_sites.csv", index=False)
    if len(out):
        print(f"\n{len(out)} sites | median delta lift {out.delta_lift.median():+.2f} | median delta AUC {out.delta_auc.median():+.3f} | "
              f"refit better on lift at {(out.delta_lift > 0).sum()}, exported better at {(out.delta_lift < 0).sum()}")


if __name__ == "__main__":
    main()
