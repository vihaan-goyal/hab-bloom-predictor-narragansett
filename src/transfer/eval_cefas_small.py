"""
eval_cefas_small.py -- run the transfer harness on the (small) open Cefas set
----------------------------------------------------------------------------
transfer_eval.run() refuses sources with < 1000 station-days. The openly
downloadable Cefas SmartBuoy data (see fetch_cefas.py) yield 904 station-days
from 2 Celtic Sea moorings over 2014-2015, so this wrapper re-runs the SAME
protocol (same functions, same bloom definitions, same bootstrap) without the
floor. With only two calendar years the rolling-origin refit (train <= T-2)
and the climatology baseline cannot run; the chl_rule baseline runs for 2015
only (c chosen on 2014). Results are written to the same output paths the
harness would use. Treat the numbers as a pilot, not a verdict.

Usage: python -m src.transfer.eval_cefas_small --min-readings 12
"""
import argparse
import pandas as pd
from src.transfer import transfer_eval as te


def run(source="cefas", min_readings=12):
    df15 = pd.read_csv(f"data/transfer/{source}_15min.csv")
    day = te.build_daily(df15, min_readings)
    print(f"[{source}] station-days={len(day):,} stations={day.station.nunique()} "
          f"years={day.year.min()}-{day.year.max()} chl median={day.chl.median():.2f}  (floor bypassed)")
    day["thr_p75"] = day.groupby("station")["chl"].transform(lambda s: s.quantile(0.75))
    day["thr_abs10"] = 10.0
    day.to_csv(f"data/transfer/{source}_daily.csv", index=False)
    gb, med, nar = te.train_reference()
    rows = []
    for thr_name in ("p75", "abs10"):
        d = te.add_label(day, f"thr_{thr_name}").rename(columns={f"thr_{thr_name}": "thr"})
        lab = d.dropna(subset=["bloom_fwd"]).copy()
        if lab.bloom_fwd.nunique() < 2 or lab.bloom_fwd.sum() < 30:
            print(f"  [{thr_name}] too few positives ({int(lab.bloom_fwd.sum())}) -- skipped"); continue
        print(f"  [{thr_name}] labeled={len(lab):,} pos={lab.bloom_fwd.mean():.3f} "
              f"onset rows={(lab.chl <= lab.thr).sum():,}")
        lab["p"] = gb.predict_proba(lab[te.TIER_A].fillna(med).values)[:, 1]
        rows += te.summarise(lab, "p", 0.5, source, "zeroshot_raw", "GB_nar", thr_name)
        qm = te.quantile_map(lab, nar.chl.values)
        lab["p_qm"] = gb.predict_proba(qm[te.TIER_A].fillna(med).values)[:, 1]
        rows += te.summarise(lab, "p_qm", 0.5, source, "zeroshot_qm", "GB_nar", thr_name)
        for mn in ("GB", "LR"):
            oof, t = te.rolling_refit(d, mn)
            if oof is None: print(f"  [{thr_name}] refit {mn}: not enough years"); continue
            rows += te.summarise(oof, "p", t, source, "refit_cv", mn, thr_name)
        oof = te.climatology_baseline(d)
        if oof is None: print(f"  [{thr_name}] climatology: not enough years")
        else: rows += te.summarise(oof, "p", 0.5, source, "baseline", "climatology", thr_name)
        oof = te.rule_baseline(d)
        if oof is None: print(f"  [{thr_name}] chl_rule: not enough years")
        else: rows += te.summarise(oof, "p", 0.5, source, "baseline", "chl_rule", thr_name)
        rows += te.summarise(lab.assign(p=1.0), "p", 0.5, source, "baseline", "always_alert", thr_name)
    res = pd.DataFrame(rows)
    res.to_csv(f"data/transfer/{source}_results.csv", index=False)
    cols = ["eval", "model", "threshold", "scope", "n_test", "n_pos", "base_rate",
            "precision", "precision_lo", "precision_hi", "pod", "lift", "lift_lo", "lift_hi", "auc"]
    print(res[cols].round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cefas")
    ap.add_argument("--min-readings", type=int, default=12)
    a = ap.parse_args()
    run(a.source, a.min_readings)
