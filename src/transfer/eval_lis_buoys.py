"""
eval_lis_buoys.py -- the frozen Narragansett model applied zero-shot to the LIS buoys (findings 25.1).
--------------------------------------------------------------------------------------------------
Question: section 12 retrained the Narragansett *recipe* on UConn LISICOS WLIS/EXRX fluorescence
(lift ~2x at a p95 label). The exported model itself was never applied to LIS. This runs it exactly
as for the 87 ERDDAP sites (section 24 protocol: own-station p75 label, 7-day horizon, onset rows,
threshold chosen on the earliest calibration years, station-year clustered bootstrap), plus the
prospective primary rule (fixed threshold 0.50) so the live test has a measured expected band.

Inputs : data/prospective/history/{WLIS_ECO_FL,EXRX_ECO_FL}.csv (night-only ECO-FL, contract format)
Outputs: data/transfer/lis_buoys_zero_shot.csv ; stdout table
Run    : ~/anaconda3/python.exe -m src.transfer.eval_lis_buoys   (fork root, BASE env)
"""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from src.deploy import prospective_sites as ps
from src.registry.run_catalog import score
from src.transfer.transfer_eval import add_label, metrics, boot_ci

pa = ps.load_pa()
import joblib, os
PACKS = {"v1 (frozen, prospective)": ps.load_pack(), "v2 (causal features)": joblib.load(pa.MODEL_PATH)}
sites = {s["site_id"]: s for s in ps.SITES if s["site_group"] == "lis_buoy"}

# 2026-09-28: leak-free scoring. Features use the prior-years climatology, and the quantile rescaling
# is fit on the calibration years only (score(..., pack=...)); the fixed-0.50 rule fits it on the
# first year and scores the later years. Before, both were fit on the whole record being scored.
rows = []
for mname, pack in PACKS.items():
    for sid, site in sites.items():
        hist = pd.read_csv(f"data/prospective/history/{sid}.csv", parse_dates=["datetime"])
        day = ps.build_site_daily(site, hist, pa, clim="prior_years")   # daily means, LIS stuck-day drop
        # (A) section-24 protocol: threshold picked on calibration years
        r = score(sid, "UConn merlin", day, pack=pack)
        if r: rows.append(dict(r, rule="t* on calibration years", station=sid, model=mname))
        # (B) prospective primary: fixed threshold, all years after the first (calibration) year
        d = day.sort_values("date").reset_index(drop=True); d["year"] = d.date.dt.year
        y0 = d.year.min()
        scored = pa.rescale_chl(d, pack["chl_quantiles"], fit_chl=d.loc[d.year == y0, "chl"])
        d["bloom_prob"] = pack["model"].predict_proba(
            scored[pack["features"]].fillna(pd.Series(pack["medians"])).fillna(0.0).values)[:, 1]
        d["thr"] = d.chl.quantile(0.75)
        lab = add_label(d, "thr").dropna(subset=["bloom_fwd"]); on = lab[lab.chl <= lab.thr]
        test = on[on.year > y0]
        tf = float(pack["threshold"])
        m = metrics(test.bloom_fwd, test.bloom_prob >= tf)
        m.update(boot_ci(test, "bloom_prob", "bloom_fwd", tf))
        m.update(rule=f"fixed {tf:.2f} (model's shipped threshold)", station=sid, model=mname, n_onset=len(test),
                 auc=roc_auc_score(test.bloom_fwd, test.bloom_prob),
                 years=round((test.date.max() - test.date.min()).days / 365.25, 1), t_star=tf, dataset_id=sid)
        rows.append(m)
out = pd.DataFrame(rows)
cols = ["model", "station", "rule", "years", "n_onset", "base_rate", "precision", "precision_lo", "precision_hi", "pod", "lift", "lift_lo", "lift_hi", "auc", "t_star"]
cols = [c for c in cols if c in out.columns]
out[cols].to_csv("data/transfer/lis_buoys_zero_shot.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
print(out[cols].to_string(index=False))
print("\nSection 12 (recipe RETRAINED on these buoys, p95 label, test 2025-26): LR lift 1.90 / HistGB 2.18, precision 0.16-0.18, base 0.08")
