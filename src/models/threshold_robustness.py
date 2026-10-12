"""
threshold_robustness.py
-----------------------
ROBUSTNESS CHECK (not a re-tune) for the locked HAB pipeline: how stable are the
results if a "bloom day" is defined at chl > 12 ug/L instead of the locked 10 ug/L?

Nothing about the model changes: same LR, C=0.05, balanced weights, same 35
features, same rolling-origin split, same 0.60 operating threshold. We only vary
the LABEL definition and re-run the full evaluation under each definition, exactly
as final_evaluation_threshold_sweep.py would.

Produces:
  - side-by-side table (positive rate, precision, recall, F1, AUC at t=0.60) for
    label=10 vs label=12, global and per-station (C1, 02, 01, A4, B3)
  - paired station-year bootstrap AUC delta (12 minus 10) with 95% CI, using the
    same clustered resampling scheme as bootstrap_ci.py
  - data/threshold_robustness.csv
  - a one-line stability verdict

Run from repo root:
    python src/models/threshold_robustness.py
"""

import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OP_THRESHOLD = 0.60          # locked operating point -- do NOT change
STATIONS = ["C1", "02", "01", "A4", "B3"]
PREDS_10 = "data/test_predictions_t10.csv"
PREDS_12 = "data/test_predictions_t12.csv"
N_BOOT = 2000
SEED = 42


def run_eval(bloom_threshold, preds_out):
    """Re-run the locked evaluation script with a given label threshold.
    Config (C, features, CV, operating threshold) is identical between runs;
    only the chl cutoff that defines a bloom day differs."""
    print(f"\n>>> Running evaluation with bloom label = chl > {bloom_threshold} ug/L")
    subprocess.run(
        [sys.executable, "src/models/final_evaluation_threshold_sweep.py",
         "--bloom-threshold", str(bloom_threshold),
         "--preds-out", preds_out],
        check=True,
    )


def metrics_at(y_true, y_prob, t=OP_THRESHOLD):
    """precision / recall / F1 at operating threshold t, plus base positive rate
    and threshold-independent AUC. NaN where undefined (single-class subset)."""
    y_pred = (y_prob >= t).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * prec * rec / (prec + rec)
          if prec and rec and not np.isnan(prec) and not np.isnan(rec)
          and (prec + rec) > 0 else np.nan)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "pos_rate": round(float(np.mean(y_true)), 3),
        "precision": round(prec, 3) if not np.isnan(prec) else np.nan,
        "recall": round(rec, 3) if not np.isnan(rec) else np.nan,
        "f1": round(f1, 3) if not np.isnan(f1) else np.nan,
        "auc": round(auc, 3) if not np.isnan(auc) else np.nan,
        "n": int(len(y_true)), "TP": tp, "FP": fp, "FN": fn,
    }


def paired_auc_bootstrap(m, n_boot=N_BOOT, seed=SEED):
    """Paired station-year cluster bootstrap of AUC(label=12) - AUC(label=10).

    Same rows are used under both label definitions (only y_true/y_prob differ),
    so the delta is paired. Resampling whole station-years with replacement
    matches the 'station_year' clustering in bootstrap_ci.py, which respects
    within-station temporal autocorrelation and keeps the CI honest (wider)."""
    m = m.reset_index(drop=True)
    year = pd.to_datetime(m["date"]).dt.year.astype(str)
    key = m["station_name"].astype(str) + "_" + year
    groups = [np.array(v) for v in m.groupby(key).indices.values()]
    n_clusters = len(groups)

    yt10, yp10 = m["y_true_10"].values, m["y_prob_10"].values
    yt12, yp12 = m["y_true_12"].values, m["y_prob_12"].values

    rng = np.random.default_rng(seed)
    deltas, a10s, a12s = [], [], []
    for _ in range(n_boot):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[c] for c in chosen])
        if len(np.unique(yt10[idx])) < 2 or len(np.unique(yt12[idx])) < 2:
            continue  # AUC undefined on a single-class resample
        a10 = roc_auc_score(yt10[idx], yp10[idx])
        a12 = roc_auc_score(yt12[idx], yp12[idx])
        a10s.append(a10); a12s.append(a12); deltas.append(a12 - a10)
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_mean": float(np.mean(deltas)),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "auc10_mean": float(np.mean(a10s)), "auc12_mean": float(np.mean(a12s)),
        "n_valid": int(len(deltas)),
    }


def main():
    # Run label=12 first, then label=10 last so the shared side-effect artifacts
    # (data/threshold_sweep_results.csv, figures/model/threshold_sweep.png) are left in
    # their canonical locked-label=10 state after this driver finishes.
    run_eval(12, PREDS_12)
    run_eval(10, PREDS_10)

    d10 = pd.read_csv(PREDS_10, dtype={"station_name": str})
    d12 = pd.read_csv(PREDS_12, dtype={"station_name": str})
    m = d10.merge(d12, on=["station_name", "date"],
                  suffixes=("_10", "_12"), how="inner")
    print(f"\nPaired test rows (identical feature rows, both labels): {len(m):,}")

    # ---- side-by-side table ----
    rows = []
    for scope in ["GLOBAL"] + STATIONS:
        sub = m if scope == "GLOBAL" else m[m["station_name"] == scope]
        if len(sub) == 0:
            print(f"WARNING: station {scope} has no test rows; skipping.")
            continue
        for lab, yt, yp in [(10, "y_true_10", "y_prob_10"),
                            (12, "y_true_12", "y_prob_12")]:
            mtr = metrics_at(sub[yt].values.astype(int), sub[yp].values)
            rows.append({"scope": scope, "label_threshold": lab, **mtr})
    table = pd.DataFrame(rows)
    table.to_csv("data/threshold_robustness.csv", index=False)

    # ---- print side-by-side ----
    print("\n" + "=" * 78)
    print("THRESHOLD ROBUSTNESS: label chl>10 vs chl>12 ug/L   (metrics at t=0.60)")
    print("=" * 78)
    print(f"{'Scope':>7} {'Label':>6} {'PosRate':>8} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'AUC':>6} {'n':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 78)

    def fmt(x):
        return f"{x:>6.3f}" if isinstance(x, float) and not np.isnan(x) else f"{'nan':>6}"

    for scope in ["GLOBAL"] + STATIONS:
        for lab in (10, 12):
            r = table[(table["scope"] == scope) & (table["label_threshold"] == lab)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"{scope:>7} {lab:>6} {r['pos_rate']:>8.3f} {fmt(r['precision'])} "
                  f"{fmt(r['recall'])} {fmt(r['f1'])} {fmt(r['auc'])} "
                  f"{int(r['n']):>6} {int(r['TP']):>4} {int(r['FP']):>4} {int(r['FN']):>4}")
        print("-" * 78)

    # ---- paired bootstrap AUC delta ----
    bs = paired_auc_bootstrap(m)
    print("\nPAIRED STATION-YEAR BOOTSTRAP  (AUC label=12 minus label=10)")
    print(f"  global AUC(10) mean = {bs['auc10_mean']:.3f}   "
          f"AUC(12) mean = {bs['auc12_mean']:.3f}")
    print(f"  delta mean = {bs['delta_mean']:+.4f}   "
          f"95% CI = [{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}]   "
          f"({bs['n_valid']}/{N_BOOT} valid resamples)")

    # ---- verdict ----
    # Two distinct questions:
    #  (1) Is the model's DISCRIMINATION (AUC) stable to the label definition?
    #      -> judged by the paired bootstrap dAUC CI covering 0.
    #  (2) Is the FIXED 0.60 operating point stable?
    #      -> precision/recall/F1 at a fixed probability threshold are prevalence-
    #         dependent, so they shift mechanically when the base rate changes.
    g10 = table[(table["scope"] == "GLOBAL") & (table["label_threshold"] == 10)].iloc[0]
    g12 = table[(table["scope"] == "GLOBAL") & (table["label_threshold"] == 12)].iloc[0]
    d_auc = abs(g12["auc"] - g10["auc"])
    d_f1 = abs(g12["f1"] - g10["f1"])
    d_rate = g10["pos_rate"] - g12["pos_rate"]
    ci_covers_zero = bs["ci_lo"] <= 0 <= bs["ci_hi"]
    auc_stable = ci_covers_zero and d_auc < 0.03

    print("\n" + "=" * 78)
    disc = "STABLE" if auc_stable else "NOT STABLE"
    print(f"VERDICT: Discrimination is {disc} across the 10-12 ug/L bloom definition "
          f"(global dAUC={d_auc:.3f}, bootstrap dAUC CI "
          f"[{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}] "
          f"{'includes' if ci_covers_zero else 'excludes'} 0); the fixed 0.60 "
          f"operating point is prevalence-sensitive (base rate "
          f"{g10['pos_rate']*100:.1f}%->{g12['pos_rate']*100:.1f}%, global dF1={d_f1:.3f}), "
          f"so it would need re-tuning under a 12 ug/L label.")
    print("=" * 78)
    print("\nSaved data/threshold_robustness.csv")


if __name__ == "__main__":
    main()
