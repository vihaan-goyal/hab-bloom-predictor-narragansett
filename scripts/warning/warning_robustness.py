"""
warning_robustness.py

More instances for the frozen early-warning operating point (t* from
warning_operating_point.py), without touching the selection years.

  1. TEST BOOTSTRAP: station-year clustered bootstrap CIs for POD / FAR /
     CSI / precision on the 2023-2025 out-of-sample rows.
  2. EXTRA YEARS: evaluates the frozen t* on the 2015-2019 out-of-sample CV
     rows, which were never used for selection. A genuinely independent
     check, with the caveat that those years straddle a different ecological
     regime (higher base rates, pre/early decoupling).
  3. PER-STATION: POD / FAR / event counts by station on the test years.

Usage:
    python scripts/warning/warning_robustness.py --t-star 0.35
"""

import argparse
import numpy as np
import pandas as pd

VAL_YEARS = (2020, 2022)
N_BOOT = 2000
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-preds", default="data/cv_pred_orig_h21.csv")
    p.add_argument("--t-star", type=float, required=True)
    p.add_argument("--out-csv", default="data/warning_robustness.csv")
    return p.parse_args()


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    far = 1 - prec if prec == prec else np.nan
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    return {"POD": pod, "FAR": far, "CSI": csi, "precision": prec,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def rnd(d):
    return {k: (round(v, 3) if isinstance(v, float) and v == v else v)
            for k, v in d.items()}


def cluster_bootstrap(df, t, n_boot=N_BOOT, seed=SEED):
    df = df.reset_index(drop=True)
    key = df["station_name"].astype(str) + "_" + df["date"].dt.year.astype(str)
    groups = [np.array(v) for v in df.groupby(key).indices.values()]
    y = df["y_true"].values.astype(int)
    p = df["y_prob"].values.astype(float)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["POD", "FAR", "CSI", "precision"]}
    skipped = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[c] for c in chosen])
        m = metrics(y[idx], p[idx], t)
        if m["POD"] != m["POD"] or m["precision"] != m["precision"]:
            skipped += 1
            continue
        for k in out:
            out[k].append(m[k])
    ci = {}
    for k, v in out.items():
        v = np.array(v)
        ci[k] = (float(np.mean(v)),
                 float(np.percentile(v, 2.5)),
                 float(np.percentile(v, 97.5)))
    return ci, skipped


def main():
    a = parse_args()
    df = pd.read_csv(a.cv_preds, dtype={"station_name": str})
    df["date"] = pd.to_datetime(df["date"])
    yr = df["date"].dt.year

    test = df[yr > VAL_YEARS[1]]
    early = df[yr < VAL_YEARS[0]]
    t = a.t_star

    print(f"Frozen t* = {t}\n")

    # ---- 1. test bootstrap ----
    m_te = metrics(test["y_true"].values.astype(int),
                   test["y_prob"].values.astype(float), t)
    print(f"TEST {test['date'].dt.year.min()}-{test['date'].dt.year.max()}  "
          f"({len(test):,} rows, {int(m_te['TP']+m_te['FN'])} events)")
    print("  point:", rnd(m_te))
    ci, skipped = cluster_bootstrap(test, t)
    print(f"  station-year clustered bootstrap ({N_BOOT} resamples, "
          f"{skipped} skipped):")
    for k, (mean, lo, hi) in ci.items():
        print(f"    {k:>9}: mean={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    # ---- 2. untouched early out-of-sample years ----
    m_ea = metrics(early["y_true"].values.astype(int),
                   early["y_prob"].values.astype(float), t)
    print(f"\nEXTRA OUT-OF-SAMPLE YEARS "
          f"{early['date'].dt.year.min()}-{early['date'].dt.year.max()}  "
          f"({len(early):,} rows, {int(m_ea['TP']+m_ea['FN'])} events, "
          f"never used for selection)")
    print("  point:", rnd(m_ea))
    print("  caveat: different ecological regime (pre/early post-TMDL), "
          "base rate "
          f"{early['y_true'].mean():.3f} vs test {test['y_true'].mean():.3f}")

    # ---- 3. per-station on test ----
    rows = []
    for st, g in test.groupby("station_name"):
        m = metrics(g["y_true"].values.astype(int),
                    g["y_prob"].values.astype(float), t)
        rows.append({"station": st, "n": len(g),
                     "events": m["TP"] + m["FN"], **rnd(m)})
    per = (pd.DataFrame(rows)
           .sort_values("events", ascending=False)
           .reset_index(drop=True))
    print(f"\nPER-STATION (test years, t*={t}; stations with 0 events "
          "show POD=nan)")
    cols = ["station", "n", "events", "POD", "FAR", "precision",
            "TP", "FP", "FN"]
    print(per[cols].to_string(index=False))

    per.to_csv(a.out_csv, index=False)
    print(f"\nSaved per-station table to {a.out_csv}")


if __name__ == "__main__":
    main()