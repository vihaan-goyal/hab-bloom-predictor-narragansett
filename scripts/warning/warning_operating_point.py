"""
warning_operating_point.py

Early-warning operating point for the LOCKED pipeline, computed entirely from
the pipeline's own exported predictions. No model is trained here; nothing can
drift from the locked implementation.

Inputs
  data/cv_pred_orig_h21.csv    rolling-origin CV predictions (out-of-sample
                               per fold) -> threshold SELECTION on 2020-2022
  data/test_predictions.csv    locked test predictions (2023-2025)
                               -> scored ONCE at the frozen threshold

Selection rule (stated before looking at test): the HIGHEST threshold whose
selection-period POD >= --target-pod. Among thresholds that meet the
detection target, the highest one gives the fewest false alarms.

Metrics reported in both vocabularies:
  ML:          precision / recall / F1
  Operational: POD (= recall), FAR (= 1 - precision, false alarm RATIO,
               NOT the false positive rate), CSI = TP/(TP+FP+FN)

Usage:
    python scripts/warning/warning_operating_point.py
    python scripts/warning/warning_operating_point.py --target-pod 0.8
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VAL_YEARS = (2020, 2022)


def parse_args():
    p = argparse.ArgumentParser(description="Model-free warning operating point.")
    p.add_argument("--cv-preds", default="data/cv_pred_orig_h21.csv")
    p.add_argument("--test-preds", default="data/test_predictions.csv")
    p.add_argument("--test-from-cv", action="store_true",
                   help="score test on the CV file's out-of-sample rows after "
                        "the selection years (keeps one label horizon "
                        "throughout; use when the separate test file was "
                        "built at a different horizon)")
    p.add_argument("--target-pod", type=float, default=0.8)
    p.add_argument("--out-csv", default="data/warning_operating_point_locked.csv")
    p.add_argument("--out-fig", default="figures/warning_operating_point_locked.png")
    return p.parse_args()


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) else np.nan)
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    far = 1 - prec if prec == prec else np.nan   # false alarm RATIO
    return {"threshold": t, "POD": rnd(rec), "FAR": rnd(far), "CSI": rnd(csi),
            "precision": rnd(prec), "F1": rnd(f1),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def rnd(x):
    return round(float(x), 3) if x == x else np.nan


def load_preds(path, ycol_candidates=("y_true",), pcol_candidates=("y_prob",)):
    df = pd.read_csv(path, dtype={"station_name": str})
    ycol = next((c for c in ycol_candidates if c in df.columns), None)
    pcol = next((c for c in pcol_candidates if c in df.columns), None)
    if ycol is None or pcol is None:
        raise SystemExit(f"{path}: need columns like y_true / y_prob; "
                         f"found {list(df.columns)}")
    if "date" not in df.columns:
        raise SystemExit(f"{path}: missing 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    return df, ycol, pcol


def main():
    a = parse_args()

    cv, ycol, pcol = load_preds(a.cv_preds)
    sel = cv[(cv["date"].dt.year >= VAL_YEARS[0])
             & (cv["date"].dt.year <= VAL_YEARS[1])]
    if len(sel) < 100:
        raise SystemExit(f"Only {len(sel)} CV rows in {VAL_YEARS}; "
                         "check the CV predictions file.")
    y_sel = sel[ycol].values.astype(int)
    p_sel = sel[pcol].values.astype(float)
    print(f"Selection set: {len(sel):,} out-of-sample CV rows "
          f"{VAL_YEARS[0]}-{VAL_YEARS[1]}  "
          f"(bloom rate {y_sel.mean():.3f}, folds {sorted(sel['fold'].unique()) if 'fold' in sel else 'n/a'})")

    rows = [metrics(y_sel, p_sel, t)
            for t in np.round(np.arange(0.05, 0.91, 0.05), 2)]
    sweep = pd.DataFrame(rows)

    print("\nSELECTION SWEEP (out-of-sample CV, 2020-2022) - choose HERE")
    print("=" * 74)
    print(sweep.to_string(index=False))

    ok = sweep[sweep["POD"] >= a.target_pod]
    if len(ok):
        chosen = ok.loc[ok["threshold"].idxmax()]
        note = f"highest threshold with selection POD >= {a.target_pod}"
    else:
        chosen = sweep.loc[sweep["POD"].idxmax()]
        note = (f"WARNING: no threshold reached POD {a.target_pod}; "
                f"best available POD = {chosen['POD']}")
    t_star = float(chosen["threshold"])
    print(f"\nFROZEN operating point: t* = {t_star}  ({note})")
    print(f"  selection: POD={chosen['POD']}  FAR={chosen['FAR']}  "
          f"CSI={chosen['CSI']}  precision={chosen['precision']}")

    if a.test_from_cv:
        te = cv[cv["date"].dt.year > VAL_YEARS[1]].copy()
        ycol_t, pcol_t = ycol, pcol
        if len(te) < 100:
            raise SystemExit(f"Only {len(te)} CV rows after {VAL_YEARS[1]}; "
                             "CV file may not cover the test years.")
        print(f"\n(test rows drawn from CV file, same horizon as selection)")
    else:
        te, ycol_t, pcol_t = load_preds(a.test_preds)
    yrs = (te["date"].dt.year.min(), te["date"].dt.year.max())
    m_te = metrics(te[ycol_t].values.astype(int),
                   te[pcol_t].values.astype(float), t_star)
    print(f"\nTEST ({yrs[0]}-{yrs[1]}, {len(te):,} rows) at frozen t* - "
          "reported once, never tuned")
    print("=" * 74)
    for k in ["POD", "FAR", "CSI", "precision", "F1", "TP", "FP", "FN", "TN"]:
        print(f"  {k:>10}: {m_te[k]}")

    out = sweep.copy()
    out["chosen"] = out["threshold"] == t_star
    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    out.to_csv(a.out_csv, index=False)
    pd.DataFrame([m_te]).to_csv(a.out_csv.replace(".csv", "_test.csv"),
                                index=False)
    print(f"\nSaved sweep + test result to {a.out_csv} (and _test.csv)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["threshold"], sweep["POD"], marker="o", ms=3,
            label="POD (recall)", color="#2a78d6")
    ax.plot(sweep["threshold"], sweep["FAR"], marker="o", ms=3,
            label="FAR (1 - precision)", color="#E24B4A")
    ax.plot(sweep["threshold"], sweep["CSI"], marker="o", ms=3,
            label="CSI", color="#1D9E75")
    ax.axvline(t_star, color="#52514e", ls="--", lw=1,
               label=f"frozen t* = {t_star}")
    ax.axhline(a.target_pod, color="#2a78d6", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Alert threshold")
    ax.set_ylabel(f"Metric (selection set {VAL_YEARS[0]}-{VAL_YEARS[1]})")
    ax.set_title("Early-warning operating point (locked model, "
                 "out-of-sample selection)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out_fig) or ".", exist_ok=True)
    fig.savefig(a.out_fig, dpi=150)
    print(f"Saved figure to {a.out_fig}")


if __name__ == "__main__":
    main()