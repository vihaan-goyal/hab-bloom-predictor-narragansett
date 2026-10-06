"""
grouped_station_report.py

Paper-ready two-group summary of test performance at the frozen t*:
stations grouped by PRE-TEST bloom history (>= --min-events events in the
2015-2022 out-of-sample rows), the same pre-registered definition used by
the gate experiment. Used here for REPORTING only -- the alert policy
remains ungated.

Usage:
    python scripts/warning/grouped_station_report.py --t-star 0.35
"""

import argparse
import numpy as np
import pandas as pd

TEST_START = 2023


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-preds", default="data/cv_pred_orig_h21.csv")
    p.add_argument("--t-star", type=float, required=True)
    p.add_argument("--min-events", type=int, default=2)
    p.add_argument("--out-csv", default="data/grouped_station_report.csv")
    return p.parse_args()


def metrics(y, p, t):
    a = (p >= t).astype(int)
    tp = int(((a == 1) & (y == 1)).sum()); fp = int(((a == 1) & (y == 0)).sum())
    fn = int(((a == 0) & (y == 1)).sum()); tn = int(((a == 0) & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    r = lambda x: round(x, 3) if x == x else np.nan
    return {"rows": len(y), "events": tp + fn, "alerts": tp + fp,
            "POD": r(pod), "precision": r(prec),
            "FAR": r(1 - prec) if prec == prec else np.nan,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def main():
    a = parse_args()
    df = pd.read_csv(a.cv_preds, dtype={"station_name": str})
    df["date"] = pd.to_datetime(df["date"])
    yr = df["date"].dt.year
    pre, test = df[yr < TEST_START], df[yr >= TEST_START]

    ev = pre.groupby("station_name")["y_true"].sum().astype(int)
    recurring = set(ev[ev >= a.min_events].index)
    print(f"Grouping (pre-test, {pre['date'].dt.year.min()}-"
          f"{pre['date'].dt.year.max()}): recurring = >= {a.min_events} events")
    print(f"  recurring ({len(recurring)}): {', '.join(sorted(recurring))}\n")

    rows = []
    for name, sub in [
        ("recurring-bloom stations", test[test["station_name"].isin(recurring)]),
        ("other stations", test[~test["station_name"].isin(recurring)]),
        ("ALL (pooled)", test),
    ]:
        m = metrics(sub["y_true"].values.astype(int),
                    sub["y_prob"].values.astype(float), a.t_star)
        rows.append({"group": name, **m})
    out = pd.DataFrame(rows)
    print(f"TEST {test['date'].dt.year.min()}-{test['date'].dt.year.max()} "
          f"at frozen t* = {a.t_star}  (reporting groups, ungated policy)")
    print("=" * 78)
    print(out.to_string(index=False))
    out.to_csv(a.out_csv, index=False)
    print(f"\nSaved to {a.out_csv}")


if __name__ == "__main__":
    main()