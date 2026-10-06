"""
warning_station_gate.py

Station-gated alerting for the frozen early-warning operating point.

PRE-REGISTERED RULE (stated before test is examined): alerts are only issued
at stations with >= --min-events observed exceedance events in the PRE-TEST
out-of-sample years (2015-2022 CV rows). At all other stations the system
stays silent. The probability threshold t* is unchanged.

The gate uses ONLY pre-2023 information. Test (2023-2025) is evaluated once,
gated vs ungated, side by side. Events at gated-out stations still count as
misses (FN) for the gated system -- silence at a station does not delete its
blooms from the scorecard.

Usage:
    python scripts/warning/warning_station_gate.py --t-star 0.35
    python scripts/warning/warning_station_gate.py --t-star 0.35 --min-events 2
"""

import argparse
import numpy as np
import pandas as pd

TEST_START = 2023


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-preds", default="data/cv_pred_orig_h21.csv")
    p.add_argument("--t-star", type=float, required=True)
    p.add_argument("--min-events", type=int, default=2,
                   help="min events in 2015-2022 for a station to be alertable")
    p.add_argument("--out-csv", default="data/warning_station_gate.csv")
    return p.parse_args()


def metrics(y, alert):
    tp = int(((alert == 1) & (y == 1)).sum())
    fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum())
    tn = int(((alert == 0) & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    far = 1 - prec if prec == prec else np.nan
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    r = lambda x: round(x, 3) if x == x else np.nan
    return {"POD": r(pod), "FAR": r(far), "CSI": r(csi),
            "precision": r(prec), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "alerts": tp + fp}


def main():
    a = parse_args()
    df = pd.read_csv(a.cv_preds, dtype={"station_name": str})
    df["date"] = pd.to_datetime(df["date"])
    yr = df["date"].dt.year

    pre = df[yr < TEST_START]
    test = df[yr >= TEST_START]

    # ---- gate from PRE-TEST years only ----
    ev = pre.groupby("station_name")["y_true"].sum().astype(int)
    gated_in = sorted(ev[ev >= a.min_events].index.tolist())
    gated_out = sorted(ev[ev < a.min_events].index.tolist())
    print(f"GATE RULE: stations with >= {a.min_events} events in "
          f"{pre['date'].dt.year.min()}-{pre['date'].dt.year.max()} "
          "(pre-test out-of-sample rows only)")
    print(f"  alertable stations ({len(gated_in)}): {', '.join(gated_in)}")
    print(f"  silenced stations  ({len(gated_out)}): {', '.join(gated_out)}")

    y = test["y_true"].values.astype(int)
    p = test["y_prob"].values.astype(float)
    in_gate = test["station_name"].isin(gated_in).values

    alert_ungated = (p >= a.t_star).astype(int)
    alert_gated = (alert_ungated & in_gate).astype(int)

    m_u = metrics(y, alert_ungated)
    m_g = metrics(y, alert_gated)

    # events at silenced stations (these become automatic FNs when gated)
    ev_out = int(y[~in_gate].sum())

    print(f"\nTEST {test['date'].dt.year.min()}-{test['date'].dt.year.max()} "
          f"at frozen t* = {a.t_star}   ({len(test):,} rows, "
          f"{int(y.sum())} events, {ev_out} of them at silenced stations)")
    print("=" * 74)
    tbl = pd.DataFrame([{"system": "ungated", **m_u},
                        {"system": "gated", **m_g}])
    print(tbl.to_string(index=False))

    d_far = m_g["FAR"] - m_u["FAR"]
    d_pod = m_g["POD"] - m_u["POD"]
    d_alerts = m_g["alerts"] - m_u["alerts"]
    print(f"\nDELTA (gated minus ungated): "
          f"FAR {d_far:+.3f}   POD {d_pod:+.3f}   alerts {d_alerts:+d}")

    tbl.to_csv(a.out_csv, index=False)
    print(f"Saved to {a.out_csv}")


if __name__ == "__main__":
    main()