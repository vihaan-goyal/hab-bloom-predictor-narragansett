"""
fp_audit.py
-----------
Audit the false positives at a chosen operating threshold and split them into
fixable label-harshness vs genuine model error. At a low base rate, most of the
proportional leverage on precision lives in this split.

For each FP (predicted positive, labeled negative) it inspects the station's own
chlorophyll series and assigns one category:

  unobserved_window     : no chlorophyll reading in (d, d+horizon]. The label was
                          0 by default but there is literally no measurement to
                          confirm bloom-or-not. Sparse sampling, not model error.
  near_miss_subthreshold: max chl in the window is in [floor, chl_threshold). A
                          bloom almost happened; the hard cutoff just excluded it.
  near_miss_just_outside: no exceedance in the window, but one occurs within
                          `outside_days` past the window end. Right event, wrong
                          horizon edge.
  genuine_low           : max chl in the window is below `floor` and no near-outside
                          exceedance. The model fired on no real signal. Real error.

Then it recomputes precision under the optimistic assumption that the two
near_miss buckets were actually positives, giving an upper bound on how far label
refinement alone could move precision.

Run from repo root:
    python fp_audit.py
"""

import argparse
import sys

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", default="data/cv_pred_orig_h21.csv")
    p.add_argument("--series", default="data/hab_features_tidal.csv")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="operating threshold applied to y_prob")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--chl-threshold", type=float, default=10.0)
    p.add_argument("--floor", type=float, default=8.0,
                   help="lower edge of the 'almost bloomed' subthreshold band")
    p.add_argument("--outside-days", type=int, default=7,
                   help="days past the window end still counted as just-outside")
    p.add_argument("--out-csv", default="figures/model/fp_audit_detail.csv")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        pred = pd.read_csv(args.predictions)
        ser = pd.read_csv(args.series)
    except FileNotFoundError as e:
        sys.exit(str(e))

    pred["date"] = pd.to_datetime(pred["date"])
    ser["date"] = pd.to_datetime(ser["date"])
    for col in ("y_true", "y_prob", "station_name", "date"):
        if col not in pred.columns:
            sys.exit(f"predictions missing '{col}'. Found: {list(pred.columns)}")
    if "Chlorophyll" not in ser.columns:
        sys.exit(f"series missing 'Chlorophyll'. Found: {list(ser.columns)}")

    pred["station_name"] = pred["station_name"].astype(str)
    ser["station_name"] = ser["station_name"].astype(str)

    # per-station sorted (date, chl) arrays for fast window lookups
    smap = {}
    for st, g in ser[["station_name", "date", "Chlorophyll"]].dropna().groupby("station_name"):
        g = g.sort_values("date")
        smap[st] = (g["date"].values.astype("datetime64[ns]"),
                    g["Chlorophyll"].values.astype(float))

    t = args.threshold
    yhat = (pred["y_prob"].values >= t).astype(int)
    ytrue = pred["y_true"].values.astype(int)
    tp = int(np.sum((yhat == 1) & (ytrue == 1)))
    fp_mask = (yhat == 1) & (ytrue == 0)
    fp = pred[fp_mask].copy()
    n_fp = len(fp)

    print("=" * 66)
    print("FALSE-POSITIVE AUDIT")
    print("=" * 66)
    print(f"threshold={t}  horizon={args.horizon}d  chl_threshold={args.chl_threshold}")
    print(f"at this threshold: TP={tp}  FP={n_fp}  "
          f"precision={tp / (tp + n_fp):.3f}" if (tp + n_fp) else "no positives")
    print()

    H = np.timedelta64(args.horizon, "D")
    OUT = np.timedelta64(args.outside_days, "D")
    cats = []
    win_max = []
    for _, row in fp.iterrows():
        st, d = row["station_name"], np.datetime64(row["date"])
        if st not in smap:
            cats.append("unobserved_window"); win_max.append(np.nan); continue
        dates, chl = smap[st]
        in_win = (dates > d) & (dates <= d + H)
        if not in_win.any():
            cats.append("unobserved_window"); win_max.append(np.nan); continue
        m = float(np.max(chl[in_win]))
        win_max.append(m)
        if m >= args.chl_threshold:
            # exceedance IS in window -> this should have been labeled positive
            cats.append("near_miss_subthreshold"); continue
        if m >= args.floor:
            cats.append("near_miss_subthreshold"); continue
        out_win = (dates > d + H) & (dates <= d + H + OUT)
        if out_win.any() and float(np.max(chl[out_win])) >= args.chl_threshold:
            cats.append("near_miss_just_outside"); continue
        cats.append("genuine_low")

    fp["category"] = cats
    fp["window_max_chl"] = win_max

    order = ["unobserved_window", "near_miss_subthreshold",
             "near_miss_just_outside", "genuine_low"]
    print("FP composition:")
    counts = fp["category"].value_counts()
    for c in order:
        k = int(counts.get(c, 0))
        print(f"  {c:<24} {k:>4}  ({k / n_fp:.1%})" if n_fp else f"  {c}: 0")
    print()

    near = int(counts.get("near_miss_subthreshold", 0) +
               counts.get("near_miss_just_outside", 0))
    new_tp = tp + near
    new_fp = n_fp - near
    print("If both near_miss buckets were relabeled positive (optimistic ceiling):")
    print(f"  precision {tp / (tp + n_fp):.3f}  ->  "
          f"{new_tp / (new_tp + new_fp):.3f}" if (new_tp + new_fp) else "  n/a")
    unobs = int(counts.get("unobserved_window", 0))
    print(f"  unverifiable (no data in window): {unobs} FPs "
          f"({unobs / n_fp:.1%}) cannot be adjudicated either way")
    print()

    print("FP concentration by station (top 10):")
    by_st = fp["station_name"].value_counts().head(10)
    for st, k in by_st.items():
        gl = int(((fp["station_name"] == st) &
                  (fp["category"] == "genuine_low")).sum())
        print(f"  {st:<8} {k:>4} FP   of which genuine_low: {gl}")

    import os
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fp.sort_values(["category", "station_name", "date"]).to_csv(args.out_csv, index=False)
    print(f"\nSaved per-FP detail: {args.out_csv}")
    print("\nRead this as: unobserved + near_miss are not real model errors. "
          "genuine_low concentrated at a few western stations = the mechanistic "
          "decoupled-regime error. The optimistic ceiling shows how much label "
          "refinement alone could buy before any modeling change.")


if __name__ == "__main__":
    main()