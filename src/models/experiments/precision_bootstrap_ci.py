"""
precision_bootstrap_ci.py

Bootstrap confidence intervals for precision and recall at fixed thresholds,
plus a precision-recall curve with confidence band and operating points marked.

Purpose
-------
Decide whether the reported precision plateau is statistically real or sitting
inside the noise, BEFORE spending effort trying to push it. This is the gate the
rest of the precision-push list has to pass.

Input
-----
Consumes the pooled predictions written by rolling_origin_cv.py
(default: data/cv_predictions.csv), one row per out-of-sample prediction.
Required columns:
    y_true : 0/1 ground-truth bloom label
    y_prob : predicted probability of bloom
Optional clustering columns (for the block bootstrap):
    fold   : test year (rolling_origin_cv.py writes this)
    station_name

Bootstrap variants reported
---------------------------
    iid   : resample rows with replacement. Assumes independence. A floor; it
            understates the true interval here.
    block : resample whole clusters with replacement. Set the cluster with
            --block-cols. Two readings worth comparing:
              --block-cols fold              (year-level: conservative/honest,
                                              respects cross-station correlation
                                              within a year, but only 11 clusters)
              --block-cols fold station_name (station_year: matches the harness,
                                              more stable but can be overconfident)
            The truth is bracketed between these two.

Note on operating point
-----------------------
This script applies a SINGLE GLOBAL threshold to the pooled probabilities to
trace the PR curve and the precision-vs-threshold CI. That is a different object
from the harness's pooled precision, which applies each fold's val-chosen
threshold. Both are valid; the numbers will not match exactly, and that is
expected. This view is the right one for "is the plateau distinguishable".

No hardcoded paths or values. Everything is configurable via argparse.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", default="data/cv_predictions.csv")
    p.add_argument("--label-col", default="y_true")
    p.add_argument("--prob-col", default="y_prob")
    p.add_argument("--block-cols", nargs="+", default=None,
                   help="Columns defining a bootstrap cluster. Default: auto "
                        "(['fold'] if present, else ['year']). Pass "
                        "'fold station_name' to match the harness's station_year.")
    p.add_argument("--locked-threshold", type=float, default=0.60)
    p.add_argument("--sweep-lo", type=float, default=0.30)
    p.add_argument("--sweep-hi", type=float, default=0.90)
    p.add_argument("--sweep-step", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Two-sided alpha. 0.05 -> 95%% CI.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-fig", default="figures/model/precision_bootstrap_ci.png")
    p.add_argument("--out-csv", default="figures/model/threshold_ci_table.csv")
    return p.parse_args()


def prec_rec_at(y_true, y_prob, t):
    """Precision, recall, counts at threshold t. Precision NaN if no positives."""
    pred = y_prob >= t
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    npp = tp + fp
    precision = tp / npp if npp > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return precision, recall, tp, fp, fn, npp


def percentile_ci(arr, alpha):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan, 0
    lo = np.percentile(arr, 100 * (alpha / 2))
    hi = np.percentile(arr, 100 * (1 - alpha / 2))
    return lo, hi, arr.size


def iid_indices(n, n_boot, rng):
    for _ in range(n_boot):
        yield rng.integers(0, n, size=n)


def block_indices(groups, n_boot, rng):
    """Resample whole clusters with replacement; concat their row indices."""
    g = len(groups)
    for _ in range(n_boot):
        chosen = rng.integers(0, g, size=g)
        yield np.concatenate([groups[c] for c in chosen])


def bootstrap_scalar(y_true, y_prob, t, index_iter):
    precs, recs = [], []
    for idx in index_iter:
        pr, rc, *_ = prec_rec_at(y_true[idx], y_prob[idx], t)
        precs.append(pr)
        recs.append(rc)
    return np.array(precs), np.array(recs)


def bootstrap_pr_band(y_true, y_prob, index_iter, recall_grid):
    curves = []
    for idx in index_iter:
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        prec, rec, _ = precision_recall_curve(yt, yp)
        order = np.argsort(rec)
        curves.append(np.interp(recall_grid, rec[order], prec[order]))
    if not curves:
        return None, None
    curves = np.vstack(curves)
    return np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)


def resolve_block_cols(df, requested):
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            sys.exit(f"--block-cols not in file: {missing}. "
                     f"Found: {list(df.columns)}")
        return requested
    for cand in (["fold"], ["year"]):
        if all(c in df.columns for c in cand):
            return cand
    return None


def main():
    args = parse_args()

    try:
        df = pd.read_csv(args.predictions)
    except FileNotFoundError:
        sys.exit(f"Not found: {args.predictions}\n"
                 "Run rolling_origin_cv.py first; it writes data/cv_predictions.csv.")

    for col in (args.label_col, args.prob_col):
        if col not in df.columns:
            sys.exit(f"Column '{col}' missing. Found: {list(df.columns)}")

    y_true = df[args.label_col].to_numpy().astype(int)
    y_prob = df[args.prob_col].to_numpy().astype(float)
    n = len(df)
    n_pos = int(y_true.sum())

    block_cols = resolve_block_cols(df, args.block_cols)
    if block_cols:
        key = df[block_cols].astype(str).agg("_".join, axis=1)
        groups = [np.array(v) for v in df.groupby(key).indices.values()]
    else:
        groups = None

    print("=" * 70)
    print("Precision bootstrap CI")
    print("=" * 70)
    print(f"rows={n}  positives={n_pos}  positive_rate={n_pos / n:.3f}")
    print(f"n_boot={args.n_boot}  CI={100 * (1 - args.alpha):.0f}%  seed={args.seed}")
    if groups:
        print(f"block cluster = {block_cols}  ->  {len(groups)} clusters")
    else:
        print("WARNING: no block columns found; only iid CIs (understated width).")
    print()

    ap = average_precision_score(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    rng = np.random.default_rng(args.seed)
    ap_b, auc_b = [], []
    for idx in iid_indices(n, args.n_boot, rng):
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        ap_b.append(average_precision_score(yt, yp))
        auc_b.append(roc_auc_score(yt, yp))
    ap_lo, ap_hi, _ = percentile_ci(ap_b, args.alpha)
    auc_lo, auc_hi, _ = percentile_ci(auc_b, args.alpha)
    print(f"Average precision (PR-AUC): {ap:.3f}  CI [{ap_lo:.3f}, {ap_hi:.3f}]")
    print(f"ROC-AUC                   : {auc:.3f}  CI [{auc_lo:.3f}, {auc_hi:.3f}]")
    print()

    thresholds = np.round(
        np.arange(args.sweep_lo, args.sweep_hi + 1e-9, args.sweep_step), 4)
    if args.locked_threshold not in thresholds:
        thresholds = np.sort(np.append(thresholds, args.locked_threshold))

    rows = []
    use_blk = groups is not None
    hdr_blk = "prec_blk_CI" if use_blk else "prec_blk_CI(NA)"
    print(f"{'thr':>5} {'npos':>5} {'prec':>6} {'prec_iid_CI':>16} "
          f"{hdr_blk:>16} {'rec':>6} {'rec_blk_CI':>16}")
    for t in thresholds:
        pr, rc, tp, fp, fn, npp = prec_rec_at(y_true, y_prob, t)

        rng_i = np.random.default_rng(args.seed)
        p_iid, _ = bootstrap_scalar(y_true, y_prob, t,
                                    iid_indices(n, args.n_boot, rng_i))
        p_iid_lo, p_iid_hi, _ = percentile_ci(p_iid, args.alpha)

        if use_blk:
            rng_b = np.random.default_rng(args.seed + 1)
            p_blk, r_blk = bootstrap_scalar(y_true, y_prob, t,
                                            block_indices(groups, args.n_boot, rng_b))
            p_blk_lo, p_blk_hi, _ = percentile_ci(p_blk, args.alpha)
            r_blk_lo, r_blk_hi, _ = percentile_ci(r_blk, args.alpha)
        else:
            p_blk_lo = p_blk_hi = r_blk_lo = r_blk_hi = np.nan

        mk = "  <-- locked" if abs(t - args.locked_threshold) < 1e-9 else ""
        print(f"{t:>5.2f} {npp:>5d} {pr:>6.3f} "
              f"[{p_iid_lo:>5.3f},{p_iid_hi:>5.3f}] "
              f"[{p_blk_lo:>5.3f},{p_blk_hi:>5.3f}] "
              f"{rc:>6.3f} [{r_blk_lo:>5.3f},{r_blk_hi:>5.3f}]{mk}")

        rows.append({"threshold": t, "n_pred_pos": npp, "tp": tp, "fp": fp,
                     "fn": fn, "precision": pr,
                     "prec_iid_lo": p_iid_lo, "prec_iid_hi": p_iid_hi,
                     "prec_blk_lo": p_blk_lo, "prec_blk_hi": p_blk_hi,
                     "recall": rc, "rec_blk_lo": r_blk_lo, "rec_blk_hi": r_blk_hi})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nSaved threshold table: {args.out_csv}")

    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
    recall_grid = np.linspace(0, 1, 101)
    band_iter = (block_indices(groups, args.n_boot, np.random.default_rng(args.seed + 2))
                 if use_blk
                 else iid_indices(n, args.n_boot, np.random.default_rng(args.seed + 2)))
    band_lo, band_hi = bootstrap_pr_band(y_true, y_prob, band_iter, recall_grid)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    axL.plot(rec_c, prec_c, color="#1f4e79", lw=2, label="PR curve")
    if band_lo is not None:
        axL.fill_between(recall_grid, band_lo, band_hi, color="#1f4e79",
                         alpha=0.18, label="95% band")
    axL.axhline(n_pos / n, ls="--", color="gray", lw=1,
                label=f"no-skill ({n_pos / n:.2f})")
    for _, r in out.iterrows():
        if not np.isnan(r["precision"]) and not np.isnan(r["recall"]):
            axL.scatter(r["recall"], r["precision"], s=18, color="#c0504d", zorder=5)
    locked = out[np.isclose(out["threshold"], args.locked_threshold)].iloc[0]
    axL.scatter(locked["recall"], locked["precision"], s=90, color="#c0504d",
                edgecolor="black", zorder=6, label=f"locked t={args.locked_threshold:.2f}")
    axL.set_xlabel("Recall"); axL.set_ylabel("Precision")
    axL.set_title(f"PR curve (AP={ap:.3f}, CI [{ap_lo:.3f}, {ap_hi:.3f}])")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axL.legend(loc="upper right", fontsize=9); axL.grid(alpha=0.25)

    axR.plot(out["threshold"], out["precision"], color="#1f4e79", lw=2,
             marker="o", ms=3, label="precision")
    blk_ok = use_blk and out["prec_blk_lo"].notna().all()
    lo_col = "prec_blk_lo" if blk_ok else "prec_iid_lo"
    hi_col = "prec_blk_hi" if blk_ok else "prec_iid_hi"
    axR.fill_between(out["threshold"], out[lo_col], out[hi_col], color="#1f4e79",
                     alpha=0.18, label="95% CI" + (" (block)" if blk_ok else " (iid)"))
    axR.plot(out["threshold"], out["recall"], color="#4f8a4f", lw=2,
             marker="s", ms=3, label="recall")
    axR.axvline(args.locked_threshold, ls="--", color="#c0504d", lw=1,
                label=f"locked t={args.locked_threshold:.2f}")
    axR.set_xlabel("Threshold"); axR.set_ylabel("Score")
    axR.set_title("Precision / recall vs threshold")
    axR.set_ylim(0, 1); axR.legend(loc="best", fontsize=9); axR.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"Saved figure: {args.out_fig}")

    print("\nRead this as: if the block precision CI at the locked threshold "
          "overlaps heavily with neighboring thresholds, the plateau is inside "
          "the noise. Compare fold-only vs fold+station_name clustering; the "
          "truth is bracketed between them.")


if __name__ == "__main__":
    main()