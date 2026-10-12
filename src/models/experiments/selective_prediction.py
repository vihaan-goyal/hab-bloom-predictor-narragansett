"""
selective_prediction.py
-----------------------
Two parts.

PART A - window-density check (folds in the audit caveat)
    Re-derives the FP categories AND records how many chlorophyll readings fell in
    each FP's forward window. If genuine_low FPs typically rest on a single reading,
    the category is thin and some of them are really unobserved. This guards against
    standing on single-reading windows before we act on the audit.

PART B - selective prediction / abstention
    Honest test of the audit's finding that error is station-concentrated. Abstention
    is decided ONLY from a station's WALK-FORWARD prior reliability (precision of its
    positive predictions in folds strictly earlier than the test fold). No test label
    ever informs the abstention rule. We then compare, on precision-vs-recall axes:
        baseline       : global probability threshold sweep
        station-select : fix the operating threshold, progressively abstain on the
                         least-reliable stations (worst prior-precision first)
    If station-select sits ABOVE baseline at matched recall, structure beats raising
    the threshold. If it lies on top, the threshold is all you need.

    Coverage is reported because abstention trades it away: a station you abstain on
    gets no alerts, so its true blooms count as missed (global recall).

Run from repo root:
    python selective_prediction.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", default="data/cv_pred_orig_h21.csv")
    p.add_argument("--series", default="data/hab_features_tidal.csv")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="operating threshold for the station-select arm")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--chl-threshold", type=float, default=10.0)
    p.add_argument("--floor", type=float, default=8.0)
    p.add_argument("--outside-days", type=int, default=7)
    p.add_argument("--min-prior", type=int, default=5,
                   help="min prior positive predictions to trust a station's "
                        "reliability; below this we COVER (never abstain for free)")
    p.add_argument("--n-boot", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-fig", default="figures/model/selective_prediction.png")
    return p.parse_args()


# ---------------- Part A ----------------

def window_density_check(pred, ser, args):
    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    ser = ser.copy(); ser["date"] = pd.to_datetime(ser["date"])
    smap = {}
    for st, g in ser[["station_name", "date", "Chlorophyll"]].dropna().groupby("station_name"):
        g = g.sort_values("date")
        smap[st] = (g["date"].values.astype("datetime64[ns]"),
                    g["Chlorophyll"].values.astype(float))

    t = args.threshold
    pp = pred["y_prob"].values >= t
    fp = pred[pp & (pred["y_true"].values == 0)].copy()
    H = np.timedelta64(args.horizon, "D")
    OUT = np.timedelta64(args.outside_days, "D")

    cat, ndens = [], []
    for _, r in fp.iterrows():
        st, d = str(r["station_name"]), np.datetime64(r["date"])
        if st not in smap:
            cat.append("unobserved_window"); ndens.append(0); continue
        dates, chl = smap[st]
        inw = (dates > d) & (dates <= d + H)
        k = int(inw.sum()); ndens.append(k)
        if k == 0:
            cat.append("unobserved_window"); continue
        m = float(np.max(chl[inw]))
        if m >= args.floor:
            cat.append("near_miss_subthreshold"); continue
        ow = (dates > d + H) & (dates <= d + H + OUT)
        if ow.any() and float(np.max(chl[ow])) >= args.chl_threshold:
            cat.append("near_miss_just_outside"); continue
        cat.append("genuine_low")
    fp["category"] = cat; fp["n_in_window"] = ndens

    print("=" * 66)
    print("PART A  window-density check on genuine_low")
    print("=" * 66)
    gl = fp[fp["category"] == "genuine_low"]["n_in_window"]
    if len(gl):
        on1 = int((gl == 1).sum())
        print(f"genuine_low FPs: {len(gl)}")
        print(f"  median readings in window: {gl.median():.0f}")
        print(f"  resting on exactly 1 reading: {on1} ({on1/len(gl):.1%})")
        print(f"  resting on >=3 readings:      {int((gl>=3).sum())} "
              f"({(gl>=3).mean():.1%})")
        if on1 / len(gl) > 0.5:
            print("  CAUTION: majority rest on a single reading -> some genuine_low")
            print("  is closer to unobserved. Treat the genuine_low count as a")
            print("  soft upper bound on real error.")
        else:
            print("  genuine_low mostly rests on multiple readings -> solid.")
    print()
    return fp


# ---------------- Part B ----------------

def station_reliability(df, t_op, min_prior):
    """Walk-forward: rel[(fold, station)] = precision of that station's positive
    predictions in folds strictly earlier than `fold`, or None if too few."""
    d = df.copy()
    d["pp"] = (d["y_prob"].values >= t_op).astype(int)
    rel = {}
    folds = sorted(d["fold"].unique())
    for T in folds:
        prior = d[d["fold"] < T]
        for s in d[d["fold"] == T]["station_name"].unique():
            sp = prior[(prior["station_name"] == s) & (prior["pp"] == 1)]
            rel[(T, s)] = sp["y_true"].mean() if len(sp) >= min_prior else None
    return rel


def evaluate_policy(df, t_op, rel, r):
    """Cover a row if its station has no trusted prior (None) or prior rel >= r.
    Abstain otherwise. Returns precision, global recall, coverage stats."""
    P = int(df["y_true"].sum())
    covered = np.array([(rel[(f, s)] is None) or (rel[(f, s)] >= r)
                        for f, s in zip(df["fold"], df["station_name"])])
    pp = df["y_prob"].values >= t_op
    acted = covered & pp
    yt = df["y_true"].values
    tp = int(np.sum(acted & (yt == 1)))
    fp = int(np.sum(acted & (yt == 0)))
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / P if P else np.nan
    cov_rows = covered.mean()
    cov_pos = np.sum(covered & (yt == 1)) / P if P else np.nan
    return dict(r=r, precision=prec, recall=rec, tp=tp, fp=fp,
               coverage_rows=cov_rows, coverage_pos=cov_pos)


def baseline_curve(df, thresholds):
    P = int(df["y_true"].sum())
    yt = df["y_true"].values; yp = df["y_prob"].values
    out = []
    for t in thresholds:
        pp = yp >= t
        tp = int(np.sum(pp & (yt == 1))); fp = int(np.sum(pp & (yt == 0)))
        out.append(dict(t=t, precision=tp/(tp+fp) if (tp+fp) else np.nan,
                        recall=tp/P if P else np.nan, tp=tp, fp=fp))
    return pd.DataFrame(out)


def bootstrap_precision_diff(df, t_op, rel, r_sel, t_base, n_boot, seed):
    """Cluster bootstrap (station-year) on precision(station-select at r_sel,t_op)
    minus precision(baseline at t_base). CI excluding 0 = real gain."""
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True)
    key = (d["station_name"].astype(str) + "_" + d["fold"].astype(str))
    groups = [np.array(v) for v in d.groupby(key).indices.values()]
    g = len(groups)
    yt = d["y_true"].values; yp = d["y_prob"].values
    folds = d["fold"].values; stn = d["station_name"].values
    covered_sel = np.array([(rel[(f, s)] is None) or (rel[(f, s)] >= r_sel)
                            for f, s in zip(folds, stn)])
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[c] for c in rng.integers(0, g, size=g)])
        # selective
        a = covered_sel[idx] & (yp[idx] >= t_op)
        tps = np.sum(a & (yt[idx] == 1)); fps = np.sum(a & (yt[idx] == 0))
        ps = tps/(tps+fps) if (tps+fps) else np.nan
        # baseline
        b = yp[idx] >= t_base
        tpb = np.sum(b & (yt[idx] == 1)); fpb = np.sum(b & (yt[idx] == 0))
        pb = tpb/(tpb+fpb) if (tpb+fpb) else np.nan
        if not (np.isnan(ps) or np.isnan(pb)):
            diffs.append(ps - pb)
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    args = parse_args()
    try:
        pred = pd.read_csv(args.predictions)
        ser = pd.read_csv(args.series)
    except FileNotFoundError as e:
        sys.exit(str(e))
    pred["station_name"] = pred["station_name"].astype(str)
    ser["station_name"] = ser["station_name"].astype(str)
    for c in ("y_true", "y_prob", "fold", "station_name"):
        if c not in pred.columns:
            sys.exit(f"predictions missing '{c}'")

    window_density_check(pred, ser, args)

    print("=" * 66)
    print("PART B  selective prediction (walk-forward station abstention)")
    print("=" * 66)
    t_op = args.threshold
    rel = station_reliability(pred, t_op, args.min_prior)

    # station-select curve: sweep reliability cutoff r
    r_grid = np.round(np.arange(0.0, 0.55, 0.05), 3)
    sel = pd.DataFrame([evaluate_policy(pred, t_op, rel, r) for r in r_grid])

    # baseline curve
    base = baseline_curve(pred, np.round(np.arange(0.30, 0.91, 0.025), 4))

    print(f"operating threshold (station-select arm): {t_op}")
    print(f"\n{'r':>5} {'cov_pos':>8} {'recall':>7} {'prec':>6} {'tp':>4} {'fp':>4}")
    for _, x in sel.iterrows():
        print(f"{x['r']:>5.2f} {x['coverage_pos']:>8.2f} {x['recall']:>7.3f} "
              f"{x['precision']:>6.3f} {int(x['tp']):>4} {int(x['fp']):>4}")

    # which stations get dropped at a moderate cutoff (actionable)
    r_show = 0.10
    dropped = sorted({s for (f, s), v in rel.items()
                      if v is not None and v < r_show})
    print(f"\nstations abstained at r={r_show} (prior precision < {r_show}): "
          f"{dropped if dropped else 'none'}")

    # matched-recall comparison + bootstrap at a sensible selective point
    # pick the selective row nearest 80% positive-coverage
    target = (sel["coverage_pos"] - 0.80).abs().idxmin()
    sr = sel.loc[target]
    # baseline threshold giving closest recall to that selective point
    bj = (base["recall"] - sr["recall"]).abs().idxmin()
    br = base.loc[bj]
    print(f"\nMatched-recall comparison (recall ~ {sr['recall']:.3f}):")
    print(f"  station-select (r={sr['r']:.2f}): precision={sr['precision']:.3f}")
    print(f"  baseline       (t={br['t']:.3f}): precision={br['precision']:.3f}")
    mean, lo, hi = bootstrap_precision_diff(
        pred, t_op, rel, sr["r"], br["t"], args.n_boot, args.seed)
    print(f"  precision diff (select - baseline): {mean:+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    verdict = ("REAL gain: structure beats threshold" if lo > 0 else
               "NO gain: CI includes 0, threshold is sufficient" if hi > 0 else
               "WORSE than threshold")
    print(f"  verdict: {verdict}")

    # plot
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(base["recall"], base["precision"], "-", color="#888", lw=2,
            label="baseline (threshold sweep)")
    ax.plot(sel["recall"], sel["precision"], "o-", color="#1f4e79", lw=2, ms=4,
            label=f"station-select (t={t_op}, sweep abstention)")
    ax.scatter([sr["recall"]], [sr["precision"]], s=110, color="#c0504d",
               edgecolor="black", zorder=6, label="matched point")
    ax.set_xlabel("Recall (global)"); ax.set_ylabel("Precision")
    ax.set_title("Selective prediction vs threshold baseline")
    ax.set_xlim(0, 1); ax.set_ylim(0, max(0.6, sel["precision"].max()*1.2))
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out_fig, dpi=150)
    print(f"\nSaved figure: {args.out_fig}")


if __name__ == "__main__":
    main()