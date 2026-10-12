"""
cadence_thinning.py
-------------------
Task 1: quantify the sampling-cadence ceiling as a measurable curve.

Design (locked model is FROZEN -- nothing here retunes anything):
  1. Load canonical dataframe, fit locked LR on rows through 2019 (native
     density, native h21 labels). Score every row once. Probabilities are
     then fixed for the whole experiment.
  2. Thinning model: random CRUISE dropout. CT DEEP samples by boat cruise,
     so station visits arrive in date-clustered batches; the realistic
     "reduced monitoring" counterfactual is fewer cruises. For each keep
     fraction p and seed, keep each unique sampling date with probability
     p (all stations visited that date are kept or dropped together).
     NOTE: hard minimum-gap thinning is degenerate here -- any cadence
     coarser than the horizon makes every verification window empty by
     construction (gap >= k > h implies zero future visits in (t, t+h]).
     That statement itself belongs in the paper; the dropout model is what
     produces a measurable curve.
  3. Recompute h21 labels on the thinned series via add_forward_label
     (verification gets sparser too -- that is the point), then evaluate
     at the frozen operating point t* = 0.35 on 2023-2025.
  3. Report POD / FAR / CSI / precision on (a) all resolvable windows and
     (b) verifiable windows only (>= 1 future visit inside the horizon),
     plus the empty-window fraction and realized median inter-sample gap.

Caveat for the paper: features for kept rows were engineered from the
full-density history, so this isolates the label/verification side of the
ceiling. True coarser monitoring would also degrade features; the measured
curve is therefore an optimistic bound.

Usage (from repo root):
    python src/models/cadence_thinning.py
Outputs:
    data/cadence_thinning_results.csv       (per cadence x seed)
    data/cadence_thinning_summary.csv       (per cadence, seed-aggregated)
    figures/model/cadence_thinning_curve.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.locked_pipeline import (          # noqa: E402
    HORIZON_DAYS, add_forward_label, fit_locked_model,
    load_locked_dataframe, predict_proba)

T_STAR = 0.35
TRAIN_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2025-12-31")
KEEP_FRACS = [1.0, 0.9, 0.75, 0.5, 0.35, 0.25]   # 1.0 = native density
N_SEEDS = 20
RESULTS_CSV = "data/cadence_thinning_results.csv"
SUMMARY_CSV = "data/cadence_thinning_summary.csv"
FIGURE_PNG = "figures/model/cadence_thinning_curve.png"


def thin_frame(df, keep_frac, rng):
    """Random cruise dropout: keep each unique sampling date with
    probability keep_frac; every station visited on a kept date stays.
    Features untouched (engineered from full-density history; see caveat)."""
    if keep_frac >= 1.0:
        return df.copy()
    dates = df["date"].unique()
    kept = dates[rng.random(len(dates)) < keep_frac]
    return df[df["date"].isin(kept)].copy()


def window_has_future_visit(df, horizon=HORIZON_DAYS):
    """Boolean per row: >= 1 visit at the same station within (t, t+h]."""
    out = pd.Series(False, index=df.index)
    for _, grp in df.groupby("station_name"):
        dates = grp["date"].values
        flags = np.zeros(len(grp), dtype=bool)
        for i in range(len(grp)):
            end = dates[i] + np.timedelta64(horizon, "D")
            flags[i] = ((dates > dates[i]) & (dates <= end)).any()
        out.loc[grp.index] = flags
    return out


def contingency(y_true, y_alert):
    tp = int(((y_true == 1) & y_alert).sum())
    fp = int(((y_true == 0) & y_alert).sum())
    fn = int(((y_true == 1) & ~y_alert).sum())
    pod = tp / (tp + fn) if (tp + fn) else np.nan
    far = fp / (tp + fp) if (tp + fp) else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    prec = 1 - far if not np.isnan(far) else np.nan
    return dict(tp=tp, fp=fp, fn=fn, pod=pod, far=far, csi=csi,
                precision=prec)


def median_gap_days(df):
    gaps = []
    for _, grp in df.groupby("station_name"):
        d = grp["date"].sort_values().diff().dt.days.dropna()
        gaps.extend(d.tolist())
    return float(np.median(gaps)) if gaps else np.nan


def main():
    df = load_locked_dataframe()
    df = df.sort_values(["station_name", "date"]).reset_index(drop=True)

    # Fit once on native-density data through 2019 (locked spec).
    df_native = add_forward_label(df, horizon=HORIZON_DAYS)
    bundle = fit_locked_model(df_native, label_col="bloom_fwd",
                              train_end=TRAIN_END)
    print(f"Locked LR trained on {bundle['n_train']:,} rows through "
          f"{TRAIN_END.date()} (bloom rate "
          f"{bundle['train_bloom_rate']*100:.1f}%)")

    # Score every row once; probabilities frozen hereafter.
    df["bloom_prob"] = predict_proba(bundle, df)
    df["alert"] = df["bloom_prob"] >= T_STAR

    rows = []
    for p in KEEP_FRACS:
        seeds = [0] if p >= 1.0 else range(N_SEEDS)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            thin = thin_frame(df, p, rng)
            thin = add_forward_label(thin, horizon=HORIZON_DAYS)
            thin["has_future"] = window_has_future_visit(thin)

            ev = thin[(thin["date"] >= TEST_START)
                      & (thin["date"] <= TEST_END)
                      & thin["bloom_fwd"].notna()].copy()
            if ev.empty:
                continue

            base = dict(keep_frac=p, seed=seed,
                        n_eval=len(ev),
                        median_gap=median_gap_days(
                            thin[(thin["date"] >= TEST_START)
                                 & (thin["date"] <= TEST_END)]),
                        empty_frac=float((~ev["has_future"]).mean()))

            all_m = contingency(ev["bloom_fwd"], ev["alert"])
            ver = ev[ev["has_future"]]
            ver_m = (contingency(ver["bloom_fwd"], ver["alert"])
                     if len(ver) else {m: np.nan for m in all_m})

            rows.append({**base,
                         **{f"all_{m}": v for m, v in all_m.items()},
                         **{f"ver_{m}": v for m, v in ver_m.items()}})
            print(f"keep={p:.2f}  seed={seed:>2}  "
                  f"gap={base['median_gap']:>5.1f}d  "
                  f"empty={base['empty_frac']*100:4.1f}%  "
                  f"POD={all_m['pod']:.3f}  FAR={all_m['far']:.3f}  "
                  f"CSI={all_m['csi']:.3f}")

    res = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    res.to_csv(RESULTS_CSV, index=False)

    summary = (res.groupby("keep_frac", sort=False)
                  .agg(median_gap=("median_gap", "mean"),
                       empty_frac=("empty_frac", "mean"),
                       pod=("all_pod", "mean"), pod_sd=("all_pod", "std"),
                       far=("all_far", "mean"), far_sd=("all_far", "std"),
                       csi=("all_csi", "mean"), csi_sd=("all_csi", "std"),
                       ver_far=("ver_far", "mean"),
                       n_eval=("n_eval", "mean"))
                  .reset_index())
    summary.to_csv(SUMMARY_CSV, index=False)
    print("\n== summary (seed means) ==")
    print(summary.to_string(index=False))

    # Figure: metrics vs realized median gap.
    os.makedirs("figures/model", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = summary["median_gap"]
    for m, c in [("pod", "tab:blue"), ("csi", "tab:green"),
                 ("far", "tab:red")]:
        ax1.errorbar(x, summary[m], yerr=summary[f"{m}_sd"],
                     marker="o", capsize=3, label=m.upper(), color=c)
    ax1.errorbar(x, summary["ver_far"], marker="^", ls="--",
        color="tab:orange", label="FAR (verifiable windows)")
    ax1.set_ylim(0, 1.05)
    ax1.axvline(HORIZON_DAYS, ls="--", color="gray", lw=1,
                label=f"h = {HORIZON_DAYS}d")
    ax1.set_xlabel("Realized median inter-sample gap (days)")
    ax1.set_ylabel("Score at t* = 0.35")
    ax1.set_title("EWS operating characteristics vs sampling cadence")
    ax1.legend()

    ax2.plot(x, summary["empty_frac"] * 100, marker="s",
             color="tab:purple")
    ax2.axvline(HORIZON_DAYS, ls="--", color="gray", lw=1)
    ax2.set_xlabel("Realized median inter-sample gap (days)")
    ax2.set_ylabel("% of verification windows with zero visits")
    ax2.set_title("Verification coverage vs cadence")
    fig.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=200)
    print(f"\nSaved {RESULTS_CSV}, {SUMMARY_CSV}, {FIGURE_PNG}")


if __name__ == "__main__":
    main()