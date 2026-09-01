"""W3 horizon sweep Narragansett -- forecast skill vs lead time.

Rebuilds the forward bloom label for each horizon h (any daily-mean chl > 10
at the same station within (date, date+h]; right-censored -> NaN), using the
same loop as src/features/build_narragansett_daily.py (copied, not imported),
then trains the locked GB tier-A model per horizon and reports test skill on
all days and onset-only (chl <= 10) days.

Outputs:
  data/horizon_sweep_nar.csv
  figures/nar_fig5_skill_vs_horizon.png
Run from repo root, BASE conda env (not `hab`).
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOOM = 10.0
TRAIN_MAX = 2020
VAL_YEARS = (2021, 2022)
TEST_YEARS = (2023,)
HORIZONS = [1, 2, 3, 5, 7, 10, 14, 21]

# copied verbatim from src/models/train_narragansett.py
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']

# LIS locked-pipeline reference points (21-day boat sampling): (h, AUC, precision)
LIS_POINTS = [(14, 0.836, 0.092), (21, 0.852, 0.175), (28, 0.815, 0.261)]

C_NAR = "#2a78d6"
C_LIS = "#8a8f98"
C_ACC = "#eb6834"


def forward_label(day, horizon):
    """Same routine as build_narragansett_daily.py, parameterised by horizon."""
    lab_all = np.full(len(day), np.nan)
    for st, grp in day.groupby("station"):
        idx = grp.index; dates = grp["date"].values; chl = grp["chl"].values
        last = dates.max()
        lab = np.full(len(grp), np.nan)
        for i in range(len(grp)):
            end = dates[i] + np.timedelta64(horizon, "D")
            m = (dates > dates[i]) & (dates <= end)
            if m.any() and (chl[m] > BLOOM).any():
                lab[i] = 1
            elif end <= last:
                lab[i] = 0
        lab_all[idx] = lab
    return lab_all


def metrics(y, alert):
    tp = int(((alert == 1) & (y == 1)).sum()); fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum()); tn = int(((alert == 0) & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    prec = 1 - far if not np.isnan(far) else np.nan
    base = (tp + fn) / (tp + fp + fn + tn)
    return dict(tp=tp, fp=fp, fn=fn, pod=pod, far=far, precision=prec,
                base_rate=base, lift=prec / base if base else np.nan)


def f1_of(mm):
    if mm["precision"] and mm["pod"] and not np.isnan(mm["far"]):
        return 2 * mm["precision"] * mm["pod"] / (mm["precision"] + mm["pod"])
    return 0.0


def run_horizon(day, h):
    y = forward_label(day, h)
    lab = day.assign(y=y).dropna(subset=["y"]).copy()
    train = lab[lab.year <= TRAIN_MAX]
    val = lab[lab.year.isin(VAL_YEARS)]
    test = lab[lab.year.isin(TEST_YEARS)]

    med = train[TIER_A].median(numeric_only=True)

    def prep(X):
        return X[TIER_A].fillna(med).values

    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                       max_iter=300, min_samples_leaf=50,
                                       l2_regularization=1.0, random_state=42,
                                       class_weight="balanced")
    m.fit(prep(train), train.y.astype(int).values)
    pv = m.predict_proba(prep(val))[:, 1]
    pt = m.predict_proba(prep(test))[:, 1]
    yv = val.y.astype(int).values; yt = test.y.astype(int).values

    ts = np.arange(0.05, 0.96, 0.05)
    f1s = [f1_of(metrics(yv, (pv >= t).astype(int))) for t in ts]
    t_star = float(ts[int(np.argmax(f1s))])

    rows = []
    for subset, mask in (("all_days", np.ones(len(test), bool)),
                         ("onset_only", (test.chl <= BLOOM).values)):
        mm = metrics(yt[mask], (pt[mask] >= t_star).astype(int))
        rows.append(dict(horizon=h, subset=subset,
                         auc=roc_auc_score(yt[mask], pt[mask]),
                         precision=mm["precision"], pod=mm["pod"],
                         base_rate=mm["base_rate"], lift=mm["lift"],
                         t_star=t_star, n_test=int(mask.sum()),
                         n_pos=int(yt[mask].sum())))
    return rows, y


def main():
    day = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
    day = day.sort_values(["station", "date"]).reset_index(drop=True)
    day["year"] = day.date.dt.year

    rows = []
    for h in HORIZONS:
        r, y = run_horizon(day, h)
        if h == 7:
            stored = day["bloom_fwd"].values
            same = (np.array_equal(np.isnan(y), np.isnan(stored)) and
                    np.array_equal(y[~np.isnan(y)], stored[~np.isnan(stored)]))
            print(f"[label check] rebuilt h=7 label identical to stored bloom_fwd: {same}")
        rows += r
        for rr in r:
            print(f"h={h:2d} {rr['subset']:10s} AUC={rr['auc']:.3f} prec={rr['precision']:.3f} "
                  f"POD={rr['pod']:.3f} base={rr['base_rate']:.3f} lift={rr['lift']:.2f} "
                  f"t*={rr['t_star']:.2f} n={rr['n_test']} pos={rr['n_pos']}")

    out = pd.DataFrame(rows)[["horizon", "subset", "auc", "precision", "pod",
                              "base_rate", "lift", "t_star", "n_test", "n_pos"]]
    os.makedirs("data", exist_ok=True); os.makedirs("figures", exist_ok=True)
    out.to_csv("data/horizon_sweep_nar.csv", index=False)

    # sanity check vs data/narragansett_model_results.csv GB_onset
    ref = pd.read_csv("data/narragansett_model_results.csv")
    ref = ref[(ref.model == "GB_onset") & (ref.features == "A_LIS_analog")].iloc[0]
    mine = out[(out.horizon == 7) & (out.subset == "onset_only")].iloc[0]
    print(f"[sanity] h=7 onset: mine AUC={mine.auc:.4f} prec={mine.precision:.4f} "
          f"t*={mine.t_star} | ref AUC={ref.auc_test:.4f} prec={ref.precision:.4f} "
          f"t*={ref.t_star}")
    ok = abs(mine.auc - ref.auc_test) < 1e-3 and abs(mine.precision - ref.precision) < 1e-3
    print(f"[sanity] {'PASS' if ok else 'FAIL'}")

    plot(out)


def plot(out):
    on = out[out.subset == "onset_only"].sort_values("horizon")
    al = out[out.subset == "all_days"].sort_values("horizon")
    lis_h = [p[0] for p in LIS_POINTS]
    lis_auc = [p[1] for p in LIS_POINTS]
    lis_prec = [p[2] for p in LIS_POINTS]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                   gridspec_kw=dict(hspace=0.12))
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.grid(True, axis="y", color="#e3e5e8", lw=0.6)
        ax.tick_params(length=0)
        ax.spines["left"].set_color("#c8cbd0"); ax.spines["bottom"].set_color("#c8cbd0")

    xl = on.horizon.iloc[-1] * 1.08  # direct-label x for Narragansett lines

    # --- top: AUC ---
    ax1.plot(on.horizon, on.auc, "-o", color=C_NAR, lw=2, ms=5.5, zorder=3)
    ax1.plot(al.horizon, al.auc, "--o", color=C_NAR, lw=1.4, ms=4.5, alpha=0.55, zorder=3,
             markerfacecolor="white")
    ax1.plot(lis_h, lis_auc, ":", color=C_LIS, lw=1, zorder=2)
    ax1.plot(lis_h, lis_auc, "s", color=C_LIS, ms=8, zorder=4, markeredgecolor="white")
    ax1.set_ylabel("Test AUC (2023)")
    ax1.set_ylim(0.6, 1.0)
    ax1.text(xl, on.auc.iloc[-1], "Narragansett\nonset-only",
             color=C_NAR, va="center", ha="left", fontsize=8.5, fontweight="bold")
    ax1.text(xl, al.auc.iloc[-1], "all days",
             color=C_NAR, va="center", ha="left", fontsize=8.5, alpha=0.75)
    ax1.text(lis_h[0], lis_auc[0] - 0.035, "LIS (21-day boat sampling)",
             color=C_LIS, ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax1.set_title("Forecast skill vs lead time  (GB, tier-A features)", loc="left",
                  fontsize=10.5, fontweight="bold", color="#222")

    # --- bottom: precision + base rate ---
    ax2.plot(on.horizon, on.precision, "-o", color=C_NAR, lw=2, ms=5.5, zorder=3)
    ax2.plot(al.horizon, al.precision, "--o", color=C_NAR, lw=1.4, ms=4.5, alpha=0.55,
             zorder=3, markerfacecolor="white")
    ax2.plot(on.horizon, on.base_rate, "--", color=C_ACC, lw=1.4, zorder=2)
    ax2.plot(al.horizon, al.base_rate, "--", color=C_ACC, lw=1.4, alpha=0.45, zorder=2)
    ax2.plot(lis_h, lis_prec, ":", color=C_LIS, lw=1, zorder=2)
    ax2.plot(lis_h, lis_prec, "s", color=C_LIS, ms=8, zorder=4, markeredgecolor="white")
    ax2.set_ylabel("Precision at t*  (val-F1 threshold)")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Forecast horizon (days, log scale)")
    ticks = HORIZONS + [28]
    ax2.set_xticks(ticks); ax2.set_xticklabels([str(t) for t in ticks])
    ax2.set_xlim(0.85, 40)
    ax2.text(xl, on.precision.iloc[-1] - 0.02, "precision, onset-only",
             color=C_NAR, va="center", ha="left", fontsize=8.5, fontweight="bold")
    ax2.text(xl, al.precision.iloc[-1] + 0.03, "all days",
             color=C_NAR, va="center", ha="left", fontsize=8.5, alpha=0.75)
    ax2.text(xl, on.base_rate.iloc[-1], "base rate,\nonset-only",
             color=C_ACC, va="center", ha="left", fontsize=8.5)
    ax2.text(xl, al.base_rate.iloc[-1], "base rate, all days",
             color=C_ACC, va="center", ha="left", fontsize=8.5, alpha=0.7)
    ax2.text(lis_h[-1], lis_prec[-1] + 0.05, "LIS", color=C_LIS, ha="center",
             va="bottom", fontsize=8.5, fontweight="bold")

    fig.savefig("figures/nar_fig5_skill_vs_horizon.png", dpi=150, bbox_inches="tight")
    print("wrote figures/nar_fig5_skill_vs_horizon.png")


if __name__ == "__main__":
    main()
