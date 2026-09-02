"""Figure 8: model probability 21 days before bloom onsets vs matched null days.

Inputs (produced by prob_before_onset.py here and prob_before_onset_lis.py in
the LIS repo, assumed to sit next to this fork as ../hab-bloom-predictor):
  data/prob_before_onset_nar.csv, data/prob_before_onset_nar_null.csv,
  data/prob_before_onset_nar_trajectory.csv,
  ../hab-bloom-predictor/data/prob_before_onset_lis.csv,
  ../hab-bloom-predictor/data/prob_before_onset_lis_null.csv
Output: figures/nar_fig8_prob_21d_before.png
Run from the fork root with the BASE conda python.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIS = os.environ.get("LIS_REPO", os.path.join("..", "hab-bloom-predictor"))
C_ON, C_NULL = "#eb6834", "#8a8f98"
BINS = np.linspace(0, 1, 21)

nar = pd.read_csv("data/prob_before_onset_nar.csv")
nar_null = pd.read_csv("data/prob_before_onset_nar_null.csv")
traj = pd.read_csv("data/prob_before_onset_nar_trajectory.csv")
lis = pd.read_csv(os.path.join(LIS, "data", "prob_before_onset_lis.csv"))
lis_null = pd.read_csv(os.path.join(LIS, "data", "prob_before_onset_lis_null.csv"))

panels = [
    ("Long Island Sound\n(21-day model, nearest visit ~21 d before)",
     lis.p_at_visit.dropna().values, lis_null.p.dropna().values, 0.35),
    ("Narragansett Bay\n(21-day model, day -21)",
     nar["p21_at_-21"].dropna().values, nar_null.p21.dropna().values, 0.50),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3),
                         gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
ax_hist = axes[:2]
ax_hist[1].sharey(ax_hist[0])
meds = {}
for ax, (title, on, nu, thr) in zip(ax_hist, panels):
    ax.hist(nu, bins=BINS, density=True, color=C_NULL, alpha=0.5,
            label=f"matched null days (n={len(nu)}, median {np.median(nu):.2f})")
    ax.hist(on, bins=BINS, density=True, color=C_ON, alpha=0.5,
            label=f"bloom onsets (n={len(on)}, median {np.median(on):.2f})")
    ax.axvline(thr, color="k", ls="--", lw=1)
    ax.text(thr + 0.015, 0.97, f"t* = {thr:.2f}", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("model bloom probability at -21 d")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.95,
              edgecolor="none", facecolor="white", bbox_to_anchor=(1.0, 0.93))
    ax.spines[["top", "right"]].set_visible(False)
    meds[title.split("\n")[0]] = (np.median(on), np.median(nu), len(on), len(nu))
ax_hist[0].set_ylabel("density")
plt.setp(ax_hist[1].get_yticklabels(), visible=False)

ax = axes[2]
ax.plot(traj.offset, traj.onset_median_p21, "o-", color=C_ON, lw=1.8, label="bloom onsets")
ax.plot(traj.offset, traj.null_median_p21, "s-", color=C_NULL, lw=1.8, label="matched null path")
ax.axhline(0.50, color="k", ls="--", lw=1)
ax.set_xticks(traj.offset)
ax.set_xticklabels([str(int(o)) for o in traj.offset])
ax.set_xlabel("days before onset")
ax.set_ylabel("median p21")
ax.set_ylim(0, 1)
ax.set_title("Narragansett Bay approach curve\n(median p21, onsets vs null)", fontsize=10.5)
ax.legend(loc="center", bbox_to_anchor=(0.5, 0.33), fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
for x, y in zip(traj.offset, traj.onset_median_p21):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=8, color=C_ON)
for x, y in zip(traj.offset, traj.null_median_p21):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -12),
                ha="center", fontsize=8, color=C_NULL)

for a, lab in zip(axes, "abc"):
    a.text(-0.08, 1.08, f"({lab})", transform=a.transAxes, fontsize=11, fontweight="bold")
fig.tight_layout(w_pad=1.5)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/nar_fig8_prob_21d_before.png", dpi=150)
print("wrote figures/nar_fig8_prob_21d_before.png")
for k, (mo, mn, no, nn) in meds.items():
    print(f"{k}: onsets n={no} median {mo:.3f} | null n={nn} median {mn:.3f}")
