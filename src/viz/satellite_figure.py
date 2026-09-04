"""
satellite_figure.py -- fig 10: satellite feasibility at the sonde stations
Reads data/transfer/satellite_{coverage,observability,agreement,skill}.csv
(from src/transfer/satellite_eval.py). Panels: A coverage and observability by
product; B satellite vs sonde agreement (Spearman) by product and source;
C onset lift by product and model (obs representation) with 95% CIs.
Writes figures/nar_fig10_satellite.png. Run from fork root, BASE env.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = pd.read_csv("data/transfer/satellite_coverage.csv")
O = pd.read_csv("data/transfer/satellite_observability.csv")
A = pd.read_csv("data/transfer/satellite_agreement.csv")
S = pd.read_csv("data/transfer/satellite_skill.csv")
order = [p for p in ["olci300", "olci750", "dineof2k", "viirs4k", "olci4k"] if p in set(C["product"])]
label = {"olci300": "OLCI 300 m", "olci750": "OLCI 750 m", "dineof2k": "gap-filled 2 km",
         "viirs4k": "VIIRS 4 km", "olci4k": "OLCI 4 km"}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
cov = C.groupby("product")["frac"].median().reindex(order)
cff = C.groupby("product")["frac_ffill3"].median().reindex(order)
ob = O.assign(w=O.obs1 * O.onsets).groupby("product").agg(w=("w", "sum"), n=("onsets", "sum"))
ob = (ob.w / ob.n).reindex(order)
x = np.arange(len(order)); w = 0.26
ax[0].bar(x - w, cov.values, w, label="valid days (raw)", color="#1f77b4")
ax[0].bar(x, cff.values, w, label="valid days (3-day fill)", color="#aec7e8")
ax[0].bar(x + w, ob.values, w, label="onsets observable (≥1 obs in 7 d)", color="#ff7f0e")
ax[0].axhline(0.6, ls="--", color="k", lw=0.8); ax[0].text(len(order) - 0.5, 0.61, "go/no-go 0.60", ha="right", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels([label[p] for p in order], rotation=20, ha="right")
ax[0].set_ylim(0, 1); ax[0].set_ylabel("fraction"); ax[0].set_title("A. Coverage and observability"); ax[0].legend(fontsize=8)
srcs = sorted(A.source.unique())
for i, p in enumerate(order):
    for j, s in enumerate(srcs):
        v = A[(A["product"] == p) & (A.source == s)].spearman.dropna()
        if len(v):
            ax[1].scatter(np.full(len(v), i) + (j - len(srcs) / 2) * 0.08, v, s=14, label=s if i == 0 else None)
ax[1].axhline(0.3, ls="--", color="k", lw=0.8); ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([label[p] for p in order], rotation=20, ha="right")
ax[1].set_ylabel("Spearman, satellite vs sonde daily chl"); ax[1].set_title("B. Agreement per station"); ax[1].legend(fontsize=7, ncol=2)
models = [("zeroshot", "Narragansett zero-shot", "#ff7f0e"), ("refit", "satellite refit", "#1f77b4"),
          ("chl_rule", "satellite chl>c rule", "#7f7f7f"), ("sonde_refit", "sonde model (upper bound)", "#2ca02c")]
s = S[S["repr"] == "obs"]; w = 0.2
for k, (m, lab, col) in enumerate(models):
    g = s[s.model == m].set_index("product").reindex(order)
    y = g.lift.values; err = np.vstack([y - g.lift_lo.values, g.lift_hi.values - y])
    ax[2].bar(x + (k - 1.5) * w, np.nan_to_num(y), w, yerr=np.nan_to_num(err), color=col, label=lab, capsize=2)
ax[2].axhline(1.0, ls="--", color="k", lw=0.8); ax[2].axhline(1.3, ls=":", color="k", lw=0.8)
ax[2].set_xticks(x); ax[2].set_xticklabels([label[p] for p in order], rotation=20, ha="right")
ax[2].set_ylabel("onset lift (sonde truth)"); ax[2].set_title("C. Skill from satellite input"); ax[2].legend(fontsize=8)
fig.suptitle("Fig 10. Can satellite chlorophyll drive the 7-day bloom model? (stations with sonde truth)", y=1.02)
fig.tight_layout(); fig.savefig("figures/nar_fig10_satellite.png", dpi=150, bbox_inches="tight")
print("wrote figures/nar_fig10_satellite.png")
