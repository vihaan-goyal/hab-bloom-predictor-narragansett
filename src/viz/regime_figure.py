"""
regime_figure.py -- fig 9: leave-one-site-out lift by regime and model
Reads data/transfer/regime_loso.csv (from src/transfer/regime_models.py).
One panel per regime; x = held-out site; bars with 95% CI for regime model,
Narragansett zero-shot, all-sites pooled, local refit; dashed line at lift 1.
Writes figures/nar_fig9_regime_loso.png.  Run from fork root, BASE env.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

r = pd.read_csv("data/transfer/regime_loso.csv")
r = r[~r.in_sample.astype(bool)]
MODELS = [("regime", "regime model", "#1f77b4"), ("narragansett", "Narragansett zero-shot", "#ff7f0e"),
          ("all_sites", "all sites pooled", "#7f7f7f"), ("local_refit", "local refit", "#2ca02c")]
regimes = [g for g in ["fresh", "estuarine", "marine", "lake"] if g in r.regime.unique()]
fig, axes = plt.subplots(1, len(regimes), figsize=(4.2 * len(regimes), 4.6), sharey=True)
axes = np.atleast_1d(axes)
for ax, reg in zip(axes, regimes):
    g = r[r.regime == reg]
    sites = sorted(g.site.unique())
    x = np.arange(len(sites)); w = 0.2
    for k, (m, lab, col) in enumerate(MODELS):
        gm = g[g.model == m].set_index("site").reindex(sites)
        y = gm.lift.values
        err = np.vstack([y - gm.lift_lo.values, gm.lift_hi.values - y])
        ax.bar(x + (k - 1.5) * w, np.nan_to_num(y), w, yerr=np.nan_to_num(err), color=col, label=lab, capsize=2)
    ax.axhline(1.0, ls="--", color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([s.split("-")[0].replace("cefas_full", "cefas") for s in sites],
                                         rotation=30, ha="right")
    ax.set_title(f"{reg} (held-out sites)")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("onset lift = precision / base rate")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Fig 9. Leave-one-site-out: does a water-type model beat the single Narragansett model?", y=1.01)
fig.tight_layout()
fig.savefig("figures/nar_fig9_regime_loso.png", dpi=150, bbox_inches="tight")
print("wrote figures/nar_fig9_regime_loso.png")
