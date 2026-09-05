"""
registry_map.py -- fig 11: where the exported model has been run, coloured by onset lift
Inputs: data/registry/site_skill.csv + data/registry/insitu_catalog.csv (new ERDDAP sites),
        data/transfer/stations_latlon.csv + data/transfer/<source>_results.csv (the seven
        known networks, zeroshot_qm onset lift from findings 19).
Output: figures/nar_fig11_coverage_map.png (plain lat/lon scatter, no basemap dependency).
Run from fork root, BASE env.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sk = pd.read_csv("data/registry/site_skill.csv") if os.path.exists("data/registry/site_skill.csv") else pd.DataFrame()
cat = pd.read_csv("data/registry/insitu_catalog.csv").drop_duplicates(subset=["server", "dataset_id"])
cat["lat"] = (cat.lat_min + cat.lat_max) / 2; cat["lon"] = (cat.lon_min + cat.lon_max) / 2
new = cat.merge(sk, on=["server", "dataset_id"], how="inner") if len(sk) else pd.DataFrame(columns=["lat", "lon", "lift"])

st = pd.read_csv("data/transfer/stations_latlon.csv")
known = []
for src in st.source.unique():
    p = f"data/transfer/{src}_results.csv"
    lift = 2.0 if src == "narragansett" else np.nan
    if os.path.exists(p):
        r = pd.read_csv(p); r = r[(r["eval"] == "zeroshot_qm") & (r.scope == "onset") & (r.threshold == "p75")]
        lift = r.lift.iloc[0] if len(r) else lift
    g = st[st.source == src]
    known.append(pd.DataFrame({"lat": g.lat, "lon": g.lon, "lift": lift, "source": src}))
known = pd.concat(known)

fig, ax = plt.subplots(figsize=(14, 6.5))
ax.set_xlim(-180, 180); ax.set_ylim(-70, 80); ax.set_aspect("equal")
ax.axhline(0, color="0.85", lw=0.8); ax.axvline(0, color="0.85", lw=0.8); ax.grid(alpha=0.25)
vmin, vmax = 0.8, 2.6
sc = ax.scatter(known.lon, known.lat, c=known.lift, cmap="viridis", vmin=vmin, vmax=vmax, s=28, marker="s",
                edgecolor="k", linewidth=0.3, label="seven networks tested in findings 19 (squares)")
n_sc = 0
if len(new):
    scored = new.dropna(subset=["lift"]); unscored = new[new.lift.isna()]; n_sc = len(scored)
    if len(scored):
        ax.scatter(scored.lon, scored.lat, c=scored.lift, cmap="viridis", vmin=vmin, vmax=vmax, s=60, marker="o",
                   edgecolor="k", linewidth=0.5, label="new ERDDAP sites, scored (circles)")
    if len(unscored):
        ax.scatter(unscored.lon, unscored.lat, color="0.6", s=40, marker="o", edgecolor="k", linewidth=0.4,
                   label="new ERDDAP sites, predictions only (<2 years)")
cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("onset lift of the exported model (precision / base rate)")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title(f"Fig 11. Every site the exported Narragansett model has been run on: "
             f"{st.station.nunique()} known-network stations + {len(new)} new ERDDAP datasets ({n_sc} scored)")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig("figures/nar_fig11_coverage_map.png", dpi=150, bbox_inches="tight")
print("wrote figures/nar_fig11_coverage_map.png")
