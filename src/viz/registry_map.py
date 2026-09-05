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
# prefer the median of the pulled coordinates (bounding-box midpoints mislocate datasets that span hemispheres)
import glob
for i, row in cat.iterrows():
    files = glob.glob(f"data/registry/raw/{row.server}/{row.dataset_id}_*.csv")
    if not files:
        continue
    try:
        d = pd.concat([pd.read_csv(f, usecols=["latitude", "longitude"]).dropna() for f in files[:3]])
        if len(d):
            cat.at[i, "lat"] = d.latitude.median(); cat.at[i, "lon"] = d.longitude.median()
    except Exception:
        pass
# known metadata sign error: UNH Great Bay (NERACOOS) is published at +70.87 E; it is 70.87 W
cat.loc[(cat.server == "NERACOOS") & (cat.lon > 0), "lon"] *= -1
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
ax.set_xlim(-180, 180); ax.set_ylim(-62, 82); ax.set_aspect("equal")
# minimalist basemap: Natural Earth 110 m land polygons (figures/data/ne_110m_land.geojson), plain matplotlib
import json
from matplotlib.patches import Polygon as MplPolygon
land = "figures/data/ne_110m_land.geojson"
if os.path.exists(land):
    for feat in json.load(open(land, encoding="utf-8"))["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "Polygon" else [p for mp in geom["coordinates"] for p in mp]
        for ring in polys:
            ax.add_patch(MplPolygon(ring, closed=True, facecolor="0.88", edgecolor="0.7", linewidth=0.4, zorder=0))
ax.set_facecolor("white")
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
vmin, vmax = 0.8, 2.6
sc = ax.scatter(known.lon, known.lat, c=known.lift, cmap="viridis", vmin=vmin, vmax=vmax, s=34, marker="s",
                edgecolor="k", linewidth=0.4, zorder=3, label="seven networks tested in findings 19 (squares)")
n_sc = 0
if len(new):
    scored = new.dropna(subset=["lift"]); unscored = new[new.lift.isna()]; n_sc = len(scored)
    if len(scored):
        ax.scatter(scored.lon, scored.lat, c=scored.lift, cmap="viridis", vmin=vmin, vmax=vmax, s=70, marker="o",
                   edgecolor="k", linewidth=0.5, zorder=4, label="new ERDDAP sites, scored (circles)")
    if len(unscored):
        ax.scatter(unscored.lon, unscored.lat, color="0.6", s=45, marker="o", edgecolor="k", linewidth=0.4, zorder=3,
                   label="new ERDDAP sites, predictions only (<2 years)")
cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("onset lift of the exported model (precision / base rate)")
ax.set_title(f"Fig 11. Every site the exported Narragansett model has been run on: "
             f"{st.station.nunique()} known-network stations + {len(new)} new ERDDAP datasets ({n_sc} scored)")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig("figures/nar_fig11_coverage_map.png", dpi=150, bbox_inches="tight")
print("wrote figures/nar_fig11_coverage_map.png")
