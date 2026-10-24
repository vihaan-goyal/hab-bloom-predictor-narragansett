"""
nitrogen_distance_nar.py -- EXPLORATORY: do bloom frequency and forecast skill fall off with distance
through the water from the two upper-bay sewage plants (Fields Point, Bucklin Point)?

Not a blind test: per-station bloom shares were already seen on the station map (findings §33) before
this was written. Report as descriptive.

Distance: shortest path along a hand-drawn water network (plants -> Seekonk / Providence River -> upper bay ->
west passage, east passage, Greenwich Bay), great-circle length per edge. Fixed before computing anything.
Excluded a priori:
  - Mount Hope Bay (B12, B12w): fed by the Taunton River and Fall River, not these plants
  - winter-only stations (B3w group, B12w): their rates reflect season, not place
Input: data/gis/nar_stations.csv (src/viz/nar_station_gis.py).
Outputs: data/gis/nar_station_distance.csv, figures/narragansett/nar_fig16_distance.png
Run from fork root, BASE env:  python src/models/nitrogen_distance_nar.py
"""
import heapq
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

N_PERM = 100_000
RNG = np.random.default_rng(20261024)
FIG = "figures/narragansett/nar_fig16_distance.png"

st = pd.read_csv("data/gis/nar_stations.csv").set_index("code")
P = {  # plants (data/gis/nar_nitrogen_sources.geojson) and channel waypoints (approximate, mid-channel)
    "BUCKLIN": (41.85139, -71.36808), "FIELDS": (41.79461, -71.39258),
    "FOX_PT": (41.8185, -71.3905),        # Seekonk meets the Providence River
    "FIELDS_CH": (41.7930, -71.3840),     # channel off Fields Point
    "GASPEE": (41.7550, -71.3780),
    "UPPER_MID": (41.6950, -71.3500),     # upper bay, north of Prudence Island
    "GB_MOUTH": (41.6800, -71.4000),      # mouth of Greenwich Bay
}
for c in ["F4", "B4", "B3", "B2", "B6", "B7", "F7", "B13", "F3", "B14", "F5"]:
    P[c] = (st.at[c, "lat"], st.at[c, "lon"])
EDGES = [("BUCKLIN", "F4"), ("F4", "FOX_PT"), ("FOX_PT", "FIELDS_CH"), ("FIELDS", "FIELDS_CH"),
         ("FIELDS_CH", "GASPEE"), ("GASPEE", "B4"), ("B4", "B3"), ("B3", "UPPER_MID"),
         ("UPPER_MID", "B2"), ("UPPER_MID", "GB_MOUTH"), ("GB_MOUTH", "B14"), ("B14", "F5"),
         ("B2", "B6"), ("B6", "B7"), ("B7", "F7"),          # west passage
         ("B2", "B13"), ("B13", "F3")]                       # east passage
SITES = ["F4", "B4", "B3", "B2", "F5", "B14", "B6", "B7", "B13", "F3", "F7"]


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def water_distance():
    g = {}
    for a, b in EDGES:
        d = km(P[a], P[b])
        g.setdefault(a, []).append((b, d)); g.setdefault(b, []).append((a, d))
    dist = {"BUCKLIN": 0.0, "FIELDS": 0.0}           # multi-source Dijkstra: nearest plant
    q = [(0.0, "BUCKLIN"), (0.0, "FIELDS")]
    while q:
        d, u = heapq.heappop(q)
        if d > dist[u]:
            continue
        for v, w in g[u]:
            if d + w < dist.get(v, math.inf):
                dist[v] = d + w; heapq.heappush(q, (d + w, v))
    return dist


def perm_p(x, y):
    """One-sided permutation p for a negative Spearman correlation (pairs with a missing value dropped)."""
    ok = pd.notna(x).to_numpy() & pd.notna(y).to_numpy()
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    obs = spearmanr(x, y).statistic
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    sims = np.array([np.corrcoef(rx, RNG.permutation(ry))[0, 1] for _ in range(N_PERM)])
    return obs, (np.sum(sims <= obs) + 1) / (N_PERM + 1)


if __name__ == "__main__":
    dist = water_distance()
    t = st.loc[SITES, ["name", "region", "n_days", "bloom_day_share", "n_pos", "auc_model", "auc_gain",
                       "onsets", "onsets_warned_share"]].copy()
    t["water_km"] = [round(dist[c], 1) for c in SITES]
    t["straight_km"] = [round(min(km(P[c], P["BUCKLIN"]), km(P[c], P["FIELDS"])), 1) for c in SITES]
    t = t.sort_values("water_km")
    t.to_csv("data/gis/nar_station_distance.csv")
    print(t.to_string())
    print(f"\nExploratory, n = {len(t)} sites; Spearman rho vs water distance, one-sided permutation p "
          f"({N_PERM:,} shuffles):")
    res = {}
    for col in ["bloom_day_share", "auc_model", "onsets_warned_share"]:
        rho, p = perm_p(t.water_km, t[col]); res[col] = (rho, p, int(t[col].notna().sum()))
        print(f"  {col:22s} rho {rho:+.2f}  p {p:.4f}  (n = {res[col][2]})")
    no_gb = t.drop(["F5", "B14"])
    rho, p = perm_p(no_gb.water_km, no_gb.bloom_day_share)
    print(f"  bloom_day_share without Greenwich Bay (F5, B14): rho {rho:+.2f}  p {p:.4f}  (n = {len(no_gb)})")
    no_f3 = t.drop(["F3"])
    rho, p = perm_p(no_f3.water_km, no_f3.bloom_day_share)
    print(f"  bloom_day_share without F3 (2010-11 only): rho {rho:+.2f}  p {p:.4f}  (n = {len(no_f3)})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, lab in [(axes[0], "bloom_day_share", "Share of days with chlorophyll > 10 µg/L"),
                         (axes[1], "onsets_warned_share", "Bloom starts warned in advance")]:
        gb = t.index.isin(["F5", "B14"])
        ax.scatter(t.water_km[~gb], t[col][~gb], s=40, color="#2a78d6", edgecolor="k", lw=0.5, zorder=3)
        ax.scatter(t.water_km[gb], t[col][gb], s=40, color="#eb6834", edgecolor="k", lw=0.5, zorder=3,
                   label="Greenwich Bay (enclosed)")
        for c, r in t.iterrows():
            ax.annotate(c, (r.water_km, r[col]), xytext=(4, 3), textcoords="offset points", fontsize=8)
        rho, p, n = res[col]
        ax.set_title(f"Spearman rho {rho:+.2f} (one-sided p {p:.3f}, n = {n})", fontsize=9)
        ax.set_xlabel("Distance through the water from nearest sewage plant (km)")
        ax.set_ylabel(lab)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Narragansett Bay: blooms and warnings vs distance from the upper-bay sewage plants "
                 "(exploratory)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
