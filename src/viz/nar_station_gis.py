"""
nar_station_gis.py -- per-station GIS package for Narragansett Bay (QGIS-ready), plus fig 15
Inputs (nothing is refit):
  data/transfer/stations_latlon.csv     station coordinates (RIDEM 2023 metadata; F7 corrected here)
  data/narragansett_daily_features.csv  days, years, bloom days (chl > 10 ug/L)
  data/calendar_baseline_nar.csv        out-of-fold GB tier-A onset predictions (= headline pooled CV set)
  data/harbor_warning_events.csv        onsets warned at the device threshold 0.45, lead days
Outputs:
  data/gis/nar_stations.csv, data/gis/nar_stations.geojson        one point per station code
  data/gis/nar_nitrogen_sources.geojson                           sewage plants (reference only)
  figures/narragansett/nar_fig15_station_map.png                  sanity map
Run from fork root, BASE env:  python src/viz/nar_station_gis.py
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T_DEVICE = 0.45          # release/narragansett_bloom_model_v2.joblib operating threshold
MIN_CLASS = 10           # AUC only when a station has >= 10 positives and >= 10 negatives
SMALL = 20               # flag stations with fewer positives than this
OUT = "data/gis"
FIG = "figures/narragansett/nar_fig15_station_map.png"
LAND = "data/gis/basemap/ne_10m_land.geojson"   # optional; downloaded separately

REPORT = ("RIDEM, Narragansett Bay Fixed-Site Monitoring Network final report (2005-08), Table 1, "
          "dem.ri.gov/sites/g/files/xkgbur861/files/bart/pdf/nbfsmin.pdf")
# Table 1 of the report (degrees, decimal minutes): independent check of every coordinate.
TABLE1 = {"F4": (41, 50.505, 71, 22.332), "B4": (41, 44.434, 71, 22.480), "B3": (41, 42.828, 71, 20.628),
          "B2": (41, 40.224, 71, 21.283), "B6": (41, 38.304, 71, 23.621), "B7": (41, 35.288, 71, 22.839),
          "F7": (41, 29.535, 71, 25.137), "B13": (41, 38.907, 71, 19.207), "F3": (41, 34.731, 71, 19.287),
          "F5": (41, 41.090, 71, 26.762), "B14": (41, 40.518, 71, 25.437), "B12": (41, 40.808, 71, 12.913)}
REGION = {"F4": "upper bay", "B4": "upper bay", "B3": "upper bay", "B3w": "upper bay", "B2": "upper bay",
          "B6": "west passage", "B7": "west passage", "F7": "west passage",
          "B13": "east passage", "F3": "east passage",
          "F5": "embayment (Greenwich Bay)", "B14": "embayment (Greenwich Bay)",
          "B12": "embayment (Mount Hope Bay)", "B12w": "embayment (Mount Hope Bay)"}
GROUP = {"B3W": "B3w", "B3a": "B3w", "UB2015": "B3w", "B5": "B6"}
# The 2023 metadata gives GSO Dock as 41 39.521 N, which puts it in Greenwich Bay; the report says 41 29.535 N
# (URI Bay Campus, Narragansett), and F7 has the bay's lowest bloom rate. Longitude agrees in both.
F7_FIX = (41 + 29.535 / 60, -(71 + 25.145 / 60))

# Main point sources of nitrogen at the head of the bay (reference layer; no distances computed here).
SOURCES = [
    {"name": "Bucklin Point WWTF (NBC)", "receiving": "Seekonk River", "lat": 41.85139, "lon": -71.36808,
     "source": "NBC plant address 102 Campbell Ave, East Providence (RIDEM facility list), geocoded with "
               "OpenStreetMap Nominatim; plant site, not the outfall"},
    {"name": "Fields Point WWTF (NBC)", "receiving": "Providence River", "lat": 41.79461, "lon": -71.39258,
     "source": "NBC plant address 2 Ernest St, Providence (RIDEM facility list), geocoded to Ernest St with "
               "OpenStreetMap Nominatim; approximate, plant site, not the outfall"},
]


def km(a, b):
    """Great-circle distance between (lat, lon) points in km."""
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def dm(t):
    return t[0] + t[1] / 60, -(t[2] + t[3] / 60)


def coords():
    s = pd.read_csv("data/transfer/stations_latlon.csv")
    s = s[s.source == "narragansett"].set_index("station")
    s.loc["F7", ["lat", "lon"]] = F7_FIX
    s.loc["F7", "coord_source"] = (REPORT + "; RIDEM 2023 metadata latitude 41 39.521 N is a typo "
                                   "(GSO Dock is on the URI Bay Campus)")
    for code, t in TABLE1.items():     # every site must agree with the report (buoys move ~1 km year to year)
        d = km((s.at[code, "lat"], s.at[code, "lon"]), dm(t))
        assert d < 1.5, f"{code} is {d:.2f} km from report Table 1"
    return s


def roc(y, p):
    return roc_auc_score(y, p) if (y.sum() >= MIN_CLASS and (1 - y).sum() >= MIN_CLASS) else np.nan


def build():
    s = coords()
    daily = pd.read_csv("data/narragansett_daily_features.csv", usecols=["station", "date", "chl"],
                        parse_dates=["date"])
    oof = pd.read_csv("data/calendar_baseline_nar.csv")
    hw = pd.read_csv("data/harbor_warning_events.csv")
    hw = hw[hw.system == "model_0.45"]
    assert len(oof) == 14703, len(oof)
    assert len(hw) == 419, len(hw)

    rows = []
    for code, d in daily.groupby("station"):
        o = oof[oof.station == code]
        h = hw[hw.station == code]
        y, p = o.y.to_numpy(), o.model.to_numpy()
        alert = p >= T_DEVICE
        tp = int((alert & (y == 1)).sum())
        base = y.mean() if len(y) else np.nan
        prec = tp / alert.sum() if alert.sum() else np.nan
        a_m, a_c = roc(y, p), roc(y, o.calendar.to_numpy())
        rows.append({
            "code": code, "name": s.at[code, "name"], "site_group": GROUP.get(code, code),
            "region": REGION.get(GROUP.get(code, code), ""),
            "lat": round(float(s.at[code, "lat"]), 6), "lon": round(float(s.at[code, "lon"]), 6),
            "coord_source": s.at[code, "coord_source"],
            "first_year": int(d.date.dt.year.min()), "last_year": int(d.date.dt.year.max()),
            "n_days": int(d.chl.notna().sum()),
            "bloom_day_share": round(float((d.chl.dropna() > 10).mean()), 3),
            "n_onset_rows": len(o), "n_pos": int(y.sum()),
            "auc_model": round(a_m, 3), "auc_calendar": round(a_c, 3), "auc_gain": round(a_m - a_c, 3),
            "alerts": int(alert.sum()), "precision": round(prec, 3),
            "pod": round(tp / y.sum(), 3) if y.sum() else np.nan,
            "base_rate": round(base, 3), "lift": round(prec / base, 2) if base and prec == prec else np.nan,
            "onsets": len(h),
            "onsets_warned_share": round(h.warned.mean(), 3) if len(h) else np.nan,
            "median_lead_days": float(h.loc[h.warned, "lead"].median()) if h.warned.any() else np.nan,
            "small_sample": bool(y.sum() < SMALL),
        })
    t = pd.DataFrame(rows).sort_values("lat", ascending=False)
    assert t.n_onset_rows.sum() == len(oof) and t.onsets.sum() == len(hw)
    pooled = roc_auc_score(oof.y, oof.model)
    print(f"pooled out-of-fold AUC {pooled:.3f} (headline 0.877); {len(t)} station codes")
    return t


def geojson(t, path, props):
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
              "properties": {k: (None if isinstance(r[k], float) and math.isnan(r[k]) else r[k]) for k in props}}
             for r in t.to_dict("records")]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, indent=1)


def figure(t):
    fig, ax = plt.subplots(figsize=(6.2, 7.5))
    if os.path.exists(LAND):
        for feat in json.load(open(LAND, encoding="utf-8"))["features"]:
            g = feat["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            for poly in polys:
                ring = np.array(poly[0])
                if ring[:, 0].max() < -71.6 or ring[:, 0].min() > -71.0 or ring[:, 1].max() < 41.3:
                    continue
                ax.fill(ring[:, 0], ring[:, 1], facecolor="0.9", edgecolor="0.6", lw=0.6)
    m = t[t.code == t.site_group]                  # one dot per site
    sc = ax.scatter(m.lon, m.lat, c=m.bloom_day_share, s=20 + m.n_days / 40, cmap="viridis",
                    vmin=0, vmax=0.6, edgecolor="k", lw=0.5, zorder=3)
    for r in m.itertuples():
        winter = r.code in ("B3w", "B12w")         # sit next to B3 / B12: short label below instead
        ax.annotate(f"{r.code} (winter)" if winter else f"{r.code} {r.name}", (r.lon, r.lat),
                    xytext=(5, -10 if winter else 3), textcoords="offset points", fontsize=7)
    for s in SOURCES:
        ax.scatter(s["lon"], s["lat"], marker="^", color="#eb6834", s=60, edgecolor="k", lw=0.5, zorder=4)
    ax.scatter([], [], marker="^", color="#eb6834", edgecolor="k", label="Sewage plant (reference)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_xlim(-71.52, -71.12); ax.set_ylim(41.46, 41.88)
    ax.set_aspect(1 / math.cos(math.radians(41.65)))
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Share of days with chlorophyll > 10 µg/L")
    ax.set_title("Narragansett Bay fixed-site stations (dot size = days of data)", fontsize=10)
    ax.set_xticks(np.arange(-71.5, -71.1, 0.1))
    ax.set_xlabel("Longitude\nCoastline: Natural Earth 1:10m, too coarse for the Providence and Seekonk Rivers\n"
                  "(F4 and the two plants are on the water)", fontsize=8)
    ax.set_ylabel("Latitude")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")


MAP_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Narragansett Bay stations</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html, body, #map { height: 100%; margin: 0; font-family: Georgia, serif; }
 .panel { background: #fff; padding: 10px 12px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.3);
          font-size: 13px; line-height: 1.35; max-width: 260px; }
 .panel h3 { margin: 0 0 6px; font-size: 15px; }
 .panel label { display: block; cursor: pointer; }
 .bar { height: 10px; border-radius: 3px; margin: 6px 0 2px;
        background: linear-gradient(90deg, #440154, #3b528b, #21918c, #5ec962, #fde725); }
 .ends { display: flex; justify-content: space-between; font-size: 11px; color: #555; }
 .note { font-size: 11px; color: #666; margin-top: 6px; }
 .leaflet-popup-content table td { padding: 1px 6px 1px 0; font-size: 12px; }
 .lbl { background: transparent; border: none; box-shadow: none; font: 11px Georgia, serif; color: #333; padding: 0; }
 .lbl::before { display: none; }
</style></head><body><div id="map"></div>
<script>
const ST = __STATIONS__, SRC = __SOURCES__;
const METRICS = {
  bloom_day_share:     {label: "Share of days with a bloom (chl > 10)", lo: 0, hi: 0.6, pct: true},
  auc_model:           {label: "Forecast skill (AUC; 0.5 = chance)", lo: 0.7, hi: 0.95},
  auc_gain:            {label: "Skill over time-of-year forecast (AUC gain)", lo: -0.05, hi: 0.2},
  onsets_warned_share: {label: "Bloom starts warned in advance", lo: 0.3, hi: 1, pct: true}};
const V = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function color(x, m) {
  if (x === null || x === undefined) return "#bbb";
  let f = Math.max(0, Math.min(1, (x - m.lo) / (m.hi - m.lo))) * (V.length - 1);
  let i = Math.min(Math.floor(f), V.length - 2), w = f - i;
  return "rgb(" + V[i].map((c, k) => Math.round(c + (V[i + 1][k] - c) * w)).join(",") + ")";
}
const fmt = (x, pct) => x === null || x === undefined ? "n/a" : pct ? Math.round(100 * x) + "%" : x.toFixed(2);
const map = L.map("map").setView([41.64, -71.33], 11);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
  {attribution: "Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors", maxZoom: 16}).addTo(map);
const layer = L.layerGroup().addTo(map);
function draw(key) {
  const m = METRICS[key]; layer.clearLayers();
  ST.forEach(s => {
    const r = 6 + Math.sqrt(s.n_days) / 6;
    L.circleMarker([s.lat, s.lon], {radius: r, color: "#222", weight: 1, fillColor: color(s[key], m), fillOpacity: .9})
     .bindTooltip(s.code.endsWith("w") ? "(winter)" : s.name, {permanent: true, className: "lbl",
       direction: s.code.endsWith("w") ? "bottom" : "right", offset: s.code.endsWith("w") ? [0, r] : [r, 0]})
     .bindPopup("<b>" + s.code + " " + s.name + "</b><br><i>" + s.region + ", " + s.first_year + "-" + s.last_year +
       ", " + s.n_days + " days</i><table>" +
       "<tr><td>Bloom days</td><td>" + fmt(s.bloom_day_share, 1) + "</td></tr>" +
       "<tr><td>Forecast AUC</td><td>" + fmt(s.auc_model) + " (calendar " + fmt(s.auc_calendar) + ")</td></tr>" +
       "<tr><td>Alerts / precision</td><td>" + s.alerts + " / " + fmt(s.precision, 1) + "</td></tr>" +
       "<tr><td>Bloom days warned</td><td>" + fmt(s.pod, 1) + " of " + s.n_pos + "</td></tr>" +
       "<tr><td>Bloom starts warned</td><td>" + fmt(s.onsets_warned_share, 1) + " of " + s.onsets + "</td></tr>" +
       "</table>" + (s.small_sample ? "<i>Small sample: read with care</i>" : ""))
     .addTo(layer);
  });
  document.getElementById("lo").textContent = fmt(m.lo, m.pct);
  document.getElementById("hi").textContent = fmt(m.hi, m.pct);
}
SRC.forEach(p => L.marker([p.lat, p.lon], {icon: L.divIcon({html: "&#9650;", className: "",
  iconSize: [16, 16], iconAnchor: [8, 8]})}).bindPopup("<b>" + p.name + "</b><br>Discharges to the " + p.receiving +
  "<br><i>Plant site, not the outfall</i>").addTo(map));
const panel = L.control({position: "topright"});
panel.onAdd = () => {
  const d = L.DomUtil.create("div", "panel");
  d.innerHTML = "<h3>Narragansett Bay stations</h3>" + Object.entries(METRICS).map(([k, m], i) =>
    '<label><input type="radio" name="m" value="' + k + '"' + (i ? "" : " checked") + "> " + m.label + "</label>").join("") +
    '<div class="bar"></div><div class="ends"><span id="lo"></span><span id="hi"></span></div>' +
    '<div class="note">Dot size = days of data. &#9650; = sewage plant. Forecast numbers: test years 2015-2023, ' +
    'device threshold 0.45, bloom-start days only. Grey = too few days to score. Click a dot for details.</div>';
  L.DomEvent.disableClickPropagation(d);
  d.addEventListener("change", e => draw(e.target.value));
  return d;
};
panel.addTo(map);
draw("bloom_day_share");
</script></body></html>
"""


def html_map(t, path):
    """Interactive Leaflet map (needs internet for the library and the background map)."""
    keep = ["code", "name", "region", "lat", "lon", "first_year", "last_year", "n_days", "bloom_day_share",
            "n_pos", "auc_model", "auc_calendar", "auc_gain", "alerts", "precision", "pod", "onsets",
            "onsets_warned_share", "small_sample"]
    m = t[t.code == t.site_group][keep]
    rows = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in m.to_dict("records")]
    html = MAP_HTML.replace("__STATIONS__", json.dumps(rows)).replace("__SOURCES__", json.dumps(
        [{k: s[k] for k in ("name", "receiving", "lat", "lon")} for s in SOURCES]))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t = build()
    t.to_csv(f"{OUT}/nar_stations.csv", index=False)
    geojson(t, f"{OUT}/nar_stations.geojson", [c for c in t.columns if c not in ("lat", "lon")])
    geojson(pd.DataFrame(SOURCES), f"{OUT}/nar_nitrogen_sources.geojson", ["name", "receiving", "source"])
    figure(t)
    html_map(t, f"{OUT}/nar_station_map.html")
    print(t[["code", "name", "region", "n_days", "bloom_day_share", "n_pos", "auc_model", "auc_gain",
             "precision", "lift", "onsets", "onsets_warned_share", "small_sample"]].to_string(index=False))
