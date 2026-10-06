"""
Quick QC of the pulled buoy ECO FL data before calibration:
  - coverage per buoy (rows, span, per-year counts)
  - logging cadence (median gap)
  - fluorescence range sanity
  - buoy coordinates (needed to match buoys to nearby CT DEEP stations)
  - a non-photochemical-quenching check: mean fluorescence by hour of day.
    A dip during local daylight = quenching, meaning midday samples should be
    down-weighted or dropped before calibrating against DEEP extracted chl-a.

Run:
  python scripts/buoy/inspect_buoy_eco_fl.py
"""

from pathlib import Path
import polars as pl

PATH = Path("data/buoy_eco_fl/all_buoys_eco_fl.parquet")

df = pl.read_parquet(PATH)
print(f"total rows: {df.height:,}")
print(f"columns: {df.columns}\n")

for key, g in df.group_by("dataset_id", maintain_order=True):
    ds = key[0] if isinstance(key, tuple) else key
    g = g.sort("time")

    tmin, tmax = g["time"].min(), g["time"].max()
    gaps = g.select(pl.col("time").diff().dt.total_seconds().alias("s")).drop_nulls()
    med_min = gaps["s"].median() / 60 if gaps.height else None

    lat = g["latitude"].drop_nulls().unique().to_list() if "latitude" in g.columns else []
    lon = g["longitude"].drop_nulls().unique().to_list() if "longitude" in g.columns else []
    fl = g["Avg_FL"].drop_nulls() if "Avg_FL" in g.columns else pl.Series([], dtype=pl.Float64)

    print(f"=== {ds} ===")
    print(f"  rows: {g.height:,}   span: {tmin} -> {tmax}")
    print(f"  median cadence: {med_min:.1f} min" if med_min is not None else "  cadence: n/a")
    print(f"  lat: {[round(x,4) for x in lat[:4]]}   lon: {[round(x,4) for x in lon[:4]]}")
    if fl.len():
        print(f"  Avg_FL: min={fl.min():.3f}  median={fl.median():.3f}  "
              f"p99={fl.quantile(0.99):.3f}  max={fl.max():.3f}")
    yc = g.group_by(pl.col("time").dt.year().alias("yr")).len().sort("yr")
    print("  by year:", {r["yr"]: r["len"] for r in yc.to_dicts()})
    print()

# NPQ check on WLIS (the workhorse western buoy)
w = df.filter(pl.col("dataset_id") == "WLIS_ECO_FL")
if w.height and "Avg_FL" in w.columns:
    hourly = (
        w.with_columns(pl.col("time").dt.hour().alias("hr"))
        .group_by("hr").agg(pl.col("Avg_FL").mean().alias("mean_fl"))
        .sort("hr")
    )
    peak = hourly["mean_fl"].max()
    print("WLIS mean Avg_FL by hour (UTC). Local = UTC-4/5; look for a daytime dip:")
    for r in hourly.to_dicts():
        val = r["mean_fl"] or 0.0
        bar = "#" * int((val / peak) * 40) if peak else ""
        print(f"  {r['hr']:02d}h {val:7.3f} {bar}")