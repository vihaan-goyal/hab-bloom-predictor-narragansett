"""
Diagnose why buoy fluorescence and DEEP extracted chlorophyll are not correlating.

Fixes two problems with the first attempt:
  1. NEAREST STATION ONLY. The first pass pooled every DEEP station within 12 km
     against the same buoy reading (9 stations for WLIS). Different chl values,
     one predictor -> manufactured scatter. C1 is 0.5 km from WLIS and A4 is
     0.9 km from EXRX; those are effectively co-located and are the only honest
     comparisons.
  2. MATCH ON DATE, NOT TIME. hab_features_final's `time` column is midnight
     local (e.g. '1994-07-05 04:00:00+00:00' = 00:00 EDT), i.e. a date with a
     zero clock. There is no real sampling hour. So we match on calendar date and
     use the buoy's de-quenched DAILY MEDIAN.

Then it asks the questions that decide whether this data is usable at all:
  - Is there ANY monotonic relationship? (Spearman, robust to nonlinearity)
  - Does the fit drift year to year? (biofouling / sensor swap signature)
  - Does a log transform help? (fluorescence is often log-linear in chl)
  - Are extreme FL outliers wrecking it?

Run from repo root:
  python scripts/buoy/diagnose_buoy_calibration.py
"""

import numpy as np
import polars as pl
from pathlib import Path

DEEP_PATH = Path("data/hab_features_final.csv")
BUOY_PATH = Path("data/buoy_eco_fl/all_buoys_eco_fl.parquet")
OUT = Path("data/buoy_calibration_pairs_nearest.csv")

BUOYS = ["WLIS_ECO_FL", "EXRX_ECO_FL"]
NIGHT_HOURS_UTC = set(range(21, 24)) | set(range(0, 10))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fit(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ c
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return c, r2


def load_deep():
    df = pl.scan_csv(
        DEEP_PATH,
        schema_overrides={"station_name": pl.String},
        infer_schema_length=50000,
        ignore_errors=True,
    ).select("station_name", "date", "latitude", "longitude",
             "depth_code", "Corrected_Chlorophyll").collect()

    df = (df.filter(pl.col("depth_code") == "S")
            .filter(pl.col("Corrected_Chlorophyll").is_not_null())
            .filter(pl.col("latitude").is_not_null())
            .with_columns(
                pl.col("date").str.to_date("%Y-%m-%d", strict=False).alias("d")
            ).filter(pl.col("d").is_not_null()))
    df = df.with_columns(
        pl.when(pl.col("station_name").str.contains(r"^\d+$"))
          .then(pl.col("station_name").str.zfill(2))
          .otherwise(pl.col("station_name")).alias("station_name")
    )
    # one chl value per station-date (DEEP can have replicate casts)
    return df.group_by(["station_name", "d"]).agg(
        pl.col("Corrected_Chlorophyll").median().alias("chl"),
        pl.col("latitude").median().alias("lat"),
        pl.col("longitude").median().alias("lon"),
    )


def diel_factors(b):
    h = (b.filter(pl.col("Avg_FL") > 0)
           .with_columns(pl.col("time").dt.hour().alias("hr"))
           .group_by("hr").agg(pl.col("Avg_FL").median().alias("med")))
    med = {int(r["hr"]): float(r["med"]) for r in h.to_dicts()}
    nights = [v for k, v in med.items() if k in NIGHT_HOURS_UTC]
    base = float(np.median(nights)) if nights else 1.0
    if base <= 0:
        base = 1.0
    return {k: (med.get(k, base) / base if med.get(k, base) > 0 else 1.0)
            for k in range(24)}


def main():
    deep = load_deep()
    buoy = pl.read_parquet(BUOY_PATH)
    all_pairs = []

    for bid in BUOYS:
        b = buoy.filter((pl.col("dataset_id") == bid) & (pl.col("Avg_FL") > 0))
        if b.height == 0:
            continue
        blat = float(b["latitude"].drop_nulls()[0])
        blon = float(b["longitude"].drop_nulls()[0])

        # ---- nearest DEEP station only ----
        sta = (deep.group_by("station_name")
                   .agg(pl.col("lat").median(), pl.col("lon").median()))
        sta = sta.with_columns(
            pl.struct(["lat", "lon"]).map_elements(
                lambda s: float(haversine_km(s["lat"], s["lon"], blat, blon)),
                return_dtype=pl.Float64).alias("km")).sort("km")
        nearest = sta.row(0, named=True)
        print(f"\n{'='*62}\n{bid} at ({blat}, {blon})")
        print(f"  nearest DEEP station: {nearest['station_name']} "
              f"({nearest['km']:.2f} km)")

        # ---- buoy de-quenched daily median ----
        fac = diel_factors(b)
        bd = (b.with_columns(
                  pl.col("time").dt.hour().alias("hr"),
                  pl.col("time").dt.date().alias("d"))
               .with_columns(
                  (pl.col("Avg_FL") / pl.col("hr").replace_strict(
                      fac, default=1.0)).alias("fl_dq"))
               .group_by("d").agg(
                  pl.col("fl_dq").median().alias("fl_dq_med"),
                  pl.col("Avg_FL").median().alias("fl_raw_med"),
                  pl.len().alias("n_obs")))

        d1 = deep.filter(pl.col("station_name") == nearest["station_name"])
        m = d1.join(bd, on="d", how="inner").filter(pl.col("n_obs") >= 10)
        print(f"  same-date matched pairs: {m.height}")
        if m.height < 8:
            print("  too few pairs to diagnose.")
            continue

        m = m.with_columns(pl.lit(bid).alias("buoy"),
                           pl.lit(nearest["station_name"]).alias("station"),
                           pl.lit(nearest["km"]).alias("km"))
        all_pairs.append(m)

        chl = m["chl"].to_numpy().astype(float)
        fl = m["fl_dq_med"].to_numpy().astype(float)
        yrs = m["d"].to_numpy().astype("datetime64[Y]").astype(int) + 1970

        print(f"\n  chl : min={chl.min():.1f} med={np.median(chl):.1f} "
              f"max={chl.max():.1f}   ({int((chl>10).sum())} of {len(chl)} > 10)")
        print(f"  FLdq: min={fl.min():.0f} med={np.median(fl):.0f} "
              f"max={fl.max():.0f}")

        print(f"\n  Pearson  r = {pearson(fl, chl):+.3f}")
        print(f"  Spearman r = {spearman(fl, chl):+.3f}   "
              f"(monotonic assoc; robust to nonlinearity)")

        c, r2 = fit(fl, chl)
        print(f"  linear   : chl = {c[0]:.2f} + {c[1]:.5f}*FL   R2={r2:.3f}")

        # log-log (fluorescence is often log-linear in chl)
        ok = (fl > 0) & (chl > 0)
        if ok.sum() >= 8:
            cl, r2l = fit(np.log(fl[ok]), np.log(chl[ok]))
            print(f"  log-log  : ln(chl) = {cl[0]:.2f} + {cl[1]:.3f}*ln(FL)  "
                  f"R2={r2l:.3f}")

        # outlier robustness: drop top/bottom 5% of FL
        lo, hi = np.percentile(fl, [5, 95])
        k = (fl >= lo) & (fl <= hi)
        if k.sum() >= 8:
            _, r2t = fit(fl[k], chl[k])
            print(f"  trimmed  : R2={r2t:.3f}  (middle 90% of FL, n={int(k.sum())})")

        # ---- biofouling / drift check: fit per year ----
        print("\n  per-year (drift / biofouling check):")
        print(f"    {'yr':<6}{'n':>4}{'r':>8}{'slope':>10}{'med_FL':>9}{'med_chl':>9}")
        for y in sorted(set(yrs)):
            s = yrs == y
            if s.sum() < 4:
                print(f"    {y:<6}{int(s.sum()):>4}{'':>8}{'(too few)':>10}"
                      f"{np.median(fl[s]):>9.0f}{np.median(chl[s]):>9.1f}")
                continue
            cy, _ = fit(fl[s], chl[s])
            print(f"    {y:<6}{int(s.sum()):>4}{pearson(fl[s], chl[s]):>+8.2f}"
                  f"{cy[1]:>10.4f}{np.median(fl[s]):>9.0f}{np.median(chl[s]):>9.1f}")

    if all_pairs:
        out = pl.concat(all_pairs, how="diagonal_relaxed").select(
            "buoy", "station", "km", "d", "chl", "fl_dq_med", "fl_raw_med", "n_obs")
        out.write_csv(OUT)
        print(f"\nwrote {out.height} nearest-station pairs -> {OUT}")

    print("\n" + "=" * 62)
    print("HOW TO READ THIS:")
    print("  Spearman near 0        -> no relationship at all; buoy FL is not")
    print("                            tracking extracted chl. Likely biofouling")
    print("                            or the sensor is measuring something else.")
    print("  Per-year slopes vary   -> sensor drift/fouling. Needs per-deployment")
    print("     wildly                 calibration, or the data is unusable pooled.")
    print("  log-log much better    -> use the log model.")
    print("  Spearman ok but R2 low -> nonlinear but monotonic; usable with a")
    print("                            rank/quantile mapping instead of OLS.")


if __name__ == "__main__":
    main()