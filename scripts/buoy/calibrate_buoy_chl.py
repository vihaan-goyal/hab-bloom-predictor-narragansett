"""
Calibrate UConn buoy ECO fluorescence (Avg_FL, raw sensor counts) against
CT DEEP lab-extracted chlorophyll, so the 15-minute buoy series can be used as a
continuous chlorophyll proxy.

WHY
Your bloom label is defined on Corrected_Chlorophyll > 10 ug/L. The buoy reports
Avg_FL in raw counts (WLIS median ~159, max ~6500), which is not ug/L. To use the
buoy to audit flagged windows, we need our own counts->ug/L fit, per buoy
(EXRX runs on a very different scale than WLIS, so they are NOT pooled).

WHAT IT DOES
  1. Loads DEEP surface discrete samples (depth_code == 'S') with real timestamps.
  2. Loads buoy 15-min data; keeps only buoys with usable history (WLIS, EXRX).
  3. For each buoy, finds DEEP stations within MAX_KM (haversine).
  4. Matches each DEEP sample to buoy readings within +/- MATCH_HOURS, taking the
     median Avg_FL in that window (robust to spikes).
  5. Fits three models per buoy and compares:
       (a) all matched pairs
       (b) night-only  (avoids non-photochemical quenching)
       (c) all pairs + a quenching correction term
  6. Reports n, slope, intercept, R^2, residual scatter. Writes the fits.

NPQ NOTE
WLIS mean fluorescence dips ~23% around midday (trough 13-15h UTC) — classic
non-photochemical quenching. DEEP samples during working hours, i.e. inside the
quenched window, so an uncorrected fit is biased. Hence models (b) and (c).

Outputs:
  data/buoy_calibration_pairs.csv  matched DEEP<->buoy pairs (audit trail)
  data/buoy_calibration_fits.csv   fitted coefficients per buoy per model

Run from repo root:
  python scripts/buoy/calibrate_buoy_chl.py
"""

import numpy as np
import polars as pl
from pathlib import Path

DEEP_PATH = Path("data/hab_features_final.csv")
BUOY_PATH = Path("data/buoy_eco_fl/all_buoys_eco_fl.parquet")
OUT_PAIRS = Path("data/buoy_calibration_pairs.csv")
OUT_FITS = Path("data/buoy_calibration_fits.csv")

# Only these buoys have history overlapping the test window.
BUOYS = ["WLIS_ECO_FL", "EXRX_ECO_FL"]

MAX_KM = 12.0        # DEEP station must be within this distance of the buoy
MATCH_HOURS = 3.0    # buoy readings within +/- this many hours of the DEEP sample
MIN_BUOY_OBS = 3     # need at least this many buoy readings in the window
NIGHT_HOURS_UTC = set(range(21, 24)) | set(range(0, 10))  # outside the quench bowl


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_deep():
    """DEEP surface discrete samples with real timestamps."""
    lf = pl.scan_csv(
        DEEP_PATH,
        schema_overrides={"station_name": pl.String},  # 'A4','B3' must not become null
        infer_schema_length=50000,
        ignore_errors=True,
    ).select(
        "station_name", "time", "date", "latitude", "longitude",
        "depth_code", "Chlorophyll", "Corrected_Chlorophyll",
    )
    df = lf.collect()

    df = (
        df.filter(pl.col("depth_code") == "S")
          .filter(pl.col("Corrected_Chlorophyll").is_not_null())
          .filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
          .with_columns(
              pl.col("time").str.replace(r"\+00:00$", "")
                .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
                .alias("dt")
          )
          .filter(pl.col("dt").is_not_null())
    )
    # Normalize station labels: '1' -> '01' so they line up with test_predictions.
    df = df.with_columns(
        pl.when(pl.col("station_name").str.contains(r"^\d+$"))
          .then(pl.col("station_name").str.zfill(2))
          .otherwise(pl.col("station_name"))
          .alias("station_name")
    )
    return df


def diel_factors(b):
    """Hour-of-day multiplicative quenching factors from the buoy's OWN climatology.

    Returns {hour: factor} where factor = median_FL(hour) / median_FL(night).
    Dividing a reading by its hour's factor removes the diel quenching bowl and
    puts every reading on a common (night-equivalent) scale.

    This is the key trick: DEEP only samples during working hours, so the matched
    pairs contain no day/night contrast to learn a correction from. But the buoy
    itself has 160k readings across all hours, so we learn the diel shape from the
    buoy and apply it before fitting.
    """
    h = (
        b.filter(pl.col("Avg_FL").is_not_null() & (pl.col("Avg_FL") > 0))
         .with_columns(pl.col("time").dt.hour().alias("hr"))
         .group_by("hr").agg(pl.col("Avg_FL").median().alias("med"))
         .sort("hr")
    )
    med = {int(r["hr"]): float(r["med"]) for r in h.to_dicts()}
    night_vals = [v for k, v in med.items() if k in NIGHT_HOURS_UTC]
    if not night_vals or not med:
        return {k: 1.0 for k in range(24)}
    base = float(np.median(night_vals))
    if base <= 0:
        return {k: 1.0 for k in range(24)}
    return {k: (med.get(k, base) / base if med.get(k, base) > 0 else 1.0)
            for k in range(24)}


def fit_ols(X, y):
    """Least squares with intercept. Returns (coefs, r2, rmse). coefs[0]=intercept."""
    X1 = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ coefs
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return coefs, r2, rmse


def main():
    deep = load_deep()
    buoy = pl.read_parquet(BUOY_PATH)
    print(f"DEEP surface samples with chl + time: {deep.height:,}")

    pair_rows = []
    fit_rows = []

    for bid in BUOYS:
        b = buoy.filter(pl.col("dataset_id") == bid).sort("time")
        if b.height == 0:
            continue
        blat = float(b["latitude"].drop_nulls()[0])
        blon = float(b["longitude"].drop_nulls()[0])
        bt = b["time"].to_numpy().astype("datetime64[s]").astype(np.int64)
        bfl = b["Avg_FL"].to_numpy().astype(float)
        bhr = b["time"].dt.hour().to_numpy().astype(int)
        fac = diel_factors(b)
        # de-quenched series: every reading rescaled to night-equivalent
        bfl_dq = np.array([f / fac.get(int(h), 1.0) if fac.get(int(h), 1.0) > 0 else f
                           for f, h in zip(bfl, bhr)])
        dip = 100 * (1 - min(fac.values()))
        print(f"  diel quench depth: {dip:.1f}% (trough hour "
              f"{min(fac, key=fac.get)}h UTC)")

        # DEEP stations near this buoy
        stations = (
            deep.group_by("station_name")
                .agg(pl.col("latitude").median().alias("lat"),
                     pl.col("longitude").median().alias("lon"))
        ).with_columns(
            pl.struct(["lat", "lon"]).map_elements(
                lambda s: float(haversine_km(s["lat"], s["lon"], blat, blon)),
                return_dtype=pl.Float64,
            ).alias("km")
        ).filter(pl.col("km") <= MAX_KM).sort("km")

        near = stations["station_name"].to_list()
        print(f"\n=== {bid} at ({blat}, {blon}) ===")
        print(f"  DEEP stations within {MAX_KM} km: "
              + ", ".join(f"{r['station_name']}({r['km']:.1f}km)"
                          for r in stations.to_dicts()) or "  none")
        if not near:
            continue

        # buoy temporal coverage limits which DEEP samples can match at all
        b_start, b_end = int(bt.min()), int(bt.max())
        cand = deep.filter(pl.col("station_name").is_in(near))
        cand_t = cand["dt"].to_numpy().astype("datetime64[s]").astype(np.int64)
        in_span = (cand_t >= b_start - 3600) & (cand_t <= b_end + 3600)
        cand = cand.filter(pl.Series(in_span))
        print(f"  DEEP samples inside buoy time span: {cand.height:,}")

        win = int(MATCH_HOURS * 3600)
        matched = 0
        for row in cand.iter_rows(named=True):
            t = np.datetime64(row["dt"], "s").astype(np.int64)
            lo, hi = np.searchsorted(bt, [t - win, t + win])
            seg = bfl[lo:hi]
            seg_dq = bfl_dq[lo:hi]
            ok = np.isfinite(seg) & (seg > 0)  # drop 0.0 sentinel dropouts
            seg, seg_dq = seg[ok], seg_dq[ok]
            if seg.size < MIN_BUOY_OBS:
                continue
            matched += 1
            pair_rows.append({
                "buoy": bid,
                "station_name": row["station_name"],
                "date": str(row["date"]),
                "dt_utc": str(row["dt"]),
                "hour_utc": row["dt"].hour,
                "km": float(stations.filter(
                    pl.col("station_name") == row["station_name"])["km"][0]),
                "deep_chl": float(row["Corrected_Chlorophyll"]),
                "buoy_fl_med": float(np.median(seg)),
                "buoy_fl_dq_med": float(np.median(seg_dq)),
                "buoy_n": int(seg.size),
            })
        print(f"  matched DEEP<->buoy pairs: {matched}")

    if not pair_rows:
        print("\nNo matched pairs. Widen MAX_KM or MATCH_HOURS.")
        return

    pairs = pl.DataFrame(pair_rows)
    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_csv(OUT_PAIRS)
    print(f"\nwrote {pairs.height} pairs -> {OUT_PAIRS}")

    print("\n--- calibration fits (deep_chl ~ buoy_fl) ---")
    for bid in BUOYS:
        p = pairs.filter(pl.col("buoy") == bid)
        if p.height < 5:
            print(f"\n{bid}: only {p.height} pairs, too few to fit.")
            continue

        fl = p["buoy_fl_med"].to_numpy().astype(float)
        fldq = p["buoy_fl_dq_med"].to_numpy().astype(float)
        chl = p["deep_chl"].to_numpy().astype(float)
        hr = p["hour_utc"].to_numpy().astype(int)
        night = np.array([h in NIGHT_HOURS_UTC for h in hr])

        print(f"\n{bid}  (n={p.height}, night pairs={int(night.sum())})")

        # (a) raw FL, all pairs -- BIASED if DEEP samples cluster in the quench window
        c, r2, rmse = fit_ols(fl.reshape(-1, 1), chl)
        print(f"  raw FL         : chl = {c[0]:8.3f} + {c[1]:.5f}*FL      "
              f"R2={r2:.3f}  RMSE={rmse:.2f} ug/L")
        fit_rows.append({"buoy": bid, "model": "raw_all", "n": p.height,
                         "intercept": c[0], "slope": c[1],
                         "r2": r2, "rmse": rmse, "apply_to": "raw Avg_FL"})

        # (b) de-quenched FL  <-- PREFERRED. Diel shape learned from the buoy's own
        #     climatology, so it works even with zero night-time DEEP samples.
        cd, r2d, rmsed = fit_ols(fldq.reshape(-1, 1), chl)
        print(f"  de-quenched FL : chl = {cd[0]:8.3f} + {cd[1]:.5f}*FL_dq   "
              f"R2={r2d:.3f}  RMSE={rmsed:.2f} ug/L   <-- preferred")
        fit_rows.append({"buoy": bid, "model": "dequenched", "n": p.height,
                         "intercept": cd[0], "slope": cd[1],
                         "r2": r2d, "rmse": rmsed, "apply_to": "Avg_FL / diel_factor[hour]"})

        # (c) night-only sanity check, only if there is real night contrast
        if night.sum() >= 5:
            cn, r2n, rmsen = fit_ols(fl[night].reshape(-1, 1), chl[night])
            print(f"  night only     : chl = {cn[0]:8.3f} + {cn[1]:.5f}*FL      "
                  f"R2={r2n:.3f}  RMSE={rmsen:.2f}  (n={int(night.sum())})")
            fit_rows.append({"buoy": bid, "model": "night_only", "n": int(night.sum()),
                             "intercept": cn[0], "slope": cn[1],
                             "r2": r2n, "rmse": rmsen, "apply_to": "raw Avg_FL (night)"})
        else:
            print(f"  night only     : SKIPPED, only {int(night.sum())} night pairs.")
            print( "                   DEEP samples sit inside the quench window, so a")
            print( "                   day/night contrast cannot be fit from the pairs.")
            print( "                   This is exactly why de-quenched (b) is the right model.")

        ratio = cd[1] / c[1] if c[1] not in (0.0,) else float("nan")
        print(f"  slope raw/dq   : raw is {1/ratio:.2f}x the de-quenched slope"
              if np.isfinite(ratio) and ratio != 0 else "")
        print(f"  deep_chl range : {chl.min():.1f} - {chl.max():.1f} ug/L "
              f"({int((chl > 10).sum())} of {len(chl)} above bloom threshold)")

    pl.DataFrame(fit_rows).write_csv(OUT_FITS)
    print(f"\nwrote fits -> {OUT_FITS}")
    print("\nUSE THE de-quenched FIT. Apply as:")
    print("    chl_hat = intercept + slope * (Avg_FL / diel_factor[hour_utc])")
    print("Applying the raw-FL fit to the full 15-min series would over-predict")
    print("chlorophyll at night, because night FL is un-quenched but the raw slope")
    print("was inflated to compensate for quenched daytime samples.")


if __name__ == "__main__":
    main()