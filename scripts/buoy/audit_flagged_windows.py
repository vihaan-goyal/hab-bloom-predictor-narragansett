"""
Audit the model's false positives using continuous buoy fluorescence WITHOUT
calibrating it.

WHY NO CALIBRATION
Per-deployment slopes swing >10x (WLIS 0.0026 -> 0.0299; EXRX 0.0022 -> 0.0350),
i.e. biofouling / sensor swaps. Absolute ug/L from these sensors is not
recoverable. But Spearman r ~ +0.5 says the RANK signal survives, and fouling
drifts over weeks-to-months while a 21-day window sees a near-constant gain.
So we ask a relative question instead of an absolute one.

THE TEST
Among windows CT DEEP labelled NEGATIVE (no observed chl > 10), do the windows
the model FLAGGED show higher buoy fluorescence than the windows it did not?

  If yes -> the "false positives" contain real biomass excursions that the
            discrete sampling cadence never caught, and the precision ceiling is
            partly an artifact of monitoring cadence, not model error.
  If no  -> the false positives look like genuine model errors, and the cadence
            explanation is not supported.

Either outcome is a publishable result. This is the honest test.

Fouling is handled by z-scoring buoy fluorescence WITHIN EACH YEAR, so a drifting
gain cannot create a spurious difference between flagged and unflagged windows
(both are drawn from the same year's distribution).

Buoy <-> station pairing (co-located, from the diagnostic):
  WLIS buoy  <-> DEEP station C1  (0.46 km)
  EXRX buoy  <-> DEEP station A4  (0.90 km)

Run from repo root:
  python scripts/buoy/audit_flagged_windows.py
"""

import math
import numpy as np
import polars as pl
from pathlib import Path

DEEP_PATH = Path("data/hab_features_final.csv")
BUOY_PATH = Path("data/buoy_eco_fl/all_buoys_eco_fl.parquet")
PRED_PATH = Path("data/test_predictions.csv")
THR_PATH = Path("data/station_thresholds.csv")
OUT = Path("data/flagged_window_audit.csv")

PAIRING = {"C1": "WLIS_ECO_FL", "A4": "EXRX_ECO_FL"}
HORIZON_DAYS = 21
NIGHT_HOURS_UTC = set(range(21, 24)) | set(range(0, 10))
MIN_BUOY_DAYS = 5   # need this many buoy-days inside a window to score it


def rank_biserial_and_u(a, b):
    """Mann-Whitney U via normal approximation. Returns (p_two_sided, effect).

    effect = common-language effect size = P(a random 'a' > a random 'b').
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    allv = np.concatenate([a, b])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            m = inv == i
            ranks[m] = ranks[m].mean()
    Ra = ranks[:na].sum()
    Ua = Ra - na * (na + 1) / 2
    effect = Ua / (na * nb)                      # = P(a > b)
    mu = na * nb / 2
    n = na + nb
    tie_term = sum(c ** 3 - c for c in cnt)
    sd = np.sqrt(na * nb / 12 * ((n + 1) - tie_term / (n * (n - 1)))) if n > 1 else 0
    if sd == 0:
        return float("nan"), effect
    z = (Ua - mu) / sd
    # two-sided normal p
    p = 2 * 0.5 * math.erfc(abs(z) / np.sqrt(2))
    return float(p), float(effect)


def diel_factors(b):
    h = (b.filter(pl.col("Avg_FL") > 0)
           .with_columns(pl.col("time").dt.hour().alias("hr"))
           .group_by("hr").agg(pl.col("Avg_FL").median().alias("med")))
    med = {int(r["hr"]): float(r["med"]) for r in h.to_dicts()}
    nights = [v for k, v in med.items() if k in NIGHT_HOURS_UTC]
    base = float(np.median(nights)) if nights else 1.0
    base = base if base > 0 else 1.0
    return {k: (med.get(k, base) / base if med.get(k, base) > 0 else 1.0)
            for k in range(24)}


def deep_coverage_check():
    """Why do matched pairs only exist in some years? Check C1/A4 sampling."""
    df = pl.scan_csv(DEEP_PATH, schema_overrides={"station_name": pl.String},
                     infer_schema_length=50000, ignore_errors=True).select(
        "station_name", "date", "depth_code", "Corrected_Chlorophyll").collect()
    df = df.with_columns(
        pl.when(pl.col("station_name").str.contains(r"^\d+$"))
          .then(pl.col("station_name").str.zfill(2))
          .otherwise(pl.col("station_name")).alias("station_name"),
        pl.col("date").str.to_date("%Y-%m-%d", strict=False).alias("d"))

    print("=" * 66)
    print("DEEP COVERAGE CHECK (why were some years missing from calibration?)")
    for st in PAIRING:
        s = df.filter(pl.col("station_name") == st)
        print(f"\n  station {st}: {s.height:,} rows total")
        for depth_only in (False, True):
            q = s.filter(pl.col("depth_code") == "S") if depth_only else s
            tag = "surface only" if depth_only else "all depths"
            byyr = (q.filter(pl.col("d").dt.year() >= 2019)
                     .group_by(pl.col("d").dt.year().alias("yr"))
                     .agg(pl.len().alias("n"),
                          pl.col("Corrected_Chlorophyll").is_not_null().sum().alias("with_chl"))
                     .sort("yr"))
            print(f"    {tag}: " + ", ".join(
                f"{r['yr']}: {r['with_chl']}/{r['n']}" for r in byyr.to_dicts()))
    print("\n  (format year: rows_with_chl/total_rows, 2019+)")


def main():
    deep_coverage_check()

    preds = pl.read_csv(PRED_PATH, schema_overrides={"station_name": pl.String})
    preds = preds.with_columns(pl.col("date").str.to_date("%Y-%m-%d").alias("d"))
    thr = pl.read_csv(THR_PATH, schema_overrides={"station": pl.String})
    thr_map = {r["station"]: float(r["threshold"]) for r in thr.to_dicts()}

    buoy = pl.read_parquet(BUOY_PATH)
    rows = []

    print("\n" + "=" * 66)
    print("FLAGGED-WINDOW AUDIT")

    for station, bid in PAIRING.items():
        b = buoy.filter((pl.col("dataset_id") == bid) & (pl.col("Avg_FL") > 0))
        if b.height == 0:
            continue
        fac = diel_factors(b)
        # de-quenched daily median, then z-scored WITHIN YEAR (kills fouling drift)
        bd = (b.with_columns(pl.col("time").dt.hour().alias("hr"),
                             pl.col("time").dt.date().alias("d"))
                .with_columns((pl.col("Avg_FL") /
                               pl.col("hr").replace_strict(fac, default=1.0)).alias("fl_dq"))
                .group_by("d").agg(pl.col("fl_dq").median().alias("fl"))
                .with_columns(pl.col("d").dt.year().alias("yr")))
        bd = bd.with_columns(
            ((pl.col("fl") - pl.col("fl").mean().over("yr")) /
             pl.col("fl").std().over("yr")).alias("fl_z"))

        bdays = bd["d"].to_numpy()
        bz = bd["fl_z"].to_numpy().astype(float)

        t = thr_map.get(station)
        p = preds.filter(pl.col("station_name") == station)
        if t is None or p.height == 0:
            print(f"\n  {station}: no predictions/threshold, skipped")
            continue

        n_scored = 0
        for r in p.iter_rows(named=True):
            d0 = np.datetime64(r["d"], "D")
            d1 = d0 + np.timedelta64(HORIZON_DAYS, "D")
            m = (bdays >= d0) & (bdays <= d1)
            z = bz[m]
            z = z[np.isfinite(z)]
            if z.size < MIN_BUOY_DAYS:
                continue
            n_scored += 1
            rows.append({
                "station": station, "buoy": bid, "date": str(r["d"]),
                "y_true": int(r["y_true"]),
                "y_prob": float(r["y_prob"]),
                "y_pred": int(float(r["y_prob"]) >= t),
                "buoy_days": int(z.size),
                "fl_z_max": float(z.max()),
                "fl_z_mean": float(z.mean()),
            })
        print(f"\n  {station} (buoy {bid}, threshold {t}): "
              f"{n_scored}/{p.height} test windows have >={MIN_BUOY_DAYS} buoy-days")

    if not rows:
        print("\nNo windows had enough buoy coverage. Buoy record does not overlap"
              "\nthe test period at these stations.")
        return

    a = pl.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    a.write_csv(OUT)
    print(f"\nwrote {a.height} audited windows -> {OUT}")

    print("\n" + "=" * 66)
    print("THE KEY TEST: among DEEP-negative windows (y_true == 0),")
    print("do FLAGGED windows show higher buoy fluorescence than unflagged ones?")

    for station in list(PAIRING) + ["ALL"]:
        s = a if station == "ALL" else a.filter(pl.col("station") == station)
        neg = s.filter(pl.col("y_true") == 0)
        fp = neg.filter(pl.col("y_pred") == 1)["fl_z_max"].to_numpy()
        tn = neg.filter(pl.col("y_pred") == 0)["fl_z_max"].to_numpy()
        if len(fp) < 3 or len(tn) < 3:
            print(f"\n  {station}: FP={len(fp)}, TN={len(tn)} -- too few to test")
            continue
        p_val, eff = rank_biserial_and_u(fp, tn)
        print(f"\n  {station}:  false-positive windows n={len(fp)}, "
              f"true-negative windows n={len(tn)}")
        print(f"    median peak FL z-score:  FP={np.median(fp):+.2f}   "
              f"TN={np.median(tn):+.2f}")
        print(f"    Mann-Whitney p={p_val:.4f}   "
              f"P(FP > TN)={eff:.2f}  (0.5 = no difference)")
        if p_val < 0.05 and eff > 0.5:
            print("    -> FLAGGED windows DO show higher continuous chlorophyll.")
            print("       Supports: false positives are partly unobserved real blooms.")
        elif p_val < 0.05 and eff < 0.5:
            print("    -> Flagged windows show LOWER fluorescence. Does not support")
            print("       the cadence explanation.")
        else:
            print("    -> No significant difference. The cadence explanation is not")
            print("       supported by the buoy data at this station.")

    # sanity: does the buoy see the blooms DEEP DID observe?
    print("\n" + "-" * 66)
    print("SANITY CHECK: does the buoy see the blooms DEEP actually recorded?")
    pos = a.filter(pl.col("y_true") == 1)["fl_z_max"].to_numpy()
    negall = a.filter(pl.col("y_true") == 0)["fl_z_max"].to_numpy()
    if len(pos) >= 3 and len(negall) >= 3:
        p_val, eff = rank_biserial_and_u(pos, negall)
        print(f"  DEEP-positive windows n={len(pos)} (median z={np.median(pos):+.2f}) vs")
        print(f"  DEEP-negative windows n={len(negall)} (median z={np.median(negall):+.2f})")
        print(f"  Mann-Whitney p={p_val:.4f}  P(pos > neg)={eff:.2f}")
        print("  If this is NOT significant, the buoy cannot see real blooms either,")
        print("  and the whole audit is uninformative -- report that honestly.")
    else:
        print("  too few DEEP-positive windows with buoy coverage to check")


if __name__ == "__main__":
    main()