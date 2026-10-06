"""
URGENT INTEGRITY CHECK.

The buoy coverage diagnostic revealed that Corrected_Chlorophyll is 100% NULL at
stations C1 and A4 for 2022, 2023 and 2024 (0 of ~2500 and ~4300 rows per year).
The test split is 2023-2025 and the bloom label is defined on
Corrected_Chlorophyll > 10. So: where do the 2023/2024 test labels come from?

This script answers, without speculating:
  1. Are Chlorophyll / Corrected_Chlorophyll null in 2022-2024 across ALL
     stations, or only C1/A4?
  2. What rule actually defines `bloom`? Test it against both columns.
  3. What years are in test_predictions.csv, and how many positives per year?
  4. For each test row, does a chlorophyll observation actually exist?
  5. Does the daily file (hab_features_daily / hab_labels_daily) have chlorophyll
     where the raw file does not?

Run from repo root:
  python scripts/data_checks/check_label_integrity.py
"""

import polars as pl
from pathlib import Path

FINAL = Path("data/hab_features_final.csv")
DAILY = Path("data/hab_features_daily.csv")
LABELS = Path("data/hab_labels_daily.csv")
PRED = Path("data/test_predictions.csv")

STR = {"station_name": pl.String}


def sect(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def norm(df):
    return df.with_columns(
        pl.when(pl.col("station_name").str.contains(r"^\d+$"))
          .then(pl.col("station_name").str.zfill(2))
          .otherwise(pl.col("station_name")).alias("station_name"))


# ---------- 1. null rates by year, ALL stations ----------
sect("1. CHLOROPHYLL NULL RATES BY YEAR (all stations, hab_features_final)")
f = pl.scan_csv(FINAL, schema_overrides=STR, infer_schema_length=50000,
                ignore_errors=True).select(
    "station_name", "date", "depth_code", "Chlorophyll",
    "Corrected_Chlorophyll", "bloom").collect()
f = norm(f).with_columns(pl.col("date").str.to_date("%Y-%m-%d", strict=False).alias("d"))

byyr = (f.group_by(pl.col("d").dt.year().alias("yr")).agg(
    pl.len().alias("rows"),
    pl.col("Chlorophyll").is_not_null().sum().alias("chl_raw"),
    pl.col("Corrected_Chlorophyll").is_not_null().sum().alias("chl_corr"),
    pl.col("bloom").is_not_null().sum().alias("has_bloom"),
    pl.col("bloom").sum().alias("n_bloom"),
).sort("yr").filter(pl.col("yr") >= 2015))
print(f"  {'yr':<6}{'rows':>8}{'raw_chl':>10}{'corr_chl':>10}{'has_lbl':>9}{'n_bloom':>9}")
for r in byyr.to_dicts():
    print(f"  {r['yr']:<6}{r['rows']:>8,}{r['chl_raw']:>10,}"
          f"{r['chl_corr']:>10,}{r['has_bloom']:>9,}{r['n_bloom'] or 0:>9,}")

# ---------- 2. what defines `bloom`? ----------
sect("2. WHAT RULE DEFINES `bloom`?")
t = f.filter(pl.col("bloom").is_not_null())
for col in ("Chlorophyll", "Corrected_Chlorophyll"):
    s = t.filter(pl.col(col).is_not_null())
    if s.height == 0:
        print(f"  {col}: no non-null rows to test")
        continue
    agree = (s.select((pl.col(col) > 10).cast(pl.Int64) == pl.col("bloom"))
              .to_series().sum())
    print(f"  bloom == ({col} > 10)  matches {agree:,}/{s.height:,} "
          f"= {100*agree/s.height:.1f}%")

# rows where bloom is set but BOTH chl columns are null -> label from nowhere
orphan = f.filter(pl.col("bloom").is_not_null()
                  & pl.col("Chlorophyll").is_null()
                  & pl.col("Corrected_Chlorophyll").is_null())
print(f"\n  rows with a bloom label but NO chlorophyll at all: {orphan.height:,}")
if orphan.height:
    oy = (orphan.group_by(pl.col("d").dt.year().alias("yr"))
                .agg(pl.len().alias("n"), pl.col("bloom").sum().alias("pos"))
                .sort("yr").filter(pl.col("yr") >= 2019))
    print("    by year (2019+):  " + ", ".join(
        f"{r['yr']}: {r['n']} rows / {r['pos'] or 0} positive" for r in oy.to_dicts()))

# ---------- 3. test_predictions ----------
sect("3. TEST PREDICTIONS: years and positives")
p = pl.read_csv(PRED, schema_overrides=STR)
p = norm(p).with_columns(pl.col("date").str.to_date("%Y-%m-%d").alias("d"))
py = (p.group_by(pl.col("d").dt.year().alias("yr")).agg(
        pl.len().alias("n"), pl.col("y_true").sum().alias("pos")).sort("yr"))
print(f"  {'yr':<6}{'rows':>8}{'y_true=1':>10}")
for r in py.to_dicts():
    print(f"  {r['yr']:<6}{r['n']:>8,}{r['pos']:>10,}")
print(f"\n  stations: {sorted(p['station_name'].unique().to_list())}")

# ---------- 4. do test rows have real chlorophyll behind them? ----------
sect("4. DO TEST ROWS HAVE OBSERVED CHLOROPHYLL? (raw file, +/-21d forward window)")
obs = (f.filter(pl.col("Corrected_Chlorophyll").is_not_null())
        .select("station_name", "d").unique())
obs_set = set(zip(obs["station_name"].to_list(), obs["d"].to_list()))
import datetime as _dt
rows = []
for r in p.iter_rows(named=True):
    hit = any((r["station_name"], r["d"] + _dt.timedelta(days=k)) in obs_set
              for k in range(0, 22))
    rows.append({"yr": r["d"].year, "y_true": r["y_true"], "has_obs": int(hit)})
chk = pl.DataFrame(rows)
g = (chk.group_by("yr").agg(pl.len().alias("n"),
                            pl.col("has_obs").sum().alias("with_obs"),
                            pl.col("y_true").sum().alias("pos")).sort("yr"))
print(f"  {'yr':<6}{'test rows':>11}{'w/ corr_chl obs':>17}{'y_true=1':>10}")
for r in g.to_dicts():
    print(f"  {r['yr']:<6}{r['n']:>11,}{r['with_obs']:>17,}{r['pos']:>10,}")
print("\n  If a year has y_true=1 rows but 0 with observed Corrected_Chlorophyll,")
print("  the labels for that year did NOT come from this column.")

# ---------- 5. daily files ----------
sect("5. DAILY FILES: do they have chlorophyll where the raw file does not?")
for path in (DAILY, LABELS):
    if not path.exists():
        print(f"  {path}: not found")
        continue
    d = pl.scan_csv(path, schema_overrides=STR, infer_schema_length=50000,
                    ignore_errors=True).select(
        "station_name", "date", "Chlorophyll", "Corrected_Chlorophyll", "bloom"
    ).collect()
    d = norm(d).with_columns(pl.col("date").str.to_date("%Y-%m-%d", strict=False).alias("d"))
    dy = (d.group_by(pl.col("d").dt.year().alias("yr")).agg(
            pl.len().alias("rows"),
            pl.col("Chlorophyll").is_not_null().sum().alias("raw"),
            pl.col("Corrected_Chlorophyll").is_not_null().sum().alias("corr"),
            pl.col("bloom").sum().alias("pos")).sort("yr").filter(pl.col("yr") >= 2019))
    print(f"\n  {path.name}")
    print(f"    {'yr':<6}{'rows':>8}{'raw':>8}{'corr':>8}{'bloom=1':>9}")
    for r in dy.to_dicts():
        print(f"    {r['yr']:<6}{r['rows']:>8,}{r['raw']:>8,}"
              f"{r['corr']:>8,}{r['pos'] or 0:>9,}")

print("\n" + "=" * 70)
print("VERDICT GUIDE")
print("  If 2023/2024 test rows carry y_true labels but the underlying")
print("  Corrected_Chlorophyll is entirely null, those labels came from")
print("  somewhere else -- find out where before trusting any test metric.")