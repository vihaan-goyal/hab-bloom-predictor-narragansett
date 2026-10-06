"""
Characterize Chlorophyll vs Corrected_Chlorophyll.

WHY THIS MATTERS
The bloom label is bloom = (Chlorophyll > 10), confirmed at 100% match. But the
paper claims to predict "chlorophyll-a > 10 ug/L". The dataset carries a SECOND
column, Corrected_Chlorophyll, which is systematically ~4-5x smaller and only
agrees with the raw column on the bloom label 85.3% of the time.

If Corrected_Chlorophyll is the properly calibrated / phaeopigment-corrected
chlorophyll-a (the usual meaning of "corrected" in oceanography), then
thresholding the RAW column at 10 is not thresholding chlorophyll-a at 10, and
the paper's central definition needs restating.

This script does not guess. It measures:
  1. Distributions of both columns (are the magnitudes plausible for LIS chl-a?)
  2. The ratio raw/corrected: constant (a unit/calibration factor) or variable
     (a per-sample physical correction)?
  3. How many bloom labels flip under each definition, and the positive rate.
  4. What raw-column threshold is equivalent to Corrected > 10.
  5. Whether the model's key EDA facts (west-east gradient) survive the swap.

NOTE: Corrected_Chlorophyll is NULL for 2022-2024, so the label CANNOT simply be
switched -- three of the test years have no corrected values. Any change would
have to be argued, not just applied. This script tells you how big the issue is.

Run from repo root:
  python scripts/data_checks/compare_chl_columns.py
"""

import numpy as np
import polars as pl
from pathlib import Path

FINAL = Path("data/hab_features_final.csv")

f = pl.scan_csv(FINAL, schema_overrides={"station_name": pl.String},
                infer_schema_length=50000, ignore_errors=True).select(
    "station_name", "date", "depth_code", "Chlorophyll",
    "Corrected_Chlorophyll", "bloom").collect()
f = f.with_columns(
    pl.when(pl.col("station_name").str.contains(r"^\d+$"))
      .then(pl.col("station_name").str.zfill(2))
      .otherwise(pl.col("station_name")).alias("station_name"),
    pl.col("date").str.to_date("%Y-%m-%d", strict=False).alias("d"))

both = f.filter(pl.col("Chlorophyll").is_not_null()
                & pl.col("Corrected_Chlorophyll").is_not_null()
                & (pl.col("Chlorophyll") > 0)
                & (pl.col("Corrected_Chlorophyll") > 0))

print("=" * 68)
print("1. DISTRIBUTIONS")
print("=" * 68)
for c in ("Chlorophyll", "Corrected_Chlorophyll"):
    s = f[c].drop_nulls()
    q = [float(s.quantile(p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]
    print(f"  {c:<24} n={s.len():>9,}")
    print(f"    p5={q[0]:7.2f}  p25={q[1]:7.2f}  med={q[2]:7.2f}  "
          f"p75={q[3]:7.2f}  p95={q[4]:7.2f}  p99={q[5]:7.2f}")
    print(f"    frac > 10: {float((s > 10).mean()):.3f}")
print("\n  Typical Long Island Sound chlorophyll-a is roughly 1-30 ug/L,")
print("  with blooms above ~10-20. Which column looks like that?")

print("\n" + "=" * 68)
print("2. RATIO raw / corrected  (constant => unit or calibration factor;")
print("                           variable => per-sample physical correction)")
print("=" * 68)
r = (both.select(
        (pl.col("Chlorophyll") / pl.col("Corrected_Chlorophyll")).alias("ratio"))
     ["ratio"])
q = [float(r.quantile(p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
print(f"  n={r.len():,}")
print(f"  p5={q[0]:.3f}  p25={q[1]:.3f}  median={q[2]:.3f}  "
      f"p75={q[3]:.3f}  p95={q[4]:.3f}")
cv = float(r.std() / r.mean()) if r.mean() else float("nan")
print(f"  coefficient of variation = {cv:.3f}")
print("    CV < ~0.1  -> essentially a constant factor (unit/calibration issue)")
print("    CV > ~0.3  -> genuinely varies per sample (physical correction)")

x = both["Chlorophyll"].to_numpy().astype(float)
y = both["Corrected_Chlorophyll"].to_numpy().astype(float)
print(f"  Pearson r(raw, corrected)          = {np.corrcoef(x, y)[0,1]:+.4f}")
print(f"  Pearson r(log raw, log corrected)  = "
      f"{np.corrcoef(np.log(x), np.log(y))[0,1]:+.4f}")
# best-fit through origin
k = float((x * y).sum() / (x * x).sum())
print(f"  best-fit corrected = {k:.4f} * raw   (i.e. raw = {1/k:.2f} * corrected)")

print("\n" + "=" * 68)
print("3. LABEL DISAGREEMENT")
print("=" * 68)
b_raw = (both["Chlorophyll"] > 10).cast(pl.Int64).to_numpy()
b_cor = (both["Corrected_Chlorophyll"] > 10).cast(pl.Int64).to_numpy()
n = len(b_raw)
print(f"  on the {n:,} rows where BOTH columns exist:")
print(f"    positive rate, raw > 10        : {b_raw.mean():.4f}")
print(f"    positive rate, corrected > 10  : {b_cor.mean():.4f}")
print(f"    agree                          : {(b_raw == b_cor).mean():.4f}")
print(f"    raw says bloom, corrected does not : "
      f"{int(((b_raw == 1) & (b_cor == 0)).sum()):,}")
print(f"    corrected says bloom, raw does not : "
      f"{int(((b_raw == 0) & (b_cor == 1)).sum()):,}")

print("\n" + "=" * 68)
print("4. EQUIVALENT THRESHOLDS")
print("=" * 68)
pos_cor = float((both["Corrected_Chlorophyll"] > 10).mean())
raw_equiv = float(both["Chlorophyll"].quantile(1 - pos_cor)) if 0 < pos_cor < 1 else float("nan")
print(f"  'corrected > 10' fires on {pos_cor*100:.1f}% of rows.")
print(f"  The RAW threshold that fires on the same fraction is: {raw_equiv:.1f}")
print(f"  Conversely, 'raw > 10' corresponds to corrected > "
      f"{float(both['Corrected_Chlorophyll'].quantile(1 - float((both['Chlorophyll']>10).mean()))):.2f}")

print("\n" + "=" * 68)
print("5. DOES THE WEST-EAST GRADIENT SURVIVE THE SWAP?")
print("=" * 68)
g = (both.group_by("station_name").agg(
        pl.len().alias("n"),
        (pl.col("Chlorophyll") > 10).mean().alias("pos_raw"),
        (pl.col("Corrected_Chlorophyll") > 10).mean().alias("pos_cor"))
     .filter(pl.col("n") >= 200).sort("pos_raw", descending=True))
print(f"  {'station':<9}{'n':>8}{'raw>10':>9}{'corr>10':>10}")
for row in g.head(8).to_dicts():
    print(f"  {row['station_name']:<9}{row['n']:>8,}"
          f"{row['pos_raw']:>9.3f}{row['pos_cor']:>10.3f}")
print("  ...")
for row in g.tail(4).to_dicts():
    print(f"  {row['station_name']:<9}{row['n']:>8,}"
          f"{row['pos_raw']:>9.3f}{row['pos_cor']:>10.3f}")
pr = g["pos_raw"].to_numpy(); pc = g["pos_cor"].to_numpy()
if len(pr) > 2:
    print(f"\n  correlation of per-station bloom rates (raw vs corrected): "
          f"{np.corrcoef(pr, pc)[0,1]:+.3f}")
    print("  High correlation -> the spatial gradient is robust to the choice,")
    print("  and only the absolute bloom RATE changes, not the geography.")

print("\n" + "=" * 68)
print("WHAT TO DO WITH THIS")
print("=" * 68)
print("  This is a question for the data steward, not something to resolve by")
print("  inference. Katie O'Brien-Clayton (CT DEEP) has already replied to you.")
print("  Ask her directly:")
print("    - Which column is chlorophyll-a in ug/L?")
print("    - What does 'Corrected' correct for?")
print("    - Why is Corrected_Chlorophyll null for 2022-2024?")
print("  Her answer determines whether the 10 ug/L threshold is stated correctly.")