"""
Inventory data/ so we can pick the right DEEP chlorophyll table and get its schema.

Run from the repo root:
  python scripts/data_checks/describe_data.py
"""

from pathlib import Path
import polars as pl

ROOT = Path("data")
TABULAR = {".csv", ".parquet", ".tsv"}

def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

files = sorted(p for p in ROOT.rglob("*") if p.is_file())
print(f"{len(files)} files under {ROOT}/\n")

# Non-tabular: just list them
other = [p for p in files if p.suffix.lower() not in TABULAR]
if other:
    print("--- non-tabular (listed only) ---")
    for p in other:
        print(f"  {p.as_posix():<55} {human(p.stat().st_size)}")
    print()

print("--- tabular (schema + row count) ---")
for p in [p for p in files if p.suffix.lower() in TABULAR]:
    size = human(p.stat().st_size)
    try:
        if p.suffix.lower() == ".parquet":
            lf = pl.scan_parquet(p)
        else:
            sep = "\t" if p.suffix.lower() == ".tsv" else ","
            lf = pl.scan_csv(p, separator=sep, infer_schema_length=10000,
                             ignore_errors=True)
        schema = lf.collect_schema()
        nrows = lf.select(pl.len()).collect().item()
        print(f"\n{p.as_posix()}  [{size}, {nrows:,} rows]")
        for name, dt in schema.items():
            print(f"    {name}: {dt}")
        # show one sample row so date formats are visible
        head = lf.head(1).collect()
        if head.height:
            print("    sample:", head.to_dicts()[0])
    except Exception as exc:  # noqa: BLE001
        print(f"\n{p.as_posix()}  [{size}]  <could not read: {exc}>")