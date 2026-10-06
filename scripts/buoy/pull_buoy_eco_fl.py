"""
Pull UConn LISICOS moored-buoy ECO fluorometer (chlorophyll fluorescence) data
from the merlin ERDDAP tabledap service.

High-frequency in-situ chlorophyll from moored buoys. Attacks the monitoring-
cadence precision ceiling: CT DEEP discrete stations give <=1 chlorophyll
reading per 21-day window; these buoys log continuously.

Variables (WLIS_ECO_FL): time, chl_ugL (chlorophyll, ug/L), Avg_FL / Min_FL /
Max_FL / StdDev_FL (raw fluorescence stats), depth, station, latitude,
longitude.

SERVER QUIRK (auto-handled)
This ERDDAP advertises the chlorophyll variable as 'chl_ugl' but the backing
Postgres column is case-sensitively 'chl_ugL', so the advertised name 500s.
resolve_variables() probes each dataset, reads Postgres's suggested name out of
the error, and remaps (or drops as a last resort) before the real pull.

CALIBRATION CAVEAT
chl_ugL is derived from buoy fluorescence and is NOT the same quantity as CT
DEEP lab-extracted chlorophyll-a (non-photochemical quenching + sensor offset).
Use it as an independent continuous proxy or calibrate against co-located DEEP
discrete samples first. See notes at the bottom of this file.

Datasets:
  WLIS_ECO_FL  Western LIS buoy (the Narrows, where the biomass signal skews)
  CLIS_ECO_FL  Central LIS buoy
  EXRX_ECO_FL  Execution Rocks buoy
  ARTG_ECO_FL  eastern buoy

Run locally (needs network to merlin.dms.uconn.edu:8080):
  python scripts/buoy/pull_buoy_eco_fl.py
"""

import io
import re
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
import polars as pl

ERDDAP = "http://merlin.dms.uconn.edu:8080/erddap"

DATASETS = ["WLIS_ECO_FL", "CLIS_ECO_FL", "EXRX_ECO_FL", "ARTG_ECO_FL"]

OUT_DIR = Path("data/buoy_eco_fl")

DEFAULT_START_YEAR = 2004
DEFAULT_END_YEAR = datetime.now(timezone.utc).year

NO_DATA_MARKERS = ("no matching", "nrows = 0", "nrows=0", "your query produced no")

# ERDDAP wraps column names in escaped quotes inside its error body
# (e.g. column \"chl_ugl\"), so tolerate any number of backslashes before the ".
_MISSING_RE = re.compile(r'column\s+\\*"([^"\\]+)')
_HINT_RE = re.compile(r'reference the column\s+\\*"([^"\\]+)')

session = requests.Session()
session.headers.update({"User-Agent": "hab-bloom-predictor/1.0 (research)"})


def get_metadata(dataset_id):
    """Return (variables, start_year, end_year). Prefers time/actual_range."""
    url = f"{ERDDAP}/info/{dataset_id}/index.json"
    r = session.get(url, timeout=60)
    r.raise_for_status()
    rows = r.json()["table"]["rows"]

    variables = []
    global_start = global_end = None
    time_actual_range = None
    for row_type, var_name, attr_name, _dtype, value in rows:
        if row_type == "variable":
            variables.append(var_name)
        elif row_type == "attribute" and var_name == "NC_GLOBAL":
            if attr_name == "time_coverage_start":
                global_start = value
            elif attr_name == "time_coverage_end":
                global_end = value
        elif row_type == "attribute" and var_name == "time" and attr_name == "actual_range":
            time_actual_range = value

    yr = _years_from_actual_range(time_actual_range)
    if yr is not None:
        return variables, yr[0], yr[1]
    return (
        variables,
        _year_from_iso(global_start, DEFAULT_START_YEAR),
        _year_from_iso(global_end, DEFAULT_END_YEAR),
    )


def _years_from_actual_range(value):
    if not value:
        return None
    try:
        parts = [float(x) for x in str(value).split(",")]
        y0 = datetime.fromtimestamp(parts[0], timezone.utc).year
        y1 = datetime.fromtimestamp(parts[1], timezone.utc).year
        if 1990 <= y0 <= y1 <= DEFAULT_END_YEAR + 1:
            return y0, y1
    except (ValueError, TypeError, OSError, OverflowError):
        pass
    return None


def _year_from_iso(value, fallback):
    if not value:
        return fallback
    try:
        return int(str(value)[:4])
    except (ValueError, TypeError):
        return fallback


def _looks_like_no_data(body):
    low = body.lower()
    return any(m in low for m in NO_DATA_MARKERS)


def _parse_missing(body):
    """From a Postgres error, return (bad_column, suggested_column|None)."""
    m = _MISSING_RE.search(body)
    bad = m.group(1) if m else None
    h = _HINT_RE.search(body)
    suggestion = h.group(1).split(".")[-1] if h else None
    return bad, suggestion


def _probe(dataset_id, vars_, year):
    """One-day probe. Return (columns_valid, body).

    404 / no-data counts as valid columns (window just has no rows); a real
    'column does not exist' error is invalid.
    """
    varstr = ",".join(vars_)
    start = quote(f"{year}-06-01T00:00:00Z", safe="")
    end = quote(f"{year}-06-02T00:00:00Z", safe="")
    url = f"{ERDDAP}/tabledap/{dataset_id}.csv?{varstr}&time>={start}&time<{end}"
    try:
        r = session.get(url, timeout=120)
    except requests.RequestException as exc:
        return False, str(exc)
    if r.status_code == 200 or r.status_code == 404 or _looks_like_no_data(r.text):
        return True, r.text[:600]
    return False, r.text[:600]


def resolve_variables(dataset_id, variables, probe_year):
    """Return a queryable variable list, remapping or dropping bad columns."""
    vars_ = list(variables)
    ok, body = _probe(dataset_id, vars_, probe_year)
    guard = 0
    while not ok and guard < len(variables) + 3:
        guard += 1
        bad, suggestion = _parse_missing(body)
        if not bad or bad not in vars_:
            print(f"[{dataset_id}] unresolved probe error:\n{body[:400]}", file=sys.stderr)
            break
        if suggestion and suggestion not in vars_:
            trial = [suggestion if v == bad else v for v in vars_]
            t_ok, t_body = _probe(dataset_id, trial, probe_year)
            if t_ok:
                print(f"[{dataset_id}] remapped column {bad} -> {suggestion}")
                return trial
            vars_ = [v for v in vars_ if v != bad]
            print(f"[{dataset_id}] dropped {bad} (remap to {suggestion} failed)", file=sys.stderr)
        else:
            vars_ = [v for v in vars_ if v != bad]
            print(f"[{dataset_id}] dropped unqueryable column {bad}", file=sys.stderr)
        ok, body = _probe(dataset_id, vars_, probe_year)
    return vars_


def pull_year(dataset_id, variables, year):
    varstr = ",".join(variables)
    start = quote(f"{year}-01-01T00:00:00Z", safe="")
    end = quote(f"{year + 1}-01-01T00:00:00Z", safe="")
    url = f"{ERDDAP}/tabledap/{dataset_id}.csv?{varstr}&time>={start}&time<{end}"

    for attempt in range(3):
        try:
            r = session.get(url, timeout=300)
        except requests.RequestException as exc:
            print(f"    {year}: request error ({exc}); retrying", file=sys.stderr)
            _time.sleep(2 * (attempt + 1))
            continue

        if r.status_code == 200:
            df = pl.read_csv(
                io.BytesIO(r.content),
                skip_rows_after_header=1,
                infer_schema_length=20000,
            )
            return df if df.height > 0 else None

        body = r.text[:600]
        if r.status_code == 404 or _looks_like_no_data(body):
            return None
        print(f"    {year}: HTTP {r.status_code}: {body.strip()[:220]}", file=sys.stderr)
        _time.sleep(2 * (attempt + 1))

    return None


def pull_dataset(dataset_id):
    print(f"[{dataset_id}] reading metadata...")
    variables, start_year, end_year = get_metadata(dataset_id)
    print(f"[{dataset_id}] variables: {variables}")
    print(f"[{dataset_id}] data years: {start_year}-{end_year}")

    probe_year = max(start_year, min(end_year, (start_year + end_year) // 2))
    variables = resolve_variables(dataset_id, variables, probe_year)
    print(f"[{dataset_id}] using variables: {variables}")

    frames = []
    for year in range(start_year, end_year + 1):
        df = pull_year(dataset_id, variables, year)
        if df is not None:
            print(f"    {year}: {df.height:,} rows")
            frames.append(df)
        _time.sleep(0.4)

    if not frames:
        print(f"[{dataset_id}] no data returned")
        return None

    out = pl.concat(frames, how="diagonal_relaxed")

    if "time" in out.columns:
        try:
            out = out.with_columns(
                pl.col("time").cast(pl.Utf8).str.to_datetime(
                    "%Y-%m-%dT%H:%M:%SZ", strict=False
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{dataset_id}] time parse skipped ({exc})", file=sys.stderr)

    out = out.with_columns(pl.lit(dataset_id).alias("dataset_id"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{dataset_id}.parquet"
    out.write_parquet(path)
    print(f"[{dataset_id}] wrote {out.height:,} rows -> {path}")
    return out


def main():
    all_frames = []
    for ds in DATASETS:
        try:
            df = pull_dataset(ds)
        except requests.HTTPError as exc:
            print(f"[{ds}] failed: {exc}", file=sys.stderr)
            continue
        if df is not None:
            all_frames.append(df)

    if all_frames:
        combined = pl.concat(all_frames, how="diagonal_relaxed")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        combined_path = OUT_DIR / "all_buoys_eco_fl.parquet"
        combined.write_parquet(combined_path)
        print(f"\nCombined: {combined.height:,} rows -> {combined_path}")
    else:
        print("\nNo data pulled from any dataset.")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# NEXT STEP: calibration against CT DEEP discrete samples (sketch)
#   1. Match CT DEEP discrete chl-a samples near each buoy in space and within a
#      tight time window (same day, within a few km).
#   2. Regress extracted_chl ~ chl_ugL (or Avg_FL) on matched pairs. The scatter
#      tells you how noisy the proxy is.
#   3. Optionally drop midday samples to reduce non-photochemical quenching.
#   4. Apply the fit to the full buoy series, then check whether flagged 21-day
#      windows near WLIS/CLIS actually had a bloom the discrete sampling missed.
#      That converts "unverifiable false positive" into "verified".
# ---------------------------------------------------------------------------