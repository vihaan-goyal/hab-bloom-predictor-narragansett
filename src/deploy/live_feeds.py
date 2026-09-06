"""
live_feeds.py -- pull recent sonde readings from the four live feeds used by the
prospective forecast and maintain one per-site history file.
-------------------------------------------------------------------------------
Purpose : turn each live source (UConn merlin ERDDAP, IOOS Sensors ERDDAP for the
          Kachemak Bay NERRS sondes and the erddap_top sites, Maryland Eyes on the
          Bay) into the predict_anywhere contract and append it to history.
Inputs  : a site dict from prospective_sites.SITES, a UTC window [start, end].
Outputs : DataFrames with exactly the columns station, datetime (naive UTC), chl,
          temp, sal, do; raw responses cached to
          data/prospective/raw/<issue_date>/<site_id>.csv (.txt for errors);
          history in data/prospective/history/<site_id>.csv.
Run     : library module (imported by prospective_forecast / score_ledger /
          prospective_freeze). Smoke test from the fork root:
          python -m src.deploy.live_feeds WLIS_ECO_FL 2026-08-30 2026-09-05
"""
import io
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import requests

from src.deploy import prospective_sites as ps

COLS = ["station", "datetime", "chl", "temp", "sal", "do"]
UA = {"User-Agent": "Mozilla/5.0 (hab-bloom-predictor research; student project)"}
RAW_DIR = os.path.join(ps.PROSPECTIVE_DIR, "raw")
HISTORY_DIR = os.path.join(ps.PROSPECTIVE_DIR, "history")
ERDDAP_EMPTY_MARKERS = ("outside of the variable's actual_range", "nRows = 0",
                        "Your query produced no matching results")


class FeedError(Exception):
    """HTTP-level failure (no 200 and not an ERDDAP empty-result 404)."""


def get(url, tries=3, timeout=300, params=None):
    """GET with retry; returns the Response (200 or 404) or raises FeedError."""
    wait, last = 5, "no attempt"
    for _ in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout, params=params)
            if r.status_code in (200, 404):
                return r
            last = f"HTTP {r.status_code}: {r.text[:120]!r}"
        except requests.RequestException as e:
            last = repr(e)[:200]
        time.sleep(wait)
        wait *= 3
    raise FeedError(last)


def _empty():
    return pd.DataFrame({c: pd.Series(dtype="float64" if c in ("chl", "temp", "sal", "do") else "object")
                         for c in COLS})


def _iso(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _erddap_frame(r):
    """ERDDAP CSV response -> DataFrame (None when the 404 means 'no rows')."""
    if r.status_code == 404:
        if any(m in r.text for m in ERDDAP_EMPTY_MARKERS):
            return None
        raise FeedError(f"HTTP 404: {r.text[:200]!r}")
    return pd.read_csv(io.StringIO(r.text), skiprows=[1])


def _finish(out):
    """Common QC: numeric chl in [0, 1000), valid times, exact column set."""
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["chl"] = pd.to_numeric(out["chl"], errors="coerce")
    for c in ("temp", "sal", "do"):
        out[c] = pd.to_numeric(out[c], errors="coerce") if c in out else np.nan
    out = out.dropna(subset=["datetime", "chl"])
    out = out[(out.chl >= 0) & (out.chl < 1000)]
    out["station"] = out["station"].astype(str)
    return out[COLS].sort_values(["station", "datetime"]).reset_index(drop=True)


def _utc_naive(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def fetch_ioos(site, start_utc, end_utc):
    """erddap_top site on IOOS Sensors ERDDAP -> (frame, raw_text)."""
    cols = ["time"] + ([site["station_var"]] if site["station_var"] else []) + [site["chl_var"]]
    cols += [v for v in (site["temp_var"], site["sal_var"], site["do_var"]) if v and v not in cols]
    url = (f"{site['base']}tabledap/{site['site_id']}.csv?{','.join(cols)}"
           f"&time>={_iso(start_utc)}&time<={_iso(end_utc)}")
    r = get(url)
    x = _erddap_frame(r)
    if x is None:
        return _empty(), r.text
    sv = site["station_var"]
    st = (x[sv].astype(str) if sv and sv in x and x[sv].notna().any() else pd.Series(site["station"], index=x.index))
    out = pd.DataFrame({"station": st, "datetime": _utc_naive(x["time"]), "chl": x[site["chl_var"]]})
    for name, v in (("temp", site["temp_var"]), ("sal", site["sal_var"]), ("do", site["do_var"])):
        out[name] = x[v] if v and v in x else np.nan
    return _finish(out), r.text


def fetch_nerrs_kac(site, start_utc, end_utc):
    """Kachemak Bay NERRS sonde (nerrs_<kacss|kach3>wq) with QARTOD filter, UTC."""
    ds = f"nerrs_{site['station'].replace('_', '')}wq"
    url = (f"{site['base']}tabledap/{ds}.csv?{','.join(ps.NERRS_VARS)}"
           f"&time>={_iso(start_utc)}&time<={_iso(end_utc)}")
    r = get(url)
    x = _erddap_frame(r)
    if x is None:
        return _empty(), r.text
    return parse_nerrs_frame(x, site["station"]), r.text


def parse_nerrs_frame(x, station):
    """Shared by the live fetch and the freeze seed (raw ERDDAP CSV, units row removed)."""
    chl = pd.to_numeric(x[ps.NERRS_VARS[1]], errors="coerce")
    qc = pd.to_numeric(x[ps.NERRS_VARS[2]], errors="coerce")
    out = pd.DataFrame({"station": station, "datetime": _utc_naive(x["time"]),
                        "chl": chl.where(qc.isin(ps.NERRS_QC_KEEP) | qc.isna()),
                        "temp": x[ps.NERRS_VARS[3]], "sal": x[ps.NERRS_VARS[4]], "do": x[ps.NERRS_VARS[5]]})
    return _finish(out)


def eastern_to_utc(local_naive):
    """Eyes on the Bay clock time (America/New_York) -> naive UTC."""
    t = pd.to_datetime(local_naive, errors="coerce")
    t = t.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    return t.dt.tz_convert("UTC").dt.tz_localize(None)


def parse_eotb_text(text, station):
    """JustDownload.cfm CSV -> contract frame (same rules as fetch_chesapeake.build)."""
    d = pd.read_csv(io.StringIO(text), dtype=str)
    ren = {"Chl_ug/L": "chl", "Temp_C": "temp", "Salinity_ppt": "sal", "DO_mg/L": "do"}
    for c in ren:
        if c not in d:
            d[c] = np.nan
    out = pd.DataFrame({"station": station,
                        "datetime": eastern_to_utc(pd.to_datetime(d["DateTime"], format="%m/%d/%Y %H:%M", errors="coerce"))})
    for src, dst in ren.items():
        v = pd.to_numeric(d[src], errors="coerce")
        out[dst] = v.where(v > -99)                 # -999 style sentinels -> NaN
    out = _finish(out)
    return out[out.chl <= 1000]


def fetch_chesapeake(site, start_utc, end_utc):
    """Maryland DNR Eyes on the Bay continuous monitor, one station, date window."""
    q = [("station", site["station"])] + [("parameter", p) for p in ps.EOTB_PARAMS] + [
        ("StartDate", (pd.Timestamp(start_utc) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
        ("EndDate", (pd.Timestamp(end_utc) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")), ("outputtype", "2")]
    r = get(site["base"], params=q)
    if r.status_code != 200:
        raise FeedError(f"HTTP {r.status_code}: {r.text[:200]!r}")
    if not r.text.lstrip().startswith("Sample_date"):
        raise ValueError(f"unexpected body: {r.text[:200]!r}")
    out = parse_eotb_text(r.text, site["station"])
    out = out[(out.datetime >= pd.Timestamp(start_utc)) & (out.datetime <= pd.Timestamp(end_utc))]
    return out.reset_index(drop=True), r.text


def night_only(frame):
    """Keep UTC hours in NIGHT_HOURS and FL_MIN < chl < FL_MAX (LIS ECO-FL de-quench)."""
    f = frame[(frame.chl > ps.FL_MIN) & (frame.chl < ps.FL_MAX)]
    return f[f.datetime.dt.hour.isin(ps.NIGHT_HOURS)]


def fetch_lisicos(site, start_utc, end_utc):
    """LIS buoy ECO-FL fluorescence from the UConn merlin ERDDAP (Avg_FL, night hours only)."""
    url = (f"{site['base']}tabledap/{site['site_id']}.csv?time,station,Avg_FL"
           f"&time>={_iso(start_utc)}&time<={_iso(end_utc)}")
    r = get(url)
    x = _erddap_frame(r)
    if x is None:
        return _empty(), r.text
    out = pd.DataFrame({"station": site["station"], "datetime": _utc_naive(x["time"]),
                        "chl": pd.to_numeric(x["Avg_FL"], errors="coerce"),
                        "temp": np.nan, "sal": np.nan, "do": np.nan})
    return night_only(_finish(out)).reset_index(drop=True), r.text


FETCHERS = {"merlin": fetch_lisicos, "ioos_nerrs": fetch_nerrs_kac, "eyesonthebay": fetch_chesapeake,
            "ioos": fetch_ioos}


def _cache_raw(issue_date, site_id, text, ok):
    d = os.path.join(RAW_DIR, str(issue_date))
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{site_id}.{'csv' if ok else 'txt'}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


def fetch_site(site, start_utc, end_utc, issue_date):
    """Dispatch on site['feed']; returns (frame, status, note).
    status in {ok, empty, http_error, parse_error}; the raw response is cached."""
    try:
        frame, text = FETCHERS[site["feed"]](site, start_utc, end_utc)
    except FeedError as e:
        _cache_raw(issue_date, site["site_id"], str(e), ok=False)
        return _empty(), "http_error", str(e)[:200]
    except Exception as e:                                       # parse/shape problems
        _cache_raw(issue_date, site["site_id"], traceback.format_exc(), ok=False)
        return _empty(), "parse_error", f"{type(e).__name__}: {str(e)[:160]}"
    _cache_raw(issue_date, site["site_id"], text, ok=True)
    if len(frame) == 0:
        return frame, "empty", "feed returned no readings in window"
    return frame, "ok", ""


def history_path(site):
    return os.path.join(HISTORY_DIR, f"{site['site_id']}.csv")


def load_history(site):
    p = history_path(site)
    if not os.path.exists(p):
        return _empty()
    h = pd.read_csv(p, parse_dates=["datetime"])
    h["station"] = h["station"].astype(str)
    return h[COLS]


def append_history(site, df):
    """Concat new rows onto the site history, dedupe on (station, datetime), write, return all."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    old = load_history(site)
    new = df[COLS].copy()
    new["datetime"] = pd.to_datetime(new["datetime"])
    full = pd.concat([old, new], ignore_index=True) if len(old) else new
    full = (full.drop_duplicates(["station", "datetime"]).sort_values(["station", "datetime"])
            .reset_index(drop=True))
    full.to_csv(history_path(site), index=False, date_format="%Y-%m-%d %H:%M:%S")
    return full


if __name__ == "__main__":
    sid, d0, d1 = sys.argv[1], pd.Timestamp(sys.argv[2]), pd.Timestamp(sys.argv[3]) + pd.Timedelta("23:59:59")
    frame, status, note = fetch_site(ps.sites_by_id()[sid], d0, d1, "smoke")
    print(status, note, len(frame))
    print(frame.head(), frame.tail(3), sep="\n")
