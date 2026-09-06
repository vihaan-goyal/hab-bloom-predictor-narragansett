"""
prospective_sites.py -- the pre-registered site list and shared helpers for the
prospective (real-time, forward-looking) forecast protocol.
-------------------------------------------------------------------------------
Purpose : one place that defines WHICH stations are forecast, which live feed
          serves each, how many sub-daily readings make a station-day, and the
          model / code / protocol version stamps written into every ledger row.
Inputs  : data/registry/site_skill.csv and data/registry/insitu_catalog.csv
          (erddap_top selection rule), release/narragansett_bloom_model.joblib,
          git working tree (code_version).
Outputs : none on disk. Exposes SITES (list of dicts), erddap_top_from_catalog(),
          model_version(), code_version(), load_pa(), load_pack(),
          build_site_daily().
Run     : python -m src.deploy.prospective_sites      (prints the site table)
          from the fork root with C:/Users/vihaa/anaconda3/python.exe
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROTOCOL_VERSION = "1.1"   # 1.0 frozen 2026-09-06T00:37Z; 1.1 amendment 2026-09-06 (see notes/PROSPECTIVE_PROTOCOL.md)
HORIZON = 7
MODEL_PATH = os.path.join(ROOT, "release", "narragansett_bloom_model.joblib")
SKILL_CSV = os.path.join(ROOT, "data", "registry", "site_skill.csv")
CAT_CSV = os.path.join(ROOT, "data", "registry", "insitu_catalog.csv")
PROSPECTIVE_DIR = os.path.join(ROOT, "data", "prospective")
P75_PATH = os.path.join(PROSPECTIVE_DIR, "site_p75.csv")

MERLIN_BASE = "http://merlin.dms.uconn.edu:8080/erddap/"
IOOS_BASE = "https://erddap.sensors.ioos.us/erddap/"
EOTB_URL = "https://eyesonthebay.dnr.maryland.gov/contmon/JustDownload.cfm"

# LIS buoy recipe constants (copied from the parent repo's
# src/models/experiments/lis_buoy_recipe.py; experiments are never imported)
NIGHT_HOURS = [21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # UTC, de-quench
MIN_READINGS_PER_DAY = 12
MAX_STUCK_FRAC = 0.5          # drop a day if >50% of night readings are one value
FL_MIN, FL_MAX = 0.0, 5000.0  # drop zero/negative and spike readings

# NERRS Kachemak Bay ERDDAP variables (as in src/transfer/fetch_nerrs.py)
NERRS_VARS = ["time", "mass_concentration_of_chlorophyll_in_sea_water",
              "mass_concentration_of_chlorophyll_in_sea_water_qc_agg",
              "sea_water_temperature", "sea_water_practical_salinity",
              "mass_concentration_of_oxygen_in_sea_water"]
# QARTOD aggregate flag filter: 1 pass, 2 not evaluated. Rows whose flag is NaN/absent are KEPT
# (protocol 1.1 (d): this was the 1.0 code behaviour for nerrs and the frozen p75 used it).
# Protocol 1.1 (c): the same rule applies to erddap_top sites via <chl_var>_qc_agg when the dataset has it.
QARTOD_KEEP = {1, 2}
# erddap_top (protocol 1.1, amended 2026-09-06): drop only QARTOD 4 (fail) and 9 (missing).
# Flag 3 (suspect) is kept because mlml_mlml_sea flags most readings 3 and the frozen p75 rests
# on unfiltered history; the Scripps flat-line zeros were flag 4 and are still excluded.
ERDDAP_TOP_QC_KEEP = {1, 2, 3}
NERRS_QC_KEEP = QARTOD_KEEP   # 1.0 name, still used by parse_nerrs_frame

EOTB_PARAMS = ["wtemp", "Salinity", "DO", "ph", "DOpctSat", "TChlPreCal"]

# Selection rule for the erddap_top group (pre-registered 2026-09-05)
ERDDAP_TOP_RULE = dict(lift_lo_gt=1.0, n_onset_min=500, server="IOOS-Sensors",
                       catalog_end_min="2026-07-01", n=10)
EXPECTED_ERDDAP_TOP = [
    "scripps-pier-automated-shore-sta-1", "newport-pier-automated-shore-sta",
    "edu_ucsc_scwharf1", "oa2-mbari-buoy", "mlml_mlml_sea", "tiburon-water-tibc1",
    "edu_calpoly_marine_morro", "edu_humboldt_humboldt",
    "indian-river-lagoon-banana-river", "indian-river-lagoon-vero-beach-i"]


def _site(site_group, site_id, feed, base, chl_var, min_readings, chl_units, station=None,
          temp_var=None, sal_var=None, do_var=None, station_var=None, t_star_site=float("nan")):
    return dict(site_group=site_group, site_id=site_id, feed=feed, base=base, chl_var=chl_var,
                temp_var=temp_var, sal_var=sal_var, do_var=do_var, station_var=station_var,
                station=station or site_id, min_readings=int(min_readings), chl_units=chl_units,
                t_star_site=float(t_star_site), fresh_site=False, seed_from=None)


# Protocol 1.1 amendments (2026-09-06, before the first issuance). Per-site overrides applied on top of
# the pre-registered erddap_top rule; every key here is a deliberate departure from insitu_catalog.csv.
#  (a) scripps-pier-automated-shore-sta-1: the catalog chl_var (..._ctd) has read a constant 0.0 flagged
#      QARTOD 4 since 2026-04-09 and NaN since 2026-09-01; the ECO fluorometer channel (..._eco, live since
#      2024-12-04) replaces it. History re-seeded from _eco only: prospective_freeze pulls the channel live
#      from seed_from (2024-12-01) to the freeze time, QARTOD {1,2,NaN} applied, raw cached under
#      data/prospective/raw/freeze_1.1/. t_star_site (chosen on _ctd) withdrawn; fresh_site=True so
#      alert_site_t is not computed and the site is reported with that caveat.
#  (b) newport-pier-automated-shore-sta: NOT changed. Its _ctd chl went NaN after 2026-08-11T15:24Z and an
#      _eco channel appeared 2026-08-13 (too short for a p75). It stays on _ctd (feed_down/warmup until
#      the CTD returns); a dated amendment may switch it to _eco once 90 days of _eco exist.
SITE_OVERRIDES = {
    "scripps-pier-automated-shore-sta-1": dict(
        chl_var="mass_concentration_of_chlorophyll_in_sea_water_eco", t_star_site=float("nan"),
        fresh_site=True, seed_from="2024-12-01"),
}


LIS_UNITS = "ECO-FL fluorescence, night-only, uncalibrated"
LIS_SITES = [_site("lis_buoy", s, "merlin", MERLIN_BASE, "Avg_FL", MIN_READINGS_PER_DAY, LIS_UNITS,
                   station_var="station") for s in ("WLIS_ECO_FL", "EXRX_ECO_FL")]
NERRS_SITES = [_site("nerrs", s, "ioos_nerrs", IOOS_BASE, NERRS_VARS[1], 48, "ug/L ChlFluor (QARTOD 1,2)",
                     temp_var=NERRS_VARS[3], sal_var=NERRS_VARS[4], do_var=NERRS_VARS[5])
               for s in ("kac_ss", "kac_h3")]
CHESAPEAKE_SITES = [_site("chesapeake", s, "eyesonthebay", EOTB_URL, "Chl_ug/L", 48, "ug/L TChlPreCal",
                          temp_var="Temp_C", sal_var="Salinity_ppt", do_var="DO_mg/L")
                    for s in ("MSC", "AWS", "AES", "MAB", "RIV", "SPS")]


def _str_or_none(v):
    return v if isinstance(v, str) and v else None


def erddap_top_from_catalog(n=10):
    """Apply the pre-registered rule to site_skill.csv + insitu_catalog.csv."""
    r = ERDDAP_TOP_RULE
    sk = pd.read_csv(SKILL_CSV)
    cat = pd.read_csv(CAT_CSV).drop_duplicates(subset=["server", "dataset_id"])
    m = sk.merge(cat, on=["server", "dataset_id"], how="left", suffixes=("", "_cat"))
    keep = ((m.lift_lo > r["lift_lo_gt"]) & (m.n_onset >= r["n_onset_min"]) & (m.server == r["server"])
            & (pd.to_datetime(m.end, errors="coerce") >= pd.Timestamp(r["catalog_end_min"])))
    top = m[keep].sort_values("lift", ascending=False).head(n)
    got = top.dataset_id.tolist()
    if set(got) != set(EXPECTED_ERDDAP_TOP[:n]):
        print(f"erddap_top rule differs from the expected list:\n  rule    : {got}\n"
              f"  expected: {EXPECTED_ERDDAP_TOP[:n]}\n  using the rule's result", file=sys.stderr)
    out = []
    for _, row in top.iterrows():
        out.append(_site("erddap_top", row.dataset_id, "ioos", row.url.rsplit("/tabledap/", 1)[0] + "/",
                         row.chl_var, 48 if row.cadence_min <= 20 else 12, "site fluorometer units (see catalog)",
                         temp_var=_str_or_none(row.temp_var), sal_var=_str_or_none(row.sal_var),
                         do_var=_str_or_none(row.do_var), station_var=_str_or_none(row.station_var),
                         t_star_site=row.t_star))
        out[-1].update(SITE_OVERRIDES.get(row.dataset_id, {}))      # protocol 1.1 (a)
    return out


SITES = LIS_SITES + NERRS_SITES + CHESAPEAKE_SITES + erddap_top_from_catalog(ERDDAP_TOP_RULE["n"])
SITE_GROUPS = ["lis_buoy", "nerrs", "chesapeake", "erddap_top"]


def sites_by_id():
    return {s["site_id"]: s for s in SITES}


def model_sha256_full():
    with open(MODEL_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def model_version():
    return model_sha256_full()[:16]


def code_version():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def load_pa():
    """Import predict_anywhere.py by path (it is a top-level script, not a package)."""
    spec = importlib.util.spec_from_file_location("pa", os.path.join(ROOT, "predict_anywhere.py"))
    pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa)
    return pa


def load_pack():
    return joblib.load(MODEL_PATH)


def lis_stuck_days(hist):
    """(station, date) pairs where >MAX_STUCK_FRAC of night readings are one value."""
    h = hist.dropna(subset=["chl"]).copy()
    h["date"] = pd.to_datetime(h["datetime"]).dt.normalize()
    frac = h.groupby(["station", "date"])["chl"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    return set(frac[frac > MAX_STUCK_FRAC].index)


def build_site_daily(site, hist, pa):
    """Station-day table with tier-A features; LIS buoys also drop stuck-sensor days."""
    if hist is None or len(hist) == 0:
        return None
    day = pa.build_daily(hist, site["min_readings"])
    if site["site_group"] == "lis_buoy" and len(day):
        stuck = lis_stuck_days(hist)
        keep = [(s, d) not in stuck for s, d in zip(day.station, day.date)]
        day = day[keep].reset_index(drop=True)
    return day


if __name__ == "__main__":
    cols = ["site_group", "site_id", "feed", "station", "chl_var", "min_readings", "t_star_site", "fresh_site"]
    print(pd.DataFrame(SITES)[cols].to_string(index=False))
    print(f"\n{len(SITES)} sites | model {model_version()} | code {code_version()} | protocol {PROTOCOL_VERSION}")


def md_table(df):
    """Plain-ASCII markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)
