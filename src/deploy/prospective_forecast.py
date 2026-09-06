"""
prospective_forecast.py -- issue one dated, forward-looking bloom forecast for
every pre-registered station and append it to the immutable ledger.
-------------------------------------------------------------------------------
Purpose : pull the last --window-days of readings per site, extend the site
          history, rebuild station-days + tier-A features on the FULL history,
          rescale to the training chl scale, apply the frozen model, and write
          one ledger row per pre-registered station (feed_down rows included).
Inputs  : data/prospective/site_p75.csv (frozen thresholds; run prospective_freeze
          first), live feeds via src/deploy/live_feeds.py, the frozen model.
Outputs : <ledger-dir>/ledger.csv (append-only; header only when new)
          <ledger-dir>/issued/<D>.csv and <D>_stationdays.csv
          <ledger-dir>/outbox/<D>_email.txt        (LIS buoys, manager language)
          notes/prospective/digest_<D>.md          (<ledger-dir>/notes/ if non-default)
          data/prospective/logs/forecast_<D>.log   (+ stdout)
Run     : python -m src.deploy.prospective_forecast [--issue-date YYYY-MM-DD]
              [--dry-run] [--force] [--sites A,B] [--window-days 35] [--stale-days 3]
              [--ledger-dir data/prospective] [--base-override SITE_ID=URL]
          from the fork root with C:/Users/vihaa/anaconda3/python.exe
Exit 2  : the ledger already holds this issue_date (use --force to add revision n+1).
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.deploy import live_feeds as lf
from src.deploy import prospective_sites as ps

LEDGER_COLS = ["issue_date", "issued_at_utc", "revision", "site_group", "site_id", "station", "feed", "status",
               "last_obs_date", "days_old", "n_days_35", "chl_today", "chl_units", "chl_p75_site", "bloom_prob",
               "threshold", "alert", "onset_row", "t_star_site", "alert_site_t", "warmup", "horizon_days",
               "model_version", "code_version", "protocol_version", "note"]
STATIONDAY_COLS = ["site_id", "station", "date", "n", "chl", "temp", "sal", "do", "bloom_prob", "alert", "warmup"]
WARMUP_DAYS = 21
SKILL_GATE = dict(min_n=30, min_pos=5)
log = logging.getLogger("prospective_forecast")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="issue the prospective bloom forecast")
    ap.add_argument("--issue-date", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--force", action="store_true", help="re-issue an existing date as revision n+1")
    ap.add_argument("--sites", default=None, help="comma list of site_id (rows get note=subset)")
    ap.add_argument("--dry-run", action="store_true", help="fetch + score + print; write no ledger/issued/digest/outbox")
    ap.add_argument("--window-days", type=int, default=35)
    ap.add_argument("--stale-days", type=int, default=3)
    ap.add_argument("--ledger-dir", default=ps.PROSPECTIVE_DIR,
                    help="where ledger/issued/outbox/notes go (default data/prospective; raw+history stay in data/prospective)")
    ap.add_argument("--base-override", default=None, metavar="SITE_ID=URL", help="test hook: replace one site's base URL")
    return ap.parse_args(argv)


def paths_for(ledger_dir):
    ledger_dir = os.path.abspath(ledger_dir)
    default = os.path.abspath(ledger_dir) == os.path.abspath(ps.PROSPECTIVE_DIR)
    return dict(ledger=os.path.join(ledger_dir, "ledger.csv"), issued=os.path.join(ledger_dir, "issued"),
                outbox=os.path.join(ledger_dir, "outbox"), skill=os.path.join(ledger_dir, "skill_summary.csv"),
                digest=os.path.join(ps.ROOT, "notes", "prospective") if default else os.path.join(ledger_dir, "notes"),
                logs=os.path.join(ps.PROSPECTIVE_DIR, "logs"))


def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def load_p75():
    if not os.path.exists(ps.P75_PATH):
        sys.exit(f"{ps.P75_PATH} missing: run python -m src.deploy.prospective_freeze first")
    return pd.read_csv(ps.P75_PATH, dtype={"station": str})


def revision_for(ledger_path, issue_date, force):
    """Duplicate guard: exit 2 if the ledger already has this issue_date, unless --force."""
    if not os.path.exists(ledger_path):
        return 1
    led = pd.read_csv(ledger_path, usecols=["issue_date", "revision"], dtype={"issue_date": str})
    dup = led[led.issue_date == issue_date]
    if len(dup) == 0:
        return 1
    if not force:
        log.error(f"ledger already has issue_date {issue_date} (revision {int(dup.revision.max())}); "
                  f"use --force to add a new revision")
        raise SystemExit(2)
    return int(dup.revision.max()) + 1


def fetch_and_update(site, start, end, issue_date):
    """Live pull for the window; history is extended only when the pull succeeded."""
    frame, fstatus, fnote = lf.fetch_site(site, start, end, issue_date)
    if fstatus == "ok":
        hist = lf.append_history(site, frame)
    else:
        hist = lf.load_history(site)
    log.info(f"{site['site_id']}: fetch {fstatus} ({len(frame)} rows){' - ' + fnote if fnote else ''}; "
             f"history {len(hist)} rows")
    return hist, fstatus, fnote


def score_day(day, pack, pa):
    scored = pa.rescale_chl(day, pack["chl_quantiles"])
    X = scored[pack["features"]].fillna(pd.Series(pack["medians"])).fillna(0.0).values
    day = day.copy()
    day["bloom_prob"] = pack["model"].predict_proba(X)[:, 1]
    day["alert"] = day.bloom_prob >= pack["threshold"]
    day["warmup"] = day.chl_roll21_mean.isna()
    return day


def row_status(fetch_failed, last, n_days, days_old, stale_days):
    if last is None or (fetch_failed and n_days == 0):
        return "feed_down"
    if n_days < WARMUP_DAYS or bool(last.warmup):
        return "warmup"
    if days_old > stale_days:
        return "stale"
    return "ok"


def station_row(site, prow, day, fstatus, fnote, D, a, pack, stamps, subset):
    """One ledger row for one pre-registered station."""
    st = str(prow.station)
    fetch_failed = fstatus != "ok"
    sd = day[day.station == st] if day is not None else None
    win_lo = D - pd.Timedelta(days=a.window_days)
    n_days = int(((sd.date >= win_lo) & (sd.date <= D)).sum()) if sd is not None else 0
    past = sd[sd.date <= D] if sd is not None else None
    last = past.iloc[-1] if past is not None and len(past) else None
    days_old = int((D - last.date).days) if last is not None else np.nan
    status = row_status(fetch_failed, last, n_days, days_old, a.stale_days)
    notes = (["subset"] if subset else []) + ([fnote] if fnote else [])
    row = dict(issue_date=D.strftime("%Y-%m-%d"), site_group=site["site_group"], site_id=site["site_id"], station=st,
               feed=site["feed"], status=status, last_obs_date=last.date.strftime("%Y-%m-%d") if last is not None else np.nan,
               days_old=days_old, n_days_35=n_days, chl_units=site["chl_units"], chl_p75_site=float(prow.chl_p75_site),
               threshold=float(pack["threshold"]), t_star_site=site["t_star_site"], horizon_days=ps.HORIZON,
               note="; ".join(notes), chl_today=np.nan, bloom_prob=np.nan, alert=np.nan, onset_row=np.nan,
               alert_site_t=np.nan, warmup=np.nan, **stamps)
    if status != "feed_down":
        p = float(last.bloom_prob)
        row.update(chl_today=float(last.chl), bloom_prob=p, alert=bool(p >= pack["threshold"]),
                   onset_row=bool(last.chl <= prow.chl_p75_site), warmup=bool(last.warmup),
                   alert_site_t=(bool(p >= site["t_star_site"])                     # fresh sites (t* NaN): NaN
                                 if site["site_group"] == "erddap_top" and np.isfinite(site["t_star_site"]) else np.nan))
    return row


def forecast_site(site, p75, D, a, pack, pa, stamps, subset):
    start = (D - pd.Timedelta(days=a.window_days)).normalize()
    end = D.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    hist, fstatus, fnote = fetch_and_update(site, start, end, D.strftime("%Y-%m-%d"))
    day = ps.build_site_daily(site, hist, pa)
    day = score_day(day, pack, pa) if day is not None and len(day) else None
    if day is None and not fnote:
        fnote = "no station-days in history"
    rows = [station_row(site, prow, day, fstatus, fnote, D, a, pack, stamps, subset)
            for prow in p75[p75.site_id == site["site_id"]].itertuples()]
    win = (day[(day.date >= start) & (day.date <= D)].assign(site_id=site["site_id"])[STATIONDAY_COLS]
           if day is not None else pd.DataFrame(columns=STATIONDAY_COLS))
    return rows, win


def write_ledger(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def write_issued(rows, stationdays, issued_dir, D):
    os.makedirs(issued_dir, exist_ok=True)
    rows.to_csv(os.path.join(issued_dir, f"{D}.csv"), index=False)
    stationdays.to_csv(os.path.join(issued_dir, f"{D}_stationdays.csv"), index=False)


def scoreboard(skill_path):
    if not os.path.exists(skill_path):
        return "No verified outcomes yet (skill_summary.csv not present)."
    sk = pd.read_csv(skill_path)
    ok = sk[(sk.n >= SKILL_GATE["min_n"]) & (sk.n_pos >= SKILL_GATE["min_pos"])]
    if len(ok) == 0:
        return (f"Verified rows exist but no stratum meets the reporting gate yet "
                f"(n >= {SKILL_GATE['min_n']} and >= {SKILL_GATE['min_pos']} positives).")
    cols = ["as_of", "scope", "site_group", "n", "n_pos", "base_rate", "precision", "lift", "lift_lo", "lift_hi", "pod", "auc"]
    return ps.md_table(ok[cols].round(3))


def write_digest(rows, D, p):
    os.makedirs(p["digest"], exist_ok=True)
    cols = ["site_id", "station", "status", "last_obs_date", "days_old", "n_days_35", "chl_today", "chl_p75_site",
            "bloom_prob", "alert", "onset_row", "alert_site_t", "note"]
    parts = [f"# Prospective bloom forecast digest, issued {D}", "",
             f"protocol {ps.PROTOCOL_VERSION} | model {rows.model_version.iloc[0]} | code {rows.code_version.iloc[0]} | "
             f"threshold {rows.threshold.iloc[0]:.2f} | horizon {ps.HORIZON} d | revision {int(rows.revision.iloc[0])}", "",
             "Status counts: " + ", ".join(f"{k}={v}" for k, v in rows.status.value_counts().items()), ""]
    for g in ps.SITE_GROUPS:
        sub = rows[rows.site_group == g]
        if len(sub):
            parts += [f"## {g}", "", ps.md_table(sub[cols].round(3)), ""]
    parts += ["## Scoreboard (verified outcomes to date)", "", scoreboard(p["skill"]), ""]
    path = os.path.join(p["digest"], f"digest_{D}.md")
    with open(path, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(parts))
    return path


def buoy_paragraph(r):
    name = {"WLIS_ECO_FL": "Western Sound buoy (WLIS)", "EXRX_ECO_FL": "Execution Rocks buoy (EXRX)"}.get(r.site_id, r.site_id)
    if r.status == "feed_down":
        return f"{name}: no usable readings this week (feed down: {r.note}). No outlook issued."
    when = f"last night-mean reading {r.chl_today:.0f} on {r.last_obs_date}"
    if r.days_old > 0:
        when += f" ({int(r.days_old)} days old)"
    kind = ("already above the buoy's own 75th percentile today (persistence)" if not r.onset_row
            else "below the 75th percentile today, so this is onset risk")
    lines = [f"{name}: {when}, versus the buoy's 75th-percentile level of {r.chl_p75_site:.0f}; {kind}.",
             f"  Probability of exceeding that level within 7 days: {r.bloom_prob:.0%}. "
             f"Alert: {'YES' if r.alert else 'no'}."]
    if r.status == "stale":
        lines.append(f"  Note: the latest reading is {int(r.days_old)} days old (stale); treat with caution.")
    if r.status == "warmup":
        lines.append("  Note: fewer than 21 recent station-days; the model is in warm-up and skill is lower.")
    return "\n".join(lines)


def write_outbox(rows, D, outbox_dir):
    lis = rows[rows.site_group == "lis_buoy"]
    body = [f"Subject: LIS buoy bloom outlook, week of {D}", "",
            "Weekly outlook from the Narragansett-trained bloom model applied to the two LIS ECO-FL buoys.", ""]
    body += [buoy_paragraph(r) + "\n" for r in lis.itertuples()]
    body += ["Standing caveats:",
             "- Chlorophyll here is ECO-FL fluorescence, night-only, uncalibrated; levels are relative to each buoy's own record.",
             "- Alert = model probability >= 0.50 of exceeding the buoy's 75th percentile within 7 days.",
             "- Expected skill: about 1.5-2x the precision of always alerting (pre-registered band 2-3x for these buoys).",
             "- A running verified score will be reported once at least 30 outcomes (>= 5 blooms) have accrued.",
             f"- Protocol {ps.PROTOCOL_VERSION}; model {rows.model_version.iloc[0]}; issued {rows.issued_at_utc.iloc[0]}."]
    os.makedirs(outbox_dir, exist_ok=True)
    path = os.path.join(outbox_dir, f"{D}_email.txt")
    with open(path, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(body) + "\n")
    return path


def main(argv=None):
    a = parse_args(argv)
    D = pd.Timestamp(a.issue_date).normalize()
    Ds = D.strftime("%Y-%m-%d")
    p = paths_for(a.ledger_dir)
    setup_logging(os.path.join(p["logs"], f"forecast_{Ds}.log"))
    log.info(f"issue_date {Ds} | dry_run={a.dry_run} | ledger_dir={os.path.abspath(a.ledger_dir)}")
    revision = revision_for(p["ledger"], Ds, a.force)
    if a.base_override:
        sid, url = a.base_override.split("=", 1)
        ps.sites_by_id()[sid]["base"] = url
        log.warning(f"TEST HOOK: base of {sid} overridden to {url}")
    sites = ps.SITES if not a.sites else [s for s in ps.SITES if s["site_id"] in a.sites.split(",")]
    subset = a.sites is not None
    p75, pack, pa = load_p75(), ps.load_pack(), ps.load_pa()
    stamps = dict(issued_at_utc=pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), revision=revision,
                  model_version=ps.model_version(), code_version=ps.code_version(), protocol_version=ps.PROTOCOL_VERSION)
    rows, wins = [], []
    for site in sites:
        try:
            r, w = forecast_site(site, p75, D, a, pack, pa, stamps, subset)
        except Exception as e:                       # never lose the other sites
            log.exception(f"{site['site_id']}: unexpected failure")
            r = [station_row(site, prow, None, "parse_error", f"{type(e).__name__}: {str(e)[:160]}", D, a, pack, stamps, subset)
                 for prow in p75[p75.site_id == site["site_id"]].itertuples()]
            w = pd.DataFrame(columns=STATIONDAY_COLS)
        rows += r
        wins.append(w)
    out = pd.DataFrame(rows).reindex(columns=LEDGER_COLS)
    stationdays = pd.concat(wins, ignore_index=True) if wins else pd.DataFrame(columns=STATIONDAY_COLS)
    show = out[["site_group", "site_id", "station", "status", "last_obs_date", "days_old", "n_days_35", "chl_today",
                "chl_p75_site", "bloom_prob", "alert", "onset_row", "alert_site_t", "note"]].copy()
    show["chl_today"] = show.chl_today.round(2)
    show["bloom_prob"] = show.bloom_prob.round(3)
    log.info("forecast table:\n" + show.to_string(index=False))
    log.info("status counts: " + ", ".join(f"{k}={v}" for k, v in out.status.value_counts().items()))
    if a.dry_run:
        log.info("dry run: nothing written to ledger/issued/digest/outbox")
        return out
    write_ledger(out, p["ledger"])
    write_issued(out, stationdays, p["issued"], Ds)
    log.info(f"digest -> {write_digest(out, Ds, p)}")
    log.info(f"outbox -> {write_outbox(out, Ds, p['outbox'])}")
    log.info(f"appended {len(out)} rows (revision {revision}) to {p['ledger']}")
    return out


if __name__ == "__main__":
    main()
