"""
score_ledger.py -- verify past prospective forecasts against what the water did
next, and summarise skill with the pre-registered protocol.
-------------------------------------------------------------------------------
Purpose : for every ledger row whose 7-day horizon has closed (plus a latency
          allowance), pull the readings that followed, rebuild station-days and
          record whether daily-mean chl exceeded the frozen site p75. Then compute
          precision / POD / lift / AUC with station-week clustered bootstrap CIs.
Inputs  : <ledger-dir>/ledger.csv, data/prospective/site_p75.csv (via the ledger
          row's chl_p75_site), live feeds (src/deploy/live_feeds.py).
Outputs : <ledger-dir>/scored.csv        (ledger columns + scored_at_utc,
          n_fwd_days, max_fwd_chl, outcome, outcome_status; atomic rewrite;
          existing non-NaN outcomes are never changed)
          <ledger-dir>/skill_summary.csv (one row per scope x site_group, all
          strata; the reporting gate is applied when printing / in the digest)
Run     : python -m src.deploy.score_ledger [--as-of YYYY-MM-DD] [--latency-days 2]
              [--rescore-only] [--min-n 30] [--min-pos 5] [--ledger-dir data/prospective]
          from the fork root with C:/Users/vihaa/anaconda3/python.exe
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.deploy import live_feeds as lf
from src.deploy import prospective_sites as ps
from src.transfer.transfer_eval import boot_ci, metrics

KEY = ["issue_date", "site_id", "station"]
EXTRA = ["scored_at_utc", "n_fwd_days", "max_fwd_chl", "outcome", "outcome_status"]
SUMMARY_COLS = ["as_of", "scope", "site_group", "n", "n_pos", "base_rate", "precision", "precision_lo", "precision_hi",
                "pod", "pod_lo", "pod_hi", "lift", "lift_lo", "lift_hi", "auc", "n_pending", "n_unverifiable"]
MIN_FWD_DAYS_FOR_ZERO = 4     # a "no bloom" verdict needs >= 4 observed station-days in the 7-day window
PENDING_DAYS = 21             # after last_obs + 21 d an unfilled outcome becomes unverifiable
N_BOOT, SEED = 2000, 42
SCORABLE_STATUS = ("ok", "stale")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="verify prospective forecasts and summarise skill")
    ap.add_argument("--as-of", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--latency-days", type=int, default=2)
    ap.add_argument("--rescore-only", action="store_true", help="recompute the summary from scored.csv; no fetching")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--ledger-dir", default=ps.PROSPECTIVE_DIR)
    return ap.parse_args(argv)


def load_ledger(path):
    """Highest revision per (issue_date, site_id, station)."""
    led = pd.read_csv(path, dtype={"issue_date": str, "station": str, "last_obs_date": str, "note": str})
    led = led.sort_values("revision").drop_duplicates(KEY, keep="last")
    return led.sort_values(["issue_date", "site_group", "site_id", "station"]).reset_index(drop=True)


def merge_existing(led, scored_path):
    """Attach previously scored columns by key so filled outcomes are preserved."""
    for c in EXTRA:
        led[c] = np.nan
    led["outcome_status"] = led["outcome_status"].astype(object)
    led["scored_at_utc"] = led["scored_at_utc"].astype(object)
    if not os.path.exists(scored_path):
        return led
    old = pd.read_csv(scored_path, dtype={"issue_date": str, "station": str, "outcome_status": str, "scored_at_utc": str})
    old = old.set_index(KEY)[EXTRA]
    idx = pd.MultiIndex.from_frame(led[KEY])
    for c in EXTRA:
        led[c] = old[c].reindex(idx).values
    return led


def due_mask(df, as_of, latency):
    last = pd.to_datetime(df.last_obs_date, errors="coerce")
    closes = last + pd.Timedelta(days=ps.HORIZON + latency)
    return (df.status != "feed_down") & last.notna() & (closes <= as_of) & ~df.outcome.isin([0.0, 1.0])


def pull_for_site(site, rows, as_of):
    """One live pull covering the union of the rows' forward windows; returns full history."""
    last = pd.to_datetime(rows.last_obs_date)
    lo = (last.min() + pd.Timedelta(days=1)).normalize()
    hi = min(last.max() + pd.Timedelta(days=ps.HORIZON), as_of) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    frame, status, note = lf.fetch_site(site, lo, hi, f"score_{as_of.strftime('%Y-%m-%d')}")
    print(f"  {site['site_id']}: pull {lo.date()}..{hi.date()} -> {status} {note} ({len(frame)} rows)", flush=True)
    return lf.append_history(site, frame) if status == "ok" else lf.load_history(site)


def outcome_for(sd, last_obs, p75, as_of):
    """(n_fwd_days, max_fwd_chl, outcome, outcome_status) for one forecast row."""
    fwd = sd[(sd.date > last_obs) & (sd.date <= last_obs + pd.Timedelta(days=ps.HORIZON))]
    n, mx = len(fwd), (float(fwd.chl.max()) if len(fwd) else np.nan)
    if n and mx > p75:
        return n, mx, 1.0, "scored"
    if n >= MIN_FWD_DAYS_FOR_ZERO:
        return n, mx, 0.0, "scored"
    return n, mx, np.nan, ("pending" if as_of <= last_obs + pd.Timedelta(days=PENDING_DAYS) else "unverifiable")


def _same(a, b):
    return (pd.isna(a) and pd.isna(b)) or a == b


def score_due_rows(df, as_of, latency, pa):
    due = due_mask(df, as_of, latency)
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    sites = ps.sites_by_id()
    for sid, rows in df[due].groupby("site_id"):
        site = sites.get(sid)
        if site is None:
            print(f"  {sid}: not in the pre-registered site list; skipped", flush=True)
            continue
        day = ps.build_site_daily(site, pull_for_site(site, rows, as_of), pa)
        for i, r in rows.iterrows():
            sd = day[day.station == str(r.station)] if day is not None else pd.DataFrame(columns=["date", "chl"])
            vals = outcome_for(sd, pd.Timestamp(r.last_obs_date), float(r.chl_p75_site), as_of)
            changed = not all(_same(v, df.at[i, c]) for v, c in zip(vals, EXTRA[1:]))
            df.loc[i, EXTRA[1:]] = vals
            if changed:
                df.at[i, "scored_at_utc"] = now
    undecided = df.outcome_status.isna()
    df.loc[undecided & (df.status == "feed_down"), "outcome_status"] = "feed_down"
    df.loc[undecided & (df.status != "feed_down"), "outcome_status"] = "pending"
    return df, int(due.sum())


def write_atomic(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def boot_pod_ci(d, pcol, t, n_boot=N_BOOT, seed=SEED):
    """Station-week clustered bootstrap of POD (same cluster loop as transfer_eval.boot_ci)."""
    rng = np.random.default_rng(seed)
    cl = (d.station.astype(str) + "_" + d.year.astype(str)).values
    groups = {c: np.where(cl == c)[0] for c in np.unique(cl)}
    keys = list(groups)
    p, y = d[pcol].values, d["outcome"].values.astype(int)
    pods = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        pods.append(metrics(y[idx], p[idx] >= t)["pod"])
    return dict(pod_lo=np.nanpercentile(pods, 2.5), pod_hi=np.nanpercentile(pods, 97.5))


def stratum(d, pcol, t, auc_col="bloom_prob"):
    """Metrics + clustered CIs for one (scope, group) slice with outcomes in {0, 1}."""
    d = d.assign(station=d.site_id.astype(str) + "|" + d.station.astype(str), year=d.issue_date.astype(str))
    r = metrics(d.outcome, d[pcol] >= t)
    r["n"] = r.pop("n_test")
    r["n_pos"] = int(d.outcome.sum())
    r["auc"] = roc_auc_score(d.outcome.astype(int), d[auc_col]) if d.outcome.nunique() == 2 else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN resamples in tiny strata
        r.update(boot_ci(d, pcol, "outcome", t, n_boot=N_BOOT, seed=SEED))
        r.update(boot_pod_ci(d, pcol, t))
    return r


def summarise(df, as_of, min_n, min_pos):
    """All strata (scope x group + pooled, plus erddap_top at its own t*) -> summary frame."""
    base = df[df.status.isin(SCORABLE_STATUS)].copy()
    base["alert_t"] = (base.bloom_prob >= base.threshold).astype(float)
    base["alert_site_t_f"] = ((base.bloom_prob >= base.t_star_site).astype(float)
                              .where(base.t_star_site.notna()))   # fresh sites (t* NaN) drop out of @t_site
    onset = base.onset_row.astype(str).str.lower().isin(["true", "1", "1.0"])
    rows = []
    for scope, sm in (("onset", onset), ("all", pd.Series(True, index=base.index))):
        groups = [(g, base.site_group == g) for g in ps.SITE_GROUPS] + [("pooled", pd.Series(True, index=base.index))]
        groups.append(("erddap_top@t_site", base.site_group == "erddap_top"))
        for g, gm in groups:
            slab = base[sm & gm]
            done = slab[slab.outcome.isin([0.0, 1.0])]
            row = dict(as_of=as_of.strftime("%Y-%m-%d"), scope=scope, site_group=g, n=len(done), n_pos=int(done.outcome.sum()),
                       n_pending=int((slab.outcome_status == "pending").sum()),
                       n_unverifiable=int((slab.outcome_status == "unverifiable").sum()))
            if len(done) and done.outcome.nunique() >= 1:
                pcol, t = ("alert_site_t_f", 0.5) if g.endswith("@t_site") else ("bloom_prob", float(done.threshold.iloc[0]))
                row.update(stratum(done, pcol, t))
            rows.append(row)
    out = pd.DataFrame(rows).reindex(columns=SUMMARY_COLS)
    for _, r in out.iterrows():
        tag = f"[{r.scope:5s} {r.site_group:18s}]"
        if r.n >= min_n and r.n_pos >= min_pos:
            print(f"{tag} n={r.n} pos={r.n_pos} base={r.base_rate:.3f} prec={r.precision:.3f} [{r.precision_lo:.3f},{r.precision_hi:.3f}] "
                  f"pod={r.pod:.3f} [{r.pod_lo:.3f},{r.pod_hi:.3f}] lift={r.lift:.2f} [{r.lift_lo:.2f},{r.lift_hi:.2f}] "
                  f"auc={r.auc:.3f} pending={r.n_pending} unverifiable={r.n_unverifiable}")
        else:
            print(f"{tag} n={r.n} (pos={r.n_pos}) below pre-registered minimum (n>={min_n}, pos>={min_pos}); "
                  f"pending={r.n_pending} unverifiable={r.n_unverifiable}")
    return out


def main(argv=None):
    a = parse_args(argv)
    as_of = pd.Timestamp(a.as_of).normalize()
    ledger, scored_p, summ_p = (os.path.join(a.ledger_dir, f) for f in ("ledger.csv", "scored.csv", "skill_summary.csv"))
    if a.rescore_only:
        if not os.path.exists(scored_p):
            sys.exit(f"{scored_p} missing; nothing to rescore")
        df = pd.read_csv(scored_p, dtype={"issue_date": str, "station": str})
        print(f"rescore-only: {len(df)} rows from {scored_p}")
    else:
        if not os.path.exists(ledger):
            print(f"{ledger} does not exist; no forecasts issued yet, nothing to score")
            return None
        df = merge_existing(load_ledger(ledger), scored_p)
        df, n_due = score_due_rows(df, as_of, a.latency_days, ps.load_pa())
        write_atomic(df, scored_p)
        print(f"scored {n_due} due rows; outcome_status counts: {df.outcome_status.value_counts().to_dict()} -> {scored_p}")
    summ = summarise(df, as_of, a.min_n, a.min_pos)
    write_atomic(summ, summ_p)
    print(f"summary -> {summ_p}")
    return df


if __name__ == "__main__":
    main()
