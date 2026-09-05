"""
run_catalog.py -- pull every catalogued sub-daily chlorophyll sonde and score the model on it
-------------------------------------------------------------------------------------------
Milestone 2, step 2. Reads data/registry/insitu_catalog.csv (from erddap_crawl.py),
keeps datasets with years >= 1, cadence <= 60 min, not already_covered, and
takes the --top N by record length x stations. For each:
  1. pulls time, station, lat, lon, chl[, temp, sal, do] year by year from
     tabledap/<id>.csv into data/registry/raw/<server>/<id>_<year>.csv (cached)
  2. writes the predict_anywhere contract data/registry/sites/<id>.csv
     (station, datetime, chl, temp, sal, do); stations with < 60 station-days dropped
  3. runs the frozen model (predict_anywhere functions imported by path)
     -> data/registry/predictions/<id>.csv
  4. scores it with the section-19 protocol where >= 1 full year exists:
     label = own-station p75 within 7 d, onset rows (chl <= p75 today),
     threshold chosen on the first year, station-year clustered bootstrap
     -> appends to data/registry/site_skill.csv
Resumable; run from fork root, BASE env:
    python -m src.registry.run_catalog --top 40
"""
import argparse
import importlib.util
import io
import os
import time

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

from src.transfer.transfer_eval import add_label, boot_ci, metrics, pick_t

UA = {"User-Agent": "Mozilla/5.0 (hab-bloom-predictor research; student project)"}
CAT = "data/registry/insitu_catalog.csv"
SKILL = "data/registry/site_skill.csv"
RAW, SITES, PRED = "data/registry/raw", "data/registry/sites", "data/registry/predictions"

spec = importlib.util.spec_from_file_location("pa", "predict_anywhere.py")
pa = importlib.util.module_from_spec(spec); spec.loader.exec_module(pa)


def get(url, tries=3, timeout=600):
    wait = 10
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code in (200, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(wait); wait *= 3
    return None


def pull(row):
    base = row.url.rsplit("/tabledap/", 1)[0] + "/"
    cols = ["time"] + ([row.station_var] if isinstance(row.station_var, str) else []) + ["latitude", "longitude", row.chl_var]
    for c in (row.temp_var, row.sal_var, row.do_var):
        if isinstance(c, str) and c not in cols:
            cols.append(c)
    d = f"{RAW}/{row.server}"; os.makedirs(d, exist_ok=True)
    y0, y1 = int(str(row.start)[:4]), int(str(row.end)[:4])
    parts = []
    for y in range(y0, y1 + 1):
        out = f"{d}/{row.dataset_id}_{y}.csv"
        if os.path.exists(out):
            parts.append(pd.read_csv(out)); continue
        u = f"{base}tabledap/{row.dataset_id}.csv?{','.join(cols)}&time>={y}-01-01T00:00:00Z&time<={y}-12-31T23:59:59Z"
        r = get(u)
        if r is None:
            print(f"    {row.dataset_id} {y}: no response", flush=True); continue
        if r.status_code == 404:
            pd.DataFrame(columns=cols).to_csv(out, index=False); continue
        try:
            x = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        except Exception:
            continue
        x.to_csv(out, index=False); parts.append(x)
        print(f"    {row.dataset_id} {y}: {len(x):,} rows", flush=True)
        time.sleep(1)
    parts = [p for p in parts if len(p)]
    if not parts:
        return None
    x = pd.concat(parts, ignore_index=True)
    st = (x[row.station_var].astype(str) if isinstance(row.station_var, str) and row.station_var in x
          and x[row.station_var].notna().any() else pd.Series(row.dataset_id, index=x.index))
    site = pd.DataFrame({"station": st, "datetime": pd.to_datetime(x["time"], utc=True, errors="coerce").dt.tz_localize(None),
                         "chl": pd.to_numeric(x[row.chl_var], errors="coerce")})
    for name, c in (("temp", row.temp_var), ("sal", row.sal_var), ("do", row.do_var)):
        site[name] = pd.to_numeric(x[c], errors="coerce") if isinstance(c, str) and c in x else np.nan
    site = site.dropna(subset=["datetime", "chl"])
    site = site[(site.chl >= 0) & (site.chl < 1000)]
    return site


def score(site_id, server, day, min_year_rows=100):
    day = day.copy().sort_values(["station", "date"]).reset_index(drop=True)   # add_label needs a RangeIndex
    day["year"] = day.date.dt.year
    day["thr"] = day.groupby("station")["chl"].transform(lambda s: s.quantile(0.75))
    lab = add_label(day, "thr").dropna(subset=["bloom_fwd"])
    on = lab[lab.chl <= lab.thr]
    years = sorted(on.year.unique())
    if len(years) < 2 or len(on) < 200:
        return None
    cal = on[on.year == years[0]]; test = on[on.year > years[0]]
    if len(cal) < min_year_rows or cal.bloom_fwd.nunique() < 2 or test.bloom_fwd.nunique() < 2:
        return None
    t = pick_t(cal.bloom_fwd.values, cal.bloom_prob.values)
    r = metrics(test.bloom_fwd, test.bloom_prob >= t)
    r.update(server=server, dataset_id=site_id, n_stations=test.station.nunique(),
             years=round((test.date.max() - test.date.min()).days / 365.25, 1), n_onset=len(test),
             auc=roc_auc_score(test.bloom_fwd, test.bloom_prob), t_star=t)
    r.update(boot_ci(test, "bloom_prob", "bloom_fwd", t))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--rescore", action="store_true", help="rebuild site_skill.csv from saved predictions and exit")
    a = ap.parse_args()
    if a.rescore:
        rescore_all(); return
    for d in (SITES, PRED):
        os.makedirs(d, exist_ok=True)
    cat = pd.read_csv(CAT)
    junk = cat.title.str.contains(r"wod\d*|ctd|dms|moving vessel|tsg|discrete|water sample|cast|profil|cruise|underway|shipboard|glider|drifter",
                                  case=False, na=False, regex=True)
    covered = cat.already_covered.astype(bool) | cat.dataset_id.str.contains(r"^URI_|NERRS", case=False, regex=True)
    cat = cat[(cat.years >= 1) & (cat.years <= 40) & (cat.cadence_min <= 60) & ~covered & ~junk].copy()
    cat = cat.drop_duplicates(subset=["title"])            # UAF mirrors PacIOOS etc.
    cat["rank"] = cat.years * cat.n_stations.clip(lower=1)
    cat = cat.sort_values("rank", ascending=False).head(a.top)
    if a.datasets != "all":
        cat = cat[cat.dataset_id.isin(a.datasets.split(","))]
    done = set(pd.read_csv(SKILL).dataset_id) if os.path.exists(SKILL) else set()
    print(f"{len(cat)} datasets to run; {len(done)} already scored", flush=True)
    pack = joblib.load(pa.MODEL_PATH)
    for _, row in cat.iterrows():
        if row.dataset_id in done:
            continue
        print(f"== {row.server} / {row.dataset_id} ({row.n_stations} st, {row.years} y, {row.cadence_min} min)", flush=True)
        try:
            site = pull(row)
        except Exception as e:
            print(f"   pull failed: {str(e)[:100]}", flush=True); continue
        if site is None or len(site) < 1000:
            print("   too little data", flush=True); continue
        site.to_csv(f"{SITES}/{row.dataset_id}.csv", index=False)
        min_readings = 48 if row.cadence_min <= 20 else 12
        day = pa.build_daily(site, min_readings)
        day = day[day.groupby("station")["chl"].transform("size") >= 60]
        if len(day) < 200:
            print("   fewer than 200 station-days after filtering", flush=True); continue
        scored = pa.rescale_chl(day, pack["chl_quantiles"])
        X = scored[pack["features"]].fillna(pd.Series(pack["medians"])).fillna(0.0).values
        day["bloom_prob"] = pack["model"].predict_proba(X)[:, 1]
        day["alert"] = day.bloom_prob >= pack["threshold"]
        day[["station", "date", "chl", "temp", "sal", "do", "bloom_prob", "alert"]].to_csv(f"{PRED}/{row.dataset_id}.csv", index=False)
        r = score(row.dataset_id, row.server, day)
        if r is None:
            print("   predictions written; not enough history to score skill", flush=True)
            r = dict(server=row.server, dataset_id=row.dataset_id, n_stations=day.station.nunique(),
                     years=round((day.date.max() - day.date.min()).days / 365.25, 1), n_onset=np.nan)
        else:
            print(f"   lift={r['lift']:.2f} [{r['lift_lo']:.2f},{r['lift_hi']:.2f}] prec={r['precision']:.3f} "
                  f"base={r['base_rate']:.3f} auc={r['auc']:.3f} n={r['n_onset']}", flush=True)
        pd.DataFrame([r]).reindex(columns=COLS).to_csv(SKILL, mode="a", header=not os.path.exists(SKILL), index=False)
    print("catalog run complete", flush=True)


COLS = ["server", "dataset_id", "n_stations", "years", "n_onset", "n_test", "tp", "fp", "fn", "base_rate",
        "precision", "pod", "lift", "lift_lo", "lift_hi", "auc", "auc_lo", "auc_hi", "precision_lo", "precision_hi", "t_star"]


def rescore_all():
    """Rebuild site_skill.csv from the saved prediction files (dedupes, fixes column order)."""
    cat = pd.read_csv(CAT).drop_duplicates(subset=["dataset_id"]).set_index("dataset_id")
    rows = []
    for f in sorted(os.listdir(PRED)):
        did = f[:-4]
        day = pd.read_csv(f"{PRED}/{f}", parse_dates=["date"])
        server = cat.server.get(did, "?")
        r = score(did, server, day)
        if r is None:
            r = dict(server=server, dataset_id=did, n_stations=day.station.nunique(),
                     years=round((day.date.max() - day.date.min()).days / 365.25, 1))
        rows.append(r)
    pd.DataFrame(rows).reindex(columns=COLS).to_csv(SKILL, index=False)
    print(f"rescored {len(rows)} sites -> {SKILL}")


if __name__ == "__main__":
    main()
