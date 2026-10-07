#!/usr/bin/env python
"""
run_site.py -- pick a site by name and score it, without remembering any flags
==============================================================================
A front end for predict_anywhere.py. It does three things the underlying tool
deliberately does not: finds a site from a fragment of its name, works out the
right --min-readings by looking at the file, and prints what that site already
scored in the registry so a live result can be read against it.

The model, the rescaling and the numbers are untouched. This resolves a name to
a path and a couple of flags, then hands over to predict_anywhere.main(), which
is the file verified to reproduce the findings harness.

USAGE
    python run_site.py                      list every site on this machine
    python run_site.py --list california    list the ones matching "california"
    python run_site.py --top                the ten highest-lift sites
    python run_site.py santa cruz           score the site whose name matches
    python run_site.py scwharf --date 2019-08-15
    python run_site.py --file my_readings.csv

Anything it does not recognise is passed straight through to predict_anywhere,
so --threshold, --date, --out and --no-rescale all still work.

Vihaan Goyal, Westhill High School.
"""
import argparse
import glob
import importlib.util
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = os.path.join(HERE, "data", "registry", "sites")
SKILL = os.path.join(HERE, "data", "registry", "site_skill.csv")
CATALOG = os.path.join(HERE, "data", "registry", "insitu_catalog.csv")
DAILY_CUTOFF = 3.0          # median readings per station-day below this = daily data


def available():
    """Every site CSV on this machine, with its catalogue title and registry skill."""
    rows = [dict(dataset_id=os.path.basename(p)[:-4], path=p, mb=os.path.getsize(p) / 1e6)
            for p in sorted(glob.glob(os.path.join(SITES, "*.csv")))]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if os.path.exists(CATALOG):
        cat = pd.read_csv(CATALOG)[["dataset_id", "title", "server", "years"]]
        df = df.merge(cat.drop_duplicates("dataset_id"), on="dataset_id", how="left")
    if os.path.exists(SKILL):
        sk = pd.read_csv(SKILL)[["dataset_id", "lift", "auc", "base_rate"]]
        df = df.merge(sk.drop_duplicates("dataset_id"), on="dataset_id", how="left")
    for c in ("title", "server"):
        if c not in df:
            df[c] = ""
    for c in ("lift", "auc", "base_rate", "years"):
        if c not in df:
            df[c] = float("nan")
    return df.fillna({"title": "", "server": ""})


def search(df, terms):
    """Match the query against the dataset id, the catalogue title or the server."""
    if not terms:
        return df
    q = " ".join(terms).lower().strip()
    hay = (df.dataset_id.str.lower() + " " + df.title.str.lower().fillna("")
           + " " + df.server.str.lower().fillna(""))
    hit = df[hay.str.contains(q, regex=False)]
    if len(hit):
        return hit
    for word in q.split():                       # fall back to any single word
        hit = df[hay.str.contains(word, regex=False)]
        if len(hit):
            return hit
    return df.iloc[0:0]


def show(df, header, hidden=0):
    if df.empty:
        print("no sites matched")
        return
    print(header)
    print(f"  {'SITE NAME (type part of this)':<44}{'DATA SOURCE':<14}"
          f"{'YEARS':>7}{'FILE MB':>9}{'LIFT':>7}{'AUC':>6}")
    print("  " + "-" * 86)
    for _, r in df.iterrows():
        lift = f"{r.lift:.2f}" if pd.notna(r.lift) else "-"
        auc = f"{r.auc:.2f}" if pd.notna(r.auc) else "-"
        yrs = f"{r.years:.1f}" if pd.notna(r.years) else "-"
        print(f"  {r.dataset_id[:43]:<44}{str(r.server)[:13]:<14}"
              f"{yrs:>7}{r.mb:>9.0f}{lift:>7}{auc:>6}")
    print(f"\n{len(df)} site(s). Score one with:  python run_site.py <part of its name>")
    print("  YEARS   length of the record at that site")
    print("  FILE MB size of the file on disk; anything over about 100 is slow to score")
    print("  LIFT    how many times better than alerting every day. 1.00 means no better")
    print("  AUC     ranking skill from 0.5 (chance) to 1.0 (perfect)")
    print("  A dash means the site has never been scored, so there is no number to compare to.")
    if hidden:
        print(f"{hidden} more on disk have no registry score (duplicate feeds of a site already "
              f"listed, or too short to score). Show them with --all; they still run.")


def readings_per_day(path, nrows=40000):
    """Read the head of the file to decide whether it is daily or sub-daily."""
    try:
        d = pd.read_csv(path, nrows=nrows, usecols=lambda c: c in ("station", "datetime"))
        d["day"] = pd.to_datetime(d.datetime, errors="coerce").dt.normalize()
        per = d.dropna(subset=["day"]).groupby(["station", "day"]).size()
        return float(per.median()) if len(per) else 1.0
    except Exception:
        return 1.0


def main():
    ap = argparse.ArgumentParser(description="score a site by name")
    ap.add_argument("name", nargs="*", help="part of a site name, e.g. 'santa cruz'")
    ap.add_argument("--list", nargs="*", default=None, metavar="TERM",
                    help="list sites, optionally filtered")
    ap.add_argument("--top", action="store_true", help="list the ten highest-lift sites")
    ap.add_argument("--file", default=None, help="score your own CSV instead of a catalogued site")
    ap.add_argument("--min-readings", type=int, default=None,
                    help="override the automatic daily/sub-daily choice")
    ap.add_argument("--all", action="store_true",
                    help="include sites with no registry score (duplicates and short records)")
    a, passthrough = ap.parse_known_args()

    df = available()

    def listing(sub, header):
        """Scored sites, best first; unscored ones are hidden unless --all."""
        if a.all:
            show(sub.sort_values("lift", ascending=False, na_position="last"), header)
            return
        keep = sub.dropna(subset=["lift"]).sort_values("lift", ascending=False)
        if keep.empty:                       # nothing scored matched, so show what did
            show(sub, header)
            return
        show(keep, header, hidden=len(sub) - len(keep))

    if a.top:
        show(df.dropna(subset=["lift"]).nlargest(10, "lift"), "Highest lift in the registry:")
        return
    if a.list is not None:
        listing(search(df, a.list), "Sites on this machine, best score first:")
        return

    row = None
    if a.file:
        path = a.file
        if not os.path.exists(path):
            sys.exit(f"no such file: {path}")
    else:
        if not a.name:
            listing(df, "Sites on this machine, best score first:")
            return
        hit = search(df, a.name)
        if hit.empty:
            sys.exit(f"nothing matched {' '.join(a.name)!r}. "
                     f"Run 'python run_site.py --list' to see them all.")
        if len(hit) > 1:
            q = " ".join(a.name).lower().strip()
            exact = hit[hit.dataset_id.str.lower() == q]
            scored = hit[hit.lift.notna()]
            if len(exact) == 1:
                hit = exact
            elif len(scored) == 1:
                hit = scored
                print(f"[{len(hit)} of the matches has a registry score; using "
                      f"{hit.iloc[0].dataset_id}]")
            else:
                show(hit, f"{' '.join(a.name)!r} matched more than one site:")
                return
        row = hit.iloc[0]
        path = row.path

    mr = a.min_readings
    if mr is None:
        rpd = readings_per_day(path)
        mr = 1 if rpd < DAILY_CUTOFF else 12
        kind = "daily" if mr == 1 else "sub-daily"
        print(f"File: {os.path.basename(path)} holds about {rpd:.0f} reading(s) per station "
              f"per day ({kind}), so a day counts once it has {mr} reading(s).")
    if row is not None and pd.notna(row.get("lift")):
        print(f"Known result here: lift {row.lift:.2f} (that many times better than alerting "
              f"every day), AUC {row.auc:.2f}, and {row.base_rate * 100:.0f}% of days at this "
              f"site precede a bloom.")
    print()

    spec = importlib.util.spec_from_file_location("pa", os.path.join(HERE, "predict_anywhere.py"))
    pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa)
    sys.argv = ["predict_anywhere.py", path, "--min-readings", str(mr)] + passthrough
    pa.main()


if __name__ == "__main__":
    main()
