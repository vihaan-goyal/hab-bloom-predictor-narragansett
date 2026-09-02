"""W9: out-of-sample bloom probability 21 days before each Narragansett onset.

Two tier-A HistGradientBoosting models trained on year <= 2020:
  GB7  : 7-day forward label (bloom_fwd, as shipped in the feature table)
  GB21 : 21-day forward label (built here with the same right-censored loop)
Both score every row in 2021-2023 (out of sample).

Onset = first station-day with chl > 10 whose previous 5 calendar days are all
present in the table and all chl <= 10. For each onset, the row at exactly
onset-21 d (same station) is used if present, else the nearest row within
+/-3 d (offset recorded). p at -14, -7, -3, -1 use the same rule.

Matched null = station-days in 2021-2023 with chl <= 10 and no chl > 10 in the
following 21 d (bloom21 == 0). Reported both as the full null distribution and
as a station+month matched resample (one null day per onset, 500 draws).

Output: data/prob_before_onset_nar.csv (+ printed summary)
Run from repo root with the BASE conda python (not the hab env).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BLOOM = 10.0
TRAIN_MAX = 2020
SCORE_YEARS = (2021, 2022, 2023)
QUIET_DAYS = 5
TOL = 3
CONTEXT = (-21, -14, -7, -3, -1)
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']

df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
df = df.sort_values(["station", "date"]).reset_index(drop=True)
df["year"] = df.date.dt.year


def forward_label(day, horizon):
    """Same loop as build_narragansett_daily.py, parameterised by horizon."""
    out = np.full(len(day), np.nan)
    for st, grp in day.groupby("station"):
        dates = grp["date"].values; chl = grp["chl"].values; last = dates.max()
        lab = np.full(len(grp), np.nan)
        for i in range(len(grp)):
            end = dates[i] + np.timedelta64(horizon, "D")
            m = (dates > dates[i]) & (dates <= end)
            if m.any() and (chl[m] > BLOOM).any():
                lab[i] = 1
            elif end <= last:
                lab[i] = 0
        out[grp.index.values] = lab
    return out


df["bloom7"] = df["bloom_fwd"]
df["bloom21"] = forward_label(df, 21)
chk = forward_label(df, 7)
assert np.allclose(np.nan_to_num(chk, nan=-1), np.nan_to_num(df.bloom7.values, nan=-1)), \
    "rebuilt 7-day label differs from shipped bloom_fwd"


def fit_gb(label):
    tr = df[(df.year <= TRAIN_MAX) & df[label].notna()]
    med = tr[TIER_A].median()
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                       min_samples_leaf=50, l2_regularization=1.0,
                                       class_weight="balanced", random_state=42)
    m.fit(tr[TIER_A].fillna(med).values, tr[label].astype(int).values)
    print(f"{label}: train n={len(tr)} pos={tr[label].mean():.3f}")
    return m, med


gb7, med7 = fit_gb("bloom7")
gb21, med21 = fit_gb("bloom21")
sc = df[df.year.isin(SCORE_YEARS)].copy()
sc["p7"] = gb7.predict_proba(sc[TIER_A].fillna(med7).values)[:, 1]
sc["p21"] = gb21.predict_proba(sc[TIER_A].fillna(med21).values)[:, 1]
lab21 = sc.bloom21.notna()
print(f"scored rows 2021-2023: {len(sc)}  AUC p21 vs bloom21 (all rows) = "
      f"{roc_auc_score(sc.bloom21[lab21].astype(int), sc.p21[lab21]):.3f}")

# ---- onsets: chl>10 after >=5 consecutive quiet calendar days ----------------
onsets = []
for st, g in df.groupby("station"):
    g = g.reset_index(drop=True)
    for i in range(QUIET_DAYS, len(g)):
        if g.chl[i] <= BLOOM or g.year[i] not in SCORE_YEARS:
            continue
        prev = g.iloc[i - QUIET_DAYS:i]
        consecutive = (g.date[i] - prev.date.iloc[0]).days == QUIET_DAYS and \
                      prev.date.diff().dropna().dt.days.eq(1).all()
        if consecutive and (prev.chl <= BLOOM).all():
            onsets.append((st, g.date[i]))
print(f"onsets 2021-2023: {len(onsets)}")

lookup = {st: g.set_index("date") for st, g in sc.groupby("station")}


def row_at(st, target):
    """Exact day, else nearest within +/-TOL days (ties -> earlier). Returns (row, offset)."""
    g = lookup[st]
    if target in g.index:
        return g.loc[target], 0
    cands = g.loc[target - pd.Timedelta(days=TOL): target + pd.Timedelta(days=TOL)]
    if cands.empty:
        return None, np.nan
    off = (cands.index - target).days.values
    k = np.lexsort((off, np.abs(off)))[0]
    return cands.iloc[k], int(off[k])


recs = []
for st, od in onsets:
    r, off = row_at(st, od - pd.Timedelta(days=21))
    rec = dict(station=st, onset_date=od.date(), day_used_offset=off,
               **{"chl_at_-21": np.nan, "p7_at_-21": np.nan, "p21_at_-21": np.nan})
    if r is not None:
        rec.update({"chl_at_-21": r.chl, "p7_at_-21": r.p7, "p21_at_-21": r.p21})
    for d in CONTEXT[1:]:
        rr, _ = row_at(st, od + pd.Timedelta(days=d))
        rec[f"p21_at_{d}"] = np.nan if rr is None else rr.p21
    recs.append(rec)
res = pd.DataFrame(recs)
res.to_csv("data/prob_before_onset_nar.csv", index=False)
print(f"wrote data/prob_before_onset_nar.csv ({len(res)} rows)")

# ---- summaries ----------------------------------------------------------------
def summ(p, thr=0.5):
    p = pd.Series(p).dropna()
    q = p.quantile([.25, .5, .75])
    return dict(n=len(p), median=q[.5], q25=q[.25], q75=q[.75],
                frac_ge=(p >= thr).mean())


null = sc[(sc.chl <= BLOOM) & (sc.bloom21 == 0)].copy()
ons = res.dropna(subset=["p21_at_-21"])
S_on, S_null = summ(ons["p21_at_-21"]), summ(null.p21)
S_on7, S_null7 = summ(ons["p7_at_-21"]), summ(null.p7)

# station+month matched resample: one null day per onset, same station & month
rng = np.random.default_rng(42)
null["month"] = null.date.dt.month
key = {k: g.p21.values for k, g in null.groupby(["station", "month"])}
ons_key = [(s, pd.Timestamp(d).month) for s, d in zip(ons.station, ons.onset_date)]
draws = []
for _ in range(500):
    samp = [rng.choice(key[k]) for k in ons_key if k in key]
    s = summ(samp); draws.append([s["median"], s["q25"], s["q75"], s["frac_ge"]])
mm = np.median(np.array(draws), axis=0)
n_matched = sum(k in key for k in ons_key)

auc = roc_auc_score(np.r_[np.ones(len(ons)), np.zeros(len(null))],
                    np.r_[ons["p21_at_-21"].values, null.p21.values])

# null trajectory analogue for the approach-curve inset (used by the figure script):
# for null days the "-21..-1" positions are the same day + 0, 7, 14, 18, 20 d,
# conditional on chl staying <= 10 through that path. Saved alongside.
null_traj = {}
for d in CONTEXT:
    shift = 21 + d  # 0, 7, 14, 18, 20
    vals = []
    for st, g in null.groupby("station"):
        gi = lookup[st]
        tgt = g.date + pd.Timedelta(days=shift)
        hit = gi.reindex(tgt)
        ok = hit.chl.notna() & (hit.chl <= BLOOM)
        vals.append(hit.p21[ok.values].values)
    null_traj[d] = np.concatenate(vals)
pd.DataFrame({"offset": list(CONTEXT),
              "onset_median_p21": [res[f"p21_at_{d}"].median() for d in CONTEXT],
              "null_median_p21": [np.median(null_traj[d]) for d in CONTEXT],
              "null_n": [len(null_traj[d]) for d in CONTEXT]}
             ).to_csv("data/prob_before_onset_nar_trajectory.csv", index=False)
null[["station", "date", "chl", "p7", "p21"]].to_csv(
    "data/prob_before_onset_nar_null.csv", index=False)

print("\n=== Narragansett: p at onset-21 d (GB tier-A, out of sample 2021-2023) ===")
print(f"onsets with a -21 row       : {S_on['n']} / {len(res)}  "
      f"(exact -21 day: {(res.day_used_offset == 0).sum()})")
print(f"p21 at -21  onsets          : median {S_on['median']:.3f}  IQR [{S_on['q25']:.3f}, {S_on['q75']:.3f}]  frac>=0.5 {S_on['frac_ge']:.3f}")
print(f"p21         null (all)      : n={S_null['n']}  median {S_null['median']:.3f}  IQR [{S_null['q25']:.3f}, {S_null['q75']:.3f}]  frac>=0.5 {S_null['frac_ge']:.3f}")
print(f"p21         null (st+month) : n={n_matched}/draw  median {mm[0]:.3f}  IQR [{mm[1]:.3f}, {mm[2]:.3f}]  frac>=0.5 {mm[3]:.3f}  (median over 500 draws)")
print(f"p7  at -21  onsets          : median {S_on7['median']:.3f}  IQR [{S_on7['q25']:.3f}, {S_on7['q75']:.3f}]  frac>=0.5 {S_on7['frac_ge']:.3f}")
print(f"p7          null (all)      : median {S_null7['median']:.3f}  IQR [{S_null7['q25']:.3f}, {S_null7['q75']:.3f}]  frac>=0.5 {S_null7['frac_ge']:.3f}")
print(f"AUC onset-21 vs null (p21)  : {auc:.3f}")
print(f"chl at -21 onsets           : median {ons['chl_at_-21'].median():.2f}  "
      f"frac chl>10 at -21: {(ons['chl_at_-21'] > BLOOM).mean():.3f}")
print("\np21 trajectory into onset (median across onsets vs null path):")
print(pd.read_csv("data/prob_before_onset_nar_trajectory.csv").round(3).to_string(index=False))
print("\nby station:")
print(res.groupby("station")["p21_at_-21"].agg(["count", "median"]).round(3).to_string())
