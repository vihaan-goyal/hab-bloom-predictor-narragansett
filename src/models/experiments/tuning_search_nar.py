"""W7 pre-registered tuning search -- Narragansett onset-alert configuration.

Goal: find the best presentable operating configuration WITHOUT selection
inflation. Every choice below was fixed before any number was looked at.

PRE-REGISTERED PROTOCOL (verbatim)
----------------------------------
Input: data/narragansett_daily_features.csv (station str, date, chl daily
mean, tier features; TIER_A / TIER_B lists copied from
src/models/train_narragansett.py; forward-label routine copied from
src/features/build_narragansett_daily.py, parameterised by horizon and
bloom threshold). Split: train year<=2020, val 2021-2022, TEST 2023 (touch
test exactly once, at the end).

GRID (all combinations):
- horizon h in {2, 3, 5, 7, 10}
- bloom threshold T in {10, 12.8, 20} ug/L (12.8 = lab-calibrated
  equivalent of 10)
- feature tier in {A, B}
- model: LR with C in {0.01, 0.05, 0.2} (balanced, StandardScaler,
  train-median impute); HistGB with max_depth in {2,3,4} x
  min_samples_leaf in {20,50,100}, lr 0.05, max_iter 300, l2 1.0, balanced,
  random_state 42
=> 5 x 3 x 2 x 12 = 360 configs. Cache labels per (h, T).

SCORING on VAL, onset-only rows (today's chl <= T): for each config sweep
alert threshold t in 0.05..0.95 step 0.05; the config's val score = max
over t of lift subject to POD >= 0.6 (if no t reaches POD 0.6, score =
-inf). Record val precision/POD/lift/AUC/base_rate/t* per config.

SELECTION RULE (fixed): maximize val onset lift subject to POD >= 0.6;
ties -> higher t*, then simpler model (LR over GB, fewer leaves). Also
record the best config per horizon and per threshold so we can show the
landscape.

TEST: evaluate ONLY the single selected config on 2023 onset rows at its
t*: precision, POD, base_rate, lift, AUC; station-clustered bootstrap (13
stations, n=2000, seed 42) CIs; and a paired bootstrap of lift difference
vs the current reference config (h7, T10, tier A, GB depth3/leaf50,
t*=0.50 -> known 0.696/0.600/2.00).

PERMUTATION NULL: 30 shuffles (seed 42). For each: permute val onset
labels WITHIN station (preserve station base rates), rerun the FULL
selection over the grid using the cached model predictions (do not refit
models; models are fit on train, which is untouched), record the best val
lift. Report the real best val lift's percentile in that null. If real <=
95th percentile, the search found nothing beyond chance and that is the
reported outcome.

OUTPUTS: data/tuning_search_nar_grid.csv (all 360 val rows),
data/tuning_search_nar_selected.csv (selected config val + test + CIs +
paired diff), data/tuning_search_nar_null.csv (30 null best-lifts + real
percentile).

Implementation notes (not part of the protocol, but fixed a priori):
- "simpler model" tie order: LR before GB; within LR smaller C; within GB
  smaller max_depth, then larger min_samples_leaf (fewer effective leaves).
- The paired bootstrap resamples the same 13 stations for both configs;
  each config's lift is computed on its own onset rows / labels (they may
  differ in h and T), then differenced per resample.
- Under the permutation null the label vector for each (h, T) is permuted
  independently within station; all 24 configs sharing (h, T) see the same
  permuted labels.

Run from repo root, BASE conda env (not `hab`):
    ~/anaconda3/python.exe src/models/experiments/tuning_search_nar.py
"""
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

TRAIN_MAX = 2020
VAL_YEARS = (2021, 2022)
TEST_YEARS = (2023,)

HORIZONS = [2, 3, 5, 7, 10]
THRESHOLDS = [10.0, 12.8, 20.0]
LR_C = [0.01, 0.05, 0.2]
GB_DEPTH = [2, 3, 4]
GB_LEAF = [20, 50, 100]
ALERT_TS = np.round(np.arange(0.05, 0.96, 0.05), 2)
POD_MIN = 0.6
N_BOOT = 2000
N_PERM = 30
SEED = 42

REFERENCE = dict(horizon=7, threshold=10.0, tier="A", model="GB",
                 depth=3, leaf=50, t_star=0.50)

# copied verbatim from src/models/train_narragansett.py
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']
TIER_B = TIER_A + ['chl_max', 'chl_std', 'chl_rate_1d', 'chl_accel',
                   'do_min', 'do_range', 'do_night_min', 'do_pct',
                   'ph', 'temp_range']
TIERS = {"A": TIER_A, "B": TIER_B}


def forward_label(day, horizon, bloom):
    """Same routine as build_narragansett_daily.py, parameterised by (h, T)."""
    lab_all = np.full(len(day), np.nan)
    for st, grp in day.groupby("station"):
        idx = grp.index; dates = grp["date"].values; chl = grp["chl"].values
        last = dates.max()
        lab = np.full(len(grp), np.nan)
        for i in range(len(grp)):
            end = dates[i] + np.timedelta64(horizon, "D")
            m = (dates > dates[i]) & (dates <= end)
            if m.any() and (chl[m] > bloom).any():
                lab[i] = 1
            elif end <= last:
                lab[i] = 0
        lab_all[idx] = lab
    return lab_all


def metrics(y, alert):
    tp = int(((alert == 1) & (y == 1)).sum()); fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum()); tn = int(((alert == 0) & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    base = (tp + fn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else np.nan
    lift = prec / base if (base and not np.isnan(prec)) else np.nan
    return dict(tp=tp, fp=fp, fn=fn, pod=pod, precision=prec, base_rate=base, lift=lift)


def safe_auc(y, p):
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return np.nan


def sweep_score(y, p):
    """Max lift over alert thresholds subject to POD >= POD_MIN.

    Returns (score, t_star, precision, pod, base_rate). score=-inf if no
    threshold reaches POD_MIN. Among thresholds with equal lift, the higher
    t is preferred (consistent with the selection tie rule)."""
    best = (-np.inf, np.nan, np.nan, np.nan, np.nan)
    for t in ALERT_TS:
        mm = metrics(y, (p >= t).astype(int))
        if np.isnan(mm["pod"]) or mm["pod"] < POD_MIN or np.isnan(mm["lift"]):
            continue
        if mm["lift"] > best[0] or (mm["lift"] == best[0] and t > best[1]):
            best = (mm["lift"], float(t), mm["precision"], mm["pod"], mm["base_rate"])
    return best


def complexity_rank(cfg):
    if cfg["model"] == "LR":
        return (0, LR_C.index(cfg["C"]), 0)
    return (1, cfg["depth"], -cfg["leaf"])


def select(rows):
    """Selection rule: max lift, then higher t*, then simpler model."""
    feasible = [r for r in rows if np.isfinite(r["val_score"])]
    if not feasible:
        return None
    return sorted(feasible, key=lambda r: (-r["val_score"], -r["t_star"],
                                           complexity_rank(r)))[0]


def model_specs():
    specs = []
    for C in LR_C:
        specs.append(dict(model="LR", C=C, depth=np.nan, leaf=np.nan,
                          name=f"LR_C{C}"))
    for d, l in itertools.product(GB_DEPTH, GB_LEAF):
        specs.append(dict(model="GB", C=np.nan, depth=d, leaf=l,
                          name=f"GB_d{d}_l{l}"))
    return specs


def fit_predict(spec, feats, train, val, test):
    med = train[feats].median(numeric_only=True)

    def prep(X):
        return X[feats].fillna(med).values

    ytr = train.y.astype(int).values
    if spec["model"] == "LR":
        sc = StandardScaler().fit(prep(train))
        m = LogisticRegression(C=spec["C"], class_weight="balanced", max_iter=1000,
                               random_state=SEED).fit(sc.transform(prep(train)), ytr)
        pv = m.predict_proba(sc.transform(prep(val)))[:, 1]
        pt = m.predict_proba(sc.transform(prep(test)))[:, 1]
    else:
        m = HistGradientBoostingClassifier(max_depth=int(spec["depth"]),
                                           learning_rate=0.05, max_iter=300,
                                           min_samples_leaf=int(spec["leaf"]),
                                           l2_regularization=1.0, random_state=SEED,
                                           class_weight="balanced")
        m.fit(prep(train), ytr)
        pv = m.predict_proba(prep(val))[:, 1]
        pt = m.predict_proba(prep(test))[:, 1]
    return pv, pt


def cfg_label(r):
    return f"h{r['horizon']} T{r['threshold']:g} tier{r['tier']} {r['name']}"


def main():
    t0 = time.time()
    day = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
    day = day.sort_values(["station", "date"]).reset_index(drop=True)
    day["year"] = day.date.dt.year

    # ---- label cache per (h, T) and split data ----------------------------
    splits = {}
    for h, T in itertools.product(HORIZONS, THRESHOLDS):
        y = forward_label(day, h, T)
        lab = day.assign(y=y).dropna(subset=["y"])
        train = lab[lab.year <= TRAIN_MAX]
        val = lab[lab.year.isin(VAL_YEARS)]
        test = lab[lab.year.isin(TEST_YEARS)]
        splits[(h, T)] = dict(
            train=train, val=val, test=test,
            val_onset=(val.chl <= T).values, test_onset=(test.chl <= T).values)
        vo = splits[(h, T)]["val_onset"]
        print(f"[labels] h={h:2d} T={T:4g}  train={len(train)} val={len(val)} "
              f"(onset {vo.sum()}, pos {int(val.y.values[vo].sum())}) test={len(test)}",
              flush=True)
    if (7, 10.0) in splits:
        s = splits[(7, 10.0)]
        rebuilt = pd.concat([s["train"].y, s["val"].y, s["test"].y])
        stored = day.loc[rebuilt.index, "bloom_fwd"]
        print(f"[label check] rebuilt h=7/T=10 identical to stored bloom_fwd: "
              f"{bool(np.array_equal(rebuilt.values, stored.values))}", flush=True)

    # ---- fit the grid; cache predictions ----------------------------------
    specs = model_specs()
    rows, cache = [], {}
    n_total = len(HORIZONS) * len(THRESHOLDS) * len(TIERS) * len(specs)
    k = 0
    for (h, T), tier, spec in itertools.product(
            itertools.product(HORIZONS, THRESHOLDS), TIERS, specs):
        s = splits[(h, T)]
        feats = TIERS[tier]
        pv, pt = fit_predict(spec, feats, s["train"], s["val"], s["test"])
        cache[(h, T, tier, spec["name"])] = (pv, pt)
        vo = s["val_onset"]; yv = s["val"].y.astype(int).values
        score, t_star, prec, pod, base = sweep_score(yv[vo], pv[vo])
        r = dict(horizon=h, threshold=T, tier=tier, model=spec["model"],
                 name=spec["name"], C=spec["C"], depth=spec["depth"], leaf=spec["leaf"],
                 val_n_onset=int(vo.sum()), val_n_pos=int(yv[vo].sum()),
                 val_base_rate=float(yv[vo].mean()),
                 val_auc_onset=safe_auc(yv[vo], pv[vo]),
                 t_star=t_star, val_precision=prec, val_pod=pod,
                 val_lift=score if np.isfinite(score) else np.nan,
                 val_score=score, feasible=bool(np.isfinite(score)))
        rows.append(r)
        k += 1
        if k % 30 == 0 or k == n_total:
            print(f"[grid] {k}/{n_total} configs  {time.time() - t0:.0f}s", flush=True)

    grid = pd.DataFrame(rows)
    grid["selected"] = False
    sel = select(rows)
    if sel is None:
        raise SystemExit("no config reached POD >= 0.6 on val onset rows")
    grid.loc[(grid.horizon == sel["horizon"]) & (grid.threshold == sel["threshold"]) &
             (grid.tier == sel["tier"]) & (grid.name == sel["name"]), "selected"] = True
    os.makedirs("data", exist_ok=True)
    grid.sort_values("val_score", ascending=False).drop(columns=["val_score"]) \
        .to_csv("data/tuning_search_nar_grid.csv", index=False)

    print("\n=== TOP-10 VAL CONFIGS (onset lift s.t. POD>=0.6) ===")
    top = grid.sort_values(["val_score", "t_star"], ascending=[False, False]).head(10)
    for _, r in top.iterrows():
        print(f"  {cfg_label(r):32s} t*={r.t_star:.2f} prec={r.val_precision:.3f} "
              f"POD={r.val_pod:.3f} base={r.val_base_rate:.3f} lift={r.val_lift:.2f} "
              f"AUC={r.val_auc_onset:.3f}")

    print("\n=== BEST PER HORIZON ===")
    for h in HORIZONS:
        b = select([r for r in rows if r["horizon"] == h])
        if b is None:
            print(f"  h={h:2d}: no feasible config"); continue
        print(f"  h={h:2d}: {cfg_label(b):32s} t*={b['t_star']:.2f} "
              f"prec={b['val_precision']:.3f} POD={b['val_pod']:.3f} "
              f"base={b['val_base_rate']:.3f} lift={b['val_lift']:.2f}")
    print("\n=== BEST PER THRESHOLD ===")
    for T in THRESHOLDS:
        b = select([r for r in rows if r["threshold"] == T])
        if b is None:
            print(f"  T={T:4g}: no feasible config"); continue
        print(f"  T={T:4g}: {cfg_label(b):32s} t*={b['t_star']:.2f} "
              f"prec={b['val_precision']:.3f} POD={b['val_pod']:.3f} "
              f"base={b['val_base_rate']:.3f} lift={b['val_lift']:.2f}")
    n_feas = int(grid.feasible.sum())
    print(f"\nfeasible configs (POD>=0.6 reachable on val): {n_feas}/{n_total}")
    print(f"\nSELECTED: {cfg_label(sel)}  t*={sel['t_star']:.2f}  "
          f"val prec={sel['val_precision']:.3f} POD={sel['val_pod']:.3f} "
          f"lift={sel['val_lift']:.2f}")

    # ---- permutation null (uses cached val predictions only) --------------
    print("\n=== PERMUTATION NULL (30 within-station shuffles of val onset labels) ===")
    rng = np.random.default_rng(SEED)
    null_rows = []
    for sidx in range(N_PERM):
        perm_rows = []
        for (h, T), s in splits.items():
            vo = s["val_onset"]
            yv = s["val"].y.astype(int).values[vo]
            stations = s["val"].station.values[vo]
            yp = yv.copy()
            for st in np.unique(stations):
                m = stations == st
                yp[m] = rng.permutation(yv[m])
            for tier in TIERS:
                for spec in specs:
                    pv = cache[(h, T, tier, spec["name"])][0][vo]
                    score, t_star, prec, pod, base = sweep_score(yp, pv)
                    perm_rows.append(dict(horizon=h, threshold=T, tier=tier,
                                          model=spec["model"], name=spec["name"],
                                          C=spec["C"], depth=spec["depth"],
                                          leaf=spec["leaf"], val_score=score,
                                          t_star=t_star))
        b = select(perm_rows)
        best = b["val_score"] if b is not None else np.nan
        null_rows.append(dict(shuffle=sidx, null_best_lift=best,
                              null_best_config=cfg_label(b) if b else "none",
                              null_best_t=b["t_star"] if b else np.nan))
        print(f"  shuffle {sidx:2d}: best null lift={best:.3f} "
              f"({null_rows[-1]['null_best_config']})", flush=True)
    null = pd.DataFrame(null_rows)
    real = sel["val_lift"]
    finite_null = null.null_best_lift.dropna().values
    pct = 100.0 * (finite_null < real).mean() if len(finite_null) else np.nan
    null_p95 = np.percentile(finite_null, 95) if len(finite_null) else np.nan
    null["real_best_lift"] = real
    null["real_percentile"] = pct
    null["null_p95"] = null_p95
    null["beats_null_p95"] = bool(real > null_p95) if len(finite_null) else False
    null.to_csv("data/tuning_search_nar_null.csv", index=False)
    print(f"  null best-lift: mean={finite_null.mean():.3f} max={finite_null.max():.3f} "
          f"p95={null_p95:.3f}")
    print(f"  REAL best val lift={real:.3f} -> percentile {pct:.1f} in the null "
          f"({'beyond chance' if real > null_p95 else 'NOT beyond chance'})")

    # ---- ONE test evaluation: selected config + reference, paired bootstrap --
    print("\n=== TEST (2023 onset rows) -- single evaluation of the selected config ===")

    def test_arrays(cfg):
        s = splits[(cfg["horizon"], cfg["threshold"])]
        to = s["test_onset"]
        yt = s["test"].y.astype(int).values[to]
        st = s["test"].station.values[to]
        pt = cache[(cfg["horizon"], cfg["threshold"], cfg["tier"], cfg["name"])][1][to]
        return yt, pt, st

    ref = dict(REFERENCE); ref["name"] = f"GB_d{ref['depth']}_l{ref['leaf']}"
    y_sel, p_sel, st_sel = test_arrays(sel)
    y_ref, p_ref, st_ref = test_arrays(ref)
    stations = np.array(sorted(set(st_sel) | set(st_ref)))
    n_st = len(stations)

    def point(y, p, t):
        mm = metrics(y, (p >= t).astype(int))
        mm["auc"] = safe_auc(y, p); mm["n"] = len(y); mm["n_pos"] = int(y.sum())
        return mm

    pt_sel = point(y_sel, p_sel, sel["t_star"])
    pt_ref = point(y_ref, p_ref, ref["t_star"])

    rng_b = np.random.default_rng(SEED)
    draws = rng_b.integers(0, n_st, size=(N_BOOT, n_st))
    idx_sel = {s_: np.where(st_sel == s_)[0] for s_ in stations}
    idx_ref = {s_: np.where(st_ref == s_)[0] for s_ in stations}
    boot = dict(precision=[], pod=[], base_rate=[], lift=[], auc=[], lift_ref=[], diff=[])
    for b in range(N_BOOT):
        pick = stations[draws[b]]
        ii = np.concatenate([idx_sel[s_] for s_ in pick])
        jj = np.concatenate([idx_ref[s_] for s_ in pick])
        m1 = point(y_sel[ii], p_sel[ii], sel["t_star"])
        m2 = point(y_ref[jj], p_ref[jj], ref["t_star"])
        for key in ("precision", "pod", "base_rate", "lift", "auc"):
            boot[key].append(m1[key])
        boot["lift_ref"].append(m2["lift"])
        boot["diff"].append(m1["lift"] - m2["lift"])

    def ci(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        return (np.percentile(v, 2.5), np.percentile(v, 97.5), len(v))

    out = []

    def add(section, cfg, metric, value, lo=np.nan, hi=np.nan, n_boot=np.nan):
        out.append(dict(section=section, config=cfg_label(cfg), t_star=cfg["t_star"],
                        metric=metric, value=value, ci_lo=lo, ci_hi=hi, n_boot=n_boot))

    for m in ("val_precision", "val_pod", "val_base_rate", "val_lift", "val_auc_onset",
              "val_n_onset", "val_n_pos"):
        add("selected_val", sel, m, sel[m])
    for m in ("precision", "pod", "base_rate", "lift", "auc"):
        lo, hi, nb = ci(boot[m])
        add("selected_test", sel, m, pt_sel[m], lo, hi, nb)
    for m in ("tp", "fp", "fn", "n", "n_pos"):
        add("selected_test", sel, m, pt_sel[m])
    for m in ("precision", "pod", "base_rate", "lift", "auc", "tp", "fp", "fn", "n", "n_pos"):
        add("reference_test", ref, m, pt_ref[m])
    lo, hi, nb = ci(boot["lift_ref"])
    add("reference_test", ref, "lift_boot_ci", pt_ref["lift"], lo, hi, nb)
    d = np.asarray(boot["diff"], float); d = d[np.isfinite(d)]
    lo, hi, nb = ci(d)
    add("paired_diff_vs_reference", sel, "lift_diff", pt_sel["lift"] - pt_ref["lift"],
        lo, hi, nb)
    add("paired_diff_vs_reference", sel, "p_diff_le_0", float((d <= 0).mean()))
    add("permutation_null", sel, "real_val_lift_percentile", pct)
    add("permutation_null", sel, "null_p95", null_p95)
    pd.DataFrame(out).to_csv("data/tuning_search_nar_selected.csv", index=False)

    print(f"  selected  {cfg_label(sel):32s} t*={sel['t_star']:.2f}")
    for m in ("precision", "pod", "base_rate", "lift", "auc"):
        lo_m, hi_m, _ = ci(boot[m])
        print(f"    {m:10s} {pt_sel[m]:.3f}  [{lo_m:.3f}, {hi_m:.3f}]")
    print(f"    tp={pt_sel['tp']} fp={pt_sel['fp']} fn={pt_sel['fn']} "
          f"n={pt_sel['n']} pos={pt_sel['n_pos']}")
    print(f"  reference {cfg_label(ref):32s} t*={ref['t_star']:.2f}")
    print(f"    precision={pt_ref['precision']:.3f} POD={pt_ref['pod']:.3f} "
          f"base={pt_ref['base_rate']:.3f} lift={pt_ref['lift']:.2f} AUC={pt_ref['auc']:.3f}"
          f"  (expected 0.696/0.600/2.00)")
    print(f"  PAIRED lift diff (selected - reference) = "
          f"{pt_sel['lift'] - pt_ref['lift']:+.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"P(diff<=0)={float((d <= 0).mean()):.3f}")
    print(f"  permutation: real val lift percentile = {pct:.1f}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
