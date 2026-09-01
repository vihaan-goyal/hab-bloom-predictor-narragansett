"""Rolling-origin (expanding-window) cross-validation for the Narragansett
7-day bloom model -- replaces the single 2023 test year with nine test years.

Fold design (test year T in 2015..2023):
  train = year <= T-2        (expanding window)
  val   = year == T-1        (one-year embargo buffer between train and test)
  test  = year == T
A fold is skipped if test has zero positives or val has < 5 positives.

Per fold and model (LR, GB; Tier-A features, locked specs from
train_narragansett.py): threshold t* = argmax val F1 over 0.10..0.90 step
0.05, chosen on (a) all val days and (b) val onset-only rows (chl <= 10).
Test is scored all-days and onset-only with the all-days t* (the onset-t*
variant is also scored and stored).

Pooling: out-of-fold test predictions are concatenated; pooled metrics use
each fold's own t*. Station-year clustered bootstrap (n_boot=2000, seed=42)
gives 95% CIs for pooled onset precision / POD / lift / AUC.

Outputs:
  data/rolling_origin_cv_nar.csv         -- one row per (fold, model)
  data/rolling_origin_cv_nar_pooled.csv  -- pooled metrics + CIs

Run from repo root with the BASE anaconda python (not the `hab` env):
  ~/anaconda3/python.exe src/models/rolling_origin_cv_nar.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BLOOM = 10.0
TEST_YEARS = range(2015, 2024)
THRESHOLDS = np.round(np.arange(0.10, 0.9001, 0.05), 2)
N_BOOT = 2000
SEED = 42
MIN_VAL_POS = 5

# Tier-A feature list, copied verbatim from src/models/train_narragansett.py
# (experiment scripts must not import from the training script).
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']


def metrics(y, alert):
    """Precision / POD / base rate / lift for a binary alert vector."""
    y = np.asarray(y).astype(int); alert = np.asarray(alert).astype(int)
    tp = int(((alert == 1) & (y == 1)).sum()); fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum()); n = len(y)
    prec = tp / (tp + fp) if tp + fp else np.nan
    pod = tp / (tp + fn) if tp + fn else np.nan
    base = (tp + fn) / n if n else np.nan
    lift = prec / base if base and not np.isnan(prec) else np.nan
    return dict(precision=prec, pod=pod, base_rate=base, lift=lift,
                tp=tp, fp=fp, fn=fn, n=n, n_pos=tp + fn)


def f1_of(y, p, t):
    m = metrics(y, p >= t)
    if np.isnan(m["precision"]) or np.isnan(m["pod"]) or (m["precision"] + m["pod"]) == 0:
        return 0.0
    return 2 * m["precision"] * m["pod"] / (m["precision"] + m["pod"])


def best_threshold(y, p):
    f1s = [f1_of(y, p, t) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(f1s))]), float(max(f1s))


def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def fit_model(name, Xtr, ytr):
    if name == "LR":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000,
                               random_state=42).fit(sc.transform(Xtr), ytr)
        return lambda X: m.predict_proba(sc.transform(X))[:, 1]
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                       max_iter=300, min_samples_leaf=50,
                                       l2_regularization=1.0, random_state=42,
                                       class_weight="balanced").fit(Xtr, ytr)
    return lambda X: m.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
df["year"] = df.date.dt.year
lab = df.dropna(subset=["bloom_fwd"]).copy()
lab["bloom_fwd"] = lab.bloom_fwd.astype(int)
lab["cluster"] = lab.station.astype(str) + "_" + lab.year.astype(str)
lab["onset"] = lab.chl <= BLOOM

fold_rows, oof = [], []   # oof: per-row out-of-fold predictions
for T in TEST_YEARS:
    train = lab[lab.year <= T - 2]
    val = lab[lab.year == T - 1]
    test = lab[lab.year == T]
    if len(train) == 0 or test.bloom_fwd.sum() == 0 or val.bloom_fwd.sum() < MIN_VAL_POS:
        print(f"skip T={T}: train={len(train)} val_pos={val.bloom_fwd.sum()} "
              f"test_pos={test.bloom_fwd.sum()}")
        continue
    med = train[TIER_A].median(numeric_only=True)
    prep = lambda d: d[TIER_A].fillna(med).values
    ytr, yv, yt = train.bloom_fwd.values, val.bloom_fwd.values, test.bloom_fwd.values
    ov, ot = val.onset.values, test.onset.values
    for mn in ("LR", "GB"):
        predict = fit_model(mn, prep(train), ytr)
        pv, pt = predict(prep(val)), predict(prep(test))
        t_all, f1v_all = best_threshold(yv, pv)
        t_on, f1v_on = best_threshold(yv[ov], pv[ov])
        m_all = metrics(yt, pt >= t_all)
        m_on = metrics(yt[ot], pt[ot] >= t_all)
        m_on_t = metrics(yt[ot], pt[ot] >= t_on)
        fold_rows.append(dict(
            test_year=T, model=mn, train_max_year=T - 2, val_year=T - 1,
            n_train=len(train), n_val=len(val), val_pos=int(yv.sum()),
            t_star=t_all, t_star_onset=t_on, val_f1=f1v_all, val_f1_onset=f1v_on,
            auc_val=safe_auc(yv, pv),
            all_auc=safe_auc(yt, pt), all_precision=m_all["precision"], all_pod=m_all["pod"],
            all_base_rate=m_all["base_rate"], all_lift=m_all["lift"],
            all_n_test=m_all["n"], all_n_pos=m_all["n_pos"],
            onset_auc=safe_auc(yt[ot], pt[ot]), onset_precision=m_on["precision"],
            onset_pod=m_on["pod"], onset_base_rate=m_on["base_rate"], onset_lift=m_on["lift"],
            onset_n_test=m_on["n"], onset_n_pos=m_on["n_pos"],
            onset_precision_tonset=m_on_t["precision"], onset_pod_tonset=m_on_t["pod"],
            onset_lift_tonset=m_on_t["lift"],
        ))
        oof.append(pd.DataFrame(dict(
            test_year=T, model=mn, cluster=test.cluster.values, onset=ot, y=yt, p=pt,
            alert=(pt >= t_all).astype(int), alert_tonset=(pt >= t_on).astype(int))))

folds = pd.DataFrame(fold_rows)
folds.to_csv("data/rolling_origin_cv_nar.csv", index=False)
oof = pd.concat(oof, ignore_index=True)

# ---------------------------------------------------------------- pooled ----
def pooled_metrics(d, subset):
    s = d[d.onset] if subset == "onset" else d
    m = metrics(s.y.values, s.alert.values)
    m["auc"] = safe_auc(s.y.values, s.p.values)
    return m


rng = np.random.default_rng(SEED)
pooled_rows = []
for mn in ("LR", "GB"):
    d = oof[oof.model == mn].reset_index(drop=True)
    for subset in ("all", "onset"):
        pm = pooled_metrics(d, subset)
        row = dict(model=mn, subset=subset, n=pm["n"], n_pos=pm["n_pos"],
                   n_folds=d.test_year.nunique(), auc=pm["auc"], precision=pm["precision"],
                   pod=pm["pod"], base_rate=pm["base_rate"], lift=pm["lift"],
                   tp=pm["tp"], fp=pm["fp"], fn=pm["fn"])
        if subset == "onset":
            s = d[d.onset].reset_index(drop=True)
            clusters = s.cluster.values
            uniq = np.unique(clusters)
            idx_by_cl = {c: np.flatnonzero(clusters == c) for c in uniq}
            boots = {k: [] for k in ("precision", "pod", "lift", "auc")}
            for _ in range(N_BOOT):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                ix = np.concatenate([idx_by_cl[c] for c in pick])
                bm = metrics(s.y.values[ix], s.alert.values[ix])
                for k in ("precision", "pod", "lift"):
                    boots[k].append(bm[k])
                boots["auc"].append(safe_auc(s.y.values[ix], s.p.values[ix]))
            for k, v in boots.items():
                v = np.asarray(v, dtype=float)
                row[f"{k}_ci_lo"] = np.nanpercentile(v, 2.5)
                row[f"{k}_ci_hi"] = np.nanpercentile(v, 97.5)
            row["n_clusters"] = len(uniq)
        pooled_rows.append(row)
pooled = pd.DataFrame(pooled_rows)
pooled.to_csv("data/rolling_origin_cv_nar_pooled.csv", index=False)

# ---------------------------------------------------------------- report ----
pd.set_option("display.width", 220)
fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
print("\n=== Per-fold (test year T; train<=T-2; val=T-1) ===")
cols = ["test_year", "model", "t_star", "t_star_onset", "all_auc", "all_precision", "all_pod",
        "all_base_rate", "all_lift", "onset_auc", "onset_precision", "onset_pod",
        "onset_base_rate", "onset_lift", "onset_n_test", "onset_n_pos"]
print(folds[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n=== Pooled out-of-fold (each fold's own t*) ===")
pc = ["model", "subset", "n", "n_pos", "auc", "precision", "pod", "base_rate", "lift",
      "precision_ci_lo", "precision_ci_hi", "pod_ci_lo", "pod_ci_hi",
      "lift_ci_lo", "lift_ci_hi", "auc_ci_lo", "auc_ci_hi"]
print(pooled[pc].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n=== Year-to-year spread of ONSET precision across folds ===")
for mn in ("LR", "GB"):
    f = folds[folds.model == mn]
    lo, hi = f.onset_precision.idxmin(), f.onset_precision.idxmax()
    print(f"{mn}: min {f.onset_precision.min():.3f} (T={int(f.loc[lo,'test_year'])})  "
          f"max {f.onset_precision.max():.3f} (T={int(f.loc[hi,'test_year'])})  "
          f"mean {f.onset_precision.mean():.3f}  sd {f.onset_precision.std(ddof=1):.3f}  "
          f"| onset lift min {f.onset_lift.min():.2f} max {f.onset_lift.max():.2f}  "
          f"| onset AUC min {f.onset_auc.min():.3f} max {f.onset_auc.max():.3f}")

print("\n=== Sanity: T=2023 GB onset fold vs single-split (prec 0.696, AUC 0.839; "
      "single split trained <=2020 with val 2021-22, this fold trains <=2021 with val 2022) ===")
g23 = folds[(folds.model == "GB") & (folds.test_year == 2023)]
if len(g23):
    r = g23.iloc[0]
    ci = pooled[(pooled.model == "GB") & (pooled.subset == "onset")].iloc[0]
    print(f"fold T=2023 GB onset: precision={r.onset_precision:.3f} AUC={r.onset_auc:.3f} "
          f"POD={r.onset_pod:.3f} lift={r.onset_lift:.2f} t*={r.t_star:.2f}")
    print(f"pooled GB onset 95% CI: precision [{ci.precision_ci_lo:.3f}, {ci.precision_ci_hi:.3f}] "
          f"AUC [{ci.auc_ci_lo:.3f}, {ci.auc_ci_hi:.3f}]")
    print(f"single-split 0.696 within pooled precision CI: "
          f"{ci.precision_ci_lo <= 0.696 <= ci.precision_ci_hi}; "
          f"0.839 within pooled AUC CI: {ci.auc_ci_lo <= 0.839 <= ci.auc_ci_hi}")
