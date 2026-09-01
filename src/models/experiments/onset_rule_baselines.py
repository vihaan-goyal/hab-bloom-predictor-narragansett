"""W2 onset rule baselines -- honesty check for the onset headline.

Question: can a trivial one-line rule match the GB tier-A model on the onset
task (rows with chl <= 10, 7-day forward bloom label)?

Reference model: HistGradientBoostingClassifier on TIER_A, refit exactly as in
src/models/train_narragansett.py (train-median imputation, max_depth=3,
lr=0.05, max_iter=300, min_samples_leaf=50, l2=1.0, balanced, seed 42),
scored on test-2023 onset rows at t*=0.50.

Rules (every parameter chosen on VAL onset rows only, never on test):
  a  chl > c                    c in 2..10 step 0.25, argmax val F1
  a2 chl > c                    c with val POD >= 0.6 and max val precision
  b  chl_rate_1d > r            r over val quantiles, argmax val F1
  c  chl_roll3_mean > c         c in 2..10 step 0.25, argmax val F1
  d  chl > c AND do < d         2-D grid on val, argmax val F1
  e  station x 15-day-DOY climatology rate from TRAIN (all labeled rows,
     as in train_narragansett.py), alert if rate >= 0.5
  e2 same but rate computed from TRAIN ONSET rows only
  f  always-alert (precision must equal base rate exactly)

For each rule on TEST onset rows: precision, POD, base rate, lift, tp/fp/fn,
plus a paired station-clustered bootstrap (13 test stations resampled with
replacement, n_boot=2000, seed 42) of lift(GB) - lift(rule) with 95%
percentile CI.

Output: data/onset_rule_baselines.csv + printed table.
Run from repo root with the BASE anaconda python (not the hab env).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BLOOM = 10.0
TRAIN_MAX = 2020
VAL_YEARS = (2021, 2022)
TEST_YEARS = (2023,)
T_STAR = 0.50
N_BOOT = 2000
SEED = 42

# verbatim from train_narragansett.py
TIER_A = ['chl', 'chl_lag1', 'chl_lag2', 'chl_lag3', 'chl_lag4',
          'chl_roll3_mean', 'chl_roll6_mean', 'chl_roll9_mean',
          'chl_roll14_mean', 'chl_roll21_mean', 'chl_trend',
          'chl_anomaly', 'chl_climatology',
          'do', 'do_lag1', 'temp', 'temp_lag1',
          'sal', 'sal_lag1', 'sal_lag2', 'sal_lag3', 'sal_lag4', 'month']

# ----------------------------------------------------------------- data
df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
df["year"] = df.date.dt.year
lab = df.dropna(subset=["bloom_fwd"]).copy()
train = lab[lab.year <= TRAIN_MAX]
val = lab[lab.year.isin(VAL_YEARS)]
test = lab[lab.year.isin(TEST_YEARS)]

val_on = val[val.chl <= BLOOM].copy()
test_on = test[test.chl <= BLOOM].copy()
yv = val_on.bloom_fwd.astype(int).values
yt = test_on.bloom_fwd.astype(int).values
print(f"train n={len(train)}  val onset n={len(val_on)} pos={yv.mean():.3f}  "
      f"test onset n={len(test_on)} pos={yt.mean():.3f} "
      f"stations={test_on.station.nunique()}")


# -------------------------------------------------------------- metrics
def metrics(y, alert):
    y = np.asarray(y); alert = np.asarray(alert).astype(int)
    tp = int(((alert == 1) & (y == 1)).sum())
    fp = int(((alert == 1) & (y == 0)).sum())
    fn = int(((alert == 0) & (y == 1)).sum())
    n = len(y)
    pod = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    base = (tp + fn) / n if n else np.nan
    lift = prec / base if (base and not np.isnan(prec)) else np.nan
    return dict(precision=prec, pod=pod, base_rate=base, lift=lift,
                tp=tp, fp=fp, fn=fn)


def f1_of(m):
    p, r = m["precision"], m["pod"]
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def val_f1(alert):
    return f1_of(metrics(yv, alert))


# ----------------------------------------------------- 1. GB tier A refit
Xtr = train[TIER_A].copy()
med = Xtr.median(numeric_only=True)
def prep(X): return X[TIER_A].fillna(med).values
gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                    max_iter=300, min_samples_leaf=50,
                                    l2_regularization=1.0, random_state=42,
                                    class_weight="balanced")
gb.fit(prep(train), train.bloom_fwd.astype(int).values)
p_test_on = gb.predict_proba(prep(test_on))[:, 1]
gb_alert = (p_test_on >= T_STAR).astype(int)
gb_m = metrics(yt, gb_alert)
print(f"\nGB tier A onset @t*={T_STAR}: prec={gb_m['precision']:.3f} "
      f"pod={gb_m['pod']:.3f} base={gb_m['base_rate']:.3f} lift={gb_m['lift']:.2f} "
      f"(known: 0.696 / 0.600 / 0.347 / 2.00)")

# ---------------------------------------------------- 2. rule baselines
rules = {}   # name -> (param_str, test_alert_vector)

# (a) chl > c, argmax val F1
c_grid = np.arange(2.0, 10.0 + 1e-9, 0.25)
f1s = [val_f1(val_on.chl.values > c) for c in c_grid]
c_a = float(c_grid[int(np.argmax(f1s))])
rules["a_chl_gt_c__valF1"] = (f"c={c_a:.2f}", test_on.chl.values > c_a)

# (a2) chl > c with val POD >= 0.6 and max val precision
best = None
for c in c_grid:
    m = metrics(yv, val_on.chl.values > c)
    if not np.isnan(m["pod"]) and m["pod"] >= 0.6 and not np.isnan(m["precision"]):
        if best is None or m["precision"] > best[1]:
            best = (float(c), m["precision"])
if best is None:
    rules["a2_chl_gt_c__valPOD0.6"] = ("no c reaches val POD>=0.6",
                                       np.zeros(len(test_on), bool))
else:
    rules["a2_chl_gt_c__valPOD0.6"] = (f"c={best[0]:.2f}", test_on.chl.values > best[0])

# (b) chl_rate_1d > r, r over val quantiles (NaN rate -> no alert)
q_grid = np.arange(0.05, 0.96, 0.05)
r_vals = np.nanquantile(val_on.chl_rate_1d.values, q_grid)
def rate_alert(frame, r):
    v = frame.chl_rate_1d.values
    return np.where(np.isnan(v), False, v > r)
f1s = [val_f1(rate_alert(val_on, r)) for r in r_vals]
i_b = int(np.argmax(f1s))
r_b = float(r_vals[i_b])
rules["b_chl_rate_1d_gt_r__valF1"] = (f"r={r_b:.3f} (val q{q_grid[i_b]:.2f})",
                                      rate_alert(test_on, r_b))

# (c) chl_roll3_mean > c (NaN -> no alert)
def roll_alert(frame, c):
    v = frame.chl_roll3_mean.values
    return np.where(np.isnan(v), False, v > c)
f1s = [val_f1(roll_alert(val_on, c)) for c in c_grid]
c_c = float(c_grid[int(np.argmax(f1s))])
rules["c_chl_roll3_gt_c__valF1"] = (f"c={c_c:.2f}", roll_alert(test_on, c_c))

# (d) chl > c AND do < d, 2-D grid on val (NaN do -> no alert)
c_grid_d = np.arange(2.0, 10.0 + 1e-9, 0.5)
d_grid = np.nanquantile(val_on["do"].values, np.arange(0.1, 1.0 + 1e-9, 0.1))
def cd_alert(frame, c, d):
    do = frame["do"].values
    return (frame.chl.values > c) & np.where(np.isnan(do), False, do < d)
best = (-1.0, None, None)
for c in c_grid_d:
    for d in d_grid:
        f = val_f1(cd_alert(val_on, c, d))
        if f > best[0]:
            best = (f, float(c), float(d))
rules["d_chl_gt_c_and_do_lt_d__valF1"] = (f"c={best[1]:.2f}, do<{best[2]:.2f}",
                                          cd_alert(test_on, best[1], best[2]))

# (e) station x 15-day-DOY climatology rate from train (all labeled rows)
def clim_alert(train_frame, frame):
    rate = (train_frame.assign(bin=(train_frame.date.dt.dayofyear - 1) // 15)
            .groupby(["station", "bin"]).bloom_fwd.mean())
    tb = frame.assign(bin=(frame.date.dt.dayofyear - 1) // 15)
    pr = tb.set_index(["station", "bin"]).index.map(rate).values
    pr = np.where(pd.isna(pr), float(train_frame.bloom_fwd.mean()), pr.astype(float))
    return pr >= 0.5
rules["e_clim_station_doy15_train_all"] = ("rate>=0.5 (train, all rows)",
                                           clim_alert(train, test_on))
# (e2) same but rate learned on train onset rows only
rules["e2_clim_station_doy15_train_onset"] = ("rate>=0.5 (train, chl<=10 rows)",
                                              clim_alert(train[train.chl <= BLOOM], test_on))

# (f) always alert
rules["f_always_alert"] = ("-", np.ones(len(test_on), bool))
m_f = metrics(yt, rules["f_always_alert"][1])
assert m_f["precision"] == m_f["base_rate"], "always-alert precision != base rate"
assert abs(m_f["lift"] - 1.0) < 1e-12

# ------------------------------------------- 3/4. test metrics + bootstrap
stations = np.array(sorted(test_on.station.unique()))
st_idx = {s: np.where(test_on.station.values == s)[0] for s in stations}
rng = np.random.default_rng(SEED)
boot_draws = [rng.choice(stations, size=len(stations), replace=True)
              for _ in range(N_BOOT)]
boot_rows = [np.concatenate([st_idx[s] for s in d]) for d in boot_draws]


def lift_on(idx, alert):
    return metrics(yt[idx], alert[idx])["lift"]


def paired_boot(rule_alert):
    diffs = np.array([lift_on(idx, gb_alert) - lift_on(idx, rule_alert)
                      for idx in boot_rows])
    ok = ~np.isnan(diffs)
    lo, hi = np.nanpercentile(diffs[ok], [2.5, 97.5])
    return float(lo), float(hi), int((~ok).sum())


out = []
out.append(dict(rule="GB_tierA_t0.50", param="reference model",
                **gb_m, lift_diff_vs_gb=0.0, ci_lo=0.0, ci_hi=0.0, n_boot_nan=0))
for name, (param, alert) in rules.items():
    alert = np.asarray(alert).astype(int)
    m = metrics(yt, alert)
    lo, hi, nn = paired_boot(alert)
    diff = (gb_m["lift"] - m["lift"]) if not np.isnan(m["lift"]) else np.nan
    out.append(dict(rule=name, param=param, **m, lift_diff_vs_gb=diff,
                    ci_lo=lo, ci_hi=hi, n_boot_nan=nn))

cols = ["rule", "param", "precision", "pod", "base_rate", "lift", "tp", "fp", "fn",
        "lift_diff_vs_gb", "ci_lo", "ci_hi", "n_boot_nan"]
res = pd.DataFrame(out)[cols]
res.to_csv("data/onset_rule_baselines.csv", index=False)
print("\n" + res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# --------------------------------------------------------------- verdict
print("\nVERDICT (paired station-clustered bootstrap, 95% CI of lift(GB)-lift(rule)):")
deflate = False
for _, r in res.iloc[1:].iterrows():
    inc0 = (r.ci_lo <= 0.0 <= r.ci_hi)
    flag = ("CI INCLUDES 0 -> rule statistically indistinguishable from GB"
            if inc0 else "GB reliably better")
    deflate |= inc0
    print(f"  {r.rule:<38s} lift={r.lift:.2f} diff={r.lift_diff_vs_gb:+.2f} "
          f"CI=[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]  {flag}")
if deflate:
    print("\n>>> At least one trivial rule falls within the model's CI. "
          "The onset lift-2.0 headline is NOT uniquely a model result; it is "
          "matched by a one-line rule.")
else:
    print("\n>>> No trivial rule falls within the model's CI; GB's onset lift "
          "exceeds every rule with the CI excluding 0.")
