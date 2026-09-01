"""Replicate the LIS-side analyses on Narragansett daily data:
1. DO / temp tercile conditioning at elevated-but-subbloom chl (counselor test)
2. Superposed-epoch composite around bloom onsets (dense-data version)
3. Seasonality of onsets; bloom rate vs temperature (<15 C Reinl check)
4. Point-of-no-return style escape probabilities (chl level -> P(reach 10))
Run from repo root, BASE conda env.
"""
import numpy as np
import pandas as pd

BLOOM = 10.0
df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
df = df.sort_values(["station", "date"]).reset_index(drop=True)

# ---- 1. DO / temp conditioning at chl in (5, 10], bloom within 7 d ----------
band = df[(df.chl > 5) & (df.chl <= BLOOM) & df.bloom_fwd.notna()]
print(f"1. chl in (5,10], n={len(band)}, base P(bloom<=7d)={band.bloom_fwd.mean():.2f}")
for col in ("temp", "do"):
    sub = band[band[col].notna()].copy()
    sub["terc"] = pd.qcut(sub[col], 3, labels=["low", "mid", "high"])
    t = sub.groupby("terc", observed=True).agg(n=("bloom_fwd", "size"),
                                               P=("bloom_fwd", "mean"),
                                               lo=(col, "min"), hi=(col, "max"))
    for name, r in t.iterrows():
        print(f"   {col:4s} {name:>4}  [{r.lo:6.2f},{r.hi:6.2f}] n={int(r.n):5d} P={r.P:.2f}")

# ---- 2. superposed epoch: onsets = first day >10 after >=5 days <=10 --------
onsets = []
for st, g in df.groupby("station"):
    g = g.reset_index(drop=True)
    above = (g.chl > BLOOM).values
    for i in range(5, len(g)):
        if above[i] and not above[i-5:i].any() \
           and (g.date.iloc[i] - g.date.iloc[i-5]).days == 5:
            onsets.append((st, g.date.iloc[i]))
print(f"\n2. clean onsets (>=5 quiet days before): {len(onsets)}")
idx = df.set_index(["station", "date"])
rows = []
for st, d0 in onsets:
    for lag in range(-21, 8):
        key = (st, d0 + pd.Timedelta(days=lag))
        if key in idx.index:
            r = idx.loc[key]
            rows.append(dict(lag=lag, chl=r.chl, do=r.do, temp=r.temp, sal=r.sal))
ep = pd.DataFrame(rows).groupby("lag").mean()
ep.to_csv("data/narragansett_epoch_composite.csv")
for lag in (-21, -14, -7, -3, -1, 0, 3, 7):
    if lag in ep.index:
        r = ep.loc[lag]
        print(f"   day {lag:+3d}: chl={r.chl:5.1f}  DO={r.do:5.2f}  temp={r.temp:5.1f}  sal={r.sal:5.2f}")

# ---- 3. seasonality + temperature ------------------------------------------
ond = pd.DataFrame(onsets, columns=["station", "date"])
print("\n3. onsets by month:", dict(ond.date.dt.month.value_counts().sort_index()))
lab = df.dropna(subset=["bloom_fwd", "temp"])
cold, warm = lab[lab.temp < 15], lab[lab.temp >= 15]
print(f"   bloom rate temp<15C: {cold.bloom_fwd.mean():.3f} (n={len(cold)}) | "
      f">=15C: {warm.bloom_fwd.mean():.3f} (n={len(warm)})")

# ---- 4. escape probabilities (PONR-style) ----------------------------------
print("\n4. P(daily-mean chl reaches >10 within 7 d) given today's chl level:")
lab2 = df.dropna(subset=["bloom_fwd"])
for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 9), (9, 10)):
    s = lab2[(lab2.chl > lo) & (lab2.chl <= hi)]
    if len(s):
        print(f"   chl ({lo:2d},{hi:2d}]: n={len(s):5d}  P={s.bloom_fwd.mean():.2f}")
