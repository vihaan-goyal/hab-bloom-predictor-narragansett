"""
bloom_rate_by_period.py -- the "2014 cliff" table (findings §15), reproducibly.
--------------------------------------------------------------------------------
Share of station-days with daily chlorophyll > 10 ug/L, per year and per period,
for LIS (parent repo, lab bottle samples) and Narragansett (sonde daily mean,
raw > 10 and lab-calibrated > 12.8, see §10). Previously computed inline; this
script replaces that so the table can be regenerated.

Inputs
  ../hab-bloom-predictor/data/hab_features_tidal.csv   (parent repo; column Chlorophyll)
  data/narragansett_daily_features.csv                 (this repo; column chl)
Output
  data/bloom_rate_by_period.csv  and the table printed to stdout.

Run from the fork root with the BASE conda env:
  python src/models/experiments/bloom_rate_by_period.py [--parent ../hab-bloom-predictor]
"""
import argparse, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--parent", default="../hab-bloom-predictor")
ap.add_argument("--cal", type=float, default=12.8, help="sonde value equal to lab 10 ug/L (§10)")
a = ap.parse_args()

lis = pd.read_csv(os.path.join(a.parent, "data/hab_features_tidal.csv"), usecols=["date", "Chlorophyll"], parse_dates=["date"])
lis = lis.dropna(subset=["Chlorophyll"])
lis["year"] = lis["date"].dt.year
nar = pd.read_csv("data/narragansett_daily_features.csv", usecols=["date", "chl"], parse_dates=["date"]).dropna(subset=["chl"])
nar["year"] = nar["date"].dt.year

yr = pd.DataFrame({
    "lis_gt10": lis.groupby("year")["Chlorophyll"].apply(lambda s: (s > 10).mean()),
    "lis_n": lis.groupby("year").size(),
    "nar_gt10": nar.groupby("year")["chl"].apply(lambda s: (s > 10).mean()),
    "nar_gt_cal": nar.groupby("year")["chl"].apply(lambda s: (s > a.cal).mean()),
    "nar_n": nar.groupby("year").size(),
}).loc[2005:2025]

def period(lo, hi):
    l = lis[(lis.year >= lo) & (lis.year <= hi)]; n = nar[(nar.year >= lo) & (nar.year <= hi)]
    ly = yr.loc[lo:hi, "lis_gt10"].dropna(); ny = yr.loc[lo:hi, "nar_gt10"].dropna(); nc = yr.loc[lo:hi, "nar_gt_cal"].dropna()
    return {"period": f"{lo}-{hi}" if lo != hi else str(lo),
            "lis_gt10_pooled": (l.Chlorophyll > 10).mean(), "lis_gt10_yr_min": ly.min(), "lis_gt10_yr_max": ly.max(), "lis_gt10_yr_mean": ly.mean(),
            "nar_gt10_pooled": (n.chl > 10).mean(), "nar_gt10_yr_min": ny.min(), "nar_gt10_yr_max": ny.max(),
            "nar_gt_cal_pooled": (n.chl > a.cal).mean(), "nar_gt_cal_yr_min": nc.min(), "nar_gt_cal_yr_max": nc.max(),
            "lis_days": len(l), "nar_days": len(n)}

per = pd.DataFrame([period(2005, 2013), period(2014, 2014), period(2015, 2023)])
os.makedirs("data", exist_ok=True)
out = pd.concat([yr.reset_index().rename(columns={"year": "period"}).assign(kind="year"), per.assign(kind="period")], ignore_index=True)
out.to_csv("data/bloom_rate_by_period.csv", index=False)

pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:.2f}")
print("Per year (share of station-days above threshold):"); print(yr)
print("\nPer period:"); print(per[["period", "lis_gt10_yr_min", "lis_gt10_yr_max", "lis_gt10_yr_mean", "nar_gt10_yr_min", "nar_gt10_yr_max", "nar_gt_cal_yr_min", "nar_gt_cal_yr_max", "lis_days", "nar_days"]])
print(f"\nToday's gap (2015-2023 pooled): raw {per.loc[2,'nar_gt10_pooled']/per.loc[2,'lis_gt10_pooled']:.1f}x, calibrated {per.loc[2,'nar_gt_cal_pooled']/per.loc[2,'lis_gt10_pooled']:.1f}x")
