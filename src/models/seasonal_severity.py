"""
seasonal_severity.py
--------------------
Task 4: Stumpf-style seasonal severity product for western LIS.

Precedent: NOAA's Lake Erie seasonal forecast predicts summer
cyanobacteria bloom severity from spring Maumee River discharge /
phosphorus load (Stumpf et al. 2012, 2016). Adaptation for LIS:

  target    : annual summer (Jun-Sep) severity of the western basin
              PRIMARY   severity_exceed = fraction of western summer
                        station-visits with Chlorophyll > 10 ug/L
              SECONDARY severity_meanchl = mean summer Chlorophyll
  predictor : PRIMARY   spring (Mar-Jun) mean log_discharge (all rivers)
              SECONDARY spring mean log_ct_discharge (Connecticut R only)
  model     : OLS linear regression (deliberately simple, following
              Stumpf 2012)
  validation: leave-one-year-out (LOYO); report r, R^2, LOYO RMSE, and
              tercile hit rate (was the year's severity tercile
              predicted correctly)
  LIS twist : post-TMDL nitrogen reduction predicts the discharge ->
              severity relationship should WEAKEN over time. Fit the
              regression in rolling 15-year windows and plot the slope
              trajectory. Either outcome (stable / decaying) is a
              result.

All choices above were fixed before any correlation was computed.

Usage (from repo root):
    python src/models/seasonal_severity.py
Outputs:
    data/seasonal_severity.csv
    figures/eda/seasonal_severity.png
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_CSV = "data/hab_features_tidal.csv"
WEST_LON = -73.4
BLOOM_THR = 10.0
SUMMER = [6, 7, 8, 9]
SPRING = [3, 4, 5, 6]
MIN_SUMMER_VISITS = 10
ROLL_WINDOW = 15          # years, for the time-varying slope
PRIMARY_Q = "log_discharge"
SECONDARY_Q = "log_ct_discharge"


def yearly_table(df):
    """One row per year: severity metrics (western summer) and spring
    discharge predictors (deduplicated by date so multi-station days
    are not overweighted)."""
    west = df[df["longitude_x"] < WEST_LON]
    summer = west[west["month"].isin(SUMMER)]

    sev = (summer.groupby("year")
                 .agg(severity_exceed=("Chlorophyll",
                                       lambda x: (x > BLOOM_THR).mean()),
                      severity_meanchl=("Chlorophyll", "mean"),
                      n_summer_visits=("Chlorophyll", "count"))
                 .reset_index())

    q = df[["date", "year", "month", PRIMARY_Q, SECONDARY_Q]] \
        .drop_duplicates("date")
    spring = q[q["month"].isin(SPRING)]
    qtab = (spring.groupby("year")
                  .agg(spring_logQ=(PRIMARY_Q, "mean"),
                       spring_logQ_ct=(SECONDARY_Q, "mean"),
                       n_spring_days=("date", "count"))
                  .reset_index())

    tab = sev.merge(qtab, on="year", how="inner")
    tab = tab[tab["n_summer_visits"] >= MIN_SUMMER_VISITS].copy()
    return tab.dropna(subset=["severity_exceed", "spring_logQ"])


def ols(x, y):
    """Slope, intercept, r for simple OLS."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, r


def loyo(tab, xcol, ycol):
    """Leave-one-year-out predictions."""
    preds = []
    for yr in tab["year"]:
        tr = tab[tab["year"] != yr]
        s, b, _ = ols(tr[xcol], tr[ycol])
        xv = float(tab.loc[tab["year"] == yr, xcol].iloc[0])
        preds.append(s * xv + b)
    return np.array(preds)


def tercile_hits(obs, pred):
    """Fraction of years whose predicted severity tercile matches the
    observed tercile (terciles from observed distribution)."""
    qs = np.quantile(obs, [1 / 3, 2 / 3])
    return float((np.digitize(obs, qs) == np.digitize(pred, qs)).mean())


def rolling_slopes(tab, xcol, ycol, window=ROLL_WINDOW):
    rows = []
    years = tab["year"].values
    for start in range(int(years.min()), int(years.max()) - window + 2):
        w = tab[(tab["year"] >= start) & (tab["year"] < start + window)]
        if len(w) < window - 3:      # tolerate a few missing years
            continue
        s, _, r = ols(w[xcol], w[ycol])
        rows.append(dict(center=start + window / 2, slope=s, r=r,
                         n=len(w)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(BASE_CSV,
                     usecols=["date", "year", "month", "Chlorophyll",
                              "longitude_x", PRIMARY_Q, SECONDARY_Q])
    tab = yearly_table(df)
    print(f"Years usable: {len(tab)} "
          f"({int(tab['year'].min())}-{int(tab['year'].max())})")

    # ---- primary regression ----
    s, b, r = ols(tab["spring_logQ"], tab["severity_exceed"])
    pred = loyo(tab, "spring_logQ", "severity_exceed")
    rmse = float(np.sqrt(np.mean((pred - tab["severity_exceed"]) ** 2)))
    hits = tercile_hits(tab["severity_exceed"].values, pred)
    print("\n== PRIMARY: severity_exceed ~ spring mean log_discharge ==")
    print(f"in-sample: slope={s:.4f}  r={r:.3f}  R^2={r*r:.3f}")
    print(f"LOYO: RMSE={rmse:.4f}  tercile hit rate={hits:.2f} "
          f"(chance = 0.33)")

    # ---- secondaries, reported regardless of outcome ----
    for xcol, ycol, label in [
            ("spring_logQ_ct", "severity_exceed", "CT-only discharge"),
            ("spring_logQ", "severity_meanchl", "mean-chl severity")]:
        sub = tab.dropna(subset=[xcol, ycol])
        _, _, r2 = ols(sub[xcol], sub[ycol])
        print(f"secondary ({label}): r={r2:.3f}  R^2={r2*r2:.3f}  "
              f"n={len(sub)}")

    # ---- rolling slope (TMDL-weakening test) ----
    rs = rolling_slopes(tab, "spring_logQ", "severity_exceed")
    print(f"\nRolling {ROLL_WINDOW}-yr slope: "
          f"{rs['slope'].iloc[0]:.4f} (first window) -> "
          f"{rs['slope'].iloc[-1]:.4f} (last window)")
    print(rs.to_string(index=False))

    os.makedirs("data", exist_ok=True)
    out = tab.copy()
    out["loyo_pred"] = pred
    out.to_csv("data/seasonal_severity.csv", index=False)

    # ---- figure ----
    os.makedirs("figures/eda", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.scatter(tab["spring_logQ"], tab["severity_exceed"],
               c=tab["year"], cmap="viridis", s=35)
    xs = np.linspace(tab["spring_logQ"].min(),
                     tab["spring_logQ"].max(), 50)
    ax.plot(xs, s * xs + b, "k--", lw=1)
    ax.set_xlabel("Spring mean log discharge (Mar-Jun)")
    ax.set_ylabel("Summer exceedance fraction (western LIS)")
    ax.set_title(f"Severity vs spring discharge (r={r:.2f})")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("year")

    ax = axes[1]
    ax.scatter(tab["severity_exceed"], pred, c=tab["year"],
               cmap="viridis", s=35)
    lim = [0, max(tab["severity_exceed"].max(), pred.max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("Observed severity")
    ax.set_ylabel("LOYO predicted severity")
    ax.set_title(f"Leave-one-year-out (RMSE={rmse:.3f}, "
                 f"tercile hits={hits:.2f})")

    ax = axes[2]
    ax.plot(rs["center"], rs["slope"], marker="o", color="tab:red")
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel(f"Window center year ({ROLL_WINDOW}-yr windows)")
    ax.set_ylabel("Discharge -> severity slope")
    ax.set_title("Time-varying relationship (TMDL-weakening test)")

    fig.tight_layout()
    fig.savefig("figures/eda/seasonal_severity.png", dpi=200)
    print("\nSaved data/seasonal_severity.csv, "
          "figures/eda/seasonal_severity.png")


if __name__ == "__main__":
    main()