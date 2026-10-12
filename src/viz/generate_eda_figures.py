"""
generate_eda_figures.py
Regenerates all EDA figures on corrected daily data (hab_labels_daily.csv).

Run from repo root:
    python generate_eda_figures.py

Outputs to figures/eda/ :
    fig2_station_bloom_rates.png
    fig3_annual_monthly_bloom_freq.png
    fig4_seasonal_chl_boxplot.png
    fig5_temp_chl_scatter.png
    fig6_lag_correlation_decay.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ---- config ----------------------------------------------------------------

DATA_PATH = "data/hab_features_daily.csv"
OUT_DIR   = "figures/eda"

BLOOM_COL   = "bloom_28d"
CHL_COL     = "Chlorophyll"
TEMP_COL    = "sea_water_temperature"
STATION_COL = "station_name"
DATE_COL    = "date"

PALETTE = {
    "bloom":    "#E63946",
    "no_bloom": "#457B9D",
    "line":     "#1D3557",
    "accent":   "#F4A261",
    "grid":     "#E8EDF2",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ---- load ------------------------------------------------------------------

print(f"Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])

# basic sanity
print(f"  Rows: {len(df):,}  |  Bloom rate: {df[BLOOM_COL].mean()*100:.1f}%")
print(f"  Columns: {list(df.columns)}")
print(f"  Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")

df["year"]  = df[DATE_COL].dt.year
df["month"] = df[DATE_COL].dt.month

# ---- Figure 2: station bloom rate bubble map --------------------------------
# Uses lat/lon if present; else plots as sorted bar chart

print("\n[Fig 2] Station bloom rates ...")

stn_stats = (
    df.groupby(STATION_COL)
    .agg(
        bloom_rate=(BLOOM_COL, "mean"),
        n_obs=(BLOOM_COL, "count"),
        lat=("latitude_x", "mean") if "latitude_x" in df.columns else (STATION_COL, "count"),
        lon=("longitude_x", "mean") if "longitude_x" in df.columns else (STATION_COL, "count"),
    )
    .reset_index()
)

has_coords = "latitude_x" in df.columns and "longitude_x" in df.columns

fig, ax = plt.subplots(figsize=(10, 5) if not has_coords else (10, 6))

if has_coords:
    sc = ax.scatter(
        stn_stats["lon"], stn_stats["lat"],
        s=stn_stats["n_obs"] / stn_stats["n_obs"].max() * 600 + 40,
        c=stn_stats["bloom_rate"],
        cmap="RdYlGn_r", vmin=0, vmax=0.5, alpha=0.85, edgecolors="white", linewidths=0.6,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Bloom rate", fontsize=10)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Station Bloom Rates Across Long Island Sound (1993-2025)", fontsize=13, fontweight="bold")

    # annotate top 5 highest
    top5 = stn_stats.nlargest(5, "bloom_rate")
    for _, row in top5.iterrows():
        ax.annotate(
            f"{row[STATION_COL]}\n{row['bloom_rate']*100:.0f}%",
            xy=(row["lon"], row["lat"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7.5, color="#1D3557",
        )
else:
    # fallback: sorted bar chart
    stn_sorted = stn_stats.sort_values("bloom_rate", ascending=False)
    colors = [PALETTE["bloom"] if r > 0.2 else PALETTE["no_bloom"]
              for r in stn_sorted["bloom_rate"]]
    ax.bar(range(len(stn_sorted)), stn_sorted["bloom_rate"] * 100, color=colors, alpha=0.8)
    ax.set_xticks(range(len(stn_sorted)))
    ax.set_xticklabels(stn_sorted[STATION_COL], rotation=90, fontsize=7)
    ax.set_ylabel("Bloom rate (%)", fontsize=11)
    ax.set_xlabel("Station", fontsize=11)
    ax.set_title("Station Bloom Rates (corrected 28-day label, 1993-2025)", fontsize=13, fontweight="bold")
    ax.axhline(df[BLOOM_COL].mean() * 100, color=PALETTE["line"], linestyle="--", lw=1.5, label="Overall mean")
    ax.legend(fontsize=9)
    ax.set_facecolor(PALETTE["grid"])
    ax.grid(axis="y", color="white", lw=0.8)

    # annotate western stations (first few after sort)
    top_rate = stn_sorted["bloom_rate"].max() * 100
    bot_rate  = stn_sorted["bloom_rate"].min() * 100
    ax.text(0.02, 0.96, f"Range: {bot_rate:.1f}% - {top_rate:.1f}%",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

fig.tight_layout()
out = os.path.join(OUT_DIR, "fig2_station_bloom_rates.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ---- Figure 3: annual + monthly bloom frequency ----------------------------

print("\n[Fig 3] Annual + monthly bloom frequency ...")

annual  = df.groupby("year")[BLOOM_COL].mean().reset_index()
monthly = df.groupby("month")[BLOOM_COL].mean().reset_index()

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig = plt.figure(figsize=(13, 5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# --- annual trend with linear fit ---
ax1.bar(annual["year"], annual[BLOOM_COL] * 100, color=PALETTE["no_bloom"], alpha=0.7, zorder=2)
# linear regression
mask = annual["year"] >= 1993
slope, intercept, r, p, se = stats.linregress(annual.loc[mask, "year"], annual.loc[mask, BLOOM_COL] * 100)
x_fit = np.array([annual["year"].min(), annual["year"].max()])
ax1.plot(x_fit, slope * x_fit + intercept, color=PALETTE["bloom"], lw=2, label=f"Trend: {slope:.2f}%/yr (p={p:.3f})")
# TMDL annotation
ax1.axvline(2014, color=PALETTE["accent"], linestyle="--", lw=1.5, label="2014 TMDL")
ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Bloom frequency (%)", fontsize=11)
ax1.set_title("Annual Bloom Frequency (corrected 28-day label)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.set_facecolor(PALETTE["grid"])
ax1.grid(axis="y", color="white", lw=0.8)

# --- monthly pattern ---
bar_colors = [PALETTE["bloom"] if r > monthly[BLOOM_COL].mean() else PALETTE["no_bloom"]
              for r in monthly[BLOOM_COL]]
ax2.bar(range(1, 13), monthly[BLOOM_COL] * 100, color=bar_colors, alpha=0.85)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names, rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("Bloom frequency (%)", fontsize=11)
ax2.set_title("Seasonal Pattern", fontsize=12, fontweight="bold")
ax2.set_facecolor(PALETTE["grid"])
ax2.grid(axis="y", color="white", lw=0.8)

fig.suptitle("Long Island Sound HAB Temporal Patterns (1993-2025)", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
out = os.path.join(OUT_DIR, "fig3_annual_monthly_bloom_freq.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ---- Figure 4: seasonal CHL boxplot ----------------------------------------

print("\n[Fig 4] Seasonal CHL boxplot ...")

if CHL_COL not in df.columns:
    print(f"  WARNING: column '{CHL_COL}' not found, skipping. Available: {list(df.columns)}")
else:
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall",   10: "Fall",  11: "Fall"}
    df["season"] = df["month"].map(season_map)
    season_order = ["Winter", "Spring", "Summer", "Fall"]

    fig, ax = plt.subplots(figsize=(9, 5))

    data_seasons = [
        df.loc[(df["season"] == s) & (df[BLOOM_COL] == 0), CHL_COL].dropna().values
        for s in season_order
    ]
    data_bloom = [
        df.loc[(df["season"] == s) & (df[BLOOM_COL] == 1), CHL_COL].dropna().values
        for s in season_order
    ]

    positions_nb = [1, 4, 7, 10]
    positions_b  = [2, 5, 8, 11]

    bp1 = ax.boxplot(data_seasons, positions=positions_nb, widths=0.7, patch_artist=True,
                     medianprops=dict(color="white", lw=2),
                     boxprops=dict(facecolor=PALETTE["no_bloom"], alpha=0.8),
                     whiskerprops=dict(color=PALETTE["line"]),
                     capprops=dict(color=PALETTE["line"]),
                     flierprops=dict(marker="o", markersize=2, alpha=0.3, color=PALETTE["no_bloom"]))
    bp2 = ax.boxplot(data_bloom, positions=positions_b, widths=0.7, patch_artist=True,
                     medianprops=dict(color="white", lw=2),
                     boxprops=dict(facecolor=PALETTE["bloom"], alpha=0.8),
                     whiskerprops=dict(color=PALETTE["line"]),
                     capprops=dict(color=PALETTE["line"]),
                     flierprops=dict(marker="o", markersize=2, alpha=0.3, color=PALETTE["bloom"]))

    ax.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax.set_xticklabels(season_order, fontsize=11)
    ax.set_ylabel("Chlorophyll-a (µg/L)", fontsize=11)
    ax.set_title("Seasonal Chlorophyll-a Distribution by Bloom Status (corrected daily data)", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.axhline(10, color=PALETTE["accent"], linestyle="--", lw=1.5, label="Bloom threshold (10 µg/L)")
    ax.legend(handles=[bp1["boxes"][0], bp2["boxes"][0], ax.get_lines()[-1]],
              labels=["No Bloom", "Bloom", "Threshold (10 µg/L)"], fontsize=9)
    ax.set_facecolor(PALETTE["grid"])
    ax.grid(axis="y", color="white", lw=0.8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_seasonal_chl_boxplot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ---- Figure 5: temperature vs CHL scatter ----------------------------------

print("\n[Fig 5] Temperature vs CHL scatter ...")

if CHL_COL not in df.columns or TEMP_COL not in df.columns:
    print(f"  WARNING: missing columns, skipping. Available: {list(df.columns)}")
else:
    # subsample for readability
    df_plot = df[[TEMP_COL, CHL_COL, BLOOM_COL]].dropna()
    if len(df_plot) > 8000:
        df_plot = df_plot.sample(8000, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))

    no_bloom = df_plot[df_plot[BLOOM_COL] == 0]
    bloom    = df_plot[df_plot[BLOOM_COL] == 1]

    ax.scatter(no_bloom[TEMP_COL], no_bloom[CHL_COL], c=PALETTE["no_bloom"],
               alpha=0.25, s=12, label="No Bloom", rasterized=True)
    ax.scatter(bloom[TEMP_COL], bloom[CHL_COL], c=PALETTE["bloom"],
               alpha=0.5, s=18, label="Bloom", rasterized=True)

    ax.axhline(10, color=PALETTE["accent"], linestyle="--", lw=1.5, label="Bloom threshold")
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("Chlorophyll-a (µg/L)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Temperature vs. Chlorophyll-a by Bloom Status (corrected daily data)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor(PALETTE["grid"])
    ax.grid(color="white", lw=0.8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_temp_chl_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

# ---- Figure 6: lag correlation decay ---------------------------------------
# NOTE: each station-date has one CHL value (no intra-cast duplication), so the
# autocorrelation is computed over station VISITS, not over calendar time. An
# earlier version of this comment claimed calendar time, which is what led the
# axis and notes/KEY_NUMBERS.md to be labelled in days. Visits are spaced a median
# of 21 days apart, so the lag axis spans roughly 0 to 2.4 years. A true
# calendar-day version is not supportable at this cadence: matched-pair counts
# swing from ~390 to ~5,300 across adjacent day-lags because visits cluster near
# 14- and 28-day spacings. See notes/KEY_NUMBERS.md Section 3.

print("\n[Fig 6] Lag correlation decay ...")

if CHL_COL not in df.columns:
    print(f"  WARNING: column '{CHL_COL}' not found, skipping.")
else:
    # For each station, build a regular time series and compute rolling lagged correlation
    # against the future bloom label.
    # We correlate chl(t-lag) with bloom_28d(t) across all station-dates.

    # NOTE: these are PRIOR VISITS, not days. `.shift(n)` below moves n rows, and
    # rows are station visits spaced a median of 21 days apart, so lag 42 is about
    # 2.4 years rather than six weeks. The variable was previously named `lag_days`
    # and the axis labelled "Lag (days)", which misrepresented every value.
    lag_visits = [0, 3, 7, 14, 21, 28, 35, 42]
    MEDIAN_GAP_DAYS = 21
    corrs = []

    df_sorted = df.sort_values([STATION_COL, DATE_COL]).copy()

    for lag in lag_visits:
        # shift CHL back by `lag` VISITS per station
        # (i.e., correlate past CHL with current bloom label)
        df_sorted["chl_lagged"] = (
            df_sorted.groupby(STATION_COL)[CHL_COL]
            .shift(lag)   # positive shift = use value from 'lag' rows ago
        )
        valid = df_sorted[["chl_lagged", BLOOM_COL]].dropna()
        r, p = stats.pearsonr(valid["chl_lagged"], valid[BLOOM_COL])
        corrs.append((lag, r, p))
        print(f"    lag={lag:2d} visits (~{lag * MEDIAN_GAP_DAYS:4d} d)  "
              f"r={r:.3f}  p={p:.2e}  n={len(valid):,}")

    lags_arr  = np.array([c[0] for c in corrs])
    r_arr     = np.array([c[1] for c in corrs])
    p_arr     = np.array([c[2] for c in corrs])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lags_arr, r_arr, "o-", color=PALETTE["line"], lw=2, markersize=8, zorder=3)

    # shade sig/nonsig
    for i, (lag, r, p) in enumerate(corrs):
        color = PALETTE["bloom"] if p < 0.05 else "#AAAAAA"
        ax.scatter(lag, r, s=80, color=color, zorder=4)

    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel(f"Lag (prior station visits; median {MEDIAN_GAP_DAYS} d apart)",
                  fontsize=11)
    ax.set_ylabel("Pearson r (CHL vs. bloom_28d)", fontsize=11)
    ax.set_title("Lagged CHL vs. Future Bloom\n"
                 "(lag in prior VISITS, not days: lag 42 is ~2.4 years)",
                 fontsize=12, fontweight="bold")
    ax.set_facecolor(PALETTE["grid"])
    ax.grid(color="white", lw=0.8)

    # annotate r=0 crossover if any
    for lag, r, _ in corrs:
        ax.annotate(f"r={r:.3f}", xy=(lag, r), xytext=(3, 6),
                    textcoords="offset points", fontsize=8, color=PALETTE["line"])

    from matplotlib.patches import Patch
    handles = [Patch(color=PALETTE["bloom"], label="p < 0.05"),
               Patch(color="#AAAAAA", label="p ≥ 0.05")]
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig6_lag_correlation_decay.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

print("\nAll EDA figures done.")