"""
generate_aeration_figures.py
Reruns the aeration intervention framework on corrected daily data and
generates figures 10 and 11.

Run from repo root:
    python generate_aeration_figures.py

Outputs to figures/intervention/:
    fig10_aeration_priority_map.png    -- bubble chart by station
    fig11_seasonal_intervention.png    -- monthly bars + aeration score line
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR      = "figures/intervention"
FEATURES_CSV = "data/hab_features_daily.csv"
LABELS_CSV   = "data/hab_labels_daily.csv"

BLOOM_COL    = "bloom_28d"
DATE_COL     = "date"
STATION_COL  = "station_name"
DO_COL       = "oxygen_concentration_in_sea_water"
TEMP_COL     = "sea_water_temperature"

# Corrected aeration score formula
# S = 0.45 * (14 - DO) / 12 + 0.30 * (T - 10) / 20 + 0.25 * p
# where p = model bloom probability (approximated by raw bloom label here for the
# intervention analysis; replace with model predict_proba output if preferred)

PALETTE = {
    "high":   "#E63946",
    "medium": "#F4A261",
    "low":    "#457B9D",
    "line":   "#1D3557",
    "grid":   "#E8EDF2",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ---- load ------------------------------------------------------------------

print("Loading data ...")
try:
    features = pd.read_csv(FEATURES_CSV, parse_dates=[DATE_COL])
    print(f"  Rows: {len(features):,}  |  Columns: {list(features.columns)}")
except FileNotFoundError as e:
    print(f"ERROR: {e}\nMake sure both CSV files exist in data/")
    sys.exit(1)

features["year"]  = features[DATE_COL].dt.year
features["month"] = features[DATE_COL].dt.month

print(f"  Rows: {len(features):,}  |  Columns: {list(features.columns)}")

# ---- check required columns ------------------------------------------------

if DO_COL not in features.columns:
    print(f"WARNING: '{DO_COL}' not found. Available: {list(features.columns)}")
    print("  Trying 'do_mgl', 'DO', 'oxygen' ...")
    for alt in ["do_mgl", "DO", "oxygen", "do"]:
        if alt in features.columns:
            DO_COL = alt
            print(f"  Using '{DO_COL}' as dissolved oxygen column.")
            break
    else:
        print("  ERROR: no dissolved oxygen column found. Cannot compute aeration scores.")
        sys.exit(1)

if TEMP_COL not in features.columns:
    for alt in ["temp", "water_temp", "temp_c", "temperature_c"]:
        if alt in features.columns:
            TEMP_COL = alt
            break

# ---- compute aeration score ------------------------------------------------
# Use bloom_28d as a proxy for p (bloom probability) in stations where
# model output isn't saved. This is conservative; real pipeline uses predict_proba.

# For intervention analysis: use validation period 2020-2022
intervention_df = features[
    (features[DATE_COL].dt.year >= 2020) &
    (features[DATE_COL].dt.year <= 2022)
].copy()

print(f"\nIntervention analysis period 2020-2022: {len(intervention_df):,} station-days")

# compute normalized aeration score
intervention_df["score_do"]   = (14 - intervention_df[DO_COL].clip(upper=14)) / 12
intervention_df["score_temp"] = ((intervention_df[TEMP_COL] - 10) / 20).clip(lower=0, upper=1)
intervention_df["score_p"]    = intervention_df[BLOOM_COL].astype(float)

intervention_df["aeration_score"] = (
    0.45 * intervention_df["score_do"] +
    0.30 * intervention_df["score_temp"] +
    0.25 * intervention_df["score_p"]
)

# classify: highly suitable = score > 0.45 AND DO < 6.0
intervention_df["high_risk"] = (
    (intervention_df["aeration_score"] > 0.45) &
    (intervention_df[DO_COL] < 6.0)
)

n_high_risk = intervention_df["high_risk"].sum()
n_total     = len(intervention_df)
print(f"  High-risk intervention days: {n_high_risk:,} ({n_high_risk/n_total*100:.1f}%)")

# ---- Figure 10: aeration priority map by station ---------------------------

print("\n[Fig 10] Aeration priority map ...")

stn_stats = (
    intervention_df.groupby(STATION_COL)
    .agg(
        n_high_risk=("high_risk", "sum"),
        mean_bloom_prob=(BLOOM_COL, "mean"),
        mean_score=("aeration_score", "mean"),
        min_do=(DO_COL, "min"),
        lat=("latitude_x", "mean") if "latitude_x" in intervention_df.columns else ("high_risk", "count"),
        lon=("longitude_x", "mean") if "longitude_x" in intervention_df.columns else ("high_risk", "count"),
    )
    .reset_index()
)

has_coords = "latitude_x" in intervention_df.columns and "longitude_x" in intervention_df.columns

fig, ax = plt.subplots(figsize=(11, 6))

if has_coords:
    sc = ax.scatter(
        stn_stats["lon"], stn_stats["lat"],
        s=stn_stats["n_high_risk"] / stn_stats["n_high_risk"].max() * 700 + 40,
        c=stn_stats["mean_score"],
        cmap="RdYlGn_r", vmin=0, vmax=0.7, alpha=0.85,
        edgecolors="white", linewidths=0.8,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Mean aeration score", fontsize=10)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    # annotate top 5
    top5 = stn_stats.nlargest(5, "n_high_risk")
    for _, row in top5.iterrows():
        ax.annotate(
            f"{row[STATION_COL]}\n({int(row['n_high_risk'])} days)",
            xy=(row["lon"], row["lat"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8, color="#1D3557",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )
else:
    # sorted bar chart
    stn_sorted = stn_stats.sort_values("n_high_risk", ascending=False)
    bar_colors = []
    for _, row in stn_sorted.iterrows():
        if row["n_high_risk"] >= stn_sorted["n_high_risk"].quantile(0.75):
            bar_colors.append(PALETTE["high"])
        elif row["n_high_risk"] >= stn_sorted["n_high_risk"].quantile(0.33):
            bar_colors.append(PALETTE["medium"])
        else:
            bar_colors.append(PALETTE["low"])

    ax.bar(range(len(stn_sorted)), stn_sorted["n_high_risk"], color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(stn_sorted)))
    ax.set_xticklabels(stn_sorted[STATION_COL], rotation=90, fontsize=7)
    ax.set_ylabel("Number of high-risk intervention days (2020-2022)", fontsize=10)
    ax.set_xlabel("Station", fontsize=11)
    ax.set_facecolor(PALETTE["grid"])
    ax.grid(axis="y", color="white", lw=0.8)

    from matplotlib.patches import Patch
    legend_els = [Patch(color=PALETTE["high"], label="High priority"),
                  Patch(color=PALETTE["medium"], label="Medium priority"),
                  Patch(color=PALETTE["low"], label="Low priority")]
    ax.legend(handles=legend_els, fontsize=9)

    # highlight top station
    top_stn = stn_sorted.iloc[0]
    ax.text(0.01, 0.97,
            f"Top station: {top_stn[STATION_COL]} "
            f"({int(top_stn['n_high_risk'])} days, min DO={top_stn['min_do']:.2f} mg/L)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

ax.set_title(f"Aeration Intervention Priority by Station (2020-2022, corrected data)\n"
             f"Total high-risk days: {n_high_risk:,} "
             f"({n_high_risk/n_total*100:.1f}% of station-days)",
             fontsize=12, fontweight="bold")

fig.tight_layout()
out = os.path.join(OUT_DIR, "fig10_aeration_priority_map.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ---- Figure 11: seasonal intervention windows ------------------------------

print("\n[Fig 11] Seasonal intervention windows ...")

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

monthly = (
    intervention_df.groupby("month")
    .agg(
        n_high_risk=("high_risk", "sum"),
        mean_aeration_score=("aeration_score", "mean"),
        n_total=("high_risk", "count"),
    )
    .reset_index()
)
monthly["high_risk_rate"] = monthly["n_high_risk"] / monthly["n_total"]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# bar: number of high-risk days
bar_colors = [PALETTE["high"] if v > monthly["n_high_risk"].mean() else PALETTE["medium"]
              for v in monthly["n_high_risk"]]
bars = ax1.bar(monthly["month"], monthly["n_high_risk"], color=bar_colors, alpha=0.75, zorder=2)

# line: mean aeration score
ax2.plot(monthly["month"], monthly["mean_aeration_score"], "o-",
         color=PALETTE["line"], lw=2.5, markersize=8, zorder=3, label="Mean aeration score")
ax2.axhline(0.45, color=PALETTE["high"], linestyle="--", lw=1.2, label="Intervention threshold (0.45)")

ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(month_names, fontsize=10)
ax1.set_ylabel("High-risk intervention days", fontsize=11, color=PALETTE["medium"])
ax1.tick_params(axis="y", labelcolor=PALETTE["medium"])
ax2.set_ylabel("Mean aeration score (S)", fontsize=11, color=PALETTE["line"])
ax2.tick_params(axis="y", labelcolor=PALETTE["line"])
ax2.set_ylim(0, 0.9)

# annotate peak month
peak_month = monthly.loc[monthly["n_high_risk"].idxmax()]
ax1.annotate(
    f"Peak: {month_names[int(peak_month['month'])-1]}\n({int(peak_month['n_high_risk'])} days)",
    xy=(peak_month["month"], peak_month["n_high_risk"]),
    xytext=(10, 15), textcoords="offset points",
    fontsize=9, color=PALETTE["high"],
    arrowprops=dict(arrowstyle="->", color=PALETTE["high"]),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
)

ax1.set_facecolor(PALETTE["grid"])
ax1.grid(axis="y", color="white", lw=0.8)
ax1.set_title("Seasonal Intervention Windows (2020-2022, corrected data)\n"
              "Bars = high-risk days per month | Line = mean aeration score",
              fontsize=12, fontweight="bold")

lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines2, labels2, fontsize=9, loc="upper left")

fig.tight_layout()
out = os.path.join(OUT_DIR, "fig11_seasonal_intervention.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ---- Detailed print summary ------------------------------------------------

print("\n" + "="*70)
print("AERATION INTERVENTION SUMMARY (2020-2022, corrected data)")
print("="*70)

n_total_station_days = len(intervention_df)
n_high = int(intervention_df["high_risk"].sum())
pct_high = n_high / n_total_station_days * 100

print(f"\nTotal station-days in intervention period : {n_total_station_days:,}")
print(f"High-risk station-days (S>0.45 & DO<6)    : {n_high:,} ({pct_high:.1f}%)")

# Per-station table
print("\n-- Per-Station Summary (top 15 by high-risk days) " + "-" * 22)
stn_summ = (
    intervention_df.groupby(STATION_COL)
    .agg(
        n_high_risk=("high_risk", "sum"),
        mean_bloom_prob=(BLOOM_COL, "mean"),
        mean_score=("aeration_score", "mean"),
        min_do=(DO_COL, "min"),
    )
    .reset_index()
    .sort_values("n_high_risk", ascending=False)
    .head(15)
    .reset_index(drop=True)
)
stn_summ["mean_bloom_prob"] = stn_summ["mean_bloom_prob"].round(3)
stn_summ["mean_score"]      = stn_summ["mean_score"].round(4)
stn_summ["min_do"]          = stn_summ["min_do"].round(2)
print(stn_summ.rename(columns={
    STATION_COL:       "station",
    "n_high_risk":     "n_high_risk",
    "mean_bloom_prob": "mean_bloom_prob",
    "mean_score":      "mean_score",
    "min_do":          "min_DO",
}).to_string(index=False))

# Per-month table
print("\n-- Per-Month Summary " + "-" * 52)
month_names_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
month_summ = (
    intervention_df.groupby("month")
    .agg(
        n_high_risk=("high_risk", "sum"),
        mean_aeration_score=("aeration_score", "mean"),
    )
    .reset_index()
)
month_summ["month_name"] = month_summ["month"].map(month_names_map)
month_summ["mean_aeration_score"] = month_summ["mean_aeration_score"].round(4)
print(month_summ[["month_name", "n_high_risk", "mean_aeration_score"]].to_string(index=False))

# Highest-priority station
top_stn = stn_summ.iloc[0]
print(
    f"\nHighest-priority station : {top_stn[STATION_COL]}"
    f"  |  {int(top_stn['n_high_risk'])} high-risk days"
    f"  |  min DO = {top_stn['min_do']:.2f} mg/L"
)

print("\nAll aeration figures done.")