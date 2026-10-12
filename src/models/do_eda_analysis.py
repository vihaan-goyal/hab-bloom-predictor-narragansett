"""
Dissolved Oxygen (DO) EDA — mirrors the chlorophyll temporal analysis.

Generates:
  figures/eda/do_lag_correlation_decay.png   — r vs. lag (0,3,7,14,21 prior VISITS,
                                           median 21 d apart, NOT days)
  figures/eda/do_distribution_by_bloom.png   — DO distribution: bloom vs. non-bloom
  figures/eda/do_seasonal_pattern.png        — monthly median DO vs. bloom frequency
  figures/eda/do_station_gradient.png        — mean DO by station (west→east)
  figures/eda/do_temporal_trend.png          — annual mean DO trend (1993–2025)

Run from project root:
    conda activate hab
    python src/models/do_eda_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/hab_features_final.csv"
FIGURES_DIR = "figures/eda"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

DO_COL = "oxygen_concentration_in_sea_water"

# Drop rows missing DO or bloom label
df = df.dropna(subset=[DO_COL, "bloom"])
print(f"Rows after dropping NA in DO/bloom: {len(df):,}")
print(f"Bloom rate: {df['bloom'].mean():.1%}")

# ── 1. Lag Correlation Decay ───────────────────────────────────────────────────
# Build lagged DO features per station (same logic as chl lags in eda_features.ipynb).
# NOTE: .shift(k) moves k ROWS = k station VISITS, not k days. Visits are spaced a
# median of 21 days apart, so "lag 21" is roughly 14 months. Labels below say visits.
print("\nBuilding DO lag features...")
df = df.sort_values(["station_name", "date"]).reset_index(drop=True)

for lag in [3, 7, 14, 21]:
    df[f"do_lag{lag}"] = df.groupby("station_name")[DO_COL].shift(lag)

df["do_roll7_mean"] = df.groupby("station_name")[DO_COL].transform(
    lambda x: x.rolling(7, min_periods=2).mean()
)

lags = [0, 3, 7, 14, 21]
lag_cols = [DO_COL, "do_lag3", "do_lag7", "do_lag14", "do_lag21"]
corrs = [df[col].corr(df["bloom"]) for col in lag_cols]
roll_corr = df["do_roll7_mean"].corr(df["bloom"])

print("\nDO lag correlations with bloom label:")
for l, c in zip(lags, corrs):
    print(f"  do_lag{l:2d} visits (~{l * 21:3d} d): {c:.3f}")
print(f"  do_roll7_mean (7 visits): {roll_corr:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lags, corrs, "o-", color="darkorange", linewidth=2, markersize=8, label="DO at lag k")
ax.fill_between(lags, corrs, alpha=0.15, color="darkorange")
ax.axhline(roll_corr, linestyle="--", color="firebrick", linewidth=1.5,
           label=f"7-visit rolling mean (r = {roll_corr:.3f})")
ax.set_xlabel("Lag (prior station visits; median 21 d apart)", fontsize=12)
ax.set_ylabel("Pearson r with bloom label", fontsize=12)
ax.set_title("Dissolved Oxygen vs. Bloom Label by Lag\n"
             "(lag in prior VISITS, not days)", fontsize=13)
ax.set_xticks(lags)
ax.grid(True, alpha=0.3)
ax.legend()

# Annotate each point
for l, c in zip(lags, corrs):
    ax.annotate(f"{c:.3f}", (l, c), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color="darkorange")

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/do_lag_correlation_decay.png", dpi=150)
plt.close()
print(f"Saved {FIGURES_DIR}/do_lag_correlation_decay.png")

# ── 2. DO Distribution by Bloom vs Non-Bloom ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bloom_vals = df.loc[df["bloom"] == 1, DO_COL].dropna()
no_bloom_vals = df.loc[df["bloom"] == 0, DO_COL].dropna()

# Histogram
ax = axes[0]
bins = np.linspace(0, 20, 60)
ax.hist(no_bloom_vals, bins=bins, alpha=0.6, color="steelblue",
        label=f"No bloom (n={len(no_bloom_vals):,})", density=True)
ax.hist(bloom_vals, bins=bins, alpha=0.6, color="firebrick",
        label=f"Bloom (n={len(bloom_vals):,})", density=True)
ax.axvline(6.0, linestyle="--", color="black", linewidth=1.2, label="6 mg/L hypoxia threshold")
ax.axvline(bloom_vals.median(), linestyle=":", color="firebrick",
           linewidth=1.5, label=f"Bloom median = {bloom_vals.median():.1f} mg/L")
ax.axvline(no_bloom_vals.median(), linestyle=":", color="steelblue",
           linewidth=1.5, label=f"No-bloom median = {no_bloom_vals.median():.1f} mg/L")
ax.set_xlabel("Dissolved Oxygen (mg/L)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("DO Distribution: Bloom vs. Non-Bloom", fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Box plot
ax = axes[1]
ax.boxplot([no_bloom_vals.clip(0, 20), bloom_vals.clip(0, 20)],
           labels=["No Bloom", "Bloom"],
           patch_artist=True,
           boxprops=dict(facecolor="none"),
           medianprops=dict(linewidth=2))
boxes = ax.patches
if len(boxes) >= 2:
    boxes[0].set_facecolor("steelblue")
    boxes[0].set_alpha(0.5)
    boxes[1].set_facecolor("firebrick")
    boxes[1].set_alpha(0.5)
ax.axhline(6.0, linestyle="--", color="black", linewidth=1.2, label="6 mg/L hypoxia threshold")
ax.set_ylabel("Dissolved Oxygen (mg/L)", fontsize=11)
ax.set_title("DO Boxplot by Bloom Status", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# Mann-Whitney U test
stat, p = stats.mannwhitneyu(bloom_vals, no_bloom_vals, alternative="two-sided")
fig.suptitle(f"Mann-Whitney U: p = {p:.2e}", fontsize=10, y=0.02)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/do_distribution_by_bloom.png", dpi=150)
plt.close()
print(f"Saved {FIGURES_DIR}/do_distribution_by_bloom.png")

# ── 3. Seasonal DO Pattern vs. Bloom Frequency ────────────────────────────────
monthly = df.groupby("month").agg(
    do_median=(DO_COL, "median"),
    do_q25=(DO_COL, lambda x: x.quantile(0.25)),
    do_q75=(DO_COL, lambda x: x.quantile(0.75)),
    bloom_freq=("bloom", "mean"),
).reset_index()

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m - 1])

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

ax1.plot(monthly["month_name"], monthly["do_median"],
         "o-", color="darkorange", linewidth=2, markersize=7, label="Median DO")
ax1.fill_between(monthly["month_name"], monthly["do_q25"], monthly["do_q75"],
                 alpha=0.2, color="darkorange", label="IQR")
ax1.axhline(6.0, linestyle="--", color="black", linewidth=1.2, alpha=0.7,
            label="6 mg/L hypoxia threshold")
ax1.set_ylabel("Dissolved Oxygen (mg/L)", fontsize=11, color="darkorange")
ax1.tick_params(axis="y", labelcolor="darkorange")

ax2.bar(monthly["month_name"], monthly["bloom_freq"],
        alpha=0.35, color="steelblue", label="Bloom frequency")
ax2.set_ylabel("Bloom Frequency", fontsize=11, color="steelblue")
ax2.tick_params(axis="y", labelcolor="steelblue")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

ax1.set_title("Seasonal Pattern: Dissolved Oxygen vs. Bloom Frequency", fontsize=13)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/do_seasonal_pattern.png", dpi=150)
plt.close()
print(f"Saved {FIGURES_DIR}/do_seasonal_pattern.png")

# ── 4. Station DO Gradient (west → east) ─────────────────────────────────────
station_stats = df.groupby("station_name").agg(
    lon=("longitude", "first"),
    do_mean=(DO_COL, "mean"),
    bloom_freq=("bloom", "mean"),
    n=("bloom", "count"),
).reset_index().sort_values("lon")

fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

x = range(len(station_stats))
ax1.bar(x, station_stats["do_mean"], alpha=0.6, color="darkorange", label="Mean DO")
ax1.axhline(6.0, linestyle="--", color="black", linewidth=1.2, label="6 mg/L threshold")
ax1.set_ylabel("Mean Dissolved Oxygen (mg/L)", fontsize=11, color="darkorange")
ax1.tick_params(axis="y", labelcolor="darkorange")

ax2.plot(x, station_stats["bloom_freq"], "o-", color="firebrick",
         linewidth=1.5, markersize=5, label="Bloom frequency")
ax2.set_ylabel("Bloom Frequency", fontsize=11, color="firebrick")
ax2.tick_params(axis="y", labelcolor="firebrick")

ax1.set_xticks(list(x))
ax1.set_xticklabels(station_stats["station_name"], rotation=90, fontsize=7)
ax1.set_xlabel("Station (west → east by longitude)", fontsize=11)
ax1.set_title("Station DO Gradient vs. Bloom Frequency (West → East)", fontsize=13)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/do_station_gradient.png", dpi=150)
plt.close()
print(f"Saved {FIGURES_DIR}/do_station_gradient.png")

# ── 5. Annual DO Trend ─────────────────────────────────────────────────────────
annual = df.groupby("year").agg(
    do_mean=(DO_COL, "mean"),
    bloom_freq=("bloom", "mean"),
).reset_index()

slope, intercept, r, p_val, se = stats.linregress(annual["year"], annual["do_mean"])
trend_line = slope * annual["year"] + intercept

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.scatter(annual["year"], annual["do_mean"], color="darkorange", s=40, zorder=3)
ax1.plot(annual["year"], trend_line, "--", color="darkorange", linewidth=1.5,
         label=f"DO trend: {slope:+.3f} mg/L/year (r={r:.2f})")
ax1.set_ylabel("Mean Dissolved Oxygen (mg/L)", fontsize=11, color="darkorange")
ax1.tick_params(axis="y", labelcolor="darkorange")

ax2.bar(annual["year"], annual["bloom_freq"], alpha=0.3, color="steelblue", label="Bloom frequency")
ax2.set_ylabel("Bloom Frequency", fontsize=11, color="steelblue")
ax2.tick_params(axis="y", labelcolor="steelblue")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

ax1.set_xlabel("Year", fontsize=11)
ax1.set_title("Annual Dissolved Oxygen Trend vs. Bloom Frequency (1993–2025)", fontsize=13)
ax1.axvline(2014, linestyle=":", color="gray", linewidth=1.2, alpha=0.7)
ax1.text(2014.2, ax1.get_ylim()[0] + 0.1, "2014\nCWA inflection", fontsize=8, color="gray")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/do_temporal_trend.png", dpi=150)
plt.close()
print(f"Saved {FIGURES_DIR}/do_temporal_trend.png")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== DO ANALYSIS SUMMARY ===")
print(f"\nLag correlations with bloom (negative = low DO predicts bloom):")
for l, c in zip(lags, corrs):
    print(f"  Lag {l:2d}d:        r = {c:.3f}")
print(f"  Roll7 mean:    r = {roll_corr:.3f}")

print(f"\nDO statistics by bloom status:")
print(f"  Bloom     — mean: {bloom_vals.mean():.2f}, median: {bloom_vals.median():.2f}, "
      f"std: {bloom_vals.std():.2f} mg/L")
print(f"  No bloom  — mean: {no_bloom_vals.mean():.2f}, median: {no_bloom_vals.median():.2f}, "
      f"std: {no_bloom_vals.std():.2f} mg/L")

pct_hypoxic_bloom = (bloom_vals < 6.0).mean()
pct_hypoxic_no_bloom = (no_bloom_vals < 6.0).mean()
print(f"\n  % below 6 mg/L hypoxia threshold:")
print(f"    During blooms:      {pct_hypoxic_bloom:.1%}")
print(f"    During non-blooms:  {pct_hypoxic_no_bloom:.1%}")

print(f"\nMann-Whitney U test (bloom vs no-bloom DO): p = {p:.2e}")
print(f"\nLinear DO trend: {slope:+.4f} mg/L/year (r = {r:.2f}, p = {p_val:.3f})")
print("\nAll figures saved to figures/eda/")