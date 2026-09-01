"""Figures for notes/NARRAGANSETT_FINDINGS.md. Agg backend, dpi=150.
Outputs to figures/:
  nar_fig1_do_temp_conditioning.png
  nar_fig2_epoch_composite.png
  nar_fig3_escape_probability.png
  nar_fig4_precision_comparison.png
Run from repo root, BASE conda env.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 150
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8f98"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
band = df[(df.chl > 5) & (df.chl <= 10) & df.bloom_fwd.notna()]

# ---- fig 1: DO / temp tercile conditioning ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, col, title, color in ((axes[0], "do", "Dissolved oxygen", BLUE),
                              (axes[1], "temp", "Temperature", ORANGE)):
    sub = band[band[col].notna()].copy()
    sub["terc"] = pd.qcut(sub[col], 3, labels=["low", "mid", "high"])
    t = sub.groupby("terc", observed=True).agg(P=("bloom_fwd", "mean"),
                                               n=("bloom_fwd", "size"))
    se = np.sqrt(t.P * (1 - t.P) / t.n)
    bars = ax.bar(t.index.astype(str), t.P, yerr=1.96 * se, capsize=4,
                  color=color, width=0.6)
    for b, (p, n) in zip(bars, zip(t.P, t.n)):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.04, f"{p:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(band.bloom_fwd.mean(), color=GRAY, ls="--", lw=1)
    ax.set_title(f"{title} tercile at chl 5–10 µg/L")
    ax.set_ylim(0, 0.85)
axes[0].set_ylabel("P(bloom within 7 d)")
axes[0].text(2.45, band.bloom_fwd.mean() + 0.015, "base rate", color=GRAY, fontsize=8)
fig.suptitle("Bloom odds at elevated chlorophyll: low DO and warm water both raise risk", y=1.0)
fig.tight_layout()
fig.savefig("figures/nar_fig1_do_temp_conditioning.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---- fig 2: superposed epoch composite -------------------------------------
ep = pd.read_csv("data/narragansett_epoch_composite.csv").set_index("lag")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
a1.plot(ep.index, ep.chl, color=BLUE, lw=2, marker="o", ms=3)
a1.axvline(0, color=GRAY, lw=1, ls="--")
a1.axhline(10, color=ORANGE, lw=1, ls=":")
a1.text(-20.5, 10.25, "bloom threshold (10 µg/L)", color=ORANGE, fontsize=8)
a1.annotate("~3-day ramp", xy=(-1.5, 9), xytext=(-9, 12.3),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
a1.set_ylabel("chlorophyll (µg/L)")
a1.set_title("Composite of 419 bloom onsets: the run-up lasts ~3 days")
a2.plot(ep.index, ep.do, color=GRAY, lw=2, marker="o", ms=3)
a2.axvline(0, color=GRAY, lw=1, ls="--")
a2.set_ylabel("DO (mg/L)")
a2.set_xlabel("days relative to onset (day 0 = first daily mean > 10 µg/L)")
fig.tight_layout()
fig.savefig("figures/nar_fig2_epoch_composite.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---- fig 3: escape probability / no PONR -----------------------------------
lab2 = df.dropna(subset=["bloom_fwd"])
bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 9), (9, 10)]
xs, ps, ns = [], [], []
for lo, hi in bins:
    s = lab2[(lab2.chl > lo) & (lab2.chl <= hi)]
    xs.append(f"{lo}–{hi}"); ps.append(s.bloom_fwd.mean()); ns.append(len(s))
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(xs, ps, color=BLUE, width=0.65)
for b, p, n in zip(bars, ps, ns):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.02, f"{p:.2f}", ha="center",
            fontsize=9, fontweight="bold")
    if p > 0.06:
        ax.text(b.get_x() + b.get_width() / 2, 0.015, f"n={n:,}", ha="center", fontsize=7, color="white")
ax.set_xlabel("today's daily-mean chlorophyll (µg/L)")
ax.set_ylabel("P(bloom within 7 d)")
ax.set_ylim(0, 0.85)
ax.set_title("No point of no return: risk climbs smoothly, no cliff\n"
             "(even at chl 9–10, 29% of days do not proceed to bloom)")
fig.tight_layout()
fig.savefig("figures/nar_fig3_escape_probability.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---- fig 4: LIS vs Narragansett onset precision ----------------------------
cis = pd.read_csv("data/narragansett_bootstrap_cis.csv").set_index("metric")
nar_p = cis.loc["GB_onset_precision"]
fig, ax = plt.subplots(figsize=(6.5, 4))
vals = [0.136, nar_p.point]
errs = [[0.136 - 0.077, nar_p.point - nar_p.lo], [0.172 - 0.136, nar_p.hi - nar_p.point]]
bars = ax.bar(["Long Island Sound\n(21-day boat sampling)",
               "Narragansett Bay\n(15-minute sondes)"],
              vals, yerr=errs, capsize=5, color=[GRAY, BLUE], width=0.5)
for i, (b, v) in enumerate(zip(bars, vals)):
    ax.text(b.get_x() + b.get_width() / 2, v + errs[1][i] + 0.035, f"{v:.2f}", ha="center",
            fontsize=12, fontweight="bold")
ax.set_ylabel("alert precision (new-bloom forecasts)")
ax.set_ylim(0, 0.95)
ax.set_title("Same model recipe, two monitoring cadences\n"
             "(95% clustered-bootstrap CIs; LIS h21 station-day, Narragansett h7 onset-only)")
fig.tight_layout()
fig.savefig("figures/nar_fig4_precision_comparison.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("wrote 4 figures to figures/")
