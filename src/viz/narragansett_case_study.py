"""Case-study figure: the GB onset model forecasting real 2023 blooms.
Fits GB tier A (train <= 2020), scores 2023, picks 3 stations/events where an
alert (P >= 0.50 on a day with chl <= 10) preceded a bloom onset, plus one
false alarm, and draws chlorophyll + probability panels for each.
Output: figures/nar_fig7_case_study.png. Agg, dpi 150.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BLUE, ORANGE, GRAY, RED = "#2a78d6", "#eb6834", "#8a8f98", "#d03b3b"
TIER_A = ['chl','chl_lag1','chl_lag2','chl_lag3','chl_lag4','chl_roll3_mean',
          'chl_roll6_mean','chl_roll9_mean','chl_roll14_mean','chl_roll21_mean',
          'chl_trend','chl_anomaly','chl_climatology','do','do_lag1','temp',
          'temp_lag1','sal','sal_lag1','sal_lag2','sal_lag3','sal_lag4','month']
TSTAR = 0.50
df = pd.read_csv("data/narragansett_daily_features.csv", parse_dates=["date"])
df["year"] = df.date.dt.year
lab = df.dropna(subset=["bloom_fwd"])
train = lab[lab.year <= 2020]
med = train[TIER_A].median(numeric_only=True)
m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                   min_samples_leaf=50, l2_regularization=1.0,
                                   random_state=42, class_weight="balanced")
m.fit(train[TIER_A].fillna(med).values, train.bloom_fwd.astype(int).values)
t23 = df[df.year == 2023].copy()
t23["p"] = m.predict_proba(t23[TIER_A].fillna(med).values)[:, 1]

# find onset events: first day > 10 after >= 5 quiet days; lead = days from first
# alert (p >= t*, chl <= 10) within the prior 7 days to onset
events = []
for st, g in t23.sort_values("date").groupby("station"):
    g = g.reset_index(drop=True)
    above = (g.chl > 10).values
    for i in range(5, len(g)):
        if above[i] and not above[i-5:i].any() and (g.date[i]-g.date[i-5]).days == 5:
            win = g.iloc[max(0, i-7):i]
            alerts = win[(win.p >= TSTAR) & (win.chl <= 10)]
            lead = (g.date[i] - alerts.date.min()).days if len(alerts) else None
            quiet = g.iloc[max(0, i-14):max(0, i-7)]
            rose = bool(len(quiet)) and (quiet.p < TSTAR).any()
            events.append((st, g.date[i], lead, g.chl[i], rose))

ev = pd.DataFrame(events, columns=["station", "onset", "lead", "peak", "rose"])
hits = ev.dropna(subset=["lead"]).sort_values("lead", ascending=False)
print(f"2023 clean onsets: {len(ev)}, alerted in prior 7 d: {len(hits)} "
      f"({len(hits)/len(ev):.0%}); median lead {hits.lead.median():.0f} d")

# pick 3 hits at different stations with lead >= 3, and one false alarm
picks = []
for _, r in hits[(hits.lead >= 3) & hits.rose].iterrows():
    if r.station not in [p[0] for p in picks]:
        picks.append((r.station, r.onset, f"alert {int(r.lead)} d before onset"))
    if len(picks) == 3: break
# false alarm: alert day (chl<=10, p>=t*) with no chl>10 in following 7 d
fa = t23[(t23.p >= TSTAR) & (t23.chl <= 10) & (t23.bloom_fwd == 0)].copy()
fa["q5"] = fa.apply(lambda r: (t23[(t23.station == r.station)
                    & (t23.date < r.date) & (t23.date >= r.date - pd.Timedelta(days=5))].chl <= 10).all(), axis=1)
fa = fa[fa.q5]
if len(fa):
    r = fa.sort_values("p", ascending=False).iloc[0]
    picks.append((r.station, r.date, "false alarm (no bloom followed)"))

fig, axes = plt.subplots(len(picks), 1, figsize=(9, 2.6 * len(picks)))
for ax, (st, d0, label) in zip(np.atleast_1d(axes), picks):
    g = t23[(t23.station == st) & (t23.date >= d0 - pd.Timedelta(days=21))
            & (t23.date <= d0 + pd.Timedelta(days=10))].sort_values("date")
    x = (g.date - d0).dt.days
    ax.plot(x, g.chl, color=BLUE, lw=2, marker="o", ms=3, label="chlorophyll (µg/L)")
    ax.axhline(10, color=ORANGE, ls=":", lw=1)
    ax.set_ylabel("chl (µg/L)", color=BLUE)
    ax.set_ylim(0, max(15, g.chl.max() * 1.15))
    ax2 = ax.twinx()   # probability on its own axis, clearly labeled, no data overlay
    ax2.fill_between(x, 0, g.p, color=GRAY, alpha=0.25)
    ax2.plot(x, g.p, color="black", lw=1.2)
    ax2.axhline(TSTAR, color="black", ls="--", lw=0.8)
    ax2.set_ylim(0, 1); ax2.set_ylabel("bloom probability")
    first_alert = g[(g.p >= TSTAR) & (g.chl <= 10) & (x <= 0) & (x >= -7)]
    if len(first_alert):
        xa = (first_alert.date.iloc[0] - d0).days
        ax.axvline(xa, color=RED, lw=1.5)
        ax.text(xa + 0.3, ax.get_ylim()[1] * 0.9, "ALERT", color=RED, fontsize=9, fontweight="bold")
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_title(f"Station {st}, {d0.date()} — {label}", loc="left", fontsize=10)
axes[-1].set_xlabel("days relative to bloom onset (chl > 10 µg/L)")
fig.suptitle("Narragansett 2023: the model's bloom probability ahead of real blooms", y=1.0)
fig.tight_layout()
fig.savefig("figures/nar_fig7_case_study.png", dpi=150, bbox_inches="tight")
print("wrote figures/nar_fig7_case_study.png")
