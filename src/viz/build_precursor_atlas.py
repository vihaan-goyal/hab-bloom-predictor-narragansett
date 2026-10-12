"""
build_precursor_atlas.py
------------------------
Export the per-event pre-onset windows that the bloom precursor atlas renders.

For each selected bloom episode this writes, for every feature, the values
observed in the 120 days before onset -- keyed by days-before-onset so all
events can be stacked on one axis.

The one thing this file is careful about: it tags every feature with its true
cadence. In-situ water-quality features are sampled on station visits with a
median gap of 21 days, so a typical event carries 3-4 readings across the whole
120-day window; the daily layers (wind, gust, river discharge, astronomy) and
the ~3.5-day satellite kd490 series are genuinely dense. The consumer must draw
the sparse ones as unconnected points. A line through four dots would read as a
trajectory that was never measured.

Values are exported twice: raw units, and standardized against the station's own
day-of-year climatology so that features with different units can be compared on
one axis.

Run from repo root:
    python src/viz/build_precursor_atlas.py
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "models"))
from bloom_precursor_events import (  # noqa: E402
    BLOOM_THRESHOLD, LOOKBACK_DAYS, add_doy_climatology_z, build_episode_index,
    load_daily_frames, select_atlas_events)
from locked_pipeline import FEATURES_ALL, load_locked_dataframe  # noqa: E402

OUT_JSON = "data/precursor_atlas.json"

# (key, label, units) grouped by what the variable physically is.
# Near-duplicates are kept adjacent so their redundancy is visible on screen.
INSITU_GROUPS = [
    ("Biology / bloom state", [
        ("Chlorophyll", "Chlorophyll a", "ug/L"),
        ("chl_anomaly", "Chl anomaly", "ug/L"),
        ("chl_climatology", "Chl climatology", "ug/L"),
        ("chl_trend", "Chl trend (slope)", "ug/L per visit"),
        ("chl_lag1", "Chl lag 1 visit", "ug/L"),
        ("chl_lag2", "Chl lag 2 visits", "ug/L"),
        ("chl_lag3", "Chl lag 3 visits", "ug/L"),
        ("chl_lag4", "Chl lag 4 visits", "ug/L"),
        ("chl_roll3_mean", "Chl rolling mean, 3 visits", "ug/L"),
        ("chl_roll6_mean", "Chl rolling mean, 6 visits", "ug/L"),
        ("chl_roll9_mean", "Chl rolling mean, 9 visits", "ug/L"),
        ("chl_roll14_mean", "Chl rolling mean, 14 visits", "ug/L"),
        ("chl_roll21_mean", "Chl rolling mean, 21 visits", "ug/L"),
        ("neighbor_chl3_mean", "Neighbour chl, 3 nearest", "ug/L"),
        ("neighbor_chl3_lag1", "Neighbour chl, 3 nearest, lag 1", "ug/L"),
    ]),
    ("Physics", [
        ("sea_water_temperature", "Water temperature", "degC"),
        ("temp_lag1", "Temperature lag 1 visit", "degC"),
        ("sea_water_salinity", "Salinity", "PSU"),
        ("sal_lag1", "Salinity lag 1 visit", "PSU"),
        ("sal_lag2", "Salinity lag 2 visits", "PSU"),
        ("sal_lag3", "Salinity lag 3 visits", "PSU"),
        ("sal_lag4", "Salinity lag 4 visits", "PSU"),
        ("tidal_gt_anom", "Tidal great-diurnal anomaly", "m"),
        ("tidal_msl_anom", "Mean sea level anomaly", "m"),
    ]),
    ("Chemistry", [
        ("oxygen_concentration_in_sea_water", "Dissolved oxygen", "mg/L"),
        ("do_lag1", "Dissolved oxygen lag 1 visit", "mg/L"),
        ("percent_saturation", "Oxygen saturation", "%"),
        ("nox_lag2", "Nitrate+nitrite lag 2 visits", "mg/L"),
        ("dip_lag2", "Dissolved inorganic P lag 2 visits", "mg/L"),
        ("dip_change", "DIP change", "mg/L per visit"),
        ("dip_x_month", "DIP x month interaction", "-"),
    ]),
    ("Context", [
        ("month", "Month of year", "1-12"),
        ("latitude_x", "Station latitude", "degN"),
        ("longitude_x", "Station longitude", "degE"),
    ]),
]

DAILY_GROUPS = [
    ("Meteorology (daily)", [
        ("max_gust_3d", "Max gust, 3-day", "m/s"),
        ("max_gust_ms", "Max gust, daily", "m/s"),
        ("wind_stress_mag", "Wind stress magnitude", "N/m2"),
        ("wsm_roll3d", "Wind stress magnitude, 3-day", "N/m2"),
        ("wsm_roll7d", "Wind stress magnitude, 7-day", "N/m2"),
        ("wind_stress_curl", "Wind stress curl", "N/m3"),
        ("wsc_roll3d", "Wind stress curl, 3-day", "N/m3"),
        ("wsc_roll7d", "Wind stress curl, 7-day", "N/m3"),
    ]),
    ("Hydrology (daily)", [
        ("connecticut_river_discharge_cfs", "Connecticut R. discharge", "cfs"),
        ("thames_river_discharge_cfs", "Thames R. discharge", "cfs"),
        ("housatonic_river_discharge_cfs", "Housatonic R. discharge", "cfs"),
    ]),
    ("Light and astronomy (daily)", [
        ("photoperiod_hrs", "Photoperiod", "hours"),
        ("moon_illum_frac", "Moon illuminated fraction", "0-1"),
        ("spring_neap", "Spring-neap phase", "-"),
    ]),
    ("Water clarity, satellite (~3.5-day)", [
        ("kd490", "Diffuse attenuation Kd(490)", "1/m"),
        ("kd490_roll7", "Kd(490), 7-day", "1/m"),
        ("kd490_roll14", "Kd(490), 14-day", "1/m"),
        ("kd490_anom", "Kd(490) anomaly", "1/m"),
    ]),
]

STATION_DAILY = {"kd490", "kd490_roll7", "kd490_roll14", "kd490_anom"}


def _round(values, digits=4):
    """JSON-safe list: NaN/inf become None, floats are trimmed."""
    out = []
    for v in values:
        if v is None or not np.isfinite(v):
            out.append(None)
        else:
            out.append(round(float(v), digits))
    return out


def _series(frame, onset, key, zkey):
    """Reshape one feature's window into days-before-onset, raw, and z."""
    t = (frame["date"] - onset).dt.days.to_numpy()
    v = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
    z = (pd.to_numeric(frame[zkey], errors="coerce").to_numpy(dtype=float)
         if zkey in frame.columns else np.full(len(frame), np.nan))
    keep = np.isfinite(v)
    if not keep.any():
        return None
    return {"t": [int(x) for x in t[keep]],
            "v": _round(v[keep]), "z": _round(z[keep])}


def build_atlas(lookback=LOOKBACK_DAYS, n_per_regime=6):
    df = load_locked_dataframe()
    episodes = build_episode_index(df, lookback=lookback)
    events = select_atlas_events(episodes, n_per_regime=n_per_regime)

    insitu_keys = [k for _, feats in INSITU_GROUPS for k, _, _ in feats
                   if k in df.columns]
    print(f"Standardizing {len(insitu_keys)} in-situ features "
          f"against station x day-of-year climatology...")
    df = pd.concat([df, add_doy_climatology_z(df, insitu_keys)], axis=1)

    regional, station_daily = load_daily_frames()
    reg_keys = [k for _, feats in DAILY_GROUPS for k, _, _ in feats
                if k in regional.columns]
    sta_keys = [k for _, feats in DAILY_GROUPS for k, _, _ in feats
                if k in station_daily.columns]
    # regional forcing has no station dimension, so its climatology is network-wide
    regional = pd.concat(
        [regional, add_doy_climatology_z(regional, reg_keys, group_col=None)],
        axis=1)
    station_daily = pd.concat(
        [station_daily, add_doy_climatology_z(station_daily, sta_keys)], axis=1)

    out_events = []
    for _, ev in events.iterrows():
        onset, station = ev["onset"], str(ev["station_name"])
        start = onset - pd.Timedelta(days=lookback)

        win_is = df[(df["station_name"].astype(str) == station)
                    & (df["date"] >= start) & (df["date"] <= onset)]
        win_reg = regional[(regional["date"] >= start)
                           & (regional["date"] <= onset)]
        win_sta = station_daily[(station_daily["station_name"] == station)
                                & (station_daily["date"] >= start)
                                & (station_daily["date"] <= onset)]

        series = {}
        for key in insitu_keys:
            s = _series(win_is, onset, key, f"{key}_z")
            if s:
                series[key] = s
        for key in reg_keys:
            s = _series(win_reg, onset, key, f"{key}_z")
            if s:
                series[key] = s
        for key in sta_keys:
            s = _series(win_sta, onset, key, f"{key}_z")
            if s:
                series[key] = s

        out_events.append({
            "id": f"{station}_{onset.date()}",
            "station": station,
            "onset": str(onset.date()),
            "regime": ev["regime"],
            "peak_chl": round(float(ev["peak_chl"]), 2),
            "onset_chl": round(float(ev["onset_chl"]), 2),
            "prev_chl": round(float(ev["prev_chl"]), 2),
            "gap_to_prev_days": int(ev["gap_to_prev_days"]),
            "n_exceedances": int(ev["n_exceedances"]),
            "sustained": bool(ev["sustained"]),
            "n_insitu_in_window": int(len(win_is)),
            "series": series,
        })
        print(f"  {station:>3} {onset.date()} {ev['regime']:<6} "
              f"peak {ev['peak_chl']:5.1f}  in-situ readings in 120 d: "
              f"{len(win_is)}")

    groups = ([{"name": name, "cadence": "insitu",
                "features": [{"key": k, "label": lab, "units": u,
                              "inModel": k in FEATURES_ALL}
                             for k, lab, u in feats if k in df.columns]}
               for name, feats in INSITU_GROUPS]
              + [{"name": name,
                  "cadence": ("satellite"
                              if any(k in STATION_DAILY for k, _, _ in feats)
                              else "daily"),
                  "features": [{"key": k, "label": lab, "units": u,
                                "inModel": k in FEATURES_ALL}
                               for k, lab, u in feats
                               if k in reg_keys or k in sta_keys]}
                 for name, feats in DAILY_GROUPS])
    groups = [g for g in groups if g["features"]]

    clean = episodes[episodes["clean_onset"]]
    meta = {
        "lookbackDays": lookback,
        "bloomThreshold": BLOOM_THRESHOLD,
        "horizonDays": 21,
        "nEpisodes": int(len(episodes)),
        "nCleanOnsets": int(len(clean)),
        "cleanByRegime": {k: int(v) for k, v
                          in clean["regime"].value_counts().items()},
        "medianGapDays": float(clean["gap_to_prev_days"].median()),
        "medianPrior30": float(clean["n_prior_30"].median()),
        "meanPrior30": round(float(clean["n_prior_30"].mean()), 2),
        "medianPrior90": float(clean["n_prior_90"].median()),
        "onsetMonthCounts": {int(k): int(v) for k, v in
                             clean["onset"].dt.month.value_counts()
                             .sort_index().items()},
        "nModelFeatures": len(FEATURES_ALL),
    }
    return {"meta": meta, "featureGroups": groups, "events": out_events}


DATA_TOKEN = "/*__ATLAS_DATA__*/null"
SE_CSV = "data/superposed_epoch.csv"
SE_NULL_CSV = "data/superposed_epoch_nullcontrol.csv"
PONR_CSV = "data/point_of_no_return.csv"
PONR_ROWS_CSV = "data/ponr_rows.csv"


def attach_analysis(atlas, se_csv=SE_CSV, se_null_csv=SE_NULL_CSV,
                    ponr_csv=PONR_CSV, ponr_rows_csv=PONR_ROWS_CSV):
    """Fold the composite and point-of-no-return results into the atlas payload.

    Both are optional: the atlas renders without them, so the page can ship
    before the pooled analyses finish.
    """
    def clean(frame):
        return frame.replace({np.nan: None}).to_dict("records")

    # Constant-within-station features have an identically zero difference, so
    # the bootstrap degenerates and flags them; drop them from both the results
    # and the control counts rather than showing a row of empty "significance".
    degenerate = ["latitude_x", "longitude_x", "month"]

    if os.path.exists(se_csv):
        se = pd.read_csv(se_csv)
        se = se[~se["feature"].isin(degenerate)]
        atlas["composite"] = {
            "rows": clean(se.round(4)),
            "leadBins": sorted(se["lead_lo"].unique().tolist()),
            "nSignificant": {r: int(se[(se["regime"] == r)
                                       & se["significant"]].shape[0])
                             for r in se["regime"].unique()},
            "nTested": {r: int(se[se["regime"] == r].shape[0])
                        for r in se["regime"].unique()},
        }
        if os.path.exists(se_null_csv):
            nc = pd.read_csv(se_null_csv)
            nc = nc[~nc["feature"].isin(degenerate)]
            atlas["composite"]["nullControl"] = {
                "nSignificant": {r: int(nc[(nc["regime"] == r)
                                           & nc["significant"]].shape[0])
                                 for r in nc["regime"].unique()},
                "nTested": {r: int(nc[nc["regime"] == r].shape[0])
                            for r in nc["regime"].unique()},
            }

    if os.path.exists(ponr_csv):
        atlas["ponr"] = {"summary": clean(pd.read_csv(ponr_csv).round(4))}
        if os.path.exists(ponr_rows_csv):
            rows = pd.read_csv(ponr_rows_csv)
            rows = rows[rows["t_star"] == rows["t_star"].max()]
            risk = rows["emp_risk"].dropna()
            hist, edges = np.histogram(risk, bins=20, range=(0, 1))
            atlas["ponr"]["empirical"] = {
                "n": int(len(risk)),
                "hist": hist.tolist(),
                "edges": [round(float(e), 3) for e in edges],
                "max": round(float(risk.max()), 3) if len(risk) else None,
                "p50": round(float(risk.median()), 3) if len(risk) else None,
                "p95": round(float(risk.quantile(0.95)), 3) if len(risk) else None,
                "p99": round(float(risk.quantile(0.99)), 3) if len(risk) else None,
            }
    return atlas


def render_html(atlas, template_path, out_path):
    """Inline the atlas JSON into the standalone HTML page."""
    with open(template_path, encoding="utf-8") as fh:
        html = fh.read()
    if DATA_TOKEN not in html:
        raise ValueError(f"{template_path} has no {DATA_TOKEN} placeholder")
    html = html.replace(DATA_TOKEN, json.dumps(atlas, separators=(",", ":")))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


FIG_DIR = "figures/precursors"


def write_figures(atlas, out_dir=FIG_DIR, dpi=150):
    """Static counterparts to the interactive page, in the repo's house style.

    Four figures, each carrying one of the findings: how thin the pre-onset
    record is, what the pooled composite shows per regime, how the model's
    point of no return depends on which levers are allowed, and how far the
    analogue-based risk actually reaches.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    C = {"winter": "#2a78d6", "summer": "#eb6834",
         "ink": "#1D3557", "muted": "#898781", "grid": "#E8EDF2",
         "crit": "#d03b3b"}
    written = []

    # --- Fig 1: sampling sparsity, the constraint behind everything else ----
    events = atlas["events"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.35, 1]})
    for row, ev in enumerate(events):
        s = ev["series"].get("Chlorophyll")
        if not s:
            continue
        ax1.scatter(s["t"], [row] * len(s["t"]), s=44,
                    color=C[ev["regime"]], zorder=3,
                    edgecolor="white", linewidth=1.1)
    ax1.set_yticks(range(len(events)))
    ax1.set_yticklabels([f"{e['station']}  {e['onset']}" for e in events],
                        fontsize=8.5, family="monospace")
    ax1.axvline(0, color=C["crit"], lw=1.4, zorder=2)
    ax1.set_xlabel("Days before bloom onset")
    ax1.set_title("Every in-situ reading in the 120 days before onset",
                  fontsize=11, fontweight="bold")
    ax1.grid(axis="x", color=C["grid"], lw=0.8)
    ax1.set_axisbelow(True)
    ax1.set_xlim(-atlas["meta"]["lookbackDays"] - 4, 8)

    meta = atlas["meta"]
    months = [meta["onsetMonthCounts"].get(str(m),
              meta["onsetMonthCounts"].get(m, 0)) for m in range(1, 13)]
    cols = [C["winter"] if m <= 4 else C["summer"] if 6 <= m <= 9
            else C["muted"] for m in range(1, 13)]
    ax2.bar(range(1, 13), months, color=cols)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O",
                         "N", "D"])
    ax2.set_xlabel("Month of onset")
    ax2.set_ylabel("Clean onsets")
    ax2.set_title(f"Onsets are bimodal ({meta['nCleanOnsets']:,} clean onsets)",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="y", color=C["grid"], lw=0.8)
    ax2.set_axisbelow(True)
    fig.suptitle("A typical onset has one reading in the month before it "
                 f"(median gap between visits: {meta['medianGapDays']:.0f} days)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_precursor_sampling.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # --- Fig 2: the pooled composite, one panel per regime -----------------
    comp = atlas.get("composite")
    if comp:
        rows = pd.DataFrame(comp["rows"])
        for regime in ("winter", "summer"):
            sub = rows[rows["regime"] == regime]
            sig = sub[sub["significant"] == True]  # noqa: E712
            feats = [f for f in sig["feature"].unique()]
            if not feats:
                continue
            order = {"insitu": 0, "satellite": 1, "daily": 2}
            cad = {f: sub[sub["feature"] == f]["cadence"].iloc[0] for f in feats}
            feats.sort(key=lambda f: (order[cad[f]], f))
            bins = sorted(sub["lead_lo"].unique(), reverse=True)
            grid = np.full((len(feats), len(bins)), np.nan)
            for i, f in enumerate(feats):
                for j, b in enumerate(bins):
                    r = sub[(sub["feature"] == f) & (sub["lead_lo"] == b)]
                    if len(r) and bool(r["significant"].iloc[0]):
                        grid[i, j] = r["diff"].iloc[0]
            lim = np.nanmax(np.abs(grid)) or 1.0
            fig, ax = plt.subplots(figsize=(9, max(4, 0.29 * len(feats) + 1.6)))
            im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels([f"{b}-{b + 15}" for b in bins], fontsize=8.5)
            ax.set_yticks(range(len(feats)))
            ax.set_yticklabels(feats, fontsize=7.5, family="monospace")
            ax.set_xlabel("Days before onset")
            ax.set_title(f"Superposed epoch composite, {regime} regime\n"
                         f"{comp['nSignificant'][regime]} of "
                         f"{comp['nTested'][regime]} feature-bins separate from "
                         "a matched null", fontsize=11, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("event minus null (sigma)", fontsize=9)
            fig.tight_layout()
            p = os.path.join(out_dir, f"fig_superposed_epoch_{regime}.png")
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(p)

    # --- Fig 3: point of no return -----------------------------------------
    ponr = atlas.get("ponr")
    if ponr:
        summ = pd.DataFrame(ponr["summary"])
        box = summ[(summ["budget"] == "box")
                   & (summ["t_star"] == summ["t_star"].max())]
        sets = list(dict.fromkeys(box["modifiable_set"]))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
        w = 0.38
        x = np.arange(len(sets))
        for k, regime in enumerate(("winter", "summer")):
            vals = [float(box[(box["modifiable_set"] == s)
                              & (box["regime"] == regime)]
                          ["frac_events_with_ponr"].fillna(0).iloc[0]) * 100
                    if len(box[(box["modifiable_set"] == s)
                               & (box["regime"] == regime)]) else 0
                    for s in sets]
            ax1.bar(x + (k - 0.5) * w, vals, w, color=C[regime],
                    label=regime.capitalize())
        ax1.set_xticks(x)
        # spell out what each rung adds; "M3" alone is meaningless standalone
        ax1.set_xticklabels(
            [s.split()[0] + "\n" + " ".join(s.split()[1:]).replace("+", "+ ")
             for s in sets], fontsize=9)
        ax1.set_ylabel("% of events reaching an irreversible state")
        ax1.set_title("Adding levers collapses the model's\npoint of no return",
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", color=C["grid"], lw=0.8)
        ax1.set_axisbelow(True)

        emp = ponr.get("empirical", {})
        if emp.get("hist"):
            edges = np.array(emp["edges"])
            ax2.bar(edges[:-1], emp["hist"], width=np.diff(edges),
                    align="edge", color=C["summer"])
            ax2.axvline(0.9, color=C["crit"], lw=1.6)
            ax2.text(0.885, max(emp["hist"]) * 0.92, "0.90 bar\nnever reached",
                     ha="right", fontsize=9, color=C["crit"])
            ax2.set_xlim(0, 1)
            ax2.set_xlabel("Fraction of 50 matched analogues that bloomed")
            ax2.set_ylabel("Pre-onset observations")
            ax2.set_title(f"Model-free test finds nothing\n(highest anywhere: "
                          f"{emp['max']}, n={emp['n']})",
                          fontsize=11, fontweight="bold")
            ax2.grid(axis="y", color=C["grid"], lw=0.8)
            ax2.set_axisbelow(True)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig_point_of_no_return.png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--n-per-regime", type=int, default=6)
    ap.add_argument("--html-template",
                    help="HTML template containing the atlas data placeholder")
    ap.add_argument("--html-out", help="where to write the rendered page")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the static PNG figures")
    args = ap.parse_args()

    atlas = build_atlas(lookback=args.lookback, n_per_regime=args.n_per_regime)
    atlas = attach_analysis(atlas)
    for section in ("composite", "ponr"):
        print(f"  {section}: {'attached' if section in atlas else 'not available yet'}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(atlas, fh, separators=(",", ":"))

    n_series = sum(len(e["series"]) for e in atlas["events"])
    print(f"\n{len(atlas['events'])} events, {n_series} feature series -> "
          f"{args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")

    if args.html_template and args.html_out:
        render_html(atlas, args.html_template, args.html_out)
        print(f"page -> {args.html_out} "
              f"({os.path.getsize(args.html_out) / 1024:.0f} KB)")

    if not args.no_figures:
        for path in write_figures(atlas):
            print(f"figure -> {path}")


if __name__ == "__main__":
    main()
