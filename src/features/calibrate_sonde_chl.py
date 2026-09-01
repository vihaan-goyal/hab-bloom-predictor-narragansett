"""
W5 sonde vs lab chlorophyll calibration (Narragansett Bay, NBFSMN 2006-2022).

QUESTION
Does the YSI sonde fluorescence chlorophyll (daily mean in
data/narragansett_daily_features.csv, column `chl`) agree with lab-extracted
chlorophyll-a from grab samples taken at the same buoys, and what sonde
daily-mean value corresponds to the lab 10 ug/L bloom threshold?

INPUTS (all under data/raw/narragansett/extracted/)
  NBFSMN 2006/Field Samples/2006.chl.compare.xls      sheet 'Data', per-station
        "Field Samples" columns aligned to 15-min sonde rows (values verified
        identical to the raw lab sheet Buoys_2006_Chl.xls, surface reps)
  NBFSMN.2007/2007_Compiled Chlorophyll field sample results.xls
        one sheet per station (QP, MV, NP, PP, MHB, CP, GB), 'Chl a (ug/L)'
  NBFSMN 2010/Other Data/Buoys_CHL Samples. GSO Stations2010.xls
  nbfsmn13/2013BuoyChla.xlsx ... nbfsmn.2021/Buoy Chlorophyll 2021.xlsx
        GSO fluorometer lab sheets: 'Sample Date', 'Sample' (station code),
        '[Chla] ug/l'.  2015 is an empty template; 2022 has raw Fo/Fa only
        (no Fs / r factors, chl column blank) -> both skipped, not imputed.
  nbfsmn.2022/NBC Fixed Site_Chlorophyll Grabs_2022.xlsx
        NBC LIMS grabs at Bullock Reach (B4) and Phillipsdale (F4), with
        Depth Stratum; surface only kept.
  data/narragansett_daily_features.csv   station, date, chl (sonde daily mean)

MATCHING
Lab replicates are averaged per (station, sample date).  A lab station-day is
matched to the sonde daily mean of the same station on the same calendar date.
Bottom samples (codes ending in B, depth 'Bottom'/'CM'/'B') are dropped.
Rows whose lab Notes say suspect/leaked/dropped are dropped.
Tributary stations with no sonde (Cole River CR*, Taunton River TR*) are
unmapped and reported as such.  'UB'/'Upper Bay'/'Winter station' is matched to
the upper-bay winter sonde (B3w, falling back to B3W / UB2015 / B3a when B3w
has no value that day).

OUTPUTS
  data/sonde_lab_calibration.csv           matched pairs
  data/sonde_lab_calibration_summary.csv   fit numbers + calibrated threshold
  figures/nar_fig6_calibration.png

Run from repo root with the base anaconda python:
  ~/anaconda3/python.exe src/features/calibrate_sonde_chl.py
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("data/raw/narragansett/extracted")
FEATURES = Path("data/narragansett_daily_features.csv")
OUT_PAIRS = Path("data/sonde_lab_calibration.csv")
OUT_SUMMARY = Path("data/sonde_lab_calibration_summary.csv")
OUT_FIG = Path("figures/nar_fig6_calibration.png")

LAB_THRESHOLD = 10.0
MIN_N = 20
N_BOOT = 5000
RNG = np.random.default_rng(20240901)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8f98"

# --------------------------------------------------------------------------
# Station name / code -> sonde station id.  Order matters (first match wins).
# --------------------------------------------------------------------------
STATION_PATTERNS = [
    (r"^(CPB|NPB|PPB|MVB|QPB|MHB2|CRB|TRB|SRB|GBB)\b", "BOTTOM"),
    (r"CONIMICUT|^CP\b|^CPS\b", "B3"),
    (r"N\.?\s*PRU|NORTH\s*PRU|^NP\b|^NPS\b", "B2"),
    (r"BULLOCK|^BR\b|^BRS\b", "B4"),
    (r"M(T|OUNT)\.?\s*VIEW|^MV\b|^MVS\b", "B6"),
    (r"QUONSET|^QP\b|^QPS\b|^QO\b", "B7"),
    (r"MHW|MT\.?\s*HOPE\s*W|MOUNT\s*HOPE\s*W", "B12w"),
    (r"M(T|OUNT)\.?\s*HOPE|^MH\b|^MHB\b|^MHS\b", "B12"),
    (r"POPPA|^PP\b|^PPS\b", "B13"),
    (r"SALLY|^SR\b|^SRS\b", "B14"),
    (r"GREENWICH|^G\s*BAY|^GBAY|^GB\b|^GBS\b|^G\.\s*BAY", "F5"),
    (r"PHILLIPSDALE|^PD\b", "F4"),
    (r"GSO\s*DOCK|^GD\b|^GSO\b", "F7"),
    (r"T-?\s*WHARF|^TW\b", "F3"),
    (r"POTTER", "F6"),
    (r"UPPER\s*BAY|WINTER|^UB\b|^UB\d|^WIN\b|^UPB", "B3w"),
]
# fall-back sonde ids for the upper-bay winter station
UB_FALLBACK = ["B3w", "B3W", "UB2015", "B3a"]
BAD_NOTE = re.compile(r"SUSPECT|LEAK|DROPPED|NOT A VALID|FELL ON|WRONG LEVEL", re.I)


def map_station(name):
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).upper().strip()
    s = re.sub(r"\s+", " ", s)
    for pat, sid in STATION_PATTERNS:
        if re.search(pat, s):
            return sid
    return None


# --------------------------------------------------------------------------
# Parsers.  Each returns a DataFrame with columns
#   station, date, lab_chl, source
# --------------------------------------------------------------------------
def parse_gso_sheet(path, source):
    """GSO fluorometer lab sheet (2010, 2013-2021 format)."""
    eng = "calamine" if path.suffix.lower() == ".xls" else None
    df = pd.read_excel(path, header=1, engine=eng)
    df.columns = [str(c).strip() for c in df.columns]
    chl_col = [c for c in df.columns if "Chla" in c]
    if not chl_col or "Sample" not in df.columns:
        print(f"  [skip] {path.name}: no Chla column")
        return pd.DataFrame()
    chl_col = chl_col[0]
    df["date"] = pd.to_datetime(df["Sample Date"], errors="coerce")
    df["lab_chl"] = pd.to_numeric(df[chl_col], errors="coerce")
    # replicate rows sometimes leave 'Sample' blank under a merged cell:
    # forward-fill only within the same sample date
    df["Sample"] = df.groupby(df["date"].dt.date, dropna=True)["Sample"].ffill()
    df = df[df["date"].notna() & df["lab_chl"].notna() & (df["lab_chl"] > 0)]
    if "Notes" in df.columns:
        bad = df["Notes"].astype(str).str.contains(BAD_NOTE)
        n_bad = int(bad.sum())
        df = df[~bad]
    else:
        n_bad = 0
    df["station"] = df["Sample"].map(map_station)
    unmapped = sorted(df.loc[df["station"].isna(), "Sample"].astype(str).str.strip().unique())
    n_bottom = int((df["station"] == "BOTTOM").sum())
    df = df[df["station"].notna() & (df["station"] != "BOTTOM")]
    print(f"  {path.name}: {len(df)} usable reps, {n_bottom} bottom dropped, "
          f"{n_bad} flagged dropped, unmapped codes: {unmapped}")
    out = df[["station", "date", "lab_chl"]].copy()
    out["source"] = source
    return out


def parse_2006_compare(path):
    df = pd.read_excel(path, sheet_name="Data", header=None, engine="calamine")
    hdr = df.iloc[4]
    body = df.iloc[7:]
    date = pd.to_datetime(body[0], errors="coerce")
    rows = []
    # Field-sample columns are the third column of each station block
    for col in range(3, df.shape[1]):
        label = str(df.iloc[5, col]).strip().lower()
        if not label.startswith("field"):
            continue
        st = map_station(hdr[col] if pd.notna(hdr[col]) else hdr[col - 2])
        vals = pd.to_numeric(body[col], errors="coerce")
        m = vals.notna()
        rows.append(pd.DataFrame({"station": st, "date": date[m].dt.normalize(),
                                  "lab_chl": vals[m]}))
    out = pd.concat(rows, ignore_index=True)
    out = out[out["station"].notna() & (out["lab_chl"] > 0)]
    out["source"] = "2006.chl.compare.xls"
    print(f"  2006.chl.compare.xls: {len(out)} field-sample reps, stations "
          f"{sorted(out.station.unique())}")
    return out


def parse_2007(path):
    xl = pd.ExcelFile(path, engine="calamine")
    rows = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=4)
        df.columns = [str(c).strip() for c in df.columns]
        if "Chl a (ug/L)" not in df.columns:
            continue
        d = pd.DataFrame({
            "station": map_station(sheet),
            "date": pd.to_datetime(df["Date"], errors="coerce"),
            "lab_chl": pd.to_numeric(df["Chl a (ug/L)"], errors="coerce"),
        })
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out = out[out["date"].notna() & out["lab_chl"].notna() & (out["lab_chl"] > 0)
              & out["station"].notna()]
    out["source"] = "2007_Compiled Chlorophyll field sample results.xls"
    print(f"  2007 compiled: {len(out)} reps, stations {sorted(out.station.unique())}")
    return out


def parse_2022_nbc(path):
    df = pd.read_excel(path, header=3)
    df.columns = [str(c).strip() for c in df.columns]
    chl_col = [c for c in df.columns if c.startswith("Chlorophyll a")][0]
    df = df[df["Depth Stratum"].astype(str).str.strip().str.lower() == "surface"]
    out = pd.DataFrame({
        "station": df["Station"].map(map_station),
        "date": pd.to_datetime(df["Collection Date"], errors="coerce"),
        "lab_chl": pd.to_numeric(df[chl_col], errors="coerce"),
    })
    out = out[out["date"].notna() & out["lab_chl"].notna() & out["station"].notna()]
    out["source"] = "NBC Fixed Site_Chlorophyll Grabs_2022.xlsx"
    print(f"  NBC 2022 grabs: {len(out)} surface reps, stations {sorted(out.station.unique())}")
    return out


def load_lab():
    print("Parsing lab chlorophyll files ...")
    parts = [
        parse_2006_compare(ROOT / "NBFSMN 2006/Field Samples/2006.chl.compare.xls"),
        parse_2007(ROOT / "NBFSMN.2007/2007_Compiled Chlorophyll field sample results.xls"),
    ]
    gso_files = [
        "NBFSMN 2010/Other Data/Buoys_CHL Samples. GSO Stations2010.xls",
        "nbfsmn13/2013BuoyChla.xlsx",
        "nbfsmn.2014/2014BuoyChla.xlsx",
        "nbfsmn.2015/2015BuoyChla.xlsx",
        "nbfsmn.2016/2016BuoyChla.xlsx",
        "nbfsmn.2017/2017BuoyChla.xlsx",
        "nbfsmn 2018/2018BuoyChla.xlsx",
        "nbfsmn2019/2019BuoyChla.xlsx",
        "nbfsmn.2020/2020BuoyChl.xlsx",
        "nbfsmn.2021/Buoy Chlorophyll 2021.xlsx",
        "nbfsmn.2022/Buoy Chl 2022.xlsx",
    ]
    for f in gso_files:
        parts.append(parse_gso_sheet(ROOT / f, Path(f).name))
    parts.append(parse_2022_nbc(ROOT / "nbfsmn.2022/NBC Fixed Site_Chlorophyll Grabs_2022.xlsx"))
    lab = pd.concat([p for p in parts if len(p)], ignore_index=True)
    lab["date"] = lab["date"].dt.normalize()
    # replicates -> one value per station-day (mean), keep replicate spread
    agg = (lab.groupby(["station", "date"])
              .agg(lab_chl=("lab_chl", "mean"), lab_sd=("lab_chl", "std"),
                   n_reps=("lab_chl", "size"), source=("source", "first"))
              .reset_index())
    print(f"Lab station-days: {len(agg)} from {len(lab)} replicate rows")
    return agg


def match_sonde(lab):
    feat = pd.read_csv(FEATURES, usecols=["station", "date", "chl"])
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.dropna(subset=["chl"])
    key = feat.set_index(["station", "date"])["chl"]

    def lookup(st, dt):
        cands = UB_FALLBACK if st == "B3w" else [st]
        for c in cands:
            if (c, dt) in key.index:
                return c, key[(c, dt)]
        return None, np.nan

    res = [lookup(s, d) for s, d in zip(lab["station"], lab["date"])]
    lab = lab.copy()
    lab["sonde_station"] = [r[0] for r in res]
    lab["sonde_chl"] = [r[1] for r in res]
    matched = lab[lab["sonde_chl"].notna()].copy()
    matched["year"] = matched["date"].dt.year
    return matched, feat


def ols(x, y):
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return a, b, 1 - ss_res / ss_tot


def main():
    lab = load_lab()
    matched, feat = match_sonde(lab)

    print("\nUnmatched lab station-days by station (no sonde value that day):")
    unm = lab[~lab.set_index(["station", "date"]).index.isin(
        matched.set_index(["station", "date"]).index)]
    print(unm.groupby("station").size().to_string())

    print("\nMatched pairs per year:")
    per_year = matched.groupby("year").size()
    print(per_year.to_string())
    print("Matched pairs per station:")
    print(matched.groupby("sonde_station").size().to_string())
    n = len(matched)
    print(f"TOTAL matched station-days: {n}")

    OUT_PAIRS.parent.mkdir(exist_ok=True, parents=True)
    cols = ["station", "sonde_station", "date", "year", "lab_chl", "lab_sd",
            "n_reps", "sonde_chl", "source"]
    matched[cols].sort_values(["date", "station"]).to_csv(OUT_PAIRS, index=False)
    print(f"Wrote {OUT_PAIRS}")

    if n < MIN_N:
        summary = pd.DataFrame([{"metric": "n_matched", "value": n},
                                {"metric": "verdict", "value": "calibration NOT possible (n < 20)"}])
        summary.to_csv(OUT_SUMMARY, index=False)
        print("Calibration NOT possible: fewer than 20 matched pairs. Stopping.")
        return

    x = matched["sonde_chl"].to_numpy(float)
    y = matched["lab_chl"].to_numpy(float)

    a_ls, b_ls, r2_ls = ols(x, y)          # lab = a + b * sonde
    a_sl, b_sl, r2_sl = ols(y, x)          # sonde = a + b * lab
    from scipy import stats
    rho, rho_p = stats.spearmanr(x, y)
    pear, pear_p = stats.pearsonr(x, y)
    sonde_eq = (LAB_THRESHOLD - a_ls) / b_ls   # inverted lab~sonde fit
    sonde_eq_direct = a_sl + b_sl * LAB_THRESHOLD
    ratio_med = float(np.median(y / x))
    rmse = float(np.sqrt(np.mean((y - (a_ls + b_ls * x)) ** 2)))
    mape = float(np.median(np.abs(y - x) / y) * 100)

    # robustness: log-log OLS and Theil-Sen (the linear OLS is leveraged by a
    # few lab values > 100 ug/L with large replicate spread)
    lx, ly = np.log10(x), np.log10(y)
    a_ll, b_ll, r2_ll = ols(lx, ly)
    sonde_eq_loglog = 10 ** ((np.log10(LAB_THRESHOLD) - a_ll) / b_ll)
    ts = stats.theilslopes(y, x)
    b_ts, a_ts = ts[0], ts[1]
    sonde_eq_ts = (LAB_THRESHOLD - a_ts) / b_ts

    # bootstrap CI on the calibrated sonde equivalent (resample pairs)
    boots = []
    idx = np.arange(n)
    for _ in range(N_BOOT):
        s = RNG.choice(idx, n, replace=True)
        bb, aa = np.polyfit(x[s], y[s], 1)
        if bb > 0:
            boots.append((LAB_THRESHOLD - aa) / bb)
    boots = np.array(boots)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    # agreement at the 10 ug/L threshold
    lab_bloom = y > LAB_THRESHOLD
    sonde_bloom10 = x > LAB_THRESHOLD
    sonde_bloom_eq = x > sonde_eq
    agree10 = float(np.mean(lab_bloom == sonde_bloom10))
    agree_eq = float(np.mean(lab_bloom == sonde_bloom_eq))
    frac_lab_bloom = float(lab_bloom.mean())
    frac_sonde_bloom10 = float(sonde_bloom10.mean())

    # population fractions over ALL sonde station-days
    frac_all_10 = float((feat["chl"] > LAB_THRESHOLD).mean())
    frac_all_eq = float((feat["chl"] > sonde_eq).mean())

    print("\n=== FIT ===")
    print(f"lab = {a_ls:.3f} + {b_ls:.3f} * sonde   r^2={r2_ls:.3f}   RMSE={rmse:.2f} ug/L")
    print(f"sonde = {a_sl:.3f} + {b_sl:.3f} * lab   r^2={r2_sl:.3f}")
    print(f"Pearson r={pear:.3f}  Spearman rho={rho:.3f} (p={rho_p:.2e})")
    print(f"median lab/sonde ratio = {ratio_med:.2f}; median |lab-sonde|/lab = {mape:.0f}%")
    print(f"Sonde daily mean equivalent to lab {LAB_THRESHOLD:g} ug/L "
          f"(inverted lab~sonde): {sonde_eq:.2f}  95% bootstrap CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  (direct sonde~lab prediction at lab=10: {sonde_eq_direct:.2f})")
    print(f"Robustness: log10(lab) = {a_ll:.3f} + {b_ll:.3f} log10(sonde), r^2={r2_ll:.3f} "
          f"-> sonde equiv {sonde_eq_loglog:.2f}; Theil-Sen lab = {a_ts:.3f} + {b_ts:.3f} sonde "
          f"-> sonde equiv {sonde_eq_ts:.2f}")
    print(f"Pairs: lab>10 in {frac_lab_bloom:.1%}, sonde>10 in {frac_sonde_bloom10:.1%}; "
          f"threshold agreement sonde>10 vs lab>10: {agree10:.1%}, "
          f"sonde>{sonde_eq:.1f} vs lab>10: {agree_eq:.1%}")
    print(f"All sonde station-days (n={len(feat)}): chl>10 in {frac_all_10:.1%}, "
          f"chl>{sonde_eq:.2f} in {frac_all_eq:.1%}")

    summary = [
        ("n_matched", n), ("n_years", matched["year"].nunique()),
        ("year_min", int(matched["year"].min())), ("year_max", int(matched["year"].max())),
        ("n_stations", matched["sonde_station"].nunique()),
        ("lab_on_sonde_intercept", a_ls), ("lab_on_sonde_slope", b_ls), ("lab_on_sonde_r2", r2_ls),
        ("lab_on_sonde_rmse_ugL", rmse),
        ("sonde_on_lab_intercept", a_sl), ("sonde_on_lab_slope", b_sl), ("sonde_on_lab_r2", r2_sl),
        ("pearson_r", pear), ("spearman_rho", rho), ("spearman_p", rho_p),
        ("median_lab_over_sonde", ratio_med), ("median_abs_pct_diff", mape),
        ("lab_threshold_ugL", LAB_THRESHOLD),
        ("sonde_equiv_of_lab10_inverted", sonde_eq),
        ("sonde_equiv_ci95_lo", ci_lo), ("sonde_equiv_ci95_hi", ci_hi),
        ("sonde_equiv_of_lab10_direct", sonde_eq_direct),
        ("loglog_intercept", a_ll), ("loglog_slope", b_ll), ("loglog_r2", r2_ll),
        ("sonde_equiv_of_lab10_loglog", sonde_eq_loglog),
        ("theilsen_intercept", a_ts), ("theilsen_slope", b_ts),
        ("sonde_equiv_of_lab10_theilsen", sonde_eq_ts),
        ("pairs_frac_lab_gt10", frac_lab_bloom), ("pairs_frac_sonde_gt10", frac_sonde_bloom10),
        ("pairs_threshold_agreement_sonde10", agree10),
        ("pairs_threshold_agreement_sonde_equiv", agree_eq),
        ("all_station_days_n", len(feat)),
        ("all_station_days_frac_sonde_gt10", frac_all_10),
        ("all_station_days_frac_sonde_gt_equiv", frac_all_eq),
    ]
    for yr, c in per_year.items():
        summary.append((f"n_matched_{yr}", int(c)))
    pd.DataFrame(summary, columns=["metric", "value"]).to_csv(OUT_SUMMARY, index=False)
    print(f"Wrote {OUT_SUMMARY}")

    make_figure(x, y, a_ls, b_ls, r2_ls, rho, sonde_eq, ci_lo, ci_hi, n, matched,
                a_ts, b_ts, sonde_eq_ts, sonde_eq_loglog)


def make_figure(x, y, a, b, r2, rho, sonde_eq, ci_lo, ci_hi, n, matched,
                a_ts, b_ts, sonde_eq_ts, sonde_eq_loglog):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lim = float(np.percentile(np.concatenate([x, y]), 99)) * 1.05
    n_clip = int(((x > lim) | (y > lim)).sum())

    fig, ax = plt.subplots(figsize=(6.4, 6.9), dpi=150)
    ax.scatter(x, y, s=18, color=BLUE, alpha=0.55, edgecolor="none", zorder=3,
               label=f"matched station-days (n={n})")
    xx = np.linspace(0, lim, 100)
    ax.plot(xx, xx, color=GREY, lw=1.2, ls="--", zorder=2, label="1:1")
    ax.plot(xx, a + b * xx, color=ORANGE, lw=2.0, zorder=4,
            label=f"OLS lab = {a:.2f} + {b:.2f}·sonde  (r² = {r2:.2f})")
    ax.plot(xx, a_ts + b_ts * xx, color=ORANGE, lw=1.2, ls="--", zorder=4,
            label=f"Theil-Sen lab = {a_ts:.2f} + {b_ts:.2f}·sonde")
    ax.axhline(LAB_THRESHOLD, color=GREY, lw=1.0, ls=":", zorder=1)
    ax.axvline(LAB_THRESHOLD, color=GREY, lw=1.0, ls=":", zorder=1)
    ax.axvline(sonde_eq, color=ORANGE, lw=1.2, ls=":", zorder=2)
    ax.axvspan(ci_lo, ci_hi, color=ORANGE, alpha=0.10, zorder=0, lw=0)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Sonde fluorescence chlorophyll, daily mean (µg/L)")
    ax.set_ylabel("Lab-extracted chlorophyll-a, grab sample (µg/L)")
    ax.set_title("Narragansett Bay: sonde vs lab chlorophyll, same station and day",
                 fontsize=10.5)

    # reference-line annotations: vertical labels start at 70% height so they
    # stay clear of the stats box (upper left) and the legend (below axes)
    ax.text(lim * 0.985, LAB_THRESHOLD + lim * 0.012, "lab 10 µg/L",
            ha="right", va="bottom", fontsize=8, color=GREY)
    ax.text(LAB_THRESHOLD - lim * 0.01, lim * 0.70, "sonde 10", rotation=90,
            ha="right", va="top", fontsize=8, color=GREY)
    ax.text(sonde_eq + lim * 0.01, lim * 0.70,
            f"sonde ≡ lab 10: {sonde_eq:.1f} µg/L\n95% CI [{ci_lo:.1f}, {ci_hi:.1f}]",
            rotation=90, ha="left", va="top", fontsize=8, color=ORANGE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))

    txt = (f"Spearman ρ = {rho:.2f}\n"
           f"robust equiv.: Theil-Sen {sonde_eq_ts:.1f}, log-log {sonde_eq_loglog:.1f} µg/L\n"
           f"{matched['year'].min()}–{matched['year'].max()}, "
           f"{matched['sonde_station'].nunique()} stations")
    if n_clip:
        txt += f"\n{n_clip} point(s) beyond axis limits"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#333333",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              fontsize=8, frameon=False)
    ax.grid(color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_aspect("equal")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(OUT_FIG)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
