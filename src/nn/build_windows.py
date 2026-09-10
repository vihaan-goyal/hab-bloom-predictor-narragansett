"""Build the 7-day 15-minute window cache for findings 26.

    python -m src.nn.build_windows            (env hab-nn, from the fork root)

One window per labelled station-day in data/narragansett_daily_features.csv:
672 slots from (t-6d) 00:00 to t 23:45, channels log1p(chl), temp, sal, DO (mg/L),
NaN preserved. Rows whose window has < 336 observed chl slots are dropped (from
every cell of the experiment). For every kept row the mean of the last 96 chl
slots must equal the daily file's `chl` (alignment assertion).

Outputs (data/nn/, gitignored): index.csv, windows_f32.npy (N x 4 x 672), norm_stats.json.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from src.nn.common import INDEX, NN_DIR, NORM, WINDOWS

RAW = "data/narragansett_surface_15min.csv"
DAILY = "data/narragansett_daily_features.csv"
CHANNELS = ["chl_ugl", "temp_c", "salinity_psu", "do_mgl"]
STEPS = 7 * 96
MIN_CHL_SLOTS = 336
BLOOM = 10.0
TRAIN_MAX, VAL_YEARS, TEST_YEAR = 2020, (2021, 2022), 2023
STEP = pd.Timedelta("15min")


def split_of(year):
    if year <= TRAIN_MAX:
        return "train"
    if year in VAL_YEARS:
        return "val"
    if year == TEST_YEAR:
        return "test"
    return "drop"


def main():
    t0 = time.time()
    os.makedirs(NN_DIR, exist_ok=True)

    day = pd.read_csv(DAILY, parse_dates=["date"], usecols=["station", "date", "chl", "bloom_fwd"])
    day = day.dropna(subset=["bloom_fwd"]).copy()
    day["year"] = day.date.dt.year
    day["split"] = day.year.map(split_of)
    day = day[day.split != "drop"].reset_index(drop=True)
    print(f"labelled rows: {len(day):,} | "
          + " | ".join(f"{s} {int((day.split == s).sum()):,}" for s in ("train", "val", "test")))

    print("loading 15-min CSV ...", flush=True)
    raw = pd.read_csv(RAW, usecols=["station", "datetime"] + CHANNELS,
                      dtype={c: "float32" for c in CHANNELS})
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce", format="mixed")
    raw = raw.dropna(subset=["datetime"])
    raw = raw[raw.datetime.dt.year >= 2000]
    raw["datetime"] = raw["datetime"].dt.floor("15min")     # floor keeps day membership exact
    print(f"  {len(raw):,} rows, {raw.station.nunique()} stations, {time.time()-t0:.0f}s", flush=True)

    n_rows = len(day)
    W = np.lib.format.open_memmap(WINDOWS, mode="w+", dtype="float32", shape=(n_rows, 4, STEPS))
    n_slots = np.zeros(n_rows, dtype=np.int32)
    last_day_mean = np.full(n_rows, np.nan, dtype="float64")
    n_last = np.zeros(n_rows, dtype=np.int32)

    for st, rows in day.groupby("station", sort=False):
        g = raw[raw.station == st]
        if g.empty:
            print(f"  station {st!r}: no 15-min rows"); continue
        g = g.groupby("datetime")[CHANNELS].mean().sort_index()
        g0, g1 = g.index.min().normalize(), g.index.max()
        grid = pd.date_range(g0, g1, freq=STEP)
        A = g.reindex(grid).to_numpy(dtype="float32")          # (T, 4), raw chl in col 0
        T = len(grid)
        for i, r in rows.iterrows():
            start = r.date.normalize() - pd.Timedelta(days=6)
            off = int((start - g0) / STEP)
            win = np.full((STEPS, 4), np.nan, dtype="float32")
            a0, a1 = max(off, 0), min(off + STEPS, T)
            if a1 > a0:
                win[a0 - off:a1 - off] = A[a0:a1]
            chl = win[:, 0].copy()                                 # raw, may be negative
            n_slots[i] = int(np.isfinite(chl).sum())
            last = chl[-96:]
            n_last[i] = int(np.isfinite(last).sum())
            if n_last[i]:
                last_day_mean[i] = float(np.nanmean(last))
            win[:, 0] = np.log1p(np.clip(chl, 0.0, None))          # stored channel
            W[i] = win.T
        print(f"  {st:<8s} rows {len(rows):>6,}  grid {T:>9,} slots  {time.time()-t0:.0f}s", flush=True)
    W.flush()

    day["n_chl_slots"] = n_slots
    day["n_last_day"] = n_last
    day["last_day_mean"] = last_day_mean
    # Alignment: the mean of the window's last 96 slots must equal the daily file's chl.
    # Sub-15-min cadence (F4 in 2005 logged every 10 min; B3w has duplicate stamps seconds
    # apart) is slot-averaged and moves the mean by < 1 %; a day shift would move it by
    # tens of %. Tolerance: 2 % relative, zero rows allowed beyond it.
    dev = (day.last_day_mean - day.chl).abs()
    rel = dev / np.maximum(1.0, day.chl.abs())
    soft, bad = rel > 1e-3, rel > 0.02
    print(f"\nalignment: max |window-day mean - daily chl| = {dev.max():.5f} "
          f"(max relative {rel.max():.4f}); rows 0.1-2% off (sub-slot cadence): {int((soft & ~bad).sum())}; "
          f"rows > 2% off: {int(bad.sum())} of {len(day):,}")
    if bad.sum() > 0:
        print(day.loc[bad, ["station", "date", "chl", "last_day_mean", "n_last_day"]].head(20).to_string())
        sys.exit("ALIGNMENT FAILURE: rows misaligned by more than 2%")

    keep = day.n_chl_slots >= MIN_CHL_SLOTS
    print(f"\ncoverage rule (>= {MIN_CHL_SLOTS} chl slots): keep {int(keep.sum()):,} / {len(day):,} "
          f"({100*(1-keep.mean()):.2f}% dropped)")
    for s in ("train", "val", "test"):
        m = day.split == s
        onset = m & (day.chl <= BLOOM)
        print(f"  {s:<5s} before {int(m.sum()):>6,} (pos {day.loc[m,'bloom_fwd'].mean():.3f})"
              f"  after {int((m & keep).sum()):>6,} (pos {day.loc[m & keep,'bloom_fwd'].mean():.3f})"
              f"  onset before {int(onset.sum()):>5,} after {int((onset & keep).sum()):>5,}"
              f" (base {day.loc[onset & keep,'bloom_fwd'].mean():.3f})")

    # compact the memmap to kept rows, in index order
    kept_pos = np.where(keep.values)[0]
    tmp = WINDOWS + ".tmp"
    W2 = np.lib.format.open_memmap(tmp, mode="w+", dtype="float32", shape=(len(kept_pos), 4, STEPS))
    for j in range(0, len(kept_pos), 2048):
        W2[j:j + 2048] = W[kept_pos[j:j + 2048]]
    W2.flush(); del W, W2
    os.replace(tmp, WINDOWS)

    idx = day.loc[keep].reset_index(drop=True)
    idx.insert(0, "row_id", np.arange(len(idx)))
    idx["doy"] = idx.date.dt.dayofyear
    idx["y"] = idx.bloom_fwd.astype(int)
    idx["onset"] = (idx.chl <= BLOOM).astype(int)
    idx["cluster"] = idx.station.astype(str) + "_" + idx.year.astype(str)
    idx = idx[["row_id", "station", "date", "year", "doy", "split", "y", "chl", "onset",
               "cluster", "n_chl_slots"]]
    idx.to_csv(INDEX, index=False, date_format="%Y-%m-%d")

    # normalisation from observed train slots only
    W = np.load(WINDOWS, mmap_mode="r")
    tr = idx.index[idx.split == "train"].to_numpy()
    mean, std = [], []
    for c in range(4):
        s = np.zeros(2); n = 0
        for j in range(0, len(tr), 2048):
            x = np.asarray(W[tr[j:j + 2048], c, :], dtype="float64")
            f = np.isfinite(x); n += int(f.sum())
            s += [x[f].sum(), (x[f] ** 2).sum()]
        mu = s[0] / n; sd = float(np.sqrt(max(s[1] / n - mu ** 2, 1e-12)))
        mean.append(float(mu)); std.append(sd)
    with open(NORM, "w") as f:
        json.dump(dict(channels=["log1p_chl", "temp_c", "salinity_psu", "do_mgl"],
                       mean=mean, std=std, steps=STEPS, min_chl_slots=MIN_CHL_SLOTS,
                       n_train=int(len(tr))), f, indent=1)
    print(f"\nnorm (train slots): mean {np.round(mean, 3).tolist()} std {np.round(std, 3).tolist()}")
    print(f"wrote {INDEX} ({len(idx):,} rows), {WINDOWS} "
          f"({os.path.getsize(WINDOWS)/1e9:.2f} GB), {NORM}; {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
