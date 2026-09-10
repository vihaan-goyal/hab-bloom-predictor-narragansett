"""Findings 26: sequence model on raw 15-min sondes vs daily-feature GB (2x2 + hybrid).

    python -m src.nn.seq_vs_daily --cells gb,mlp          # then
    python -m src.nn.seq_vs_daily --cells cnn,hyb
    python -m src.nn.seq_vs_daily --score                 # paired bootstrap, CSVs, fig 12

Cells: gb (reference GB_KW on tier-A), mlp (tier-A MLP, architecture control),
cnn (7-d 15-min window CNN, the hypothesis), hyb (CNN embedding + tier-A).
All cells use the identical row set from data/nn/index.csv (build_windows.py).
Per-cell probabilities are cached in data/nn/preds_<cell>.npz so cells can be run
in separate invocations; --score reads whatever cells exist.
Pre-registered criteria and design: notes/NARRAGANSETT_FINDINGS.md section 26.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from src.nn.common import (NN_DIR, SEEDS, WINDOWS, boot_paired, load_index, load_norm,
                           n_params, predict, set_seed, tab_batches, train_loop)
from src.nn.models import MLP, SeqCNN, HybridCNN
from src.transfer.transfer_eval import GB_KW, TIER_A, pick_t

DAILY = "data/narragansett_daily_features.csv"
RESULTS = os.path.join(NN_DIR, "seq_vs_daily_results.csv")
PREDS_OUT = os.path.join(NN_DIR, "seq_vs_daily_predictions.csv")
FIG = "figures/nar_fig12_seq_vs_daily.png"
NN_CELLS = {"mlp": "MLP-daily", "cnn": "CNN-15min", "hyb": "Hybrid"}
LABEL = {"gb": "GB-daily", "gb_fulltrain": "GB-daily (full train, sanity)", **NN_CELLS}
BATCH = 256


# ----------------------------------------------------------------- data
def load_data():
    idx = load_index()
    day = pd.read_csv(DAILY, parse_dates=["date"])
    d = idx.drop(columns=["chl"]).merge(day[["station", "date"] + TIER_A], on=["station", "date"],
                  how="left", validate="1:1")
    assert len(d) == len(idx) and d["chl"].notna().all(), "index/daily merge mismatch"
    return d, day


def tabular(d):
    tr = (d.split == "train").to_numpy()
    med = d.loc[tr, TIER_A].median()
    X = d[TIER_A].fillna(med).fillna(0.0)
    mu, sd = X[tr].mean(), X[tr].std().replace(0, 1.0)
    Z = ((X - mu) / sd).to_numpy(dtype="float32")
    return X.to_numpy(dtype="float64"), Z, med


class WindowSource:
    """Standardised (value + mask) windows and sin/cos DOY, read from the memmap."""

    def __init__(self, d, norm):
        self.W = np.load(WINDOWS, mmap_mode="r")
        assert self.W.shape[0] == len(d), "windows/index length mismatch"
        self.mean = np.asarray(norm["mean"], dtype="float32")[None, :, None]
        self.std = np.asarray(norm["std"], dtype="float32")[None, :, None]
        ang = 2 * np.pi * d.doy.to_numpy(dtype="float32") / 365.25
        self.static = np.stack([np.sin(ang), np.cos(ang)], 1).astype("float32")

    def get(self, rows):
        x = np.asarray(self.W[rows], dtype="float32")             # (B, 4, 672)
        m = np.isfinite(x)
        z = np.where(m, (x - self.mean) / self.std, 0.0)
        z = np.clip(z, -6.0, 6.0).astype("float32")
        return np.concatenate([z, m.astype("float32")], axis=1), self.static[rows]


def seq_batches(src, rows, y=None, Z=None, rng=None):
    order = rng.permutation(len(rows)) if rng is not None else np.arange(len(rows))
    for i in range(0, len(rows), BATCH):
        j = np.sort(rows[order[i:i + BATCH]])
        xw, xs = src.get(j)
        xs_ = [torch.from_numpy(xw), torch.from_numpy(xs)]
        if Z is not None:
            xs_.append(torch.from_numpy(Z[j]))
        if y is None:
            yield tuple(xs_)
        else:
            yield tuple(xs_), torch.from_numpy(y[j])


# ---------------------------------------------------------------- cells
def preds_path(cell):
    return os.path.join(NN_DIR, f"preds_{cell}.npz")


def save_preds(cell, val, test, meta):
    np.savez(preds_path(cell), **{f"val_{k}": v for k, v in val.items()},
             **{f"test_{k}": v for k, v in test.items()}, meta=json.dumps(meta))


def run_gb(d, day, X, tr, va, te):
    y = d.y.to_numpy()
    gb = HistGradientBoostingClassifier(**GB_KW).fit(X[tr], y[tr])
    save_preds("gb", {"s0": gb.predict_proba(X[va])[:, 1]},
               {"s0": gb.predict_proba(X[te])[:, 1]}, dict(n_train=int(tr.sum())))
    # sanity: GB on ALL labelled train rows of the daily file (before the coverage rule)
    full = day.dropna(subset=["bloom_fwd"])
    full = full[full.date.dt.year <= 2020]
    med = full[TIER_A].median()
    gbf = HistGradientBoostingClassifier(**GB_KW).fit(full[TIER_A].fillna(med).fillna(0.0),
                                                       full.bloom_fwd.astype(int))
    Xf = d[TIER_A].fillna(med).fillna(0.0).to_numpy()
    save_preds("gb_fulltrain", {"s0": gbf.predict_proba(Xf[va])[:, 1]},
               {"s0": gbf.predict_proba(Xf[te])[:, 1]}, dict(n_train=int(len(full))))
    on = te & (d.onset.to_numpy() == 1)
    print(f"  GB-daily  filtered-train n={int(tr.sum()):,}  test-onset AUC "
          f"{roc_auc_score(y[on], gb.predict_proba(X[on])[:, 1]):.4f}")
    print(f"  GB-daily  full-train     n={len(full):,}  test-onset AUC "
          f"{roc_auc_score(y[on], gbf.predict_proba(Xf[on])[:, 1]):.4f}")


def run_nn(cell, d, Z, src, tr, va, te, seeds, max_epochs):
    y = d.y.to_numpy().astype("float32")
    rows_tr, rows_va, rows_te = (np.where(m)[0] for m in (tr, va, te))
    pos_w = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    val, test, meta = {}, {}, {}
    for seed in seeds:
        t0 = time.time(); set_seed(seed)
        if cell == "mlp":
            model = MLP(len(TIER_A))
            batches = lambda rng: tab_batches(Z[rows_tr], y[rows_tr], BATCH, rng)
            pv = lambda m: predict(m, tab_batches(Z[rows_va]))
            pt = lambda m: predict(m, tab_batches(Z[rows_te]))
        elif cell == "cnn":
            model = SeqCNN()
            batches = lambda rng: seq_batches(src, rows_tr, y, None, rng)
            pv = lambda m: predict(m, seq_batches(src, rows_va))
            pt = lambda m: predict(m, seq_batches(src, rows_te))
        elif cell == "hyb":
            model = HybridCNN(n_tab=len(TIER_A))
            batches = lambda rng: seq_batches(src, rows_tr, y, Z, rng)
            pv = lambda m: predict(m, seq_batches(src, rows_va, None, Z))
            pt = lambda m: predict(m, seq_batches(src, rows_te, None, Z))
        else:
            raise ValueError(cell)
        log = os.path.join(NN_DIR, f"train_log_{cell}_s{seed}.csv")
        model, best_ep, best_auc = train_loop(model, batches, pv, y[rows_va], pos_w, seed, log,
                                              max_epochs=max_epochs)
        val[f"s{seed}"], test[f"s{seed}"] = pv(model), pt(model)
        meta[f"s{seed}"] = dict(best_epoch=best_ep, val_auc=best_auc, n_params=n_params(model),
                                seconds=round(time.time() - t0))
        print(f"  {NN_CELLS[cell]:<10s} seed {seed}  best epoch {best_ep:>2d}  val AUC {best_auc:.4f}"
              f"  params {n_params(model):,}  {time.time()-t0:.0f}s", flush=True)
        save_preds(cell, val, test, meta)


# --------------------------------------------------------------- scoring
def score(d):
    va, te = (d.split == "val").to_numpy(), (d.split == "test").to_numpy()
    on = d.onset.to_numpy() == 1
    y = d.y.to_numpy()
    va_on, te_on = va & on, te & on
    probs_val, probs_test, meta = {}, {}, {}
    for cell in ("gb", "gb_fulltrain", "mlp", "cnn", "hyb"):
        p = preds_path(cell)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        keys = sorted(k[4:] for k in z.files if k.startswith("val_"))
        for k in keys:
            name = cell if cell.startswith("gb") else f"{cell}_{k}"
            probs_val[name], probs_test[name] = z[f"val_{k}"], z[f"test_{k}"]
            meta[name] = m.get(k, m) if isinstance(m, dict) else {}
        if not cell.startswith("gb") and len(keys) > 1:
            probs_val[f"{cell}_ens"] = np.mean([z[f"val_{k}"] for k in keys], axis=0)
            probs_test[f"{cell}_ens"] = np.mean([z[f"test_{k}"] for k in keys], axis=0)
            meta[f"{cell}_ens"] = dict(n_seeds=len(keys))
    assert "gb" in probs_test, "run --cells gb first"
    for k, v in probs_test.items():
        assert len(v) == int(te.sum()), f"{k}: test length mismatch"
    va_on_local, te_on_local = on[va], on[te]
    thr = {k: pick_t(y[va_on], probs_val[k][va_on_local]) for k in probs_val}
    P = {k: v[te_on_local] for k, v in probs_test.items()}
    res = boot_paired(y[te_on], P, thr, d.cluster.to_numpy()[te_on], ref="gb")
    rows = []
    for k, r in res.items():
        cell = k if k.startswith("gb") else k.split("_")[0]
        seed = "s0" if k.startswith("gb") else k.split("_")[1]
        rows.append(dict(model=k, cell=LABEL.get(cell, cell), seed=seed, scope="onset", **r,
                         auc_all=roc_auc_score(y[te], probs_test[k]),
                         **{f"meta_{a}": b for a, b in meta.get(k, {}).items()}))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS, index=False)
    pred = d.loc[te, ["row_id", "station", "date", "year", "y", "chl", "onset", "cluster"]].copy()
    for k, v in probs_test.items():
        pred[f"p_{k}"] = v
    pred.to_csv(PREDS_OUT, index=False, date_format="%Y-%m-%d")

    pd.set_option("display.width", 250)
    cols = ["model", "n_test", "t_star", "precision", "pod", "base_rate", "lift", "lift_lo",
            "lift_hi", "auc", "auc_lo", "auc_hi", "dauc", "dauc_lo", "dauc_hi", "dlift",
            "dlift_lo", "dlift_hi", "auc_all"]
    print("\nTest-2023 onset rows, paired station-year clustered bootstrap (ref = gb):")
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nPRE-REGISTERED VERDICT (section 26):")
    if "cnn_ens" in res:
        r = res["cnn_ens"]
        go = (r["dauc_lo"] > 0) and (r["dauc"] >= 0.02)
        harm = r["dauc_hi"] < 0
        seeds = [res[k]["auc"] for k in res if k.startswith("cnn_s")]
        above = sum(a > res["gb"]["auc"] for a in seeds)
        verdict = "GO" if go else ("HARM" if harm else "NO-GO")
        print(f"  primary  dAUC(CNN ens - GB) = {r['dauc']:+.4f} [{r['dauc_lo']:+.4f}, {r['dauc_hi']:+.4f}]"
              f"  -> {verdict}")
        excl = not (r["dlift_lo"] <= 0 <= r["dlift_hi"])
        print(f"  secondary dLift = {r['dlift']:+.3f} [{r['dlift_lo']:+.3f}, {r['dlift_hi']:+.3f}]"
              f"  CI excludes 0: {excl}")
        print(f"  seeds above GB on point AUC: {above}/{len(seeds)}  "
              f"(seed AUC range {min(seeds):.4f}-{max(seeds):.4f})")
        tie = (not go) and (not harm) and r["dauc"] > 0 and r["dauc_lo"] <= 0
        print(f"  tie-breaker (rolling-origin 2019-23) triggered: {tie}")
    else:
        print("  cnn cell not run yet")
    make_figure(y[te_on], P, res)
    print(f"\nwrote {RESULTS}, {PREDS_OUT}, {FIG}")


def make_figure(y, P, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    order = [k for k in ("gb", "mlp_ens", "cnn_ens", "hyb_ens") if k in P]
    names = {"gb": "GB-daily", "mlp_ens": "MLP-daily", "cnn_ens": "CNN-15min", "hyb_ens": "Hybrid"}
    for k in order:
        fpr, tpr, _ = roc_curve(y, P[k])
        a1.plot(fpr, tpr, label=f"{names[k]}  AUC {res[k]['auc']:.3f}")
    a1.plot([0, 1], [0, 1], "k:", lw=0.8)
    a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate")
    a1.set_title(f"Test 2023 onset rows (n = {len(y)})"); a1.legend(frameon=False, fontsize=8)
    xs = {"mlp": 0, "cnn": 1, "hyb": 2}
    for cell, x in xs.items():
        seeds = [res[k]["auc"] for k in res if k.startswith(f"{cell}_s")]
        a2.scatter([x] * len(seeds), seeds, s=18, color="0.6", zorder=2)
        if f"{cell}_ens" in res:
            r = res[f"{cell}_ens"]
            a2.errorbar(x, r["auc"], yerr=[[r["auc"] - r["auc_lo"]], [r["auc_hi"] - r["auc"]]],
                        fmt="o", color="C0", capsize=3, zorder=3)
            a2.annotate(f"dAUC {r['dauc']:+.3f}\n[{r['dauc_lo']:+.3f}, {r['dauc_hi']:+.3f}]",
                        (x, r["auc_hi"]), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7)
    g = res["gb"]
    a2.axhline(g["auc"], color="C3", lw=1.2, label=f"GB-daily {g['auc']:.3f}")
    a2.axhspan(g["auc_lo"], g["auc_hi"], color="C3", alpha=0.08)
    a2.set_xticks(list(xs.values())); a2.set_xticklabels(["MLP-daily", "CNN-15min", "Hybrid"])
    a2.set_ylabel("AUC (onset rows)"); a2.set_title("Seeds (grey), 5-seed ensemble (blue, 95% CI)")
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Findings 26: 15-min sequence model vs daily-feature GB, paired on identical rows",
                 fontsize=10)
    fig.tight_layout(); os.makedirs("figures", exist_ok=True); fig.savefig(FIG, dpi=160)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma list of gb,mlp,cnn,hyb")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    cells = [c for c in a.cells.split(",") if c]
    seeds = [int(s) for s in a.seeds.split(",")]
    d, day = load_data()
    tr, va, te = ((d.split == s).to_numpy() for s in ("train", "val", "test"))
    print(f"rows: train {tr.sum():,} val {va.sum():,} test {te.sum():,} "
          f"(test onset {(te & (d.onset.to_numpy() == 1)).sum():,}); torch {torch.__version__}")
    X, Z, _ = tabular(d)
    src = WindowSource(d, load_norm()) if any(c in ("cnn", "hyb") for c in cells) else None
    for c in cells:
        print(f"\n== cell {c} ==", flush=True)
        if c == "gb":
            run_gb(d, day, X, tr, va, te)
        else:
            run_nn(c, d, Z, src, tr, va, te, seeds, a.epochs)
    if a.score or not cells:
        score(d)


if __name__ == "__main__":
    main()
