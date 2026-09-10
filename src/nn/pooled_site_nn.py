"""Findings 27: pooled multi-site MLP with a site embedding, tested blind on Narragansett.

    python -m src.nn.pooled_site_nn            (env hab-nn, from the fork root)

Training rows are exactly the section-20 pooled rows (six foreign daily files, own-station
p75 label, chl quantile-mapped to Narragansett). Narragansett is never in training.
Early stopping uses a seed-42 random 15 % holdout of foreign station-years; Narragansett
val 2021-22 is used only to choose t*; the test set is the same test-2023 onset rows as
findings 26 (data/nn/index.csv). Pre-registered criteria: NARRAGANSETT_FINDINGS.md section 27.
Cells: gb_pooled (reference, re-run), mlp_noemb, mlp_unk (UNK-token embedding),
mlp_meanemb (sensitivity: mean of the six site vectors at test time).
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.nn.common import (NN_DIR, SEEDS, boot_paired, load_index, n_params, predict,
                           set_seed, train_loop)
from src.nn.models import SiteMLP
from src.transfer.pooled_model_test import SITES, load_site
from src.transfer.transfer_eval import GB_KW, TIER_A, metrics, pick_t

DAILY = "data/narragansett_daily_features.csv"
RESULTS = os.path.join(NN_DIR, "pooled_site_nn_results.csv")
PREDS_OUT = os.path.join(NN_DIR, "pooled_site_nn_predictions.csv")
FIG = "figures/nar_fig13_pooled_site_nn.png"
BATCH = 256
P_UNK = 0.3
EMB_DIM = 4
ES_FRAC = 0.15
GO_AUC, GO_LIFT = 0.80, 1.82


def site_batches(X, S, y=None, rng=None, unk=None, p_unk=0.0):
    n = len(X)
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, BATCH):
        j = order[i:i + BATCH]
        s = S[j].copy()
        if p_unk > 0 and rng is not None:
            s[rng.random(len(j)) < p_unk] = unk
        xs = (torch.from_numpy(X[j]), torch.from_numpy(s))
        if y is None:
            yield xs
        else:
            yield xs, torch.from_numpy(y[j])


def load_all():
    nar = pd.read_csv(DAILY, parse_dates=["date"])
    pooled = pd.concat([s for s in (load_site(n, nar.chl.values) for n in SITES) if s is not None],
                       ignore_index=True)
    pooled["date"] = pd.to_datetime(pooled["date"])
    pooled["year"] = pooled.date.dt.year
    pooled["site_idx"] = pooled.site.map({s: i for i, s in enumerate(SITES)}).astype("int64")
    idx = load_index()
    d = idx.drop(columns=["chl"]).merge(nar[["station", "date"] + TIER_A], on=["station", "date"],
                                        how="left", validate="1:1")
    return pooled, d


def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [("gb_pooled", "Pooled GB (sec. 20)"), ("mlp_noemb_ens", "Pooled MLP, no embedding"),
             ("mlp_unk_ens", "Pooled MLP + UNK embedding"), ("mlp_meanemb_ens", "Pooled MLP, mean embedding"),
             ("local_gb_ref", "Local Narragansett GB (ref)")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (k, lab) in enumerate(order):
        r = res[k]
        ax.errorbar(r["lift"], i, xerr=[[r["lift"] - r["lift_lo"]], [r["lift_hi"] - r["lift"]]],
                    fmt="o", color="C3" if k == "local_gb_ref" else "C0", capsize=3)
        ax.annotate(f"AUC {r['auc']:.2f}", (r["lift_hi"], i), textcoords="offset points",
                    xytext=(5, -3), fontsize=8)
    ax.axvline(1.0, color="k", lw=0.8, ls=":", label="always alert")
    ax.axvline(GO_LIFT, color="C2", lw=1, ls="--", label=f"GO bar {GO_LIFT}")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([lab for _, lab in order]); ax.invert_yaxis()
    ax.set_xlabel("onset lift on Narragansett test 2023 (station-year clustered 95% CI)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Findings 27: pooled foreign models, blind on Narragansett", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG, dpi=160)


def main():
    t0 = time.time()
    pooled, d = load_all()
    print(f"pooled rows {len(pooled):,} from {pooled.site.nunique()} sites, "
          f"{pooled.station.nunique()} stations, pos rate {pooled.bloom_fwd.mean():.3f}")
    va, te = (d.split == "val").to_numpy(), (d.split == "test").to_numpy()
    on = d.onset.to_numpy() == 1
    y_nar = d.y.to_numpy()
    print(f"Narragansett val onset {int((va & on).sum()):,}, test onset {int((te & on).sum()):,}")

    med = pooled[TIER_A].median()
    Xp = pooled[TIER_A].fillna(med).fillna(0.0)
    mu, sd = Xp.mean(), Xp.std().replace(0, 1.0)
    Zp = ((Xp - mu) / sd).to_numpy(dtype="float32")
    Xn = d[TIER_A].fillna(med).fillna(0.0)
    Zn = ((Xn - mu) / sd).to_numpy(dtype="float32")
    yp = pooled.bloom_fwd.to_numpy().astype("float32")
    Sp = pooled.site_idx.to_numpy()
    UNK = len(SITES)
    Sn_unk = np.full(len(d), UNK, dtype="int64")

    cl = (pooled.station.astype(str) + "_" + pooled.year.astype(str)).to_numpy()
    ucl = np.unique(cl)
    es_cl = set(np.random.default_rng(42).choice(ucl, int(round(ES_FRAC * len(ucl))), replace=False))
    es = np.array([c in es_cl for c in cl]); fit = ~es
    print(f"early-stop holdout: {len(es_cl)} of {len(ucl)} station-years, {int(es.sum()):,} rows")

    probs_val, probs_test, meta = {}, {}, {}
    gb = HistGradientBoostingClassifier(**GB_KW).fit(Xp.to_numpy(), pooled.bloom_fwd.astype(int))
    p = gb.predict_proba(Xn.to_numpy())[:, 1]
    probs_val["gb_pooled"], probs_test["gb_pooled"] = p[va], p[te]
    print(f"gb_pooled test-onset AUC {roc_auc_score(y_nar[te & on], p[te & on]):.4f}  {time.time()-t0:.0f}s")

    pos_w = float((yp[fit] == 0).sum() / max((yp[fit] == 1).sum(), 1))
    S_es = np.full(int(es.sum()), UNK, dtype="int64")
    for cell, use_emb in (("mlp_noemb", False), ("mlp_unk", True)):
        vals, tests, vals_m, tests_m = [], [], [], []
        for seed in SEEDS:
            ts = time.time(); set_seed(seed)
            model = SiteMLP(len(TIER_A), len(SITES), EMB_DIM, use_emb=use_emb)
            batches = lambda rng: site_batches(Zp[fit], Sp[fit], yp[fit], rng, UNK,
                                               P_UNK if use_emb else 0.0)
            pv_es = lambda m: predict(m, site_batches(Zp[es], S_es if use_emb else Sp[es]))
            log = os.path.join(NN_DIR, f"train_log_{cell}_s{seed}.csv")
            model, best_ep, best_auc = train_loop(model, batches, pv_es, yp[es], pos_w, seed, log)
            pn = predict(model, site_batches(Zn, Sn_unk))
            vals.append(pn[va]); tests.append(pn[te])
            meta[f"{cell}_s{seed}"] = dict(best_epoch=best_ep, es_auc=best_auc, n_params=n_params(model))
            probs_val[f"{cell}_s{seed}"], probs_test[f"{cell}_s{seed}"] = pn[va], pn[te]
            if use_emb:
                model.eval()
                with torch.no_grad():
                    mean_vec = model.emb.weight[:len(SITES)].mean(0)
                    pm = np.concatenate([torch.sigmoid(model(*xs, emb_vec=mean_vec)).numpy()
                                         for xs in site_batches(Zn, Sn_unk)]).astype(float)
                vals_m.append(pm[va]); tests_m.append(pm[te])
                probs_val[f"mlp_meanemb_s{seed}"], probs_test[f"mlp_meanemb_s{seed}"] = pm[va], pm[te]
            print(f"  {cell:<10s} seed {seed}  best epoch {best_ep:>2d}  es AUC {best_auc:.4f}  "
                  f"Nar test-onset AUC {roc_auc_score(y_nar[te & on], pn[te & on]):.4f}  {time.time()-ts:.0f}s",
                  flush=True)
        probs_val[f"{cell}_ens"], probs_test[f"{cell}_ens"] = np.mean(vals, 0), np.mean(tests, 0)
        if use_emb:
            probs_val["mlp_meanemb_ens"] = np.mean(vals_m, 0)
            probs_test["mlp_meanemb_ens"] = np.mean(tests_m, 0)

    z = np.load(os.path.join(NN_DIR, "preds_gb.npz"))
    probs_val["local_gb_ref"], probs_test["local_gb_ref"] = z["val_s0"], z["test_s0"]

    va_on_l, te_on_l = on[va], on[te]
    thr = {k: pick_t(y_nar[va & on], v[va_on_l]) for k, v in probs_val.items()}
    P = {k: v[te_on_l] for k, v in probs_test.items()}
    res = boot_paired(y_nar[te & on], P, thr, d.cluster.to_numpy()[te & on], ref="gb_pooled")
    aa = metrics(y_nar[te & on], np.ones(int((te & on).sum())))
    rows = [dict(model="always_alert", **aa, auc=0.5)]
    for k, r in res.items():
        rows.append(dict(model=k, **r, auc_all=roc_auc_score(y_nar[te], probs_test[k]),
                         **{f"meta_{a}": b for a, b in meta.get(k, {}).items()}))
    out = pd.DataFrame(rows); out.to_csv(RESULTS, index=False)
    pred = d.loc[te, ["row_id", "station", "date", "year", "y", "chl", "onset", "cluster"]].copy()
    for k, v in probs_test.items():
        pred[f"p_{k}"] = v
    pred.to_csv(PREDS_OUT, index=False, date_format="%Y-%m-%d")

    pd.set_option("display.width", 250)
    cols = ["model", "n_test", "t_star", "precision", "pod", "base_rate", "lift", "lift_lo", "lift_hi",
            "auc", "auc_lo", "auc_hi", "dlift", "dlift_lo", "dlift_hi", "dauc", "dauc_lo", "dauc_hi"]
    show = out[out.model.isin(["always_alert", "gb_pooled", "mlp_noemb_ens", "mlp_unk_ens",
                               "mlp_meanemb_ens", "local_gb_ref"])]
    print("\nNarragansett test-2023 onset rows, paired bootstrap (ref = gb_pooled):")
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    r = res["mlp_unk_ens"]
    excl = not (r["dlift_lo"] <= 0 <= r["dlift_hi"])
    if r["auc"] >= GO_AUC and r["lift"] >= GO_LIFT and excl and r["dlift"] > 0:
        verdict = "GO"
    elif r["auc"] >= GO_AUC and r["lift"] >= GO_LIFT:
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    print(f"\nPRE-REGISTERED VERDICT (section 27): {verdict}  "
          f"(UNK ens AUC {r['auc']:.3f} vs >= {GO_AUC}; lift {r['lift']:.2f} vs >= {GO_LIFT}; "
          f"dLift vs pooled GB {r['dlift']:+.2f} [{r['dlift_lo']:+.2f}, {r['dlift_hi']:+.2f}])")
    seeds = [res[k]["auc"] for k in res if k.startswith("mlp_unk_s")]
    print(f"  UNK seeds AUC range {min(seeds):.3f}-{max(seeds):.3f}")
    make_figure(res)
    print(f"\nwrote {RESULTS}, {PREDS_OUT}, {FIG}; {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
