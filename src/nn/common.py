"""Shared helpers for the NN experiments (findings 26-27).

- paired bootstrap that reproduces `transfer_eval.boot_ci` resampling and adds
  paired difference CIs (pattern from experiments/onset_rule_baselines_cv.py)
- deterministic seeding, a generic early-stopping training loop, prediction
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.transfer.transfer_eval import metrics

NN_DIR = "data/nn"
INDEX = os.path.join(NN_DIR, "index.csv")
WINDOWS = os.path.join(NN_DIR, "windows_f32.npy")
NORM = os.path.join(NN_DIR, "norm_stats.json")
SEEDS = (42, 43, 44, 45, 46)
N_THREADS = 12


# ----------------------------------------------------------------- data
def load_index():
    return pd.read_csv(INDEX, parse_dates=["date"])


def load_norm():
    with open(NORM) as f:
        return json.load(f)


# ------------------------------------------------------------ bootstrap
def boot_paired(y, probs, thresholds, clusters, ref, n_boot=2000, seed=42):
    """Station-year clustered bootstrap, every model on the SAME resamples.

    Resampling is identical to transfer_eval.boot_ci (default_rng(seed), keys in
    np.unique order, rng.choice(keys, len(keys))), so marginal CIs reproduce it.
    Returns {model: {auc, precision, lift, *_lo, *_hi, dauc, dauc_lo, dauc_hi,
    dprec.., dlift..}} with differences taken as model - ref.
    """
    y = np.asarray(y).astype(int)
    cl = np.asarray(clusters)
    groups = {c: np.where(cl == c)[0] for c in np.unique(cl)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    names = list(probs)
    reps = {m: {"auc": [], "precision": [], "lift": []} for m in names}
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        yy = y[idx]
        two = 0 < yy.mean() < 1
        for m in names:
            p = probs[m][idx]
            mm = metrics(yy, p >= thresholds[m])
            reps[m]["auc"].append(roc_auc_score(yy, p) if two else np.nan)
            reps[m]["precision"].append(mm["precision"])
            reps[m]["lift"].append(mm["lift"])
    for m in names:
        for k in reps[m]:
            reps[m][k] = np.asarray(reps[m][k], dtype=float)

    def point(m):
        mm = metrics(y, probs[m] >= thresholds[m])
        return dict(auc=roc_auc_score(y, probs[m]), precision=mm["precision"],
                    lift=mm["lift"], pod=mm["pod"], base_rate=mm["base_rate"],
                    tp=mm["tp"], fp=mm["fp"], fn=mm["fn"])

    ref_pt = point(ref)
    out = {}
    for m in names:
        pt = point(m)
        r = dict(n_test=len(y), n_pos=int(y.sum()), t_star=thresholds[m], **pt)
        for k in ("auc", "precision", "lift"):
            r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(reps[m][k], [2.5, 97.5])
        for k, kk in (("auc", "dauc"), ("precision", "dprec"), ("lift", "dlift")):
            if m == ref:
                r[kk] = r[f"{kk}_lo"] = r[f"{kk}_hi"] = 0.0
                r[f"{kk}_nboot_nan"] = 0
            else:
                d = reps[m][k] - reps[ref][k]
                ok = ~np.isnan(d)
                r[kk] = pt[k] - ref_pt[k]
                r[f"{kk}_lo"], r[f"{kk}_hi"] = np.percentile(d[ok], [2.5, 97.5])
                r[f"{kk}_nboot_nan"] = int((~ok).sum())
        out[m] = r
    return out


# -------------------------------------------------------------- training
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(N_THREADS)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def train_loop(model, batches_fn, val_fn, y_val, pos_weight, seed, log_path,
               lr=1e-3, wd=1e-4, max_epochs=60, patience=8):
    """Generic early-stopping loop.

    batches_fn(rng) -> iterator of (inputs_tuple, y_tensor) for one epoch
    val_fn(model) -> np.ndarray of val probabilities
    Early stop on val AUC; best weights restored. Returns (model, best_epoch, best_val_auc).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)]))
    rng = np.random.default_rng(seed)
    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
    rows = []
    for ep in range(1, max_epochs + 1):
        model.train(); tot, nb = 0.0, 0
        for xs, yb in batches_fn(rng):
            opt.zero_grad()
            loss = crit(model(*xs), yb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if not np.isfinite(tot):
            raise RuntimeError("non-finite training loss")
        pv = val_fn(model)
        auc = roc_auc_score(y_val, pv)
        rows.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_auc=auc))
        if auc > best_auc:
            best_auc, best_epoch, bad = auc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    pd.DataFrame(rows).to_csv(log_path, index=False)
    return model, best_epoch, best_auc


@torch.no_grad()
def predict(model, batches):
    model.eval()
    out = [torch.sigmoid(model(*xs)).numpy() for xs in batches]
    return np.concatenate(out).astype(float)


def tab_batches(X, y=None, batch=256, rng=None):
    """Tabular batches from numpy float32 arrays (shuffled if rng given)."""
    n = len(X)
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch):
        j = order[i:i + batch]
        xs = (torch.from_numpy(X[j]),)
        if y is None:
            yield xs
        else:
            yield xs, torch.from_numpy(y[j])
