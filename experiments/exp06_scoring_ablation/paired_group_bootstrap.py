# -*- coding: utf-8 -*-
"""Paired cluster bootstrap on per-group metric differences + Holm correction.

Reads oof_<task>.parquet files (structure_id, group_id, y_true, model, y_pred),
recomputes per-group scores, then bootstraps group-level differences between
model pairs.  Usage: python paired_group_bootstrap.py [task ...]
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _P
_F = _P(__file__).resolve()
_R = _F.parent
while _R != _R.parent and not ((_R / "src").is_dir() and (_R / "experiments").is_dir()):
    _R = _R.parent
for _d in (_R / "src", _R / "tasks" / "rna", _R / "tasks" / "protein_dna",
           _R / "tasks" / "membrane"):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
for _e in sorted((_R / "experiments").glob("exp*")):
    if _e.is_dir() and str(_e) not in _sys.path:
        _sys.path.insert(0, str(_e))
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

H3 = Path(r"H:\2026try\11.3GSPH\GSPH_H3")
OUT = H3 / "results" / "ablation"
N_BOOT = 10_000
SEED = 1

# pre-specified comparison set (Holm family) for rank tasks
RANK_PAIRS = [
    ("A4_PT", "A1_P", "delta_T   = P+T - P"),
    ("A5_PH", "A1_P", "delta_H   = P+H - P"),
    ("A7_PTH", "A1_P", "delta_TH  = P+T+H - P"),
    ("A7_PTH", "A4_PT", "delta_H|T = P+T+H - P+T"),
    ("A7_PTH", "A5_PH", "delta_T|H = P+T+H - P+H"),
    ("A2_T", "A3_H", "T vs H (solo)"),
    ("A7_PTH", "A8_S", "full vs size-only"),
    ("A7_PTH", "A9_Pcompact", "full vs compact-P"),
]


def per_group_spearman(oof: pd.DataFrame, higher_better: bool) -> pd.Series:
    if not higher_better:
        oof = oof.assign(y_true=-oof.y_true)
    vals = {}
    for g, sub in oof.groupby("group_id"):
        if sub.y_true.nunique() > 1 and sub.y_pred.nunique() > 1:
            r = spearmanr(sub.y_true, sub.y_pred).statistic
            if np.isfinite(r):
                vals[g] = r
    return pd.Series(vals)


def bootstrap_diff(a: pd.Series, b: pd.Series, rng) -> dict:
    common = a.index.intersection(b.index).sort_values()
    d = (a[common] - b[common]).to_numpy()
    n = len(d)
    if n == 0:
        return dict(n=0)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p_two = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)),
                ci95=[float(lo), float(hi)], p_boot=float(p_two), n=int(n))


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run_max = 0.0
    for rank, i in enumerate(order):
        run_max = max(run_max, (m - rank) * pvals[i])
        adj[i] = min(run_max, 1.0)
    return adj.tolist()


def main(tasks):
    rng = np.random.default_rng(SEED)
    rows = []
    for task in tasks:
        f = OUT / f"oof_{task}.parquet"
        if not f.exists():
            print(f"skip {task}: no oof"); continue
        oof = pd.read_parquet(f)
        higher = task != "rna"  # RNA label is RMSD (lower better)
        if task == "membrane":
            # per-protein spearman for each <model>::<predictor> key
            scores = {}
            for m, sub in oof.groupby("model"):
                scores[m] = per_group_spearman(sub, higher_better=True)
            pairs = []
            for pred in ("ridge", "rf", "xgb", "enet"):
                pairs += [
                    (f"A4_PT::{pred}", f"A1_P::{pred}", f"dT={pred}"),
                    (f"A5_PH::{pred}", f"A1_P::{pred}", f"dH={pred}"),
                    (f"A7_PTH::{pred}", f"A1_P::{pred}", f"dTH={pred}"),
                ]
        else:
            scores = {m: per_group_spearman(sub, higher)
                      for m, sub in oof.groupby("model")}
            pairs = [(a, b, lbl) for a, b, lbl in RANK_PAIRS
                     if a in scores and b in scores]
        res = []
        for a, b, lbl in pairs:
            r = bootstrap_diff(scores[a], scores[b], rng)
            r.update(task=task, model_a=a, model_b=b, comparison=lbl)
            res.append(r)
        adj = holm([r.get("p_boot", 1.0) for r in res])
        for r, p in zip(res, adj):
            r["p_holm"] = p
        rows.extend(res)
        for r in res:
            print(f"[{task}] {r['comparison']:26s} {r['model_a']:18s}-{r['model_b']:18s} "
                  f"d={r.get('mean', float('nan')):+.4f} CI{r.get('ci95')} p_holm={r['p_holm']:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "paired_bootstrap.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["rna", "membrane", "pd_bound", "pd_unbound"])
