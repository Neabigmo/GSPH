# -*- coding: utf-8 -*-
"""Step-6 statistics for v2 models on oof_v2_<task>.parquet  (R2, corrected).

Fixes vs. first pass:
  * top-1 metric oriented so HIGHER = better for every task
    (RNA uses -best_RMSD, PD uses best_CAD) -> diffs are unambiguous.
  * Holm correction applied to BOTH spearman and top-1 families.
  * Cross-compression interaction arms added (localise T×H source).
  * CleanFull-PTHS attribution: vs A1_P and vs B15_PfullS.
  * homology-cluster resampling: NO family labels exist in the candidate
    metadata (verified) -> per-target bootstrap is the finest available
    grouping; documented as a limitation.

Interacions (target-level, all from same OOF):
  I_full    = A7_PTH - A4_PT - A5_PH + A1_P
  I_compact = B9_PcT2H2 - B7_PcT2 - B8_PcH2 + B2_PcV2
  I_cH      = X_P_T_H2 - A4_PT - X_P_H2 + A1_P      (only H compressed)
  I_cT      = X_P_T2_H - X_P_T2 - A5_PH + A1_P      (only T compressed)
  I_cP      = X_Pc_T_H - X_Pc_T - X_Pc_H + B2_PcV2  (only P compressed)
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
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\ablation")
N = 10_000
rng = np.random.default_rng(11)

PAIRS = [
    ("B7_PcT2", "B2_PcV2", "dT=Pc+T2-Pc"),
    ("B8_PcH2", "B2_PcV2", "dH=Pc+H2-Pc"),
    ("B9_PcT2H2", "B2_PcV2", "dTH=Pc+T2+H2-Pc"),
    ("B9_PcT2H2", "B7_PcT2", "dH|T"),
    ("B9_PcT2H2", "B8_PcH2", "dT|H"),
    ("B9_PcT2H2", "B10_Pc_T2res_H2res", "full-vs-resid"),
    ("B9_PcT2H2", "B11_Sv2", "full-vs-strictsize"),
    ("B9_PcT2H2", "A1_P", "compactfull-vs-Pfull"),
    ("B2_PcV2", "A1_P", "PcV2-vs-Pfull"),
    ("B4_T2", "A2_T", "T2-vs-Told"),
    ("B6_H2", "A3_H", "H2-vs-Hold"),
    ("B11_Sv2", "A8_S", "strictS-vs-mixedS"),
    ("B14_PcNoLJ", "B2_PcV2", "noLJ-effect"),
    ("A1_P__nolj", "A1_P", "Pfull-noLJ-effect"),
    ("A7_PTH__nolj", "A7_PTH", "PTH-noLJ-effect"),
    ("CleanFull_PTHS", "A1_P", "PTHS-vs-P"),
    ("CleanFull_PTHS", "B15_PfullS", "PTHS-vs-P+S"),
    ("CleanFull_PTHS", "A7_PTH", "PTHS-vs-PTH(dS)"),
    ("B13_PcT2H2S", "B12_PcS", "PcTHS-vs-PcS(dTH|S)"),
    ("B13_PcT2H2S", "B9_PcT2H2", "PcTHS-vs-PcTH(dS)"),
    ("P_PCA142", "A1_P", "PCA142-vs-Pfull"),
]


def per_target(oof, higher=True):
    vals = {}
    for g, sub in oof.groupby("group_id"):
        y = sub.y_true if higher else -sub.y_true
        if y.nunique() > 1 and sub.y_pred.nunique() > 1:
            r = spearmanr(y, sub.y_pred).statistic
            if np.isfinite(r):
                vals[g] = r
    return pd.Series(vals)


def per_target_top1(oof, higher=True):
    """top-1 metric oriented higher=better (RNA: -bestRMSD; PD: bestCAD)."""
    vals = {}
    for g, sub in oof.groupby("group_id"):
        t = sub.nlargest(1, "y_pred")["y_true"]
        vals[g] = float(t.max() if higher else -t.min())
    return pd.Series(vals)


def boot(a, b):
    common = a.index.intersection(b.index).sort_values()
    d = (a[common] - b[common]).to_numpy()
    n = len(d)
    if n == 0:
        return dict(n=0)
    bs = d[rng.integers(0, n, (N, n))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    flips = rng.choice([-1, 1], (N, n))
    p_perm = float((np.abs((flips * d).mean(1)) >= abs(d.mean())).mean())
    return dict(mean=float(d.mean()), ci95=[round(float(lo), 4), round(float(hi), 4)],
                p_perm=p_perm, n=n)


def holm(res_list):
    order = np.argsort([r["p_perm"] for r in res_list])
    mx = 0.0; adj = np.empty(len(res_list))
    for rank, i in enumerate(order):
        mx = max(mx, (len(res_list) - rank) * res_list[i]["p_perm"])
        adj[i] = min(mx, 1.0)
    for r, p in zip(res_list, adj):
        r["p_holm"] = float(p)


def interaction(sp, terms, label, task, out):
    """terms = [(model, sign), ...]; I = sum(sign * sp[model]) per target."""
    common = None
    for m, s in terms:
        if m not in sp:
            return None
        common = sp[m].index if common is None else common.intersection(sp[m].index)
    common = common.sort_values()
    I = sum(s * sp[m][common] for m, s in terms).to_numpy()
    n = len(I)
    bs = I[rng.integers(0, n, (N, n))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    flips = rng.choice([-1, 1], (N, n))
    p = float((np.abs((flips * I).mean(1)) >= abs(I.mean())).mean())
    r = dict(task=task, interaction=label, n_targets=n, mean=float(I.mean()),
             ci95=[round(float(lo), 4), round(float(hi), 4)], p_perm=p)
    out.append(r)
    print(f"[{task}] {label}: I={I.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}] p_perm={p:.4f}")
    return r


def main():
    rows, inter = [], []
    INTER_ACTIONS = {
        "I_full   (P,T,H full)": [("A7_PTH", 1), ("A4_PT", -1), ("A5_PH", -1), ("A1_P", 1)],
        "I_compact(Pc,T2,H2)": [("B9_PcT2H2", 1), ("B7_PcT2", -1), ("B8_PcH2", -1), ("B2_PcV2", 1)],
        "I_cH     (only H comp)": [("X_P_T_H2", 1), ("A4_PT", -1), ("X_P_H2", -1), ("A1_P", 1)],
        "I_cT     (only T comp)": [("X_P_T2_H", 1), ("X_P_T2", -1), ("A5_PH", -1), ("A1_P", 1)],
        "I_cP     (only P comp)": [("X_Pc_T_H", 1), ("X_Pc_T", -1), ("X_Pc_H", -1), ("B2_PcV2", 1)],
    }
    for task in ("rna", "pd_bound", "pd_unbound"):
        f = OUT / f"oof_v2_{task}.parquet"
        if not f.exists():
            continue
        oof = pd.read_parquet(f)
        higher = task != "rna"
        sp = {m: per_target(oof[oof.model == m], higher) for m in oof.model.unique()}
        t1 = {m: per_target_top1(oof[oof.model == m], higher) for m in oof.model.unique()}
        sp_res, t1_res = [], []
        for a_, b_, lbl in PAIRS:
            if a_ in sp and b_ in sp:
                r = boot(sp[a_], sp[b_]); r.update(task=task, metric="spearman",
                                                 model_a=a_, model_b=b_, comparison=lbl)
                sp_res.append(r)
                r2 = boot(t1[a_], t1[b_]); r2.update(task=task, metric="top1_oriented",
                                                   model_a=a_, model_b=b_, comparison=lbl)
                t1_res.append(r2)
        holm(sp_res); holm(t1_res)
        res = sp_res + t1_res
        rows.extend(res)
        for r in res:
            print(f"[{task}] {r['metric']:14s} {r['comparison']:28s} "
                  f"d={r['mean']:+.4f} CI{r['ci95']} p={r['p_perm']:.4f} "
                  f"holm={r['p_holm']:.3f}")
        for lbl, terms in INTER_ACTIONS.items():
            interaction(sp, terms, lbl, task, inter)
    if inter:
        order = np.argsort([r["p_perm"] for r in inter])
        mx = 0.0; adj = np.empty(len(inter))
        for rank, i in enumerate(order):
            mx = max(mx, (len(inter) - rank) * inter[i]["p_perm"])
            adj[i] = min(mx, 1.0)
        for r, p in zip(inter, adj):
            r["p_holm"] = float(p)
            print(f"  -> {r['task']} {r['interaction']} holm={p:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "topk_bootstrap_v2.csv", index=False)
    pd.DataFrame(inter).to_csv(OUT / "interaction_test_compact_v2.csv", index=False)


if __name__ == "__main__":
    main()
