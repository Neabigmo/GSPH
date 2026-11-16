# -*- coding: utf-8 -*-
"""Formal T×H interaction test on OOF predictions.

  I_TH(g) = [S(P+T+H) - S(P+T)] - [S(P+H) - S(P)]   per target g

  = conditional Hodge increment (given T) minus unconditional Hodge increment.
  Positive I_TH  =>  Hodge is worth MORE when topology is present (synergy).

  Statistics: mean, paired cluster bootstrap CI (10k), sign-flip permutation
  p-value (10k), Holm correction across the three tasks.
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
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\ablation")
N = 10_000
rng = np.random.default_rng(7)


def per_target(oof, higher=True):
    vals = {}
    for g, sub in oof.groupby("group_id"):
        y = sub.y_true if higher else -sub.y_true
        if y.nunique() > 1 and sub.y_pred.nunique() > 1:
            r = spearmanr(y, sub.y_pred).statistic
            if np.isfinite(r):
                vals[g] = r
    return pd.Series(vals)


def test_task(task):
    f = OUT / f"oof_{task}.parquet"
    if not f.exists():
        return None
    oof = pd.read_parquet(f)
    higher = task != "rna"
    need = ["A1_P", "A4_PT", "A5_PH", "A7_PTH"]
    sc = {m: per_target(oof[oof.model == m], higher) for m in need}
    if any(s.empty for s in sc.values()):
        return None
    common = sc["A1_P"].index
    for s in sc.values():
        common = common.intersection(s.index)
    common = common.sort_values()
    I = ((sc["A7_PTH"][common] - sc["A4_PT"][common])
         - (sc["A5_PH"][common] - sc["A1_P"][common])).to_numpy()
    n = len(I)
    idx = rng.integers(0, n, size=(N, n))
    bs = I[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # sign-flip permutation: flip signs of per-target I values
    flips = rng.choice([-1, 1], size=(N, n))
    ps = (flips * I).mean(axis=1)
    p_perm = float((np.abs(ps) >= abs(I.mean())).mean())
    return dict(task=task, n_targets=n, interaction_mean=float(I.mean()),
                ci95=[float(lo), float(hi)], p_perm=p_perm,
                per_target_I=I.tolist(),
                delta_H_given_T=float((sc["A7_PTH"][common] - sc["A4_PT"][common]).mean()),
                delta_H_no_T=float((sc["A5_PH"][common] - sc["A1_P"][common]).mean()))


def main():
    res = [test_task(t) for t in ("rna", "pd_bound", "pd_unbound")]
    res = [r for r in res if r]
    # Holm across tasks
    order = np.argsort([r["p_perm"] for r in res])
    m = len(res); run_max = 0.0; adj = np.empty(m)
    for rank, i in enumerate(order):
        run_max = max(run_max, (m - rank) * res[i]["p_perm"])
        adj[i] = min(run_max, 1.0)
    rows = []
    for r, p in zip(res, adj):
        r["p_holm"] = float(p)
        rows.append({k: v for k, v in r.items() if k != "per_target_I"})
        for g, v in zip(range(r["n_targets"]), r["per_target_I"]):
            pass
        print(f"[{r['task']}] I_TH={r['interaction_mean']:+.4f} "
              f"CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"p_perm={r['p_perm']:.4f} p_holm={p:.3f} "
              f"(dH|T={r['delta_H_given_T']:+.4f} vs dH={r['delta_H_no_T']:+.4f})")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "interaction_test_v2.csv", index=False)
    # per-target detail
    det = []
    for r in res:
        for i, v in enumerate(r["per_target_I"]):
            det.append(dict(task=r["task"], target_idx=i, I_TH=v))
    pd.DataFrame(det).to_csv(OUT / "interaction_per_target_v2.csv", index=False)


if __name__ == "__main__":
    main()
