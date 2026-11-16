# -*- coding: utf-8 -*-
"""Hard-pair discrimination: does GSPH order physically-ambiguous pairs?

For each target, consider all decoy pairs whose TRUE labels differ by at
least the target's 75th percentile of |Δlabel| ("hard" quality contrast).
A model "wins" a pair if sign(pred_i - pred_j) == sign(y_i - y_j).

We then restrict further to pairs that are PHYSICALLY AMBIGUOUS:
|ŷ_P(i) - ŷ_P(j)| below the target's 50th percentile of |Δŷ_P| — i.e. the
physical-only model cannot separate them confidently.  On those pairs we
compare A1_P vs A7_PTH (and B2_PcV2 vs B9_PcT2H2) pair-win rates.

Paired target bootstrap + sign-flip on the per-target win-rate difference.
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

OUT = Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\ablation")
N = 10_000
rng = np.random.default_rng(3)


def pair_stats(oof, m_a, m_b, higher=True):
    """per-target: win rates on hard+physically-ambiguous pairs."""
    res = {}
    for g, sub in oof.groupby("group_id"):
        pa = sub[sub.model == m_a].set_index("structure_id")
        pb = sub[sub.model == m_b].set_index("structure_id")
        pth = sub[sub.model == m_b + "__X"] if False else None
        common = pa.index.intersection(pb.index)
        if len(common) < 10:
            continue
        ya = pa.loc[common, "y_true"].to_numpy(float)
        yb = pb.loc[common, "y_true"].to_numpy(float)
        sa = pa.loc[common, "y_pred"].to_numpy(float)
        sb = pb.loc[common, "y_pred"].to_numpy(float)
        # need a third model for ambiguity -> handled by caller variant
        res[g] = (ya, sa, sb)
    return res


def winrate(y, s):
    n = len(y)
    iu = np.triu_indices(n, 1)
    dy = y[iu[0]] - y[iu[1]]
    ds = s[iu[0]] - s[iu[1]]
    hard = np.abs(dy) >= np.quantile(np.abs(dy), 0.75)
    amb = np.abs(ds) <= np.quantile(np.abs(ds), 0.5)   # ambiguous for THIS model's partner? placeholder
    return dy, ds, hard


def oriented_label(y, task):
    """Return label oriented so HIGHER = better.

    RNA y_true is official RMSD (lower is better) while y_pred is a learned
    quality score that is NEGATIVELY correlated with RMSD (verified:
    spearman(y_pred, RMSD) ~ -0.66).  The correct pair win is therefore
    sign(dy_pred) == sign(-dRMSD).  PD labels (CAD) are already higher=better.
    """
    return -np.asarray(y, dtype=float) if task == "rna" else np.asarray(y, dtype=float)


def per_target_diffs(oof, m_phys, m_full, task):
    """Per-target win-rate difference (full - phys) on hard+ambiguous pairs.

    Returns dict group_id -> (d, n_pairs, win_phys, win_full).
    """
    out = {}
    for g, sub in oof.groupby("group_id"):
        pp = sub[sub.model == m_phys].set_index("structure_id")
        pf = sub[sub.model == m_full].set_index("structure_id")
        common = pp.index.intersection(pf.index)
        if len(common) < 15:
            continue
        y = oriented_label(pp.loc[common, "y_true"].to_numpy(float), task)
        sp_ = pp.loc[common, "y_pred"].to_numpy(float)
        sf = pf.loc[common, "y_pred"].to_numpy(float)
        if np.ptp(y) == 0:
            continue
        iu = np.triu_indices(len(y), 1)
        dy = y[iu[0]] - y[iu[1]]
        dsp = sp_[iu[0]] - sp_[iu[1]]
        dsf = sf[iu[0]] - sf[iu[1]]
        hard = np.abs(dy) >= np.quantile(np.abs(dy), 0.75)
        # physically ambiguous: phys-only score diff in bottom half
        amb = np.abs(dsp) <= np.quantile(np.abs(dsp), 0.5)
        m = hard & amb
        if m.sum() < 5:
            continue
        wa = (np.sign(dsp[m]) == np.sign(dy[m])).mean()
        wb = (np.sign(dsf[m]) == np.sign(dy[m])).mean()
        out[g] = (float(wb - wa), int(m.sum()), float(wa), float(wb))
    return out


def summarize_diffs(per_t, rng, n_boot=N):
    pt = pd.DataFrame(per_t, index=["d", "n", "phys", "full"]).T
    d = pt["d"].to_numpy()
    n = len(d)
    if n == 0:
        return None
    bs = d[rng.integers(0, n, (n_boot, n))].mean(1)
    fl = rng.choice([-1, 1], (n_boot, n))
    p = float((np.abs((fl * d).mean(1)) >= abs(d.mean())).mean())
    return dict(n_targets=n, mean_d=float(d.mean()),
                ci_lo=float(np.percentile(bs, 2.5)),
                ci_hi=float(np.percentile(bs, 97.5)), p_perm=p,
                mean_phys=float(pt["phys"].mean()),
                mean_full=float(pt["full"].mean()),
                mean_pairs=float(pt["n"].mean()))


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "pd_bound"
    oof_file = sys.argv[2] if len(sys.argv) > 2 else f"oof_v2_{task}.parquet"
    oof = pd.read_parquet(OUT / oof_file)
    pairs_cfg = [("A1_P", "A7_PTH"), ("B2_PcV2", "B9_PcT2H2")]
    if task == "rna":
        pairs_cfg = [("A1_P", "CleanFull_PTHS"), ("B2_PcV2", "B9_PcT2H2")]
    rows = []
    for m_phys, m_full in pairs_cfg:
        per_t = per_target_diffs(oof, m_phys, m_full, task)
        s = summarize_diffs(per_t, rng)
        if s is None:
            continue
        s.update(task=task, contrast=f"{m_full}-vs-{m_phys}", oof_file=oof_file)
        rows.append(s)
        print(f"[{task}] {m_full} vs {m_phys}: d={s['mean_d']:+.4f} "
              f"CI[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] p={s['p_perm']:.4f} "
              f"(phys={s['mean_phys']:.3f} full={s['mean_full']:.3f})")
    tag = oof_file.replace(".parquet", "")
    pd.DataFrame(rows).to_csv(OUT / f"hardpair_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
