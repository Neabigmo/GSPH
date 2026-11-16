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
#!/usr/bin/env python3
"""Evaluate the official FTDMP ranking against CAD-score / ligand-RMSD labels.

Data: GSPH_H3/data/protein_dna/ftdmp_labels/Protein-DNA/
  {Bound_docking,Unbound_docking}/<T>/
    scoring_results-ranks.txt    ID <rank>   (rank 1 = FTDMP's best pick)
    rmsd_results.txt             model.pdb <ligand_rmsd>
    cad_scores.txt               model.pdb <iface_cad> <iface_site_cad>
    binding_site_CAD-scores.txt  model.pdb <site_cad>

Metrics per target: top-1/top-5 CAD & RMSD of the official ranking, success
rates (CAD>=0.5 acceptable), Spearman(rank, label).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(ROOT, "..", "..", "data", "protein_dna",
                                     "ftdmp_labels", "Protein-DNA"))


def load_target(pdir):
    ranks = pd.read_csv(os.path.join(pdir, "scoring_results-ranks.txt"),
                        sep=r"\s+")
    ranks["model"] = ranks["ID"] + ".pdb"
    ranks["relaxed_RJS_rank"] = pd.to_numeric(ranks["relaxed_RJS_rank"],
                                            errors="coerce")
    cad = pd.read_csv(os.path.join(pdir, "cad_scores.txt"), sep=r"\s+")
    cad.columns = ["model", "iface_cad", "iface_site_cad"][:len(cad.columns)]
    for col in ("iface_cad", "iface_site_cad"):
        cad[col] = pd.to_numeric(cad[col], errors="coerce")
    rmsd = pd.read_csv(os.path.join(pdir, "rmsd_results.txt"), sep=r"\s+",
                       header=None, names=["model", "ligand_rmsd"])
    df = ranks[["model", "relaxed_RJS_rank"]].merge(cad, on="model").merge(
        rmsd, on="model")
    return df


def main():
    rows = []
    for proto in ("Bound_docking", "Unbound_docking"):
        pdir0 = os.path.join(DATA, proto)
        if not os.path.isdir(pdir0):
            continue
        for target in sorted(os.listdir(pdir0)):
            pdir = os.path.join(pdir0, target)
            if not os.path.isfile(os.path.join(pdir, "scoring_results-ranks.txt")):
                continue
            df = load_target(pdir)
            if len(df) < 5:
                continue
            df = df.sort_values("relaxed_RJS_rank")
            top1, top5 = df.iloc[0], df.iloc[:5]
            rows.append({
                "target": target, "protocol": proto, "n": len(df),
                "spearman_rank_cad": spearmanr(-df["relaxed_RJS_rank"],
                                               df["iface_cad"])[0],
                "top1_cad": top1["iface_cad"],
                "top1_site_cad": top1["iface_site_cad"],
                "top1_rmsd": top1["ligand_rmsd"],
                "best_in_top5_cad": top5["iface_cad"].max(),
                "best_possible_cad": df["iface_cad"].max(),
                "n_acceptable_pool": int((df["iface_cad"] >= 0.5).sum()),
            })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ROOT, "ftdmp_baseline_per_target.csv"), index=False)
    s = res.groupby("protocol").agg(
        n_targets=("target", "count"), n_models=("n", "sum"),
        mean_spearman_cad=("spearman_rank_cad", "mean"),
        mean_top1_cad=("top1_cad", "mean"),
        frac_top1_acceptable=("top1_cad", lambda x: float((x >= 0.5).mean())),
        frac_top5_acceptable=("best_in_top5_cad", lambda x: float((x >= 0.5).mean())),
        mean_top1_rmsd=("top1_rmsd", "mean"),
        mean_best_possible=("best_possible_cad", "mean"),
    ).reset_index()
    s.insert(0, "method", "ftdmp_official_rank")
    s.to_csv(os.path.join(ROOT, "ftdmp_baseline_summary.csv"), index=False)
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
