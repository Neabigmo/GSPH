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
"""Evaluate official RNA scoring baselines on the PM decoy set.

Joins baseline score CSVs (baselines/rna/*_scores.csv) with official
RMSD/DI labels (external/rsRNASP/PM_decoy_set/{RMSD,DI}/<target>.*) and
reports per-target Spearman correlation and top-1 quality.

Score direction: all of rsRNASP/cgRNASP are energies where lower = better.
Lower RMSD / lower DI = better structure (check paired PM labels before reporting).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PM = os.path.normpath(os.path.join(ROOT, "..", "..", "external", "rsRNASP", "PM_decoy_set"))

SCORE_FILES = {
    "rsRNASP": "rsrnasp_scores.csv",
    "cgRNASP": "cgRNASP_scores.csv",
    "cgRNASP-C": "cgRNASP-C_scores.csv",
    "cgRNASP-PC": "cgRNASP-PC_scores.csv",
    # Statistical potentials re-run on this dataset (HPC build from source,
    # no pretrained weights): baselines_ext/rna_statpot_scores.jsonl
    "3dRNAscore": "3dRNAscore_scores.csv",
    "DFIRE-RNA": "DFIRE-RNA_scores.csv",
    # RASP excluded: rasp_fd is a protein-only all-atom potential; it cannot
    # parse RNA atom names (outputs -nan/garbage columns) so its numbers are
    # meaningless on this dataset.
    "RNA-BRiQ": "RNA-BRiQ_scores.csv",
}


def load_labels():
    rows = []
    for target in sorted(os.listdir(PM)):
        tdir = os.path.join(PM, target)
        if not os.path.isdir(tdir):
            continue
        for kind, fname in (("rmsd", os.path.join(PM, "RMSD", f"{target}.rmsd")),
                            ("di", os.path.join(PM, "DI", f"{target}.DI"))):
            if not os.path.isfile(fname):
                continue
            for line in open(fname):
                parts = line.split()
                if len(parts) < 2:
                    continue
                rows.append({"target": target, "structure": parts[0],
                             kind: float(parts[1])})
    df = pd.DataFrame(rows)
    df = df.groupby(["target", "structure"], as_index=False).first()
    return df


def spearman(a, b):
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return r


def evaluate(df_scores, labels, name):
    df = labels.merge(df_scores, on=["target", "structure"], how="inner")
    out = []
    for target, g in df.groupby("target"):
        g = g.dropna(subset=["score_kbt", "rmsd", "di"])
        if len(g) < 5:
            continue
        # score lower better -> negate so higher predicts better quality
        neg = -g["score_kbt"]
        r_rmsd = spearman(neg, -g["rmsd"])   # vs quality = -RMSD
        r_di = spearman(neg, -g["di"])       # DI lower = better, like RMSD
        # top-1 pick quality: RMSD of lowest-scoring decoy (excluding native)
        d = g[g["structure"] != f"{target}.pdb"]
        top1_rmsd = d.loc[d["score_kbt"].idxmin(), "rmsd"] if len(d) else np.nan
        native_rank = int((d["score_kbt"] < g.loc[g["structure"] == f"{target}.pdb", "score_kbt"].iloc[0]).sum() + 1) if (g["structure"] == f"{target}.pdb").any() else np.nan
        out.append({"target": target, "n": len(g), "spearman_rmsd": r_rmsd,
                    "spearman_di": r_di, "top1_rmsd": top1_rmsd,
                    "native_rank": native_rank})
    res = pd.DataFrame(out)
    res["method"] = name
    return res, len(df)


def main():
    labels = load_labels()
    frames = []
    summary = []
    for name, fname in SCORE_FILES.items():
        path = os.path.join(ROOT, fname)
        if not os.path.isfile(path):
            print(f"[skip] {name}: {fname} not found")
            continue
        sc = pd.read_csv(path)
        res, n = evaluate(sc, labels, name)
        frames.append(res)
        summary.append({
            "method": name, "n_structures": n, "n_targets": len(res),
            "mean_spearman_rmsd": res["spearman_rmsd"].mean(),
            "mean_spearman_di": res["spearman_di"].mean(),
            "mean_top1_rmsd": res["top1_rmsd"].mean(),
            "median_native_rank": res["native_rank"].median(),
        })
    allres = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    allres.to_csv(os.path.join(ROOT, "rna_baseline_per_target.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(ROOT, "rna_baseline_summary.csv"), index=False)
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    sys.exit(main())
