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
"""Evaluate VoroMQA scores (re-run on our exact FTDMP decoys on the HPC)
against the same CAD/RMSD labels and protocol as eval_ftdmp_baseline.py.

Input: baselines_ext/pd_voromqa_scores.jsonl
  {"pdb": "Bound_docking/1A74/relaxed_top_complexes/x.pdb", "voromqa": 0.42, ...}
VoroMQA: higher = better quality.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(ROOT, "..", "..", "data", "protein_dna",
                                     "ftdmp_labels", "Protein-DNA"))
JSONL = os.path.normpath(os.path.join(ROOT, "..", "..", "baselines_ext",
                                      "pd_voromqa_scores.jsonl"))


def load_labels(pdir):
    cad = pd.read_csv(os.path.join(pdir, "cad_scores.txt"), sep=r"\s+")
    cad.columns = ["model", "iface_cad", "iface_site_cad"][:len(cad.columns)]
    rmsd = pd.read_csv(os.path.join(pdir, "rmsd_results.txt"), sep=r"\s+",
                       header=None, names=["model", "ligand_rmsd"])
    for col in ("iface_cad", "iface_site_cad", "ligand_rmsd"):
        if col in cad.columns or col in rmsd.columns:
            pass
    cad["iface_cad"] = pd.to_numeric(cad["iface_cad"], errors="coerce")
    cad["iface_site_cad"] = pd.to_numeric(cad["iface_site_cad"],
                                          errors="coerce")
    rmsd["ligand_rmsd"] = pd.to_numeric(rmsd["ligand_rmsd"], errors="coerce")
    return cad.merge(rmsd, on="model")


def main():
    scores = pd.DataFrame([
        json.loads(l) for l in open(JSONL) if l.strip()])
    scores["voromqa"] = pd.to_numeric(scores["voromqa"], errors="coerce")
    scores = scores.dropna(subset=["voromqa"])
    def split_pdb(p):
        toks = str(p).replace("\\", "/").split("/")
        for i, t in enumerate(toks):
            if t in ("Bound_docking", "Unbound_docking") and i + 2 < len(toks):
                return t, toks[i + 1], toks[-1]
        return None, None, None
    tr = scores["pdb"].map(split_pdb)
    scores["protocol"] = tr.map(lambda x: x[0])
    scores["target"] = tr.map(lambda x: x[1])
    scores["model"] = tr.map(lambda x: x[2])
    scores = scores.dropna(subset=["protocol"])

    rows = []
    for (proto, target), sc in scores.groupby(["protocol", "target"]):
        pdir = os.path.join(DATA, proto, target)
        if not os.path.isfile(os.path.join(pdir, "cad_scores.txt")):
            continue
        df = sc.merge(load_labels(pdir), on="model").dropna(
            subset=["iface_cad"])
        if len(df) < 5:
            continue
        df = df.sort_values("voromqa", ascending=False)
        top1, top5 = df.iloc[0], df.iloc[:5]
        rows.append({
            "target": target, "protocol": proto, "n": len(df),
            "spearman_rank_cad": spearmanr(df["voromqa"],
                                           df["iface_cad"])[0],
            "top1_cad": top1["iface_cad"],
            "top1_site_cad": top1["iface_site_cad"],
            "top1_rmsd": top1["ligand_rmsd"],
            "best_in_top5_cad": top5["iface_cad"].max(),
            "best_possible_cad": df["iface_cad"].max(),
            "n_acceptable_pool": int((df["iface_cad"] >= 0.5).sum()),
        })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ROOT, "ftdmp_voromqa_per_target.csv"),
               index=False)
    s = res.groupby("protocol").agg(
        n_targets=("target", "count"), n_models=("n", "sum"),
        mean_spearman_cad=("spearman_rank_cad", "mean"),
        mean_top1_cad=("top1_cad", "mean"),
        frac_top1_acceptable=("top1_cad", lambda x: float((x >= 0.5).mean())),
        frac_top5_acceptable=("best_in_top5_cad",
                              lambda x: float((x >= 0.5).mean())),
        mean_top1_rmsd=("top1_rmsd", "mean"),
        mean_best_possible=("best_possible_cad", "mean"),
    ).reset_index()
    s.insert(0, "method", "voromqa_rerun")
    s.to_csv(os.path.join(ROOT, "ftdmp_voromqa_summary.csv"), index=False)
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
