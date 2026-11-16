# -*- coding: utf-8 -*-
"""EXP-05: aggregate per-path support records into landscape tables.

Reads complex_paths.parquet + tau_table.parquet dumps and produces
results/tables/path_landscape.parquet with per-(structure, graph_mode,
path_order) summaries plus per-canonical-patch-set tau distributions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = [
    Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\deep_subset_pd_complex"),
    Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\deep_subset_rna_complex_clash06"),
]
OUT = Path(__file__).resolve().parents[2] / "results" / "tables" / "path_landscape.parquet"


def main():
    frames = []
    for root in ROOTS:
        if not root.exists():
            print("missing", root); continue
        for f in sorted(root.glob("*.complex_paths.parquet")):
            sid = f.name.split(".")[0]
            df = pd.read_parquet(f)
            df["structure_id"] = sid
            task = "pd" if "pd" in root.name else "rna"
            df["task"] = task
            frames.append(df)
    if not frames:
        print("no data"); return
    df = pd.concat(frames, ignore_index=True)
    df["tau"] = pd.to_numeric(df["tau"], errors="coerce")
    g = df.groupby(["task", "structure_id", "graph", "order"], dropna=False)
    summ = g.agg(n_paths=("tau", "size"),
                 n_valid=("valid", "sum") if "valid" in df else ("tau", "count"),
                 tau_mean=("tau", "mean"), tau_median=("tau", "median"),
                 tau_p90=("tau", lambda s: s.quantile(0.9)),
                 tau_max=("tau", "max"))
    if "eta3" in df.columns:
        summ["eta3_pos_frac"] = g.apply(
            lambda x: float((pd.to_numeric(x["eta3"], errors="coerce") > 0).mean()))
    summ = summ.reset_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summ.to_parquet(OUT, index=False)
    print(summ.head(20).to_string())
    print("rows:", len(summ), "->", OUT)


if __name__ == "__main__":
    main()
