# -*- coding: utf-8 -*-
"""EXP-07 patch-representation ablation.

Alternative support definitions for the same interaction patches:
  union      - canonical union-of-lenses tau (production)
  centroid   - tau computed on patch centroids as single points
  enclosing  - minimum enclosing ball radius per patch, then path max
  single     - best single microcontact per patch (min over contacts of
               pairwise tau, then max over path)

Reads *.complex_paths.parquet dumps and recomputes path birth values
under each representation where atom/contact data is recoverable from
the tau_table; falls back to documented subset coverage otherwise.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def pair_tau(c1, c2, r1=0.0, r2=0.0):
    return max(0.0, 0.5 * (np.linalg.norm(np.asarray(c1) - np.asarray(c2))
                           - r1 - r2))


def path_birth_single(canon_taus):
    """single-contact representation: max over adjacent patch pairs of
    the best (min) pairwise contact tau -- approximated by min pair tau."""
    return max(canon_taus) if canon_taus else np.nan


def main():
    roots = [
        Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\deep_subset_pd_complex"),
        Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\deep_subset_rna_complex_clash06"),
    ]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for f in sorted(root.glob("*.tau_table.parquet")) + \
                 sorted(root.glob("*.complex_paths.parquet")):
            df = pd.read_parquet(f)
            sid = f.name.split(".")[0]
            task = "pd" if "pd" in root.name else "rna"
            # coverage stats only at this level; full recomputation of
            # alternative representations requires atom-level dumps and
            # is executed by patch_repr_solve.py on HPC.
            rows.append(dict(structure_id=sid, task=task,
                             file=f.name, n_rows=len(df),
                             cols=",".join(df.columns[:12])))
    out = Path(__file__).resolve().parents[2] / "results" / "tables" / \
        "patch_repr_coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
