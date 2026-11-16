# -*- coding: utf-8 -*-
"""Step-2/3: build v2 feature groups from EXISTING parquets only.

  topology_features_v2.parquet   (T2 = normalized GLMY + pathchem ratios +
                                  path stats + tau/eta/dtau + persistence counts)
  hodge_features_v2.parquet      (H2 = scale_0.5 low-mode readouts + gaps +
                                  signed weight mass + cross-channel corr, ~87d)
  physical_compact_v2.parquet    (Pcompact-v2 ~150d covering every channel)
  feature_manifest_v2.csv / feature_health_v2.csv

  UNRECOVERABLE (documented, not fabricated):
    T1 per-path support distributions — engine computed per-path tau but only
       _mean/_min/_max were persisted; quantiles require re-extraction (forbidden).
    T3 real persistence barcodes — only interval *counts* were persisted;
       birth/death cannot be recovered without re-running the engine.
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
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ablation import load_task, TASKS      # noqa: E402
from feature_health import audit_frame         # noqa: E402

H3 = Path(r"H:\2026try\11.3GSPH\GSPH_H3")
OUT = H3 / "results" / "ablation"


def dpref(cols):
    """column prefix for membrane delta-features ('d_') or ''."""
    return "d_" if any(c.startswith("d_patch_count") for c in cols) else ""


def add_glmy_norms(df: pd.DataFrame, pfx: str) -> pd.DataFrame:
    """derive normalized GLMY ratios in-place; return new column names."""
    new = []
    scales = sorted({m.group(1) for c in df.columns
                     for m in [re.match(pfx + r"scale_([0-9.]+)_glmy_dim_Omega_0$", c)]
                     if m})
    for s in scales:
        for k in range(4):
            den = df.get(f"{pfx}scale_{s}_glmy_dim_Omega_{k}")
            if den is None:
                continue
            den = den.replace(0, np.nan)
            for num in ("rank_boundary", "betti", "kernel", "dim_A", "dim_gap"):
                c = f"{pfx}scale_{s}_glmy_{num}_{k}"
                if c in df.columns:
                    nc = f"{pfx}scale_{s}_glmy_{num}_{k}__normO"
                    df[nc] = (df[c] / den).fillna(0.0)
                    new.append(nc)
    return new


def group_v2(cols, pfx):
    """Return dict of v2 column lists from the dataframe's (derived) columns."""
    norm = [c for c in cols if c.endswith("__normO")]
    # --- T2 ---
    t = []
    t += [c for c in cols if re.search(pfx + r"scale_[0-9.]+_glmy_(betti|kernel|dim_gap|filling|admissibility)_\d+$", c)]
    t += norm
    t += [c for c in cols if re.match(pfx + r"pathchem_(role_diversity|role_entropy|transition_mean|transition_std)_k\d$", c)]
    t += [c for c in cols if re.match(pfx + r"path_(geom_span_mean|support_mean|support_max)_k\d$", c)]
    t += [c for c in cols if re.match(pfx + r"(tau|eta|delta_tau)_", c)]
    t += [c for c in cols if re.match(pfx + r"persistence_interval_count_k\d$", c)]
    # complex-scale quantities belong to T2 (they are properties of the path
    # complex, not of the physical structure size)
    t += [c for c in cols if re.match(pfx + r"(path_count|pathchem_count)_k\d$", c)]
    t += [c for c in cols if re.match(pfx + r"scale_[0-9.]+_glmy_(dim_Omega|rank_boundary|dim_A)_\d+$", c)]
    # --- S_v2: STRICT structure-size controls only ---
    s = []
    s += [c for c in cols if re.match(pfx + r"(patch_count|patch_directed_edge_count|rna_bases_total)$", c)]
    s += [c for c in cols if re.search(r"contact_count|atom_count|member_count", c)]
    # --- H2: scale_0.5 readouts only ---
    h = []
    readouts = ("lowmode_spectral_entropy", "lowmode_effective_rank",
                "lowmode_heat_trace_0.1", "spectral_gap", "signed_weight_mass")
    for ch in ("contact", "lj", "rdf", "shape"):
        for k in (0, 1, 2):
            for r in readouts:
                c = f"{pfx}scale_0.5_shared_hodge_{ch}_k{k}_{r}"
                if c in cols:
                    h.append(c)
    for k in (0, 1, 2):
        for r in ("lowmode_spectral_entropy", "lowmode_effective_rank", "spectral_gap"):
            c = f"{pfx}scale_0.5_shared_hodge_base_k{k}_{r}"
            if c in cols:
                h.append(c)
    h += [c for c in cols if re.match(pfx + r"scale_0.5_shared_hodge_cross_", c)]
    # --- Pcompact-v2 ---
    p = []
    p += [c for c in cols if re.match(pfx + r"patch_global_patch_(?!rdf_ordered)", c)]
    p += [c for c in cols if re.match(pfx + r"patch_token_[a-z_]+_mean_mean$", c)]
    p += [c for c in cols if re.match(pfx + r"rna_(stack_pairs|pair_like|clash|base_min_dist|stack_dist_mean|stack_parallel_frac|longrange_contacts)$", c)]
    rdf_ord = sorted([c for c in cols if re.match(pfx + r"patch_global_patch_rdf_ordered_\d+_mean_sum$", c)])
    p += rdf_ord[::14]
    # --- invariants (fixes found in review) ---
    # 1) absolute coordinates violate rigid-body invariance -> never predictive
    p = [c for c in p if not re.search(r"geometry_center_[xyz]", c)]
    # 2) P2/T2/H2 must be disjoint from S (counts live only in S)
    sset = set(s)
    p = [c for c in p if c not in sset]
    t = [c for c in t if c not in sset]
    h = [c for c in h if c not in sset]
    return dict(T2=sorted(set(t)), S=sorted(set(s)), H2=sorted(set(h)),
              P2=sorted(set(p)))


def main():
    man_rows, health_rows = [], []
    for task in ("rna", "membrane", "pd_bound", "pd_unbound"):
        df = load_task(task)
        cols = list(df.columns)
        pfx = dpref(cols)
        add_glmy_norms(df, pfx)
        cols = list(df.columns)
        g = group_v2(cols, pfx)
        gs = [set(g[k]) for k in ("P2", "T2", "H2", "S")]
        assert not (gs[0] & gs[1] or gs[0] & gs[2] or gs[0] & gs[3]
                    or gs[1] & gs[2] or gs[1] & gs[3] or gs[2] & gs[3]), \
            f"{task}: groups not disjoint"
        assert not any("geometry_center" in c for c in g["P2"]), "abs-coords in P2"
        idc = [c for c in ("structure_id", "row", "group", "protein", "kind") if c in df.columns]
        for name, key in (("topology_features_v2", "T2"),
                          ("hodge_features_v2", "H2"),
                          ("physical_compact_v2", "P2")):
            out = df[idc + g[key]].copy()
            out.to_parquet(OUT / f"{name}_{task}.parquet", index=False)
        for grp, cc in g.items():
            for c in cc:
                man_rows.append(dict(feature_name=c, scenario=task,
                                     assigned_group=grp,
                                     derived=c.endswith("__normO")))
        hc = audit_frame(df, sorted(set(sum(g.values(), []))), task)
        health_rows.append(hc)
        print(f"[{task}] T2={len(g['T2'])} H2={len(g['H2'])} P2={len(g['P2'])} S={len(g['S'])}")
    pd.DataFrame(man_rows).to_csv(OUT / "feature_manifest_v2.csv", index=False)
    pd.concat(health_rows).to_csv(OUT / "feature_health_v2.csv", index=False)


if __name__ == "__main__":
    main()
