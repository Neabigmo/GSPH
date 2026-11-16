# -*- coding: utf-8 -*-
"""Feature group manifest builder for the GSPH attribution ablation.

Assigns every column in a task feature table to exactly one of:
  P   physical / geometric patch features
  T   GSPH topology (GLMY + directed-path chemistry + geom. common support + persistence)
  H   Shared-Hodge spectral readouts (mathematically meaningful only)
  S   structure-size controls (counts; kept disjoint from P/T/H)
  META    identifiers / labels / audit status  (never features)
  DIAG    numerical-solver diagnostics          (never features)
  EXCL    excluded for a documented reason      (never features)

Membrane columns are WT->mut deltas prefixed with `d_`; grouping rules
operate on the de-prefixed name so all three tasks share one ruleset.
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
import re
import pandas as pd

# ---------------------------------------------------------------- meta ----
_META_EXACT = {
    "structure_id", "row", "group", "kind", "gsph_mode",
    "pdbid", "resid", "wt_aa", "mut_aa", "protein", "ddG",
    "bound_protocol",
}
_META_PREFIX = ("label_", "base_")

# ------------------------------------------------------ diagnostics --------
_DIAG_SUBSTR = (
    "_converged", "estimation_error", "_residual", "method_code",
    "spectrum_complete", "omitted_modes", "lowmode_n_modes", "_n_modes",
    "_dimension", "dense_limit", "engine_sparse_exact",
    "boundary_square_residual", "omega_truncated",
    "wt_status", "mut_status", "topology_valid",
)

# ------------------------------------------------ absolute coords ----------
_COORD_RE = re.compile(r"center_[xyz]")

# --------------------------------------------------------- size group ------
_SIZE_EXACT = {"patch_count", "patch_directed_edge_count", "rna_bases_total"}
_SIZE_SUBSTR = ("contact_count", "atom_count", "member_count", "path_count")

# --------------------------------------------------------------- hodge -----
# readouts that carry mathematical meaning (solver flags already removed above)
_HODGE_KEEP = (
    "lowmode_spectral_entropy", "lowmode_effective_rank",
    "lowmode_heat_trace", "signed_weight_mass",
    "spectral_entropy", "spectral_gap", "effective_rank", "heat_trace",
    "cross_",
)

# ----------------------------------------------------------------- T -------
_T_PREFIX = ("pathchem_",)
_T_SUBSTR = ("_glmy_",)
_T_KEEP_PREFIX = ("path_geom_span_", "path_support_mean_", "path_support_max_")
_T_TAUS = re.compile(r"^(tau|eta|delta_tau)_")
_T_PERSIST = "persistence_interval_count"

# ---------------------------------------------------------------- RNA P ----
_RNA_P = {
    "rna_stack_pairs", "rna_pair_like", "rna_clash", "rna_base_min_dist",
    "rna_stack_dist_mean", "rna_stack_parallel_frac", "rna_longrange_contacts",
}


def norm_name(col: str) -> str:
    """Strip the membrane WT->mut delta prefix so all tasks share rules."""
    return col[2:] if col.startswith("d_") else col


def classify(col: str) -> tuple[str, str]:
    """Return (group, reason).  group in {P,T,H,S,META,DIAG,EXCL}."""
    n = norm_name(col)

    if n in _META_EXACT or n.startswith(_META_PREFIX):
        return "META", "identifier/label/status column"
    if _COORD_RE.search(n):
        return "EXCL", "absolute patch centroid coordinate (not rigid-motion invariant)"
    if any(s in n for s in _DIAG_SUBSTR):
        return "DIAG", "numerical-solver diagnostic / status flag"
    if n in _SIZE_EXACT or any(s in n for s in _SIZE_SUBSTR):
        return "S", "extensive size/count measure (structure-size control)"

    if "shared_hodge" in n:
        if any(k in n for k in _HODGE_KEEP):
            return "H", "shared-Hodge spectral readout"
        return "DIAG", "hodge field without spectral interpretation"
    if any(s in n for s in _T_SUBSTR):
        return "T", "GLMY directed-flag invariant (dim/rank/kernel/betti/filling/admissibility)"
    if n.startswith(_T_PREFIX):
        return "T", "directed-path chemical-role statistics"
    if n.startswith(_T_KEEP_PREFIX):
        return "T", "path geometry/support statistics (normalised, non-count)"
    if _T_TAUS.match(n):
        return "T", "geometric common-support invariant (tau/eta/delta-tau)"
    if _T_PERSIST in n:
        return "T", "persistent-homology interval count"

    if n.startswith("patch_"):
        return "P", "interaction-patch physical/geometric statistic"
    if n in _RNA_P:
        return "P", "RNA secondary-structure physical statistic"
    return "EXCL", "no mathematical interpretation — excluded by default"


def build_manifest(cols, task: str) -> pd.DataFrame:
    rows = []
    for c in cols:
        g, why = classify(c)
        n = norm_name(c)
        if n.startswith("patch_global_patch_"):
            block = n.split("_")[2] if len(n.split("_")) > 2 else "global"
            block = "patch_global:" + n.replace("patch_global_patch_", "").split("_")[0]
        elif n.startswith("patch_token_"):
            block = "patch_token:" + n.replace("patch_token_", "").split("_")[0]
        elif "shared_hodge" in n:
            block = "shared_hodge:" + re.search(r"shared_hodge_([a-z]+)", n).group(1)
        elif "_glmy_" in n:
            block = "glmy"
        elif n.startswith("scale_"):
            block = "scale:" + re.match(r"scale_([0-9.]+)", n).group(1)
        elif n.startswith("pathchem_"):
            block = "pathchem"
        elif n.startswith("path"):
            block = "path"
        elif _T_TAUS.match(n):
            block = "geom_support"
        elif n.startswith("persistence"):
            block = "persistence"
        elif n.startswith("rna_"):
            block = "rna_phys"
        else:
            block = "misc"
        rows.append(dict(
            feature_name=c, normalized_name=n, scenario=task,
            original_schema_block=block, assigned_group=g,
            mathematical_definition=why,
            included=g in "PTHS",
            exclusion_reason="" if g in "PTHS" else why,
            rigid_motion_invariant=(not _COORD_RE.search(n)),
            diagnostic_or_structural=("diagnostic" if g in ("DIAG", "META") else "structural"),
        ))
    df = pd.DataFrame(rows)
    # disjointness assertion: a feature belongs to at most one of P/T/H/S
    assert df["assigned_group"].isin(["P", "T", "H", "S", "META", "DIAG", "EXCL"]).all()
    return df


def group_columns(manifest: pd.DataFrame) -> dict[str, list[str]]:
    return {g: manifest.loc[manifest.assigned_group == g, "feature_name"].tolist()
            for g in ("P", "T", "H", "S")}
