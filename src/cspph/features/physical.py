"""Patch-level physical tokens; all microcontacts contribute."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from cspph.chemistry.patch_builder import InteractionPatch


def _stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {f"{prefix}_{name}": 0.0 for name in ("sum", "mean", "std", "min", "max", "median", "q25", "q75", "heterogeneity")}
    arr = np.asarray(values, dtype=float)
    q25, med, q75 = np.quantile(arr, (0.25, 0.5, 0.75))
    return {f"{prefix}_sum": float(arr.sum()), f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_std": float(arr.std()), f"{prefix}_min": float(arr.min()),
            f"{prefix}_max": float(arr.max()), f"{prefix}_median": float(med),
            f"{prefix}_q25": float(q25), f"{prefix}_q75": float(q75),
            f"{prefix}_heterogeneity": float(arr.std() / max(abs(arr.mean()), 1e-8))}


def aggregate_patch_physical_features(patch: InteractionPatch) -> Dict[str, float]:
    """Aggregate RDF/contact/LJ/shape over every microcontact in a patch."""
    attrs = [dict(c.attributes) for c in patch.contacts]
    out: Dict[str, float] = {
        "patch_contact_count": float(len(attrs)),
        "patch_donor_member_count": float(len(patch.donor_members)),
        "patch_acceptor_member_count": float(len(patch.acceptor_members)),
    }
    for key in ("rdf_mass", "contact_mass", "lj_att", "lj_rep", "lj_net", "shape_facing", "shape_gap", "shape_curvature"):
        out.update(_stats([float(a.get(key, 0.0)) for a in attrs], f"patch_{key}"))
    out.update(_stats([abs(float(a.get("lj_att", 0.0))) + abs(float(a.get("lj_rep", 0.0))) for a in attrs], "patch_lj_abs"))
    # Ordered RDF is intentionally not sorted: donor->acceptor orientation is scientific information.
    rdf_keys = sorted({k for a in attrs for k in a if k.startswith("rdf_ordered_")})
    for key in rdf_keys:
        out[f"patch_{key}_mean"] = float(np.mean([float(a.get(key, 0.0)) for a in attrs]))
        out[f"patch_{key}_max"] = float(np.max([float(a.get(key, 0.0)) for a in attrs]))
    types = Counter(str(c.interaction_type) for c in patch.contacts)
    total = float(sum(types.values()))
    out["patch_interaction_type_entropy"] = float(-sum((n / total) * np.log(n / total) for n in types.values())) if total else 0.0
    # Absolute centre coordinates are internal bookkeeping for
    # path_geom_span (graph.py) but are NOT rigid-motion invariants and must
    # never become model features.  Only invariant descriptors are exported.
    out.update({f"patch_geometry_{k}": float(v) for k, v in patch.geometry.items()
                if not k.startswith("center_")})
    return out


def aggregate_all_patches(patches: Sequence[InteractionPatch]) -> Dict[str, float]:
    out: Dict[str, float] = {"patch_count": float(len(patches))}
    for patch in patches:
        patch.physical = aggregate_patch_physical_features(patch)
        for key, value in patch.physical.items():
            out[f"patch_global_{key}_sum"] = out.get(f"patch_global_{key}_sum", 0.0) + float(value)
    return out


__all__ = ["aggregate_patch_physical_features", "aggregate_all_patches"]
