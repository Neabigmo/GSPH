"""GSPH-S: task-adaptive sparse event selection (mGLI-style task differentiation).

The frozen gsph6 math core is unchanged.  Sparsity is applied UPSTREAM of
patch construction: a deterministic, chemically stratified subset of
microcontact events enters the identical patch -> transfer graph -> GLMY ->
persistence pipeline.  Every filtered run records ``events_before_filter`` /
``events_after_filter`` in audit metadata; a sparse result is a
*task-defined sparse GSPH*, never claimed equivalent to the full-atom GSPH.

Two selectors are provided:

* :func:`rna_event_filter` -- nucleotide-region stratified sampling for
  ``rna_quality``.  Events are scored by donor/acceptor chemical region
  (base / sugar / phosphate), sequence separation and contact tightness;
  the top fraction is kept deterministically.  ``fraction=1.0`` is the
  exact reference (no events dropped).

* :func:`local_event_filter` -- mutation-site local selection for
  ``protein_stability``.  Only events whose centre lies within ``radius``
  Angstrom of a focus point (e.g. the mutated residue C-alpha) are kept.
  WT and mutant use the same spatial rule so ``Delta X`` stays consistent.
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

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Region assignment for RNA atoms.  Base = ring/edge heteroatoms of the
# nucleotide; sugar = ribose carbons/oxygens; phosphate = P and its oxygens.
_RNA_PHOSPHATE = {"P", "OP1", "OP2", "O1P", "O2P", "O3P", "O5P"}
_RNA_SUGAR_NAMES = {
    "C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'",
    "C1*", "C2*", "C3*", "C4*", "C5*", "O2*", "O3*", "O4*", "O5*",
    "HO2'", "HO3'", "HO5'",
}


def _rna_region(atom) -> str:
    name = str(getattr(atom, "name", "")).strip().upper().replace("*", "'")
    if name in _RNA_PHOSPHATE or name.startswith("OP") or name.startswith("O1P") or name.startswith("O2P"):
        return "phosphate"
    if "'" in name or name in _RNA_SUGAR_NAMES:
        return "sugar"
    return "base"


_REGION_WEIGHT = {
    ("base", "base"): 4.0,        # base pairing / stacking chemistry
    ("base", "sugar"): 2.0,
    ("sugar", "base"): 2.0,
    ("phosphate", "base"): 2.5,   # backbone-base long-range contacts
    ("base", "phosphate"): 2.5,
    ("phosphate", "sugar"): 1.5,
    ("sugar", "phosphate"): 1.5,
    ("phosphate", "phosphate"): 0.5,
    ("sugar", "sugar"): 0.5,
}


def _seq_sep(atom_a, atom_b) -> float:
    """Sequence separation when both atoms share a chain, else chain-change bonus."""
    ca, cb = str(getattr(atom_a, "chain", "")), str(getattr(atom_b, "chain", ""))
    if ca != cb:
        return 50.0
    try:
        return float(abs(int(getattr(atom_a, "resid", 0)) - int(getattr(atom_b, "resid", 0))))
    except (TypeError, ValueError):
        return 0.0


def rna_event_scores(atoms: Sequence, events: Sequence) -> np.ndarray:
    """Deterministic priority score per event (higher = keep first)."""
    atom_map = {int(a.index): a for a in atoms}
    scores = np.zeros(len(events), dtype=float)
    for i, ev in enumerate(events):
        da, ac = atom_map.get(int(ev.donor)), atom_map.get(int(ev.acceptor))
        if da is None or ac is None:
            continue
        ra, rb = _rna_region(da), _rna_region(ac)
        w = _REGION_WEIGHT.get((ra, rb), _REGION_WEIGHT.get((rb, ra), 1.0))
        sep = min(_seq_sep(da, ac), 30.0)
        # tighter contact (smaller tau) ranks higher within a class
        tight = -float(getattr(ev, "tau", 0.0))
        scores[i] = w * 1000.0 + sep * 2.0 + tight
    return scores


def rna_event_filter(atoms: Sequence, fraction: float = 1.0,
                     per_nt_quota: Optional[int] = None):
    """Return ``(filter_fn, meta)`` for stratified RNA event subsampling.

    ``fraction`` in [0, 1]; 1.0 keeps every event (reference mode).
    ``per_nt_quota`` additionally caps how many kept events may touch any
    single nucleotide residue (counted on both donor and acceptor sides).
    """
    frac = float(fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    def _filter(events, atoms_):
        n = len(events)
        if frac >= 1.0 and per_nt_quota is None:
            return list(events), {"filter": "rna_sparse", "fraction": frac,
                                  "kept": n, "dropped": 0,
                                  "per_nt_quota": per_nt_quota}
        scores = rna_event_scores(atoms_, events)
        order = np.argsort(-scores, kind="stable")
        keep_count = n if frac >= 1.0 else max(1, int(math.ceil(frac * n)))
        keep_idx = set(int(i) for i in order[:keep_count])
        if per_nt_quota is not None:
            atom_map = {int(a.index): a for a in atoms_}
            res_count: Dict[Tuple[str, int, str], int] = {}

            def _res_key(atom_index: int):
                a = atom_map.get(int(atom_index))
                if a is None:
                    return None
                return (str(getattr(a, "chain", "")), int(getattr(a, "resid", 0)),
                        str(getattr(a, "icode", "") or ""))

            def _over_quota(ev) -> bool:
                for side in (int(ev.donor), int(ev.acceptor)):
                    key = _res_key(side)
                    if key is not None and res_count.get(key, 0) >= int(per_nt_quota):
                        return True
                return False

            # re-fill in priority order, respecting the per-residue quota
            keep_idx = set()
            for i in order:
                ev = events[int(i)]
                if _over_quota(ev):
                    continue
                keep_idx.add(int(i))
                for side in (int(ev.donor), int(ev.acceptor)):
                    key = _res_key(side)
                    if key is not None:
                        res_count[key] = res_count.get(key, 0) + 1
                if len(keep_idx) >= keep_count:
                    break
        kept = [events[i] for i in sorted(keep_idx)]
        return kept, {"filter": "rna_sparse", "fraction": frac,
                      "kept": len(kept), "dropped": n - len(kept),
                      "per_nt_quota": per_nt_quota}

    return _filter, {"name": "rna_sparse", "fraction": frac,
                     "per_nt_quota": per_nt_quota}


def local_event_filter(atoms: Sequence, focus_xyz,
                       radius: float = 10.0):
    """Keep events whose centre is within ``radius`` of ``focus_xyz``.

    ``focus_xyz`` is an (x, y, z) point -- e.g. the C-alpha of the mutated
    residue.  Used identically for WT and mutant so the differential feature
    row stays spatially consistent.
    """
    center = np.asarray(focus_xyz, dtype=float).reshape(3)
    rad = float(radius)

    def _filter(events, atoms_):
        kept, dropped = [], 0
        for ev in events:
            c = np.asarray(getattr(ev, "center", None), dtype=float)
            if c.size != 3:
                kept.append(ev)
                continue
            if float(np.linalg.norm(c - center)) <= rad:
                kept.append(ev)
            else:
                dropped += 1
        return kept, {"filter": "local_radius", "radius": rad,
                      "kept": len(kept), "dropped": dropped,
                      "focus": [float(x) for x in center]}

    return _filter, {"name": "local_radius", "radius": rad}


def focus_ca_xyz(atoms: Sequence, chain: str, resid: int,
                 icode: str = "") -> Optional[np.ndarray]:
    """C-alpha (fallback: first atom) coordinates of one residue."""
    want_icode = str(icode or "").strip()
    res_atoms = [a for a in atoms
                 if str(getattr(a, "chain", "")) == str(chain)
                 and int(getattr(a, "resid", -1)) == int(resid)
                 and str(getattr(a, "icode", "") or "").strip() == want_icode]
    if not res_atoms:
        return None
    for a in res_atoms:
        if str(getattr(a, "name", "")).strip().upper() in {"CA", "C*", "C1*"}:
            return np.asarray(a.xyz, dtype=float)
    return np.asarray(res_atoms[0].xyz, dtype=float)


__all__ = ["rna_event_filter", "local_event_filter", "focus_ca_xyz",
           "rna_event_scores"]
