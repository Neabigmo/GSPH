"""Deterministic side-chain mutation builder for protein_stability.

Protocol (fixed, auditable):
  - WT backbone (N, CA, C, O) and every other residue are kept verbatim.
  - Side-chain atoms whose names exist in BOTH WT and mutant residue are
    copied from WT coordinates (preserves shared chi angles exactly).
  - Remaining atoms are placed with idealized bond lengths/angles; the free
    dihedral is scanned over a fixed grid and the candidate minimizing
    heavy-atom clash against already-placed atoms is chosen (local clash
    resolution, no MD, no minimization).
  - GLY mutants keep backbone only; PRO keeps its pyrrolidine template.

This matches the spec: same backbone / frame / radii for WT and mutant,
local rotameric placement only, deterministic version-pinned rules.
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

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parents[2]
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))
from runners.gsph1 import Atom  # noqa: E402

_BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HN", "HA"}

# (atom, parent, grandparent_for_angle, great_for_dihedral_anchor)
# bond lengths in angstrom, angles in degrees; dihedral is the free variable.
# Template order is topological: every parent is defined earlier.
_SIDECHAIN_BUILD = {
    "ALA": [("CB", "CA", "N", "C")],
    "ARG": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("CD", "CG", "CB", "CA"), ("NE", "CD", "CG", "CB"),
            ("CZ", "NE", "CD", "CG"), ("NH1", "CZ", "NE", "CD"),
            ("NH2", "CZ", "NE", "NH1")],
    "ASN": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("OD1", "CG", "CB", "CA"), ("ND2", "CG", "CB", "OD1")],
    "ASP": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("OD1", "CG", "CB", "CA"), ("OD2", "CG", "CB", "OD1")],
    "CYS": [("CB", "CA", "N", "C"), ("SG", "CB", "CA", "N")],
    "GLN": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("CD", "CG", "CB", "CA"), ("OE1", "CD", "CG", "CB"),
            ("NE2", "CD", "CG", "OE1")],
    "GLU": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("CD", "CG", "CB", "CA"), ("OE1", "CD", "CG", "CB"),
            ("OE2", "CD", "CG", "OE1")],
    "GLY": [],
    "HIS": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("ND1", "CG", "CB", "CA"), ("CD2", "CG", "CB", "ND1"),
            ("CE1", "ND1", "CG", "CB"), ("NE2", "CD2", "CG", "ND1")],
    "ILE": [("CB", "CA", "N", "C"), ("CG1", "CB", "CA", "N"),
            ("CG2", "CB", "CA", "CG1"), ("CD1", "CG1", "CB", "CA")],
    "LEU": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("CD1", "CG", "CB", "CA"), ("CD2", "CG", "CB", "CD1")],
    "LYS": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("CD", "CG", "CB", "CA"), ("CE", "CD", "CG", "CB"),
            ("NZ", "CE", "CD", "CG")],
    "MET": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("SD", "CG", "CB", "CA"), ("CE", "SD", "CG", "CB")],
    "PHE": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("CD1", "CG", "CB", "CA"), ("CD2", "CG", "CB", "CD1"),
            ("CE1", "CD1", "CG", "CB"), ("CE2", "CD2", "CG", "CD1"),
            ("CZ", "CE1", "CD1", "CG")],
    "PRO": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("CD", "CG", "CB", "N")],
    "SER": [("CB", "CA", "N", "C"), ("OG", "CB", "CA", "N")],
    "THR": [("CB", "CA", "N", "C"), ("OG1", "CB", "CA", "N"),
            ("CG2", "CB", "CA", "OG1")],
    "TRP": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("CD1", "CG", "CB", "CA"), ("CD2", "CG", "CB", "CD1"),
            ("NE1", "CD1", "CG", "CB"), ("CE2", "CD2", "CG", "CD1"),
            ("CE3", "CD2", "CG", "CE2"), ("CZ2", "CE2", "CD2", "CG"),
            ("CZ3", "CE3", "CD2", "CE2"), ("CH2", "CZ2", "CE2", "CD2")],
    "TYR": [("CB", "CA", "N", "C"), ("CG", "CB", "CA", "N"),
            ("CD1", "CG", "CB", "CA"), ("CD2", "CG", "CB", "CD1"),
            ("CE1", "CD1", "CG", "CB"), ("CE2", "CD2", "CG", "CD1"),
            ("CZ", "CE1", "CD1", "CG"), ("OH", "CZ", "CE1", "CE2")],
    "VAL": [("CB", "CA", "N", "C"), ("CG1", "CB", "CA", "N"),
            ("CG2", "CB", "CA", "CG1")],
}

_BOND_LEN = {"C": 1.53, "N": 1.46, "O": 1.43, "S": 1.81}
_SP2_CENTERS = {("CG", "OD1"), ("CG", "OD2"), ("CG", "OE1"), ("CG", "NE2"),
                ("CD", "OE1"), ("CD", "OE2"), ("CD", "NE2"),
                ("CZ", "NH1"), ("CZ", "NH2")}
_ELEMENT_BY_NAME = {"C": "C", "N": "N", "O": "O", "S": "S"}
_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}
_CHI_GRID = np.deg2rad([60.0, 180.0, -60.0, 0.0, 120.0, -120.0, 30.0, -30.0,
                       90.0, -90.0, 150.0, -150.0])


def _element(name: str) -> str:
    return _ELEMENT_BY_NAME.get(name[0].upper(), "C")


def _place(atom_a: np.ndarray, atom_b: np.ndarray, atom_c: np.ndarray,
           bond: float, angle_deg: float, dihedral: float) -> np.ndarray:
    """Place D given C-B-A frame: |D-C|=bond, angle(D,C,B), dihedral(D,C,B,A)."""
    bc = atom_c - atom_b
    bc /= np.linalg.norm(bc)
    # vector from C making angle with -BC direction
    v = -bc
    perp = atom_a - atom_b - np.dot(atom_a - atom_b, bc) * bc
    n = np.linalg.norm(perp)
    if n < 1e-6:
        perp = np.cross(bc, np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(perp)
        if n < 1e-6:
            perp = np.cross(bc, np.array([0.0, 1.0, 0.0]))
            n = np.linalg.norm(perp)
    perp /= n
    perp2 = np.cross(bc, perp)
    a = np.deg2rad(angle_deg)
    direction = (np.cos(a) * v + np.sin(a) *
                 (np.cos(dihedral) * perp + np.sin(dihedral) * perp2))
    direction /= np.linalg.norm(direction)
    return atom_c + bond * direction


def build_mutant_atoms(wt_atoms: Sequence[Atom], chain: str, resid: int,
                       icode: str, mut_resname: str,
                       clash_cutoff: float = 0.4) -> Tuple[List[Atom], dict]:
    """Return (mutant atom list, audit dict)."""
    mut_resname = mut_resname.upper()
    if mut_resname not in _SIDECHAIN_BUILD:
        raise ValueError(f"unsupported mutant residue {mut_resname}")
    icode_norm = str(icode or "").strip()
    target = [a for a in wt_atoms
              if a.chain == chain and int(a.resid) == int(resid)
              and str(getattr(a, "icode", "") or "").strip() == icode_norm]
    if not target:
        raise ValueError(f"residue {chain}:{resid}{icode} not found")
    frame = {a.name: a.xyz for a in target}
    for need in ("N", "CA", "C"):
        if need not in frame:
            raise ValueError(f"residue {chain}:{resid} missing backbone atom {need}")

    target_ids = {a.index for a in target}
    others = [a for a in wt_atoms if a.index not in target_ids]
    placed: Dict[str, np.ndarray] = dict(frame)
    out_atoms: List[Atom] = []
    next_index = max(a.index for a in wt_atoms) + 1
    wt_sidechain = {a.name: a for a in target if a.name not in _BACKBONE}
    audit = {"residue": f"{chain}:{resid}{icode}", "wt": target[0].resname,
             "mut": mut_resname, "copied_atoms": 0, "built_atoms": 0,
             "clash_rescans": 0, "min_clash": 0.0}
    # keep the original residue slot's backbone atoms in place
    new_target_atoms: List[Atom] = []
    for a in target:
        if a.name in _BACKBONE:
            new_target_atoms.append(Atom(a.index, a.chain, a.name, mut_resname,
                                         a.resid, a.element, a.xyz,
                                         a.serial, a.hetero, a.icode))
    for name, parent, grand, great in _SIDECHAIN_BUILD[mut_resname]:
        if name in wt_sidechain:
            # shared side-chain atom: keep WT coordinates exactly
            xyz = wt_sidechain[name].xyz
            audit["copied_atoms"] += 1
        else:
            xyz = None
            best, best_score = None, np.inf
            angle = 109.5 if (name, parent) not in _SP2_CENTERS else 120.0
            anchor = placed.get(great, placed.get(grand, placed[parent]))
            for chi in _CHI_GRID:
                cand = _place(anchor, placed[grand], placed[parent],
                              _BOND_LEN.get(_element(name), 1.53), angle, chi)
                score = 0.0
                el = _element(name)
                for o in others:
                    d = np.linalg.norm(cand - o.xyz)
                    overlap = (_VDW.get(el, 1.7) + _VDW.get(o.element, 1.7)
                               - clash_cutoff - d)
                    if overlap > 0:
                        score += overlap
                for p_name, p_xyz in placed.items():
                    if p_name in (parent, grand, great, name):
                        continue
                    d = np.linalg.norm(cand - p_xyz)
                    overlap = (2 * _VDW.get(el, 1.7) - clash_cutoff - d)
                    if overlap > 0:
                        score += overlap
                if score < best_score:
                    best_score, best = score, cand
            xyz = best
            audit["built_atoms"] += 1
            audit["min_clash"] = max(audit["min_clash"], float(best_score))
            audit["clash_rescans"] += int(best_score > 0.5)
        placed[name] = xyz
        next_index += 1
        new_target_atoms.append(Atom(next_index, chain, name, mut_resname,
                                     resid, _element(name), np.asarray(xyz),
                                     target[0].serial, target[0].hetero,
                                     target[0].icode))
    # preserve original ordering position of the mutated residue
    first_idx = min(i for i, a in enumerate(wt_atoms)
                    if a.index in target_ids)
    result = list(wt_atoms[:first_idx]) + new_target_atoms + \
        [a for a in wt_atoms[first_idx:] if a.index not in target_ids]
    for i, a in enumerate(result):
        a.index = i
    return result, audit
