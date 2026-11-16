"""H3 intramolecular task adapter: RNA quality / protein (membrane) stability.

Reuses the frozen gsph6 math core unchanged.  The only structural difference
vs. the docking scenarios is the side definition: both sides are the SAME
molecule, so microcontacts become intramolecular contacts.  Covalently
linked atom pairs (< ``intra_min_bond_sep`` bonds apart) are excluded from
event generation -- they are backbone/side-chain chemistry, not contacts.
Direction still comes exclusively from real donor/acceptor roles; the
relay graph still uses the real covalent graph (phosphodiester/peptide
backbone + CONECT records).  Base stacking stays a symmetric physical
channel -- no direction is fabricated.
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

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_WORKSPACE = os.environ.get(
    "GSPH_WORKSPACE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
for _p in (_WORKSPACE, os.path.join(_WORKSPACE, "dl_baselines")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runners.gsph1 import Atom, _parse_atoms  # noqa: E402

# Bond-graph distance below which a pair is chemistry, not a contact.
# 4 covers direct bonds + 2-/3-neighbour shells (intra-residue + backbone
# linkage) while keeping base-base contacts of adjacent nucleotides.
DEFAULT_INTRA_SEP = int(os.environ.get("GSPH_H3_INTRA_SEP", "4"))


@dataclass
class StructureSpec:
    """One real structure -> one extraction unit."""
    structure_id: str
    path: str
    group: str           # target id (rna target / protein id / complex id)
    kind: str            # 'native' | 'decoy' | 'wt' | 'mutant'
    label: Optional[float] = None
    meta: Optional[dict] = None


def parse_structure(path: str) -> List[Atom]:
    """First-model PDB/mmCIF atoms via the H2 parser (water excluded)."""
    return _parse_atoms(str(path))


def extract_intra(atoms: Sequence[Atom], scenario: str,
                  config=None, intra_sep: Optional[int] = None,
                  static_bond_graph=None, target_cache=None,
                  event_filter=None) -> dict:
    """Run the frozen gsph6 engine on one single-molecule structure."""
    from gsph6.engine import GSPH6Config, extract_gsph6_atoms
    import dataclasses
    cfg = config or GSPH6Config(
        interaction_backend=os.environ.get("GSPH_H3_INTERACTION_BACKEND", "auto"),
        intra_min_bond_sep=DEFAULT_INTRA_SEP if intra_sep is None else intra_sep)
    # Steric-clash floor applies to every intramolecular extraction, including
    # callers that pass an explicit config (production drivers, deep subsets).
    # GSPH_H3_CLASH_RATIO=0 disables; default 0.6 of summed vdW radii.
    clash = float(os.environ.get("GSPH_H3_CLASH_RATIO", "0.6"))
    if float(getattr(cfg, "contact_min_vdw_ratio", 0.0) or 0.0) == 0.0:
        cfg = dataclasses.replace(cfg, contact_min_vdw_ratio=clash)
    return extract_gsph6_atoms(
        list(atoms), scenario=scenario, config=cfg,
        static_bond_graph=static_bond_graph, target_cache=target_cache,
        event_filter=event_filter)


# ------------------- RNA intramolecular physical block --------------------

_PURINE_RING = ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4")
_PYRIMIDINE_RING = ("N1", "C2", "N3", "C4", "C5", "C6")
_PURINE_RES = {"A", "G", "I", "DA", "DG", "DI", "ADE", "GUA", "1MA", "OMG", "M2G", "7MG"}
_PYRIMIDINE_RES = {"C", "U", "T", "DC", "DT", "DU", "CYT", "URA", "THY", "PSU", "5MC", "OMC"}
_NA_RESNAMES = _PURINE_RES | _PYRIMIDINE_RES
_NA_BASE_EXCLUDE = {"P", "OP1", "OP2", "O1P", "O2P", "O3P", "O5P", "HO5'"}


def _residue_bases(atoms: Sequence[Atom]) -> Dict[tuple, dict]:
    """Base-ring centroid + Newell normal per nucleotide (same math as H2)."""
    groups: Dict[tuple, list] = {}
    for a in atoms:
        if a.resname in _NA_RESNAMES:
            groups.setdefault((a.chain, int(a.resid), str(getattr(a, "icode", "") or "")), []).append(a)
    out = {}
    for key, res_atoms in groups.items():
        base = [a for a in res_atoms if "'" not in a.name
                and a.name not in _NA_BASE_EXCLUDE and a.element != "H"]
        if len(base) < 4:
            continue
        ring = _PURINE_RING if res_atoms[0].resname in _PURINE_RES else _PYRIMIDINE_RING
        order = {n: i for i, n in enumerate(ring)}
        base = sorted(base, key=lambda a: (order.get(a.name, len(order)), a.name))
        xyz = np.asarray([a.xyz for a in base], dtype=float)
        centroid = xyz.mean(axis=0)
        ring_xyz = xyz[:min(len(ring), len(xyz))]
        n = np.zeros(3)
        for i in range(len(ring_xyz)):
            p, q = ring_xyz[i], ring_xyz[(i + 1) % len(ring_xyz)]
            n[0] += (p[1] - q[1]) * (p[2] + q[2])
            n[1] += (p[2] - q[2]) * (p[0] + q[0])
            n[2] += (p[0] - q[0]) * (p[1] + q[1])
        norm = np.linalg.norm(n)
        out[key] = {"centroid": centroid,
                    "normal": (n / norm if norm > 1e-9 else np.zeros(3)),
                    "resname": res_atoms[0].resname}
    return out


def rna_intra_features(atoms: Sequence[Atom]) -> Dict[str, float]:
    """Intramolecular base stacking / pairing block for a single RNA.

    Same thresholds as the H2 rna_rna block but pairs are unordered (i<j) and
    self-pairs excluded -- the symmetric stacking channel carries no direction.
    """
    bases = _residue_bases(atoms)
    out = {k: 0.0 for k in (
        "rna_stack_pairs", "rna_pair_like", "rna_clash", "rna_base_min_dist",
        "rna_stack_dist_mean", "rna_stack_parallel_frac", "rna_bases_total",
        "rna_longrange_contacts")}
    out["rna_bases_total"] = float(len(bases))
    keys = sorted(bases)
    if len(keys) < 2:
        return out
    cents = np.asarray([bases[k]["centroid"] for k in keys])
    resids = np.asarray([k[1] for k in keys])
    stack_d = []
    n_stack = n_pair = n_clash = n_long = 0
    min_d = 1e9
    for i in range(len(keys)):
        d = np.linalg.norm(cents - cents[i], axis=1)
        for j in np.where(d <= 8.0)[0]:
            if j <= i:
                continue
            dd = float(d[j])
            min_d = min(min_d, dd)
            seq_sep = abs(int(resids[j]) - int(resids[i]))
            if seq_sep >= 4:
                n_long += 1
            if dd < 2.6:
                n_clash += 1
            na, nb = bases[keys[i]]["normal"], bases[keys[j]]["normal"]
            par = abs(float(np.dot(na, nb))) if np.linalg.norm(na) and np.linalg.norm(nb) else 0.0
            if 3.2 <= dd <= 4.6 and par >= 0.7:
                n_stack += 1
                stack_d.append((dd, par))
            elif 2.8 <= dd <= 4.2 and par <= 0.45:
                n_pair += 1
    out["rna_stack_pairs"] = float(n_stack)
    out["rna_pair_like"] = float(n_pair)
    out["rna_clash"] = float(n_clash)
    out["rna_longrange_contacts"] = float(n_long)
    out["rna_base_min_dist"] = float(min_d) if min_d < 1e8 else 0.0
    out["rna_stack_dist_mean"] = float(np.mean([s[0] for s in stack_d])) if stack_d else 0.0
    out["rna_stack_parallel_frac"] = (float(sum(1 for s in stack_d if s[1] >= 0.85) / n_stack)
                                      if n_stack else 0.0)
    return out


# ------------------------------ labels -----------------------------------

def _trace_atoms(atoms: Sequence[Atom], names=("C4'", "C4*", "P")) -> np.ndarray:
    pts = {a.name: a.xyz for a in atoms if a.name in names}
    pref = "C4'" if any(a.name == "C4'" for a in atoms) else ("C4*" if any(a.name == "C4*" for a in atoms) else "P")
    sel = [a for a in atoms if a.name == pref]
    return np.asarray([a.xyz for a in sel], dtype=float)


def _kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p - p.mean(axis=0); q = q - q.mean(axis=0)
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((p @ rot - q) ** 2).sum(axis=1).mean()))


def c4_rmsd(native: Sequence[Atom], decoy: Sequence[Atom]) -> float:
    pn, pd = _trace_atoms(native), _trace_atoms(decoy)
    n = min(len(pn), len(pd))
    if n < 3:
        return float("nan")
    return _kabsch_rmsd(pn[:n], pd[:n])


def tm_score(native: Sequence[Atom], decoy: Sequence[Atom]) -> float:
    pn, pd = _trace_atoms(native), _trace_atoms(decoy)
    n = min(len(pn), len(pd))
    if n < 3:
        return float("nan")
    pn, pd = pn[:n], pd[:n]
    d0 = max(0.5, 1.24 * (n - 15) ** (1 / 3) - 1.8) if n > 15 else 0.5
    cn, cd = pn - pn.mean(axis=0), pd - pd.mean(axis=0)
    h = cn.T @ cd
    u, _, vt = np.linalg.svd(h)
    rot = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    dist = np.linalg.norm(cn @ rot - cd, axis=1)
    return float(np.mean(1.0 / (1.0 + (dist / d0) ** 2)))


def lddt_c4(native: Sequence[Atom], decoy: Sequence[Atom], cutoff: float = 15.0) -> float:
    """lDDT-style local-distance agreement on trace atoms (lite version).

    Fraction of native pairwise distances (within ``cutoff``) preserved in the
    decoy at 0.5/1/2/4 A tolerances.  Same-molecule atoms only; standard lDDT
    uses all atoms -- documented as lite.
    """
    pn, pd = _trace_atoms(native), _trace_atoms(decoy)
    n = min(len(pn), len(pd))
    if n < 3:
        return float("nan")
    pn, pd = pn[:n], pd[:n]
    dn = np.linalg.norm(pn[:, None, :] - pn[None, :, :], axis=2)
    dd = np.linalg.norm(pd[:, None, :] - pd[None, :, :], axis=2)
    mask = (dn < cutoff) & (dn > 0.5)
    iu = np.triu_indices(n, 1)
    mask = mask[iu]
    if not mask.any():
        return float("nan")
    diff = np.abs(dn[iu][mask] - dd[iu][mask])
    scores = [np.mean(diff < t) for t in (0.5, 1.0, 2.0, 4.0)]
    return float(np.mean(scores))


def featurize_structure(path: str, scenario: str = "rna_quality",
                        intra_sep: Optional[int] = None, config=None,
                        atoms: Optional[Sequence[Atom]] = None,
                        event_filter=None) -> Tuple[Dict[str, float], dict]:
    """One structure -> (features, audit metadata).

    ``atoms`` may be supplied by the caller to avoid a second parse; when
    omitted the structure is parsed from ``path`` exactly once.
    """
    if atoms is None:
        atoms = parse_structure(path)
    result = extract_intra(atoms, scenario, config=config, intra_sep=intra_sep,
                           event_filter=event_filter)
    meta = result.get("metadata") or {}
    features = {k: float(v) for k, v in (result.get("features") or {}).items()}
    if str(scenario).startswith("rna_quality"):
        features.update(rna_intra_features(atoms))
    status = str(meta.get("status", "unknown"))
    audit = {
        "status": status,
        "events_truncated": bool(meta.get("events_truncated")),
        "paths_truncated": bool(meta.get("paths_truncated")),
        "support_incomplete": bool(meta.get("support_incomplete")),
        "hodge_incomplete": bool(meta.get("hodge_incomplete")),
        "persistence_valid": bool(meta.get("persistence_valid")),
        "timing": meta.get("timing") or {},
        "intra_min_bond_sep": int((meta.get("event_stats") or {}).get(
            "intra_min_bond_sep", -1)),
        "clash_filtered_count": int((meta.get("event_stats") or {}).get(
            "clash_filtered_count", 0) or 0),
        "event_filter": meta.get("event_filter"),
        "event_filter_stats": meta.get("event_filter_stats"),
        "sparse_gsph": bool(meta.get("sparse_gsph")),
        "path_engine": meta.get("path_engine"),
        "patch_count": meta.get("patch_count"),
        "directed_edge_count": meta.get("directed_edge_count"),
        "support_solver_routes": meta.get("support_solver_routes"),
        "support_fallback_used": bool(meta.get("support_fallback_used")),
        "support_certificate_passed": bool(
            meta.get("support_certificate_passed", True)),
        "support_stats": meta.get("support_stats"),
    }
    return features, audit
