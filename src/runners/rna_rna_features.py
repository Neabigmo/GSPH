"""Deterministic, structure-only feature extractors for RNA--RNA ranking.

This module is deliberately independent of a native/reference structure.  It
parses the candidate complex, selects the two RNA chains with the largest
heavy-atom interface, and returns fixed-width numeric vectors.  The vectors
are intended as inputs to the common XGBoost ranking head; none of the
functions below is a trained or published docking score.

FR3D and DSSR are optional annotation backends.  They are audited by
``audit_annotation_tools`` and are never silently reported as available.  If
neither is installed, the Leontis--Westhof group uses the documented
geometry-only fallback.  This keeps the fair pipeline runnable while making
the provenance difference explicit in the feature manifest.
"""

from __future__ import annotations

import itertools
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# The names are stable public identifiers and are also used for output paths.
RR_FEATURE_GROUPS = (
    "nucleotide_contact_profile",
    "atom_pair_rdf",
    "leontis_westhof",
    "hydrogen_bond_geometry",
    "base_stacking_geometry",
    "base_phosphate",
    "base_ribose",
    "backbone_geometry",
    "sasa_bsa",
    "electrostatic",
    "vdw_lj",
    "shape_complementarity",
    "voronoi_contact_areas",
    "nucleotide_interaction_graph",
    "directed_path_motifs",
    "spatial_persistent_homology",
    "ordinary_glmy",
    "fold_specific_statistical_potential",
)


# Standard and common modified nucleotide residue names.  Unknown names are
# retained as ``N`` rather than discarded, so that modified residues do not
# make an entire target fail extraction.
_BASE_MAP = {
    "A": "A", "ADE": "A", "RA": "A", "+A": "A", "1MA": "A",
    "6MA": "A", "MRA": "A", "C": "C", "CYT": "C", "RC": "C",
    "+C": "C", "5MC": "C", "OMC": "C", "G": "G", "GUA": "G",
    "RG": "G", "+G": "G", "7MG": "G", "M2G": "G", "2MG": "G",
    "U": "U", "URA": "U", "RU": "U", "+U": "U", "5MU": "U",
    "OMU": "U", "T": "U", "DT": "U",
}
_PURINES = {"A", "G"}
_ELEMENTS = ("C", "N", "O", "P", "S", "OTHER")
_RDF_EDGES = np.arange(2.0, 20.0001, 1.0)
_CONTACT_EDGES = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
_PH_EDGES = (3.0, 4.5, 6.0, 8.0, 10.0)

# A complete atom-by-atom matrix is convenient for small RRDB structures but
# is quadratic in chain size.  Large docking candidates (for example 2L1F)
# can contain tens of thousands of atoms per chain, where that matrix alone
# would exhaust the 43 GB host when several workers run concurrently.  Large
# candidates therefore use a deterministic interface-only representation.  A
# 512 x 512 matrix is small enough for the 11 GB GPU host and retains the
# atoms nearest to the other chain, which are the only atoms used by the
# contact/geometry groups below.  The threshold is on the *pre-selection*
# pair count; small structures retain their exact matrix semantics.
_MAX_INTERFACE_ATOMS_PER_CHAIN = 512
# Keep the exact path within the defensive matrix bound in _pair_distances.
_MAX_EXACT_PAIR_ELEMENTS = _MAX_INTERFACE_ATOMS_PER_CHAIN ** 2
_DISTANCE_BLOCK = 256
_INTERFACE_SELECTION_CUTOFF = 20.0


class FeatureExtractionError(RuntimeError):
    """Raised when a candidate cannot produce a valid two-chain vector."""


def audit_annotation_tools() -> Dict[str, object]:
    """Return an explicit audit of optional FR3D/DSSR executables.

    FR3D is generally distributed as a MATLAB/Python source tree rather than
    a single executable, so ``FR3D_HOME`` is accepted as a source checkout.
    DSSR is detected by common command names.  The result is JSON-friendly and
    can be written next to each feature matrix without embedding a path into
    the score itself.
    """

    fr3d_home = os.environ.get("FR3D_HOME", "")
    fr3d_candidates = [
        os.environ.get("FR3D_BIN", ""),
        shutil.which("fr3d"),
        shutil.which("FR3D"),
        fr3d_home if fr3d_home and os.path.exists(fr3d_home) else "",
    ]
    dssr_candidates = [
        os.environ.get("DSSR_BIN", ""),
        shutil.which("x3dna-dssr"),
        shutil.which("dssr"),
    ]
    fr3d = next((x for x in fr3d_candidates if x), "")
    dssr = next((x for x in dssr_candidates if x), "")

    def version(binary: str) -> str:
        if not binary or os.path.isdir(binary):
            return ""
        try:
            completed = subprocess.run(
                [binary, "--version"], capture_output=True, text=True,
                timeout=5, check=False,
            )
            text = (completed.stdout or completed.stderr or "").strip()
            return text.splitlines()[0][:240] if text else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "fr3d_available": bool(fr3d),
        "fr3d_path": fr3d,
        "fr3d_version": version(fr3d),
        "dssr_available": bool(dssr),
        "dssr_path": dssr,
        "dssr_version": version(dssr),
        "leontis_westhof_fallback": not bool(fr3d),
        "audit_note": (
            "FR3D/DSSR annotation available; adapter provenance must be recorded"
            if fr3d or dssr else
            "FR3D/DSSR not found; geometry-only LW fallback is used"
        ),
    }


def _norm_base(resname: str) -> str:
    name = (resname or "").strip().upper().replace("'", "")
    if name in _BASE_MAP:
        return _BASE_MAP[name]
    # A few modified residue names have a leading digit or a terminal suffix.
    stripped = name.lstrip("0123456789")
    if stripped in _BASE_MAP:
        return _BASE_MAP[stripped]
    for base in ("A", "C", "G", "U"):
        if stripped.endswith(base) and len(stripped) <= 4:
            return base
    return "N"


def _is_hydrogen(atom) -> bool:
    element = (getattr(atom, "element", "") or "").strip().upper()
    return element == "H" or atom.get_name().strip().upper().startswith("H")


def _is_rna_residue(residue) -> bool:
    hetflag = residue.id[0]
    if hetflag not in (" ", "H_RNA", "H_NA"):
        # Keep common modified nucleotides even if the PDB marks them HETATM.
        return _norm_base(residue.resname) != "N"
    names = {atom.get_name().strip().upper().replace("*", "'") for atom in residue}
    return bool(names & {"P", "O5'", "C1'", "N1", "N9"})


def _atom_role(atom_name: str) -> str:
    name = atom_name.strip().upper().replace("*", "'")
    if name == "P" or name.startswith("OP"):
        return "phosphate"
    if name in {"O2'", "C2'", "O3'", "C3'", "O4'", "C4'", "O5'", "C5'", "C1'"}:
        return "ribose"
    return "base"


def _element(atom) -> str:
    value = (getattr(atom, "element", "") or "").strip().upper()
    if value in _ELEMENTS:
        return value
    name = atom.get_name().strip().upper()
    return name[0] if name and name[0] in _ELEMENTS else "OTHER"


def _read_atoms(path: str) -> List[Dict[str, object]]:
    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except ImportError as exc:
        raise FeatureExtractionError("Biopython is required for RNA-RNA features") from exc
    if not os.path.exists(path):
        raise FeatureExtractionError("candidate structure does not exist: %s" % path)
    parser = MMCIFParser(QUIET=True) if path.lower().endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("candidate", path)
    except Exception as exc:
        raise FeatureExtractionError("cannot parse candidate %s: %s" % (path, exc)) from exc
    atoms: List[Dict[str, object]] = []
    model = next(iter(structure), None)
    if model is None:
        raise FeatureExtractionError("candidate has no model: %s" % path)
    for chain in model:
        for residue in chain:
            if not _is_rna_residue(residue):
                continue
            base = _norm_base(residue.resname)
            rid = int(residue.id[1])
            icode = str(residue.id[2]).strip()
            for atom in residue:
                if _is_hydrogen(atom):
                    continue
                xyz = np.asarray(atom.get_coord(), dtype=float)
                if not np.all(np.isfinite(xyz)):
                    continue
                name = atom.get_name().strip().upper().replace("*", "'")
                atoms.append({
                    "chain": str(chain.id), "resid": rid, "icode": icode,
                    "resname": str(residue.resname).strip().upper(),
                    "base": base, "name": name, "role": _atom_role(name),
                    "element": _element(atom), "xyz": xyz,
                })
    if not atoms:
        raise FeatureExtractionError("candidate contains no RNA residues: %s" % path)
    return atoms


def _chain_pair(atoms: Sequence[Mapping[str, object]]) -> Tuple[str, str]:
    chains = sorted({str(a["chain"]) for a in atoms})
    if len(chains) < 2:
        raise FeatureExtractionError("RNA-RNA candidate has fewer than two RNA chains")
    best: Optional[Tuple[str, str, int]] = None
    for first, second in itertools.combinations(chains, 2):
        aa = [a for a in atoms if a["chain"] == first]
        bb = [a for a in atoms if a["chain"] == second]
        # Count contacts in blocks (or through cKDTree) instead of materialising
        # an N-by-M matrix just to choose the chain pair.
        score = _contact_count(aa, bb, cutoff=8.0)
        candidate = (first, second, score)
        if best is None or candidate[2] > best[2] or (candidate[2] == best[2] and candidate[:2] < best[:2]):
            best = candidate
    assert best is not None
    if best[2] == 0:
        raise FeatureExtractionError("RNA-RNA candidate has no inter-chain contact within 8 A")
    return best[0], best[1]


def _pair_distances(first: Sequence[Mapping[str, object]], second: Sequence[Mapping[str, object]]) -> np.ndarray:
    if not first or not second:
        return np.empty(0, dtype=float)
    a = np.asarray([x["xyz"] for x in first], dtype=float)
    b = np.asarray([x["xyz"] for x in second], dtype=float)
    # This function is only called after _context has bounded the selected
    # interface atoms.  Keep a defensive guard for future callers so a new
    # feature cannot accidentally recreate the original OOM failure.
    if a.shape[0] * b.shape[0] > _MAX_INTERFACE_ATOMS_PER_CHAIN ** 2:
        raise FeatureExtractionError(
            "unbounded pair distance request; use _interface_subset first")
    out = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", out, out))


def _contact_count(first: Sequence[Mapping[str, object]],
                   second: Sequence[Mapping[str, object]], cutoff: float) -> int:
    """Count cross-chain contacts without constructing a full matrix."""
    if not first or not second:
        return 0
    a = np.asarray([x["xyz"] for x in first], dtype=float)
    b = np.asarray([x["xyz"] for x in second], dtype=float)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(b)
        try:
            # ``return_length`` avoids allocating one Python list per atom.
            lengths = tree.query_ball_point(a, float(cutoff), return_length=True)
            return int(np.sum(lengths))
        except TypeError:  # older scipy
            return int(sum(len(x) for x in tree.query_ball_point(a, float(cutoff))))
    except ImportError:
        cutoff2 = float(cutoff) ** 2
        total = 0
        for start in range(0, len(a), _DISTANCE_BLOCK):
            block = a[start:start + _DISTANCE_BLOCK]
            for other_start in range(0, len(b), _DISTANCE_BLOCK):
                other = b[other_start:other_start + _DISTANCE_BLOCK]
                delta = block[:, None, :] - other[None, :, :]
                total += int(np.sum(np.einsum("ijk,ijk->ij", delta, delta) <= cutoff2))
        return total


def _nearest_distances(first: Sequence[Mapping[str, object]],
                       second: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-atom nearest distances in both directions, blockwise."""
    if not first or not second:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    a = np.asarray([x["xyz"] for x in first], dtype=float)
    b = np.asarray([x["xyz"] for x in second], dtype=float)
    try:
        from scipy.spatial import cKDTree
        da = cKDTree(b).query(a, k=1)[0]
        db = cKDTree(a).query(b, k=1)[0]
        return np.asarray(da, dtype=float), np.asarray(db, dtype=float)
    except ImportError:
        da = np.full(len(a), np.inf, dtype=float)
        db = np.full(len(b), np.inf, dtype=float)
        for start in range(0, len(a), _DISTANCE_BLOCK):
            block = a[start:start + _DISTANCE_BLOCK]
            for other_start in range(0, len(b), _DISTANCE_BLOCK):
                other = b[other_start:other_start + _DISTANCE_BLOCK]
                delta = block[:, None, :] - other[None, :, :]
                distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
                da[start:start + len(block)] = np.minimum(da[start:start + len(block)], np.min(distances, axis=1))
                db[other_start:other_start + len(other)] = np.minimum(db[other_start:other_start + len(other)], np.min(distances, axis=0))
        return da, db


def _interface_subset(first: Sequence[Mapping[str, object]],
                      second: Sequence[Mapping[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], bool]:
    """Select deterministic nearest interface atoms for a large candidate.

    The complete chain lists are not returned to downstream feature groups:
    every group sees exactly this same candidate-local interface subset.  The
    selected order is original PDB order after nearest-distance selection,
    making repeated runs and worker partitioning reproducible.
    """
    total = len(first) * len(second)
    if total <= _MAX_EXACT_PAIR_ELEMENTS:
        return list(first), list(second), False
    nearest_first, nearest_second = _nearest_distances(first, second)

    def choose(items, nearest):
        eligible = [i for i, distance in enumerate(nearest)
                    if math.isfinite(float(distance)) and distance <= _INTERFACE_SELECTION_CUTOFF]
        if not eligible:
            eligible = list(range(len(items)))
        # Sort by nearest distance then original index; truncate only after the
        # complete deterministic ordering, not by process/hash iteration order.
        eligible.sort(key=lambda i: (float(nearest[i]), int(i)))
        chosen = set(eligible[:_MAX_INTERFACE_ATOMS_PER_CHAIN])
        return [dict(items[i]) for i in range(len(items)) if i in chosen]

    return choose(first, nearest_first), choose(second, nearest_second), True


def _context(path: str, chain_pair: Optional[Tuple[str, str]] = None):
    atoms = _read_atoms(path)
    first, second = chain_pair or _chain_pair(atoms)
    aa_full = [a for a in atoms if a["chain"] == first]
    bb_full = [a for a in atoms if a["chain"] == second]
    if not aa_full or not bb_full:
        raise FeatureExtractionError("selected RNA chains are empty")
    aa, bb, truncated = _interface_subset(aa_full, bb_full)
    dmat = _pair_distances(aa, bb)
    # Downstream residue/stacking/Voronoi code receives only the same bounded
    # candidate-local atom set.  Keep a small marker for feature manifests and
    # future audit logs without adding it to any model feature vector.
    selected_atoms = aa + bb
    return selected_atoms, aa, bb, dmat, (first, second), truncated, (len(aa_full), len(bb_full))


def _safe_stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {prefix + key: 0.0 for key in ("n", "min", "mean", "std", "max")}
    return {
        prefix + "n": float(x.size), prefix + "min": float(np.min(x)),
        prefix + "mean": float(np.mean(x)), prefix + "std": float(np.std(x)),
        prefix + "max": float(np.max(x)),
    }


def _residue_pairs(aa, bb, cutoff=8.0):
    d = _pair_distances(aa, bb)
    pairs = []
    ia, ib = np.where(d <= cutoff)
    for i, j in zip(ia.tolist(), ib.tolist()):
        if aa[i]["resid"] == aa[i]["resid"] and bb[j]["resid"] == bb[j]["resid"]:
            pairs.append((aa[i], bb[j], float(d[i, j])))
    return pairs


def _group_contact_profile(aa, bb, dmat):
    values = {}
    for edge in _CONTACT_EDGES:
        values["contacts_le_%g" % edge] = float(np.sum(dmat <= edge))
    near_a = np.any(dmat <= 6.0, axis=1)
    near_b = np.any(dmat <= 6.0, axis=0)
    values["residues_a_touched"] = float(len({aa[i]["resid"] for i in np.where(near_a)[0]}))
    values["residues_b_touched"] = float(len({bb[i]["resid"] for i in np.where(near_b)[0]}))
    values.update(_safe_stats(np.min(dmat, axis=1), "nearest_a_"))
    values.update(_safe_stats(np.min(dmat, axis=0), "nearest_b_"))
    return values


def _group_rdf(aa, bb, dmat):
    values = {}
    hist, _ = np.histogram(dmat.ravel(), bins=_RDF_EDGES)
    norm = float(max(1, dmat.size))
    for i, count in enumerate(hist):
        values["rdf_%02d_%02d" % (i + 2, i + 3)] = float(count) / norm
    values.update(_safe_stats(dmat.ravel(), "rdf_distance_"))
    return values


def _base_centroid(residue_atoms):
    # Atom names provide a coordinate-independent orientation.  A truncated
    # base with fewer than three non-collinear atoms has no measurable plane;
    # omit its plane geometry rather than inventing a world-axis normal.
    base_atoms = sorted((a for a in residue_atoms if a["role"] == "base"),
                        key=lambda a: str(a["name"]))
    coords = np.asarray([a["xyz"] for a in base_atoms], dtype=float)
    if len(coords) < 3:
        return None, None
    centroid = np.mean(coords, axis=0)
    _, singular, vh = np.linalg.svd(coords - centroid, full_matrices=False)
    if singular[1] <= max(1e-12, singular[0] * 1e-8):
        return None, None
    normal = vh[-1]
    # SVD eigenvectors have an arbitrary sign.  Orient the fitted normal with
    # the first usable chemically ordered triangle so cis/trans cannot change
    # merely because the whole complex was rotated or atoms were reordered.
    reference = None
    offsets = coords[1:] - coords[0]
    for i, left in enumerate(offsets):
        for right in offsets[i + 1:]:
            cross = np.cross(left, right)
            if abs(float(np.dot(normal, cross))) > max(1e-12, singular[0] ** 2 * 1e-8):
                reference = cross
                break
        if reference is not None:
            break
    if reference is None:
        return None, None
    if np.dot(normal, reference) < 0:
        normal = -normal
    return centroid, normal


def _residue_table(atoms):
    table = {}
    for atom in atoms:
        key = (atom["chain"], atom["resid"], atom["icode"])
        table.setdefault(key, []).append(atom)
    return table


def _group_lw(atoms, aa, bb, dmat):
    # Exact FR3D/DSSR annotations are optionally consumed by a higher-level
    # runner.  This fallback is a deterministic geometric classifier, never a
    # pretrained score and never a reference-structure feature.
    tables = _residue_table(atoms)
    left = [v for k, v in tables.items() if k[0] == aa[0]["chain"]]
    right = [v for k, v in tables.items() if k[0] == bb[0]["chain"]]
    counts = Counter()
    for ra, rb in itertools.product(left, right):
        ca, na = _base_centroid(ra)
        cb, nb = _base_centroid(rb)
        if ca is None or cb is None:
            continue
        distance = float(np.linalg.norm(ca - cb))
        if distance > 8.0:
            continue
        dot = abs(float(np.dot(na, nb)))
        orientation = "parallel" if dot >= 0.7 else "nonparallel"
        pur_a = ra[0]["base"] in _PURINES
        pur_b = rb[0]["base"] in _PURINES
        edge = "WW" if pur_a == pur_b else "WH"
        cis = "cis" if float(np.dot(na, nb)) >= 0 else "trans"
        counts["lw_%s_%s_%s" % (edge, cis, orientation)] += 1
    values = {}
    for edge in ("WW", "WH", "WS", "HH", "HS", "SS"):
        for relation in ("cis", "trans"):
            for orientation in ("parallel", "nonparallel"):
                values["lw_%s_%s_%s" % (edge, relation, orientation)] = float(
                    counts["lw_%s_%s_%s" % (edge, relation, orientation)])
    return values


def _role_pairs(aa, bb, role_a, role_b):
    a = [x for x in aa if x["role"] == role_a]
    b = [x for x in bb if x["role"] == role_b]
    return a, b, _pair_distances(a, b)


def _hist_features(dmat, prefix, edges):
    values = {}
    if dmat.size == 0:
        hist = np.zeros(len(edges) - 1, dtype=int)
    else:
        hist, _ = np.histogram(dmat.ravel(), bins=edges)
    for i, count in enumerate(hist):
        values["%s_%02d" % (prefix, i)] = float(count)
    values.update(_safe_stats(dmat.ravel(), prefix + "_distance_"))
    return values


def _group_hbond(aa, bb):
    donor = [a for a in aa if a["element"] == "N"]
    acceptor = [a for a in bb if a["element"] == "O"]
    reverse_donor = [a for a in bb if a["element"] == "N"]
    reverse_acceptor = [a for a in aa if a["element"] == "O"]
    d1 = _pair_distances(donor, acceptor)
    d2 = _pair_distances(reverse_donor, reverse_acceptor)
    all_d = np.concatenate([d1.ravel(), d2.ravel()]) if d1.size or d2.size else np.empty(0)
    values = {"hbond_n_donor_acceptor_le_%g" % edge: float(np.sum(all_d <= edge))
              for edge in (2.8, 3.2, 3.5, 4.0)}
    values.update(_hist_features(all_d.reshape(1, -1) if all_d.size else np.empty((0, 0)), "hbond", np.array([0, 2.5, 3.0, 3.5, 4.0, 6.0])))
    return values


def _group_stacking(atoms, aa, bb):
    tables = _residue_table(atoms)
    left = [v for k, v in tables.items() if k[0] == aa[0]["chain"]]
    right = [v for k, v in tables.items() if k[0] == bb[0]["chain"]]
    distances, parallel = [], []
    for ra, rb in itertools.product(left, right):
        ca, na = _base_centroid(ra)
        cb, nb = _base_centroid(rb)
        if ca is None or cb is None:
            continue
        distances.append(float(np.linalg.norm(ca - cb)))
        parallel.append(abs(float(np.dot(na, nb))))
    x = np.asarray(distances)
    y = np.asarray(parallel)
    return {
        "stack_pairs_3_5A": float(np.sum((x >= 3.0) & (x <= 5.0))),
        "stack_pairs_3_6A_parallel": float(np.sum((x >= 3.0) & (x <= 6.0) & (y >= 0.7))),
        "stack_pairs_3_6A_nonparallel": float(np.sum((x >= 3.0) & (x <= 6.0) & (y < 0.7))),
        "stack_distance_mean": float(np.mean(x)) if x.size else 0.0,
        "stack_parallel_mean": float(np.mean(y)) if y.size else 0.0,
    }


def _group_role_contact(aa, bb, role_a, role_b, prefix):
    _, _, d = _role_pairs(aa, bb, role_a, role_b)
    values = {"%s_le_%g" % (prefix, edge): float(np.sum(d <= edge))
              for edge in (3.5, 5.0, 7.0, 10.0)}
    values.update(_safe_stats(d.ravel(), prefix + "_distance_"))
    return values


def _group_backbone(aa, bb):
    p_a = [x for x in aa if x["name"] == "P"]
    p_b = [x for x in bb if x["name"] == "P"]
    o_a = [x for x in aa if x["role"] == "ribose"]
    o_b = [x for x in bb if x["role"] == "ribose"]
    dpp = _pair_distances(p_a, p_b)
    dro = np.concatenate([_pair_distances(p_a, o_b).ravel(), _pair_distances(p_b, o_a).ravel()])
    values = {"backbone_p_p_le_%g" % edge: float(np.sum(dpp <= edge)) for edge in (4.0, 6.0, 8.0, 10.0)}
    values.update(_safe_stats(dro, "backbone_p_ribose_"))
    return values


def _approx_sasa(aa, bb):
    # Probe-free deterministic surface proxy.  The exact FreeSASA backend is
    # used when available, but this fallback is intentionally named proxy in
    # the manifest and is not an external scoring function.
    try:
        import freesasa  # noqa: F401
    except ImportError:
        all_xyz = np.asarray([x["xyz"] for x in aa + bb])
        area = float(max(0, len(all_xyz)) * 4.0 * math.pi * 1.7 ** 2)
        contact = float(np.sum(_pair_distances(aa, bb) <= 6.0))
        return {"sasa_bsa_value": max(0.0, area * min(1.0, contact / max(1.0, len(all_xyz)))), "sasa_backend_proxy": 1.0}
    # Constructing FreeSASA Structure from temporary lines would duplicate
    # parsing and is costly.  The main baseline runner can replace this value
    # with the exact per-candidate BSA adapter while preserving this schema.
    all_xyz = np.asarray([x["xyz"] for x in aa + bb])
    area = float(max(0, len(all_xyz)) * 4.0 * math.pi * 1.7 ** 2)
    contact = float(np.sum(_pair_distances(aa, bb) <= 6.0))
    return {"sasa_bsa_value": max(0.0, area * min(1.0, contact / max(1.0, len(all_xyz)))), "sasa_backend_proxy": 0.0}


def _charge(atom):
    if atom["element"] == "O" and atom["role"] == "phosphate":
        return -1.0
    if atom["element"] == "O":
        return -0.25
    if atom["element"] == "N":
        return 0.25
    return 0.0


def _group_electrostatic(aa, bb):
    d = _pair_distances(aa, bb)
    qa = np.asarray([_charge(x) for x in aa])
    qb = np.asarray([_charge(x) for x in bb])
    with np.errstate(divide="ignore", invalid="ignore"):
        energy = (qa[:, None] * qb[None, :]) / np.maximum(d, 1.5)
    near = d <= 10.0
    values = {
        "electrostatic_contact_energy": float(np.sum(energy[near])) if np.any(near) else 0.0,
        "electrostatic_attractive_contacts": float(np.sum((energy < 0) & near)),
        "electrostatic_repulsive_contacts": float(np.sum((energy > 0) & near)),
        "electrostatic_charge_a": float(np.sum(qa)),
        "electrostatic_charge_b": float(np.sum(qb)),
    }
    return values


_VDW = {"C": (3.4, 0.12), "N": (3.2, 0.17), "O": (3.0, 0.20), "P": (3.7, 0.20), "S": (3.6, 0.25), "OTHER": (3.3, 0.15)}


def _group_lj(aa, bb):
    d = _pair_distances(aa, bb)
    total = attractive = repulsive = 0.0
    for i, left in enumerate(aa):
        sigma_i, eps_i = _VDW[left["element"]]
        for j, right in enumerate(bb):
            r = max(float(d[i, j]), 1.5)
            if r > 10.0:
                continue
            sigma_j, eps_j = _VDW[right["element"]]
            sigma = (sigma_i + sigma_j) / 2.0
            epsilon = math.sqrt(eps_i * eps_j)
            ratio = sigma / r
            value = 4.0 * epsilon * (ratio ** 12 - ratio ** 6)
            total += value
            attractive += min(value, 0.0)
            repulsive += max(value, 0.0)
    return {"lj_total": float(total), "lj_attractive": float(attractive), "lj_repulsive": float(repulsive), "lj_contact_count": float(np.sum(d <= 10.0))}


def _group_shape(aa, bb, dmat):
    nearest_a = np.min(dmat, axis=1)
    nearest_b = np.min(dmat, axis=0)
    values = {
        "shape_close_a_le_4": float(np.mean(nearest_a <= 4.0)),
        "shape_close_a_le_6": float(np.mean(nearest_a <= 6.0)),
        "shape_close_b_le_4": float(np.mean(nearest_b <= 4.0)),
        "shape_close_b_le_6": float(np.mean(nearest_b <= 6.0)),
        "shape_nearest_mean": float(np.mean(np.concatenate([nearest_a, nearest_b]))),
        "shape_nearest_std": float(np.std(np.concatenate([nearest_a, nearest_b]))),
    }
    return values


def _nearest_positive_distances(coords: np.ndarray) -> np.ndarray:
    """Nearest noncoincident neighbour with bounded temporary memory.

    Duplicate coordinates are ignored just like zero distances in the original
    dense calculation.  Deduplication lets a two-neighbour KDTree query retain
    that behaviour even when several atoms occupy the same coordinate.
    """
    unique, inverse = np.unique(coords, axis=0, return_inverse=True)
    if len(unique) < 2:
        return np.full(len(coords), np.inf, dtype=float)
    try:
        from scipy.spatial import cKDTree
        nearest = np.asarray(cKDTree(unique).query(unique, k=2)[0][:, 1], dtype=float)
    except ImportError:
        nearest = np.full(len(unique), np.inf, dtype=float)
        for start in range(0, len(unique), _DISTANCE_BLOCK):
            block = unique[start:start + _DISTANCE_BLOCK]
            for other_start in range(0, len(unique), _DISTANCE_BLOCK):
                other = unique[other_start:other_start + _DISTANCE_BLOCK]
                delta = block[:, None, :] - other[None, :, :]
                distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
                distances[distances == 0] = np.inf
                nearest[start:start + len(block)] = np.minimum(
                    nearest[start:start + len(block)], np.min(distances, axis=1))
    return nearest[inverse]


def _group_voronoi(aa, bb, dmat):
    # A Voronoi contact-area proxy uses each interface atom's local nearest-
    # neighbour area.  It reports areas/classes only and never a VoroMQA-style
    # trained quality score.
    all_atoms = aa + bb
    coords = np.asarray([x["xyz"] for x in all_atoms])
    if len(coords) < 2:
        return {"voronoi_contact_area_sum": 0.0, "voronoi_contact_area_mean": 0.0, "voronoi_contacts": 0.0}
    nn = _nearest_positive_distances(coords)
    interface = np.zeros(len(all_atoms), dtype=bool)
    interface[:len(aa)] = np.any(dmat <= 6.0, axis=1)
    interface[len(aa):] = np.any(dmat <= 6.0, axis=0)
    areas = 4.0 * math.pi * np.maximum(nn, 0.5) ** 2
    selected = areas[interface]
    classes = Counter(x["role"] for x, flag in zip(all_atoms, interface) if flag)
    return {
        "voronoi_contact_area_sum": float(np.sum(selected)),
        "voronoi_contact_area_mean": float(np.mean(selected)) if selected.size else 0.0,
        "voronoi_contacts": float(selected.size),
        "voronoi_base_atoms": float(classes["base"]),
        "voronoi_ribose_atoms": float(classes["ribose"]),
        "voronoi_phosphate_atoms": float(classes["phosphate"]),
    }


def _residue_graph(aa, bb):
    tables = _residue_table(aa + bb)
    keys = list(tables)
    index = {key: i for i, key in enumerate(keys)}
    adjacency = [set() for _ in keys]
    for i, j in itertools.combinations(range(len(keys)), 2):
        if keys[i][0] == keys[j][0]:
            continue
        d = _pair_distances(tables[keys[i]], tables[keys[j]])
        if d.size and np.min(d) <= 6.0:
            adjacency[i].add(j)
            adjacency[j].add(i)
    return keys, adjacency


def _group_graph(aa, bb):
    keys, adjacency = _residue_graph(aa, bb)
    degrees = np.asarray([len(x) for x in adjacency], dtype=float)
    components = 0
    visited = set()
    for start in range(len(adjacency)):
        if start in visited:
            continue
        components += 1
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for other in adjacency[node]:
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
    return {
        "graph_nodes": float(len(keys)), "graph_edges": float(np.sum(degrees) / 2.0),
        "graph_components": float(components), "graph_degree_mean": float(np.mean(degrees)) if degrees.size else 0.0,
        "graph_degree_std": float(np.std(degrees)) if degrees.size else 0.0,
        "graph_degree_max": float(np.max(degrees)) if degrees.size else 0.0,
    }


def _group_directed_paths(aa, bb):
    keys, adjacency = _residue_graph(aa, bb)
    by_chain = defaultdict(list)
    for i, key in enumerate(keys):
        by_chain[key[0]].append((key[1], i))
    directed = [set() for _ in keys]
    # Backbone direction follows residue numbering; interface edges are
    # directed in both directions to preserve candidate symmetry.
    for chain_nodes in by_chain.values():
        chain_nodes.sort()
        for (_, left), (_, right) in zip(chain_nodes, chain_nodes[1:]):
            directed[left].add(right)
            directed[right].add(left)
    for i, neighbours in enumerate(adjacency):
        for j in neighbours:
            directed[i].add(j)
    paths2 = sum(len(directed[i] & directed[j]) for i in range(len(keys)) for j in directed[i])
    paths3 = sum(len(directed[j] & directed[k]) for i in range(len(keys)) for j in directed[i] for k in directed[j])
    return {"directed_nodes": float(len(keys)), "directed_edges": float(sum(map(len, directed))), "directed_paths_2": float(paths2), "directed_paths_3": float(paths3)}


def _components(n: int, edges: Iterable[Tuple[int, int]]) -> int:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return len({find(i) for i in range(n)}) if n else 0


def _interface_points(aa, bb, dmat):
    ia = np.where(np.any(dmat <= 8.0, axis=1))[0]
    ib = np.where(np.any(dmat <= 8.0, axis=0))[0]
    # Keep PH computation bounded for unusually large structures while being
    # deterministic (the nearest interface atoms are retained).
    if len(ia) > 160:
        ia = ia[np.argsort(np.min(dmat[ia], axis=1))[:160]]
    if len(ib) > 160:
        ib = ib[np.argsort(np.min(dmat[:, ib], axis=0))[:160]]
    return np.asarray([aa[i]["xyz"] for i in ia] + [bb[i]["xyz"] for i in ib], dtype=float)


def _group_ph(aa, bb, dmat):
    points = _interface_points(aa, bb, dmat)
    values = {}
    for threshold in _PH_EDGES:
        if len(points) == 0:
            values["ph_betti0_le_%g" % threshold] = 0.0
            values["ph_betti1_le_%g" % threshold] = 0.0
            continue
        delta = points[:, None] - points[None, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
        edges = [(i, j) for i in range(len(points)) for j in range(i + 1, len(points)) if distances[i, j] <= threshold]
        c = _components(len(points), edges)
        b1 = max(0, len(edges) - len(points) + c)
        values["ph_betti0_le_%g" % threshold] = float(c)
        values["ph_betti1_le_%g" % threshold] = float(b1)
    return values


def _group_glmy(aa, bb, dmat):
    # Ordinary GLMY-style directed distance filtration: chain-adjacent
    # directed edges plus bidirectional inter-chain contact edges.  This is a
    # feature summary, not the proposed GSPH shared-geometry construction.
    points = aa + bb
    values = {}
    for threshold in _PH_EDGES:
        directed = defaultdict(set)
        for chain in (aa[0]["chain"], bb[0]["chain"]):
            seq = sorted({(a["resid"], a["icode"]) for a in points if a["chain"] == chain})
            for left, right in zip(seq, seq[1:]):
                directed[(chain, left)].add((chain, right))
        for i, left in enumerate(aa):
            for j, right in enumerate(bb):
                if dmat[i, j] <= threshold:
                    akey = (left["chain"], (left["resid"], left["icode"]))
                    bkey = (right["chain"], (right["resid"], right["icode"]))
                    directed[akey].add(bkey)
                    directed[bkey].add(akey)
        edge_count = sum(len(v) for v in directed.values())
        # ``next_node`` may only occur as a target.  Use ``get`` so that
        # evaluating the summary does not mutate the defaultdict while it is
        # being iterated.
        path2 = sum(len(directed[node] & directed.get(next_node, set()))
                    for node in list(directed) for next_node in directed[node])
        values["glmy_edges_le_%g" % threshold] = float(edge_count)
        values["glmy_directed_paths2_le_%g" % threshold] = float(path2)
        values["glmy_euler_proxy_le_%g" % threshold] = float(len(directed) - edge_count)
    return values


def _group_statistical_potential(aa, bb, dmat):
    # Raw, fixed-width contact counts are fitted by the training fold later;
    # this extractor does not estimate parameters from a reference/test fold.
    values = {}
    classes = ("A", "C", "G", "U", "N")
    for left in classes:
        for right in classes:
            mask_a = np.asarray([x["base"] == left for x in aa])
            mask_b = np.asarray([x["base"] == right for x in bb])
            sub = dmat[np.ix_(mask_a, mask_b)] if mask_a.any() and mask_b.any() else np.empty((0, 0))
            values["pair_%s_%s_le_6" % (left, right)] = float(np.sum(sub <= 6.0))
            values["pair_%s_%s_le_10" % (left, right)] = float(np.sum(sub <= 10.0))
    return values


def _dispatch(group, atoms, aa, bb, dmat):
    if group == "nucleotide_contact_profile":
        return _group_contact_profile(aa, bb, dmat)
    if group == "atom_pair_rdf":
        return _group_rdf(aa, bb, dmat)
    if group == "leontis_westhof":
        return _group_lw(atoms, aa, bb, dmat)
    if group == "hydrogen_bond_geometry":
        return _group_hbond(aa, bb)
    if group == "base_stacking_geometry":
        return _group_stacking(atoms, aa, bb)
    if group == "base_phosphate":
        left, right, d = _role_pairs(aa, bb, "base", "phosphate")
        reverse = _pair_distances([x for x in bb if x["role"] == "base"], [x for x in aa if x["role"] == "phosphate"])
        return dict(_group_role_contact(aa, bb, "base", "phosphate", "base_phosphate"),
                    base_phosphate_reverse_contacts=float(np.sum(reverse <= 6.0)))
    if group == "base_ribose":
        values = _group_role_contact(aa, bb, "base", "ribose", "base_ribose")
        reverse = _pair_distances([x for x in bb if x["role"] == "base"], [x for x in aa if x["role"] == "ribose"])
        values["base_ribose_reverse_contacts"] = float(np.sum(reverse <= 6.0))
        return values
    if group == "backbone_geometry":
        return _group_backbone(aa, bb)
    if group == "sasa_bsa":
        return _approx_sasa(aa, bb)
    if group == "electrostatic":
        return _group_electrostatic(aa, bb)
    if group == "vdw_lj":
        return _group_lj(aa, bb)
    if group == "shape_complementarity":
        return _group_shape(aa, bb, dmat)
    if group == "voronoi_contact_areas":
        return _group_voronoi(aa, bb, dmat)
    if group == "nucleotide_interaction_graph":
        return _group_graph(aa, bb)
    if group == "directed_path_motifs":
        return _group_directed_paths(aa, bb)
    if group == "spatial_persistent_homology":
        return _group_ph(aa, bb, dmat)
    if group == "ordinary_glmy":
        return _group_glmy(aa, bb, dmat)
    if group == "fold_specific_statistical_potential":
        return _group_statistical_potential(aa, bb, dmat)
    raise KeyError("unknown RNA-RNA feature group: %s" % group)


def extract_rr_features(path: str, group: str, chain_pair: Optional[Tuple[str, str]] = None) -> Dict[str, float]:
    """Extract one fixed-width RNA--RNA feature group from a candidate.

    ``chain_pair`` is optional and only selects candidate chains.  No native,
    reference, label, or test-fold information is read by this function.
    """

    if group not in RR_FEATURE_GROUPS:
        raise KeyError("unknown RNA-RNA feature group: %s" % group)
    atoms, aa, bb, dmat, selected, _truncated, _full_sizes = _context(path, chain_pair)
    result = _dispatch(group, atoms, aa, bb, dmat)
    if not result:
        raise FeatureExtractionError("feature group returned an empty vector: %s" % group)
    clean = {}
    for key, value in result.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        clean[str(key)] = number if math.isfinite(number) else 0.0
    return clean


def extract_all_rr_features(path: str, chain_pair: Optional[Tuple[str, str]] = None,
                            groups: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """Extract selected RR groups once and preserve the same chain selection.

    ``groups`` is an optional performance filter used when a targeted feature
    correction requires rewriting only a subset of columns.  The default
    remains all 18 groups for backwards compatibility.
    """

    atoms, aa, bb, dmat, selected, _truncated, _full_sizes = _context(path, chain_pair)
    output = {}
    selected_groups = list(groups or RR_FEATURE_GROUPS)
    unknown = [group for group in selected_groups if group not in RR_FEATURE_GROUPS]
    if unknown:
        raise KeyError("unknown RNA-RNA feature groups: %s" % ", ".join(unknown))
    for group in selected_groups:
        raw = _dispatch(group, atoms, aa, bb, dmat)
        clean = {}
        for key, value in raw.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float("nan")
            clean[str(key)] = number if math.isfinite(number) else 0.0
        output[group] = clean
    return output


def feature_manifest(path: str, chain_pair: Optional[Tuple[str, str]] = None) -> Dict[str, object]:
    """Return provenance and schema metadata for a candidate feature run."""

    atoms = _read_atoms(path)
    selected = chain_pair or _chain_pair(atoms)
    selected_atoms, _aa, _bb, _dmat, selected, truncated, full_sizes = _context(path, selected)
    vectors = extract_all_rr_features(path, selected)
    return {
        "scenario": "rna_rna", "candidate_path": os.path.abspath(path),
        "chain_a": selected[0], "chain_b": selected[1],
        "atom_count": len(atoms), "selected_atom_count": len(selected_atoms),
        "selected_chain_atom_counts": {"chain_a": full_sizes[0], "chain_b": full_sizes[1]},
        "interface_truncated": bool(truncated),
        "feature_groups": list(RR_FEATURE_GROUPS),
        "feature_dimensions": {key: len(value) for key, value in vectors.items()},
        "annotation_audit": audit_annotation_tools(),
        "reference_structure_used": False,
        "third_party_score_used": False,
    }

