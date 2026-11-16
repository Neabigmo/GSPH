"""Reproducible local RNA--RNA candidate generation and quality labels.

The RRDB archive contains native/partner structures but does not expose a
single uniform 1,000-candidate pool.  This module supplies the documented
fixed-version local fallback for smoke tests and for targets whose official
decoys are unavailable: perturb one RNA partner by deterministic rigid-body
motions.  It is a candidate *source*, never a scoring method.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .rna_rna_features import FeatureExtractionError, _chain_pair, _read_atoms


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ], dtype=float)


def _candidate_transforms(count: int, seed: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield (rotation, translation) pairs from near-native to broad poses."""
    if count <= 0:
        return
    rng = np.random.default_rng(seed)
    yield np.eye(3), np.zeros(3)
    for index in range(1, count):
        # First quarter provides near-native perturbations; the remainder
        # spans broader rigid-body poses.  No candidate is an exact duplicate.
        if index < max(2, count // 4):
            angle = float(rng.uniform(0.02, 0.25))
            translation_scale = float(rng.uniform(0.1, 1.5))
        else:
            angle = float(rng.uniform(0.25, math.pi))
            translation_scale = float(rng.uniform(1.5, 15.0))
        axis = rng.normal(size=3)
        translation = rng.normal(size=3)
        translation = translation / max(float(np.linalg.norm(translation)), 1e-12) * translation_scale
        yield _rotation(axis, angle), translation


def generate_decoys(native_path: str, output_dir: str, target_id: Optional[str] = None,
                    count: int = 100, seed: int = 20261103) -> List[Dict[str, str]]:
    """Write ``count`` deterministic coordinate candidates and a manifest.

    The native structure is copied as ``decoy_0000`` for calibration and is
    not duplicated beyond that one candidate.  The second RNA chain (selected
    by largest interface when chain order is ambiguous) is moved around its
    own centroid; all other chains remain unchanged.
    """
    try:
        from Bio.PDB import PDBIO, PDBParser
    except ImportError as exc:
        raise FeatureExtractionError("Biopython is required to generate RR decoys") from exc
    atoms = _read_atoms(native_path)
    first, second = _chain_pair(atoms)
    target_id = target_id or os.path.splitext(os.path.basename(native_path))[0].upper()
    os.makedirs(output_dir, exist_ok=True)
    parser = PDBParser(QUIET=True)
    native = parser.get_structure("native", native_path)
    model = next(iter(native))
    if first not in model or second not in model:
        raise FeatureExtractionError("selected chains absent from parsed structure")
    movable = model[second]
    movable_coords = np.asarray([atom.get_coord() for residue in movable for atom in residue], dtype=float)
    if movable_coords.size == 0:
        raise FeatureExtractionError("movable RNA chain contains no atoms")
    centroid = np.mean(movable_coords, axis=0)
    rows: List[Dict[str, str]] = []
    for index, (rotation, translation) in enumerate(_candidate_transforms(count, seed)):
        # Parse a fresh copy for every candidate so transforms cannot leak into
        # subsequent candidates or into the native reference.
        structure = parser.get_structure("candidate_%04d" % index, native_path)
        candidate_model = next(iter(structure))
        moving_chain = candidate_model[second]
        for residue in moving_chain:
            for atom in residue:
                xyz = np.asarray(atom.get_coord(), dtype=float)
                atom.set_coord(np.dot(xyz - centroid, rotation.T) + centroid + translation)
        path = os.path.join(output_dir, "decoy_%04d.pdb" % index)
        io = PDBIO()
        io.set_structure(structure)
        io.save(path)
        rows.append({
            "scenario": "rna_rna", "dataset": "RRDB", "target_id": str(target_id),
            "decoy_id": "decoy_%04d" % index, "structure_path": os.path.abspath(path),
            "reference_path": os.path.abspath(native_path), "candidate_source": "fixed_rigid_body_generator",
            "generator_version": "rr-rigidbody-v1", "seed": str(seed),
            "chain_a": first, "chain_b": second,
        })
    manifest = os.path.join(output_dir, "samples.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["target_id"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _atom_key(atom: Mapping[str, object]) -> Tuple[str, int, str, str]:
    return str(atom["chain"]), int(atom["resid"]), str(atom["icode"]), str(atom["name"])


def _kabsch(mobile: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[0] < 3:
        return np.eye(3), np.zeros(3)
    cm = np.mean(mobile, axis=0)
    cr = np.mean(reference, axis=0)
    covariance = (mobile - cm).T @ (reference - cr)
    u, _, vh = np.linalg.svd(covariance)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    translation = cr - cm @ rotation
    return rotation, translation


def _aligned_coords(reference_path: str, candidate_path: str):
    ref = _read_atoms(reference_path)
    cand = _read_atoms(candidate_path)
    ref_by_key = {_atom_key(x): x for x in ref}
    cand_by_key = {_atom_key(x): x for x in cand}
    first, second = _chain_pair(ref)
    common_a = [key for key in ref_by_key if key in cand_by_key and key[0] == first]
    if len(common_a) < 3:
        raise FeatureExtractionError("candidate/reference have fewer than 3 common receptor atoms")
    mobile = np.asarray([cand_by_key[key]["xyz"] for key in common_a], dtype=float)
    reference = np.asarray([ref_by_key[key]["xyz"] for key in common_a], dtype=float)
    rotation, translation = _kabsch(mobile, reference)
    aligned = {}
    for key, atom in cand_by_key.items():
        aligned[key] = np.asarray(atom["xyz"], dtype=float) @ rotation + translation
    return ref_by_key, aligned, first, second


def quality_labels(reference_path: str, candidate_path: str) -> Dict[str, object]:
    """Compute reference-only quality labels for one candidate.

    Receptor alignment is estimated from the candidate/reference receptor;
    interface residues and native contacts are then fixed from the reference.
    These labels are never available to feature extraction or model fitting.
    """
    ref, aligned, first, second = _aligned_coords(reference_path, candidate_path)
    ref_a = [x for x in ref.values() if x["chain"] == first]
    ref_b = [x for x in ref.values() if x["chain"] == second]
    interface_residues = {
        int(b["resid"]) for b in ref_b
        if any(np.linalg.norm(np.asarray(a["xyz"]) - np.asarray(b["xyz"])) <= 10.0 for a in ref_a)
    }
    ref_pairs = set()
    cand_pairs = set()
    for a in ref_a:
        for b in ref_b:
            if np.linalg.norm(np.asarray(a["xyz"]) - np.asarray(b["xyz"])) <= 6.0:
                ref_pairs.add((int(a["resid"]), int(b["resid"])))
            ak = _atom_key(a)
            bk = _atom_key(b)
            if ak in aligned and bk in aligned and np.linalg.norm(aligned[ak] - aligned[bk]) <= 6.0:
                cand_pairs.add((int(a["resid"]), int(b["resid"])))
    interface_keys = [key for key in ref if key[0] == second and int(key[1]) in interface_residues and key in aligned]
    mobile = np.asarray([aligned[key] for key in interface_keys], dtype=float)
    target = np.asarray([ref[key]["xyz"] for key in interface_keys], dtype=float)
    irmsd = float(np.sqrt(np.mean(np.sum((mobile - target) ** 2, axis=1)))) if len(mobile) else float("nan")
    recovery = float(len(ref_pairs & cand_pairs) / len(ref_pairs)) if ref_pairs else float("nan")
    if not math.isfinite(irmsd):
        status = "no_reference_interface"
    elif irmsd < 2.0:
        status = "high"
    elif irmsd < 5.0:
        status = "medium"
    else:
        status = "low"
    return {
        "quality_irmsd": irmsd, "quality_contact_recovery": recovery,
        "quality_capri": status, "quality_status": "ok" if math.isfinite(irmsd) else "invalid",
        "label_reference_used": True,
    }


def label_manifest(manifest_path: str, output_path: Optional[str] = None) -> List[Dict[str, object]]:
    """Append labels to a generated samples CSV without changing features."""
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        row = dict(row)
        try:
            row.update(quality_labels(row["reference_path"], row["structure_path"]))
        except Exception as exc:
            row.update({"quality_irmsd": "", "quality_contact_recovery": "", "quality_capri": "", "quality_status": "failed", "label_error": str(exc)})
        out.append(row)
    output_path = output_path or os.path.splitext(manifest_path)[0] + "_labels.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        fields = list(out[0]) if out else ["target_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    return out

