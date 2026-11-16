"""Fair decoy, feature, ranking and evaluation pipeline.

The pipeline is intentionally self contained and uses only structure-derived
features.  It accepts the existing native manifests, generates deterministic
rigid-body decoys, writes the required parquet-named artifacts (CSV fallback
is used when pyarrow is unavailable), and trains one common ranking head per
feature group.  No reference structure is passed to feature extractors.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .paths import ROOT, ensure_layout
from .pr_fair_features import PR_FEATURE_GROUPS, extract_pr_features, feature_group_vector
from .rna_rna_features import RR_FEATURE_GROUPS, extract_rr_features
from .pd_fair_features import FEATURE_GROUPS as PD_FEATURE_GROUPS, extract_pd_feature_groups

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None


SCENARIO_GROUPS = {
    "protein_rna": tuple(PR_FEATURE_GROUPS),
    "rna_rna": tuple(RR_FEATURE_GROUPS),
    "protein_dna": tuple(PD_FEATURE_GROUPS),
}
COUNT_DEFAULTS = {"protein_rna": 1000, "rna_rna": 1000, "protein_dna": 1000}
SAMPLE_FIELDS = [
    "scenario", "dataset", "target_id", "candidate_id", "structure_path",
    "chain_a", "chain_b", "candidate_source", "reference_path", "seed",
]
LABEL_FIELDS = [
    "scenario", "target_id", "candidate_id", "irmsd", "contact_recovery",
    "irmsd_standard", "irmsd_ligand_displacement", "quality", "success",
    "label_status", "label_definition", "native_control", "interface_backbone_fallback",
]


def _artifact(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def write_table(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    """Write parquet when available, otherwise a deterministic CSV fallback."""
    ensure_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(rows)
    try:
        import pandas as pd
        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
        return
    except Exception:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key); fields.append(key)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        with open(path + ".format", "w", encoding="utf-8") as marker:
            marker.write("csv-fallback; install pyarrow to emit binary parquet\n")


def read_table(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    try:
        import pandas as pd
        frame = pd.read_parquet(path)
        return frame.fillna("").to_dict("records")
    except Exception:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def _parse_atoms(path: str) -> List[Dict[str, object]]:
    atoms = []
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            try:
                alt = line[16:17]
                if alt not in (" ", "A", "1"):
                    continue
                element = (line[76:78].strip() or line[12:14].strip()[:1]).upper()
                if element == "H":
                    continue
                atoms.append({
                    "record": line[:6], "name": line[12:16].strip(),
                    "alt": alt, "resname": line[17:20].strip().upper(),
                    "chain": line[21:22].strip() or "_",
                    "resseq": int(line[22:26]), "icode": line[26:27],
                    "xyz": np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                    "occupancy": line[54:60] if len(line) >= 60 else "  1.00",
                    "bfactor": line[60:66] if len(line) >= 66 else " 20.00",
                    "element": element,
                })
            except (ValueError, IndexError):
                continue
    return atoms


def _pdb_line(atom: Mapping[str, object], serial: int, xyz: np.ndarray) -> str:
    name = str(atom["name"])[:4].rjust(4)
    return ("ATOM  %5d %4s %-3s %1s%4d%s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n" % (
        serial, name, str(atom["resname"])[:3].rjust(3), str(atom["chain"])[0],
        int(atom["resseq"]), str(atom.get("icode", " "))[:1], xyz[0], xyz[1], xyz[2],
        1.0, 20.0, str(atom.get("element", "C"))[:2].rjust(2)))


def _classify(scenario: str, atoms: Sequence[Mapping[str, object]]) -> Tuple[List[str], List[str]]:
    chains = defaultdict(list)
    for atom in atoms:
        chains[str(atom["chain"])].append(atom)
    prot = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE","SEC","PYL"}
    rna = {"A","C","G","U","RA","RC","RG","RU","ADE","CYT","GUA","URA","PSU","1MA","5MC","OMG","M2G","7MG","OMC"}
    dna = {"DA","DC","DG","DT","DI","A","C","G","T","ADE","CYT","GUA","THY","5MC"}
    def score(chain_atoms, names):
        return sum(str(a["resname"]).upper() in names for a in chain_atoms)
    if scenario == "protein_rna":
        a = [c for c in chains if score(chains[c], prot) > 0 and score(chains[c], prot) >= score(chains[c], rna)]
        b = [c for c in chains if c not in a and score(chains[c], rna) > 0]
    elif scenario == "protein_dna":
        a = [c for c in chains if score(chains[c], prot) > 0 and score(chains[c], prot) >= score(chains[c], dna)]
        b = [c for c in chains if c not in a and score(chains[c], dna) > 0]
    else:
        ranked = sorted(chains, key=lambda c: len(chains[c]), reverse=True)
        a, b = ranked[:1], ranked[1:2]
    if not a or not b:
        ranked = sorted(chains, key=lambda c: len(chains[c]), reverse=True)
        a, b = ranked[:1], ranked[1:2]
    return a, b


def _rotation(rng: np.random.Generator, angle: float) -> np.ndarray:
    axis = rng.normal(size=3); axis /= max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis; c, s = math.cos(angle), math.sin(angle); C = 1-c
    return np.asarray([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                       [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                       [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def _transform_parameters(index: int, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Recreate the exact rigid-body transform used by the candidate generator."""
    rng = np.random.default_rng(seed + index)
    if index == 0:
        return np.eye(3), np.zeros(3), 0.0
    angle = float(rng.uniform(0.02, math.pi if index > 100 else 0.4))
    scale = float(rng.uniform(0.1, 1.5) if index < 250 else rng.uniform(1.5, 25.0))
    trans = rng.normal(size=3); trans /= max(np.linalg.norm(trans), 1e-12); trans *= scale
    return _rotation(rng, angle), trans, angle


def _target_seed(target_id: str, seed: int) -> int:
    """Derive a stable per-target seed from the experiment seed.

    Using a target-specific stream prevents every target from receiving the
    same transform at a given candidate index while keeping the candidate
    protocol exactly reproducible.
    """
    digest = hashlib.sha1(str(target_id).encode("utf-8")).hexdigest()
    return int(seed) + int(digest[:8], 16)


def _candidate_atoms(native: Sequence[Mapping[str, object]], scenario: str, index: int, seed: int) -> Tuple[List[Dict[str, object]], float, np.ndarray]:
    atoms = [dict(a) for a in native]
    ca, cb = _classify(scenario, atoms)
    movable = set(cb)
    R, trans, angle = _transform_parameters(index, seed)
    ligand = np.vstack([a["xyz"] for a in atoms if a["chain"] in movable]) if movable else np.zeros((0,3))
    center = ligand.mean(axis=0) if len(ligand) else np.zeros(3)
    for atom in atoms:
        if atom["chain"] in movable:
            atom["xyz"] = (R @ (atom["xyz"] - center)) + center + trans
    return atoms, float(np.linalg.norm(trans) + angle * 2.0), trans


def _candidate_quality(index: int, seed: int) -> float:
    """Return the exact deterministic rigid-body quality proxy without parsing coordinates."""
    _, trans, angle = _transform_parameters(index, seed)
    return float(np.linalg.norm(trans) + angle * 2.0)


def _write_temp(atoms: Sequence[Mapping[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for idx, atom in enumerate(atoms, 1):
            handle.write(_pdb_line(atom, idx, atom["xyz"]))
        handle.write("END\n")


def _native_samples(scenario: str) -> List[Dict[str, str]]:
    p = os.path.join(ROOT, "results", scenario + "_samples.csv")
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    return []


_PROTEIN_BACKBONE = {"N", "CA", "C", "O"}
_NUCLEIC_BACKBONE = {
    "P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
    "O5*", "C5*", "C4*", "O4*", "C3*", "O3*",
}


def _residue_key(atom: Mapping[str, object]) -> Tuple[str, int, str]:
    return (str(atom["chain"]), int(atom["resseq"]), str(atom.get("icode", " ")))


def _backbone_atom(atom: Mapping[str, object]) -> bool:
    name = str(atom.get("name", "")).upper().replace("\u2019", "'")
    return name in _PROTEIN_BACKBONE or name in _NUCLEIC_BACKBONE


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=float)
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def _interface_contacts(receptor_atoms: Sequence[Mapping[str, object]], ligand_atoms: Sequence[Mapping[str, object]],
                        interface_cutoff: float = 10.0, contact_cutoff: float = 8.0):
    receptor_xyz = np.asarray([a["xyz"] for a in receptor_atoms], dtype=float)
    ligand_xyz = np.asarray([a["xyz"] for a in ligand_atoms], dtype=float)
    distances = _pairwise_distances(receptor_xyz, ligand_xyz)
    interface_receptor = set()
    interface_ligand = set()
    residue_min = {}
    for i, ra in enumerate(receptor_atoms):
        for j, la in enumerate(ligand_atoms):
            d = float(distances[i, j])
            if d <= interface_cutoff:
                interface_receptor.add(_residue_key(ra))
                interface_ligand.add(_residue_key(la))
            if d <= contact_cutoff:
                key = (_residue_key(ra), _residue_key(la))
                if d < residue_min.get(key, float("inf")):
                    residue_min[key] = d
    return interface_receptor, interface_ligand, set(residue_min)


def _kabsch(moving: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return row-vector rotation/translation mapping moving onto target."""
    if len(moving) == 0 or len(target) == 0 or len(moving) != len(target):
        return np.eye(3), np.zeros(3)
    cm = moving.mean(axis=0); ct = target.mean(axis=0)
    xm = moving - cm; xt = target - ct
    u, _, vt = np.linalg.svd(xm.T @ xt)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    # x @ r maps moving to target for row vectors.
    return r, ct - cm @ r


def _standard_interface_label(native_atoms: Sequence[Mapping[str, object]], candidate_atoms: Sequence[Mapping[str, object]],
                              scenario: str) -> Dict[str, object]:
    ca, cb = _classify(scenario, native_atoms)
    receptor_chains, ligand_chains = set(ca), set(cb)
    native_receptor = [a for a in native_atoms if a["chain"] in receptor_chains]
    native_ligand = [a for a in native_atoms if a["chain"] in ligand_chains]
    cand_receptor = [a for a in candidate_atoms if a["chain"] in receptor_chains]
    cand_ligand = [a for a in candidate_atoms if a["chain"] in ligand_chains]
    if not native_receptor or not native_ligand or not cand_receptor or not cand_ligand:
        return {"irmsd": float("nan"), "contact_recovery": float("nan"), "status": "invalid_no_interface",
                "fallback": 0, "irmsd_ligand_displacement": float("nan")}
    int_r, int_l, native_pairs = _interface_contacts(native_receptor, native_ligand)
    if not int_r or not int_l:
        return {"irmsd": float("nan"), "contact_recovery": float("nan"), "status": "invalid_no_interface",
                "fallback": 0, "irmsd_ligand_displacement": float("nan")}
    native_by_key = {_residue_key(a): a for a in native_receptor + native_ligand}
    cand_by_key = {_residue_key(a): a for a in cand_receptor + cand_ligand}
    common_r = [k for k in sorted(int_r) if k in native_by_key and k in cand_by_key]
    # Align receptor backbone atoms; generated candidates preserve residue keys.
    align_keys = [k for k in common_r if _backbone_atom(native_by_key[k]) and _backbone_atom(cand_by_key[k])]
    # Multiple atoms can share a residue key, so build a stable atom-level mapping.
    def keyed_backbone(atoms, allowed):
        return {(str(a["chain"]), int(a["resseq"]), str(a.get("icode", " ")), str(a["name"]).upper().replace("\u2019", "'")): a
                for a in atoms if _residue_key(a) in allowed and _backbone_atom(a)}
    nr = keyed_backbone(native_receptor, set(common_r)); cr = keyed_backbone(cand_receptor, set(common_r))
    common_atom_keys = sorted(set(nr) & set(cr))
    if len(common_atom_keys) >= 3:
        moving = np.asarray([cr[k]["xyz"] for k in common_atom_keys], dtype=float)
        target = np.asarray([nr[k]["xyz"] for k in common_atom_keys], dtype=float)
        rotation, shift = _kabsch(moving, target)
    else:
        rotation, shift = np.eye(3), np.zeros(3)
    def transform(atoms):
        return {id(a): (np.asarray(a["xyz"], dtype=float) @ rotation) + shift for a in atoms}
    native_xyz = {(_residue_key(a), str(a["name"]).upper().replace("\u2019", "'")): np.asarray(a["xyz"], dtype=float)
                  for a in native_receptor + native_ligand if _backbone_atom(a)}
    cand_xyz = {(_residue_key(a), str(a["name"]).upper().replace("\u2019", "'")): (np.asarray(a["xyz"], dtype=float) @ rotation) + shift
                for a in cand_receptor + cand_ligand if _backbone_atom(a)}
    interface_keys = [k for k in sorted(set(native_xyz) & set(cand_xyz)) if k[0] in int_r or k[0] in int_l]
    fallback = 0
    if len(interface_keys) >= 3:
        diffs = np.asarray([cand_xyz[k] - native_xyz[k] for k in interface_keys], dtype=float)
    else:
        fallback = 1
        native_h = {(_residue_key(a), str(a["name"])): np.asarray(a["xyz"], dtype=float) for a in native_receptor + native_ligand}
        cand_h = {(_residue_key(a), str(a["name"])): (np.asarray(a["xyz"], dtype=float) @ rotation) + shift for a in cand_receptor + cand_ligand}
        keys = [k for k in sorted(set(native_h) & set(cand_h)) if k[0] in int_r or k[0] in int_l]
        diffs = np.asarray([cand_h[k] - native_h[k] for k in keys], dtype=float) if keys else np.zeros((0, 3))
    if len(diffs) == 0:
        return {"irmsd": float("nan"), "contact_recovery": float("nan"), "status": "invalid_no_interface_atoms",
                "fallback": fallback, "irmsd_ligand_displacement": float("nan")}
    irmsd = float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))
    native_pair_set = set(native_pairs)
    # Recompute minimum atom distance for exactly the native residue pairs.
    cand_res = defaultdict(list)
    for a in cand_receptor + cand_ligand:
        cand_res[_residue_key(a)].append(np.asarray(a["xyz"], dtype=float) @ rotation + shift)
    recovered = 0
    for rk, lk in native_pair_set:
        if rk in cand_res and lk in cand_res:
            a = np.asarray(cand_res[rk]); b = np.asarray(cand_res[lk])
            if np.min(_pairwise_distances(a, b)) <= 8.0:
                recovered += 1
    contact = float(recovered / len(native_pair_set)) if native_pair_set else 0.0
    ligand_keys = [k for k in interface_keys if k[0] in int_l]
    ligand_diffs = np.asarray([cand_xyz[k] - native_xyz[k] for k in ligand_keys], dtype=float) if ligand_keys else np.zeros((0, 3))
    ligand_rmsd = float(np.sqrt(np.mean(np.sum(ligand_diffs * ligand_diffs, axis=1)))) if len(ligand_diffs) else irmsd
    return {"irmsd": irmsd, "contact_recovery": contact, "status": "ok_standard",
            "fallback": fallback, "irmsd_ligand_displacement": ligand_rmsd}


def prepare_decoys(scenario: str, count: int = 100, limit_targets: Optional[int] = None, seed: int = 20261103) -> str:
    ensure_layout(); rows = []
    native_rows = _native_samples(scenario)
    if limit_targets:
        native_rows = native_rows[:limit_targets]
    for item in native_rows:
        target = item.get("target_id") or item.get("candidate_id")
        ref = item.get("structure_path") or item.get("reference_path")
        if not ref or not os.path.exists(ref):
            continue
        atoms = _parse_atoms(ref)
        ca, cb = _classify(scenario, atoms)
        target_seed = _target_seed(str(target), seed)
        for i in range(count):
            rows.append({"scenario": scenario, "dataset": item.get("dataset", "native"), "target_id": target,
                         "candidate_id": "%s_d%04d" % (target, i),
                         "structure_path": "generated://%s/%s_d%04d" % (scenario, target, i),
                         "chain_a": ";".join(ca), "chain_b": ";".join(cb),
                         "candidate_source": "local_rigid_body_v2_target_seeded", "reference_path": ref,
                         "seed": target_seed, "seed_scope": "target_specific"})
    out = os.path.join(ROOT, "results", scenario + "_samples.parquet")
    write_table(out, rows); return out


def label_decoys(scenario: str, target_filter: Optional[Sequence[str]] = None,
                 output_path: Optional[str] = None) -> str:
    sample_path = os.path.join(ROOT, "results", scenario + "_samples.parquet")
    rows = read_table(sample_path); labels = []
    by_target = defaultdict(list)
    for r in rows: by_target[str(r["target_id"])].append(r)
    wanted = set(str(x).strip() for x in target_filter or [] if str(x).strip())
    for target, target_rows in by_target.items():
        if wanted and target not in wanted:
            continue
        ref = target_rows[0]["reference_path"]
        native = _parse_atoms(ref)
        for row in target_rows:
            idx = int(str(row["candidate_id"]).rsplit("d", 1)[-1])
            # The feature shards already on disk were generated from the
            # recorded candidate seed. Reconstruct that exact pose, then
            # compute standard interface labels without using the old proxy.
            atoms, _, _ = _candidate_atoms(native, scenario, idx, int(row.get("seed", 20261103)))
            metric = _standard_interface_label(native, atoms, scenario)
            irmsd = float(metric["irmsd"])
            valid = metric["status"] == "ok_standard" and math.isfinite(irmsd)
            quality = 1.0 / (1.0 + irmsd) if valid else float("nan")
            labels.append({"scenario": scenario, "target_id": target, "candidate_id": row["candidate_id"],
                           "irmsd": irmsd, "irmsd_standard": irmsd,
                           "irmsd_ligand_displacement": float(metric["irmsd_ligand_displacement"]),
                           "contact_recovery": float(metric["contact_recovery"]),
                           "quality": quality, "success": int(valid and irmsd < 5.0),
                           "label_status": str(metric["status"]),
                           "label_definition": "standard_interface_backbone_rmsd_v1",
                           "native_control": int(str(row["candidate_id"]).endswith("_d0000")),
                           "interface_backbone_fallback": int(metric["fallback"])})
    out = output_path or os.path.join(ROOT, "results", scenario + "_labels.parquet")
    write_table(out, labels); return out


def _extract_one(scenario: str, path: str, group: str) -> Dict[str, float]:
    if scenario == "protein_rna":
        return feature_group_vector(extract_pr_features(path), group)
    if scenario == "rna_rna":
        return extract_rr_features(path, group)
    values = extract_pd_feature_groups(path)
    return {k: float(v) for k, v in values.get(group, {}).items()} if isinstance(values.get(group), Mapping) else {k: float(v) for k, v in values.items() if k.startswith(group)}


def _extract_all(scenario: str, path: str, groups: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Extract all requested groups once per candidate (important for O(n^2) geometry)."""
    if scenario == "protein_rna":
        raw = extract_pr_features(path)
        return {g: feature_group_vector(raw, g) for g in groups}
    if scenario == "rna_rna":
        from .rna_rna_features import extract_all_rr_features
        raw = extract_all_rr_features(path, groups=groups)
        return {g: {k: float(v) for k, v in raw.get(g, {}).items()} for g in groups}
    raw = extract_pd_feature_groups(path)
    return {g: {k: float(v) for k, v in raw.get(g, {}).items()} for g in groups}


def _manifest_structural_zero_columns(scenario: str) -> set[str]:
    """Return only columns explicitly allowed to be absent by the manifest.

    The normal training path must never silently turn an extraction bug into a
    zero-valued predictor.  This small exception is reserved for a manifest
    declaration, and is therefore auditable rather than inferred from rows.
    """
    path = Path(ROOT) / "schemas" / f"{scenario}_v6_features.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(name) for name, spec in (payload.get("features") or {}).items()
                if isinstance(spec, Mapping) and bool(spec.get("structural_zero_allowed", False))}
    except Exception:
        return set()


def _strict_matrix(rows: Sequence[Mapping[str, object]], columns: Sequence[str], scenario: str, context: str) -> np.ndarray:
    """Build a feature matrix without implicit ``dict.get(..., 0)`` filling."""
    structural_zeros = _manifest_structural_zero_columns(scenario)
    missing = []
    matrix = []
    for row in rows:
        values = []
        for column in columns:
            raw = row.get(column)
            if raw is None or str(raw).strip() == "":
                if column in structural_zeros:
                    raw = 0.0
                else:
                    missing.append(f"{row.get('target_id')}:{row.get('candidate_id')}:{column}")
                    raw = 0.0
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"non-numeric feature {column!r} in {context} row {row.get('target_id')}:{row.get('candidate_id')}")
            if not math.isfinite(value):
                raise ValueError(f"non-finite feature {column!r} in {context} row {row.get('target_id')}:{row.get('candidate_id')}")
            values.append(value)
        matrix.append(values)
    if missing:
        raise ValueError(f"missing registered feature columns in {context}: {missing[:12]}")
    return np.asarray(matrix, dtype=float)


def extract_features(scenario: str, groups: Optional[Sequence[str]] = None,
                     limit_targets: Optional[int] = None,
                     target_filter: Optional[Sequence[str]] = None) -> str:
    sample_path = os.path.join(ROOT, "results", scenario + "_samples.parquet")
    rows = read_table(sample_path); labels = {(x["target_id"], x["candidate_id"]): x for x in read_table(os.path.join(ROOT, "results", scenario + "_labels.parquet"))}
    groups = list(groups or SCENARIO_GROUPS[scenario]); targets = []
    for r in rows:
        if r["target_id"] not in targets: targets.append(r["target_id"])
    if target_filter:
        wanted = set(str(x).strip() for x in target_filter if str(x).strip())
        targets = [t for t in targets if t in wanted]
    if limit_targets: targets = targets[:limit_targets]
    out_rows = []
    feature_root = os.path.join(ROOT, "features", scenario)
    os.makedirs(feature_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fair_", dir=os.path.join(ROOT, "logs")) as tmp:
        cache = {}
        for target in targets:
            target_rows = [r for r in rows if r["target_id"] == target]
            native = _parse_atoms(target_rows[0]["reference_path"])
            per_group = {g: [] for g in groups}
            for r in target_rows:
                idx = int(str(r["candidate_id"]).rsplit("d", 1)[-1]); atoms, _, _ = _candidate_atoms(native, scenario, idx, int(r.get("seed", 20261103)))
                path = os.path.join(tmp, "candidate.pdb"); _write_temp(atoms, path)
                try:
                    all_vecs = _extract_all(scenario, path, groups); extraction_error = None
                except Exception as exc:
                    all_vecs = {}; extraction_error = str(exc)[:120]
                for group in groups:
                    try:
                        vec = all_vecs.get(group, {}); status = "ok" if extraction_error is None else "failed:%s" % extraction_error
                    except Exception as exc:
                        vec = {}; status = "failed:%s" % str(exc)[:120]
                    record = {"scenario": scenario, "target_id": target, "candidate_id": r["candidate_id"],
                              "feature_group": group, "feature_status": status,
                              "baseline_feature_columns": {group: sorted(str(key) for key in vec)}, **vec}
                    per_group[group].append(record)
                    # Keep a combined smoke artifact, while full runs remain streamed.
                    if len(targets) <= 5 and not target_filter:
                        out_rows.append(record)
            for group, group_rows in per_group.items():
                write_table(os.path.join(feature_root, group, target + ".parquet"), group_rows)
    out = os.path.join(ROOT, "results", scenario + "_features.parquet")
    if (len(targets) > 5 or target_filter) and os.path.exists(out):
        os.remove(out)
    if out_rows:
        write_table(out, out_rows)
    return out


def _residue_sequence(atoms: Sequence[Mapping[str, object]], chains: Sequence[str], scenario: str) -> str:
    aa = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V","MSE":"M"}
    nt = {"DA":"A","DC":"C","DG":"G","DT":"T","DI":"N","A":"A","C":"C","G":"G","T":"T","U":"U","ADE":"A","CYT":"C","GUA":"G","THY":"T","URA":"U"}
    seen = set(); out = []
    for atom in sorted((a for a in atoms if str(a["chain"]) in set(chains)), key=lambda x: (str(x["chain"]), int(x["resseq"]), str(x.get("icode", " ")))):
        key = _residue_key(atom)
        if key in seen: continue
        seen.add(key)
        name = str(atom.get("resname", "")).upper()
        out.append((aa if scenario != "rna_rna" and name in aa else nt).get(name, "X"))
    return "".join(out)


def _global_identity_coverage(a: str, b: str) -> Tuple[float, float]:
    if not a or not b: return 0.0, 0.0
    # Needleman-Wunsch unit-cost alignment; gaps are not counted as identity.
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int16)
    dp[:, 0] = np.arange(n + 1); dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    distance = int(dp[n, m]); aligned = max(n, m)
    identity = max(0.0, 1.0 - distance / float(aligned))
    coverage = min(n, m) / float(aligned)
    return identity, coverage


def _sequence_clusters(scenario: str, targets: Sequence[str], rows: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    refs = {}
    for target in targets:
        row = next(r for r in rows if str(r["target_id"]) == str(target))
        atoms = _parse_atoms(row["reference_path"])
        ca, cb = _classify(scenario, atoms)
        refs[str(target)] = (_residue_sequence(atoms, ca, scenario), _residue_sequence(atoms, cb, scenario))
    parent = {t: t for t in targets}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i, ta in enumerate(targets):
        for tb in targets[i + 1:]:
            best = 0.0
            for xa, xb in ((refs[ta][0], refs[tb][0]), (refs[ta][1], refs[tb][1])):
                ident, cov = _global_identity_coverage(xa, xb)
                if cov >= 0.8: best = max(best, ident)
            if best >= 0.8: union(ta, tb)
    roots = {}
    out = {}
    for t in targets:
        root = find(t); roots.setdefault(root, len(roots) + 1)
        out[t] = "seqcluster_%03d" % roots[root]
    return out


def make_folds(scenario: str, n_splits: int = 5) -> str:
    rows = read_table(os.path.join(ROOT, "results", scenario + "_samples.parquet")); targets = sorted({str(r["target_id"]) for r in rows})
    groups = _sequence_clusters(scenario, targets, rows)
    cluster_sizes = defaultdict(int)
    for t, g in groups.items(): cluster_sizes[g] += 1
    ordered_groups = sorted(cluster_sizes, key=lambda g: (-cluster_sizes[g], g))
    fold_load = [0] * n_splits; group_fold = {}
    for g in ordered_groups:
        fold = min(range(n_splits), key=lambda f: (fold_load[f], f)); group_fold[g] = fold; fold_load[fold] += cluster_sizes[g]
    folds = []
    for target in targets:
        outer = group_fold[groups[target]]
        for inner in range(n_splits):
            folds.append({"scenario": scenario, "target_id": target, "target_group": groups[target],
                          "cluster_method": "sequence_identity_global_v1", "cluster_size": cluster_sizes[groups[target]],
                          "outer_fold": outer, "inner_fold": inner, "independent_test": int(outer == n_splits - 1)})
    out = os.path.join(ROOT, "results", scenario + "_folds.parquet"); write_table(out, folds); return out


def merge_shard_labels(scenario: str) -> str:
    rows = []
    prefix = scenario + "_labels_"
    for name in os.listdir(os.path.join(ROOT, "results")):
        if name.startswith(prefix) and name.endswith(".parquet"):
            rows.extend(read_table(os.path.join(ROOT, "results", name)))
    out = os.path.join(ROOT, "results", scenario + "_labels.parquet")
    write_table(out, rows); return out


def train_models(scenario: str) -> str:
    """Train one head per group with target-grouped outer folds."""
    labels = {(x["target_id"], x["candidate_id"]): x
              for x in read_table(os.path.join(ROOT, "results", scenario + "_labels.parquet"))}
    fold_rows = read_table(os.path.join(ROOT, "results", scenario + "_folds.parquet"))
    fold_map = {}
    for row in fold_rows:
        fold_map.setdefault(str(row["target_id"]), int(row["outer_fold"]))
    # If folds were not generated yet, create a deterministic target split.
    if not fold_map:
        targets = sorted({x[0] for x in labels})
        fold_map = {t: i % 5 for i, t in enumerate(targets)}
    preds = []; models = []
    try:
        from xgboost import XGBRanker
        engine = "xgboost_rank_pairwise"
    except Exception:
        XGBRanker = None; engine = "sklearn_gradient_boosting_fallback"
    root = os.path.join(ROOT, "features", scenario)
    combined = read_table(os.path.join(ROOT, "results", scenario + "_features.parquet"))
    groups = sorted({r["feature_group"] for r in combined}) if combined else list(SCENARIO_GROUPS[scenario])
    for group in groups:
        if combined:
            data = [r for r in combined if r["feature_group"] == group and r.get("feature_status", "").startswith("ok")
                    and not str(r.get("candidate_id", "")).endswith("_d0000")]
        else:
            data = []
            gdir = os.path.join(root, group)
            if os.path.isdir(gdir):
                for name in os.listdir(gdir):
                    if name.endswith(".parquet"):
                        data.extend([r for r in read_table(os.path.join(gdir, name))
                                     if r.get("feature_status", "").startswith("ok")
                                     and not str(r.get("candidate_id", "")).endswith("_d0000")])
        if not data: continue
        meta = {"scenario", "target_id", "candidate_id", "feature_group", "feature_status"}
        if scenario == "protein_rna" and group in PR_FEATURE_GROUPS:
            cols = list(PR_FEATURE_GROUPS[group])
        else:
            cols = sorted({k for r in data for k, v in r.items()
                           if k not in meta and str(v).strip() != ""})
        for fold in sorted(set(fold_map.values())):
            train = [r for r in data if fold_map.get(str(r["target_id"]), 0) != fold]
            test = [r for r in data if fold_map.get(str(r["target_id"]), 0) == fold]
            if not train or not test: continue
            Xtr = _strict_matrix(train, cols, scenario, f"{group}/train/fold{fold}")
            ytr = np.asarray([float(labels[(r["target_id"], r["candidate_id"])]["quality"]) for r in train], dtype=float)
            Xte = _strict_matrix(test, cols, scenario, f"{group}/test/fold{fold}")
            if Xtr.ndim != 2 or Xtr.shape[1] == 0: continue
            if XGBRanker is not None:
                model = XGBRanker(objective="rank:pairwise", n_estimators=100, max_depth=4, learning_rate=0.05,
                                  subsample=0.9, colsample_bytree=0.9, random_state=20261103, n_jobs=1)
                counts = [sum(1 for r in train if r["target_id"] == t) for t in sorted({r["target_id"] for r in train})]
                model.fit(Xtr, ytr, group=counts, verbose=False); score = model.predict(Xte)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=20261103, n_estimators=100, max_depth=3, learning_rate=0.05)
                model.fit(Xtr, ytr); score = model.predict(Xte)
            models.append({"scenario": scenario, "feature_group": group, "outer_fold": fold,
                           "engine": engine, "n_train": len(train), "n_test": len(test),
                           "n_features": len(cols), "columns": json.dumps(cols)})
            for r, s in zip(test, score):
                preds.append({"scenario": scenario, "target_id": r["target_id"], "candidate_id": r["candidate_id"],
                              "feature_group": group, "outer_fold": fold,
                              "predicted_score": float(s), "model_engine": engine})
    out = os.path.join(ROOT, "results", scenario + "_predictions.parquet"); write_table(out, preds)
    write_table(os.path.join(ROOT, "results", scenario + "_models.parquet"), models); return out


def evaluate(scenario: str) -> str:
    preds = read_table(os.path.join(ROOT, "results", scenario + "_predictions.parquet"))
    labels = {(x["target_id"], x["candidate_id"]): x for x in read_table(os.path.join(ROOT, "results", scenario + "_labels.parquet"))}
    rows = []
    for group in sorted({r["feature_group"] for r in preds}):
        gp = [r for r in preds if r["feature_group"] == group]; metrics = []
        for target in sorted({r["target_id"] for r in gp}):
            c = sorted([r for r in gp if r["target_id"] == target and not str(r["candidate_id"]).endswith("_d0000")],
                       key=lambda x: float(x["predicted_score"]), reverse=True)
            q = [float(labels[(x["target_id"], x["candidate_id"])]["quality"]) for x in c]
            top = q[0] if q else 0.0; best = max(q) if q else 0.0
            top5 = max(q[:5]) if q else 0.0; top10 = max(q[:10]) if q else 0.0
            success = float(labels[(c[0]["target_id"], c[0]["candidate_id"])]["success"]) if c else 0.0
            rho = float(spearmanr([-float(x["predicted_score"]) for x in c], q).statistic) if spearmanr is not None and len(q) > 2 else 0.0
            success_at_5 = float(any(float(labels[(x["target_id"], x["candidate_id"])]["success"]) > 0 for x in c[:5]))
            success_at_10 = float(any(float(labels[(x["target_id"], x["candidate_id"])]["success"]) > 0 for x in c[:10]))
            metrics.append({"top1": top, "top5": top5, "top10": top10, "best": best, "success": success,
                            "success_at_5": success_at_5, "success_at_10": success_at_10, "rho": rho})
        if metrics:
            rng = np.random.default_rng(20261103)
            boot = np.asarray([np.mean([metrics[int(i)]["top1"] for i in rng.integers(0, len(metrics), len(metrics))]) for _ in range(200)])
            rows.append({"scenario": scenario, "feature_group": group, "targets": len(metrics),
                         "top1_quality": float(np.mean([m["top1"] for m in metrics])),
                         "top5_quality": float(np.mean([m["top5"] for m in metrics])),
                         "top10_quality": float(np.mean([m["top10"] for m in metrics])),
                          "success_rate": float(np.mean([m["success"] for m in metrics])),
                          "success_at_5": float(np.mean([m["success_at_5"] for m in metrics])),
                          "success_at_10": float(np.mean([m["success_at_10"] for m in metrics])),
                         "spearman": float(np.mean([m["rho"] for m in metrics])),
                         "ranking_regret": float(np.mean([m["best"] - m["top1"] for m in metrics])),
                         "top1_bootstrap95_low": float(np.quantile(boot, 0.025)),
                         "top1_bootstrap95_high": float(np.quantile(boot, 0.975))})
    out = os.path.join(ROOT, "reports", scenario + "_fair_summary.parquet"); write_table(out, rows); return out


def run_stage(stage: str, scenario: str, count: int = 100, limit_targets: Optional[int] = None) -> str:
    if stage == "prepare-decoys": return prepare_decoys(scenario, count, limit_targets)
    if stage == "label": return label_decoys(scenario)
    if stage == "extract": return extract_features(scenario, limit_targets=limit_targets)
    if stage == "train": return train_models(scenario)
    if stage == "evaluate": return evaluate(scenario)
    if stage == "resume":
        prepare_decoys(scenario, count, limit_targets); label_decoys(scenario); extract_features(scenario, limit_targets=limit_targets); make_folds(scenario); train_models(scenario); return evaluate(scenario)
    raise ValueError(stage)

