"""GSPH1: typed, geometrically supported GLMY path features.

This module is deliberately independent of the fair-baseline predictors.  It
turns a candidate complex into a traceable feature dictionary consisting of:

* directed donor--acceptor interaction events with local RDF/contact/LJ/shape
  attributes;
* directed chemical-transfer paths with a common geometric-support birth;
* finite GLMY path-chain algebra (A_k, Omega_k, boundary ranks, Betti numbers);
* path-level attribute lifting and attribute-weighted Hodge signatures.

The implementation is bounded and deterministic.  ``truncated`` is emitted
whenever the configured event/path limits are reached; paths are never
silently discarded.  The implementation uses only structure-derived data and
fixed public constants.  It is a research prototype, not an Amber energy or
official FR3D/DSSR scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from cspph.chemistry.interaction_backend import InteractionPairBatch, build_interaction_pair_batch
except ImportError:  # pragma: no cover - direct script execution
    from cspph.chemistry.interaction_backend import InteractionPairBatch, build_interaction_pair_batch

try:
    from Bio.PDB import MMCIFParser, PDBParser
except Exception:  # pragma: no cover - optional dependency in minimal installs
    MMCIFParser = PDBParser = None


ELEMENTS = ("C", "N", "O", "P", "S", "X")
ELEMENT_PAIRS = tuple((a, b) for i, a in enumerate(ELEMENTS) for b in ELEMENTS[i:])
ELEMENT_PAIR_INDEX = {pair: i for i, pair in enumerate(ELEMENT_PAIRS)}
ORDERED_ELEMENT_PAIRS = tuple((a, b) for a in ELEMENTS for b in ELEMENTS)
ORDERED_ELEMENT_PAIR_INDEX = {pair: i for i, pair in enumerate(ORDERED_ELEMENT_PAIRS)}
RDF_EDGES = np.asarray((2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0), dtype=float)
RDF_BINS = len(RDF_EDGES) - 1
HEAT_TIMES = (0.1, 0.5, 1.0, 2.0)

# Fixed, transparent Lennard--Jones-like constants.  They are intentionally
# generic element-class constants, not an Amber force field energy protocol.
LJ_SIGMA = {"C": 3.40, "N": 3.25, "O": 3.00, "P": 3.70, "S": 3.60, "X": 3.50}
LJ_EPSILON = {"C": 0.105, "N": 0.170, "O": 0.210, "P": 0.200, "S": 0.250, "X": 0.100}
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "S": 1.80, "X": 1.70}
COVALENT_RADII = {"C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07, "S": 1.05, "X": 0.77}


def _vdw_radius(element) -> float:
    """Van der Waals radius with a conservative fallback for metals/exotics.

    ``X`` (1.70 A) is the generic fallback already used for unknown elements;
    a metal ion must never raise KeyError and silently abort bond-graph or
    support construction.
    """
    return float(VDW_RADII.get(str(element).strip().upper(), VDW_RADII["X"]))

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "MSE",
}
NUCLEIC_RESIDUES = {
    "A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "ADE", "CYT", "GUA",
    "URA", "RA", "RC", "RG", "RU", "DI", "DU", "5MC", "1MA", "7MG",
}
DNA_RESIDUES = {"DA", "DC", "DG", "DT"}
RNA_RESIDUES = NUCLEIC_RESIDUES - DNA_RESIDUES - {"T", "DT"}
WATER_RESIDUES = {"HOH", "WAT", "H2O", "DOD", "TIP3", "SOL"}
BASE_MAP = {
    "A": "A", "DA": "A", "ADE": "A", "RA": "A", "1MA": "A", "C": "C",
    "DC": "C", "CYT": "C", "RC": "C", "5MC": "C", "G": "G", "DG": "G",
    "GUA": "G", "RG": "G", "7MG": "G", "U": "U", "DT": "U", "T": "U",
    "URA": "U", "RU": "U",
}
RESIDUE_CLASSES = ("charged", "polar", "hydrophobic", "aromatic", "nucleic", "other")
RESIDUE_CLASS_INDEX = {name: i for i, name in enumerate(RESIDUE_CLASSES)}


@dataclass(frozen=True)
class GSPH1Config:
    """Bounded deterministic settings for one extraction."""

    event_cutoff: float = 5.0
    transfer_cutoff: float = 6.0
    local_radius: float = 4.0
    scales: Tuple[float, ...] = (2.5, 3.5, 5.5, 7.0, 12.0)
    # The default is intentionally conservative because Hodge signatures use
    # dense linear algebra.  Increase these limits for a controlled study and
    # inspect the emitted ``*_truncated`` flags.
    max_events: int = 32
    max_paths_per_order: int = 64
    max_local_pairs_per_event: int = 3000
    max_hodge_dimension: int = 64
    svd_tol: float = 1e-9
    max_bond_transfer_distance: int = 3
    mode: str = "debug"  # debug retains bounded diagnostics; benchmark rejects incomplete topology
    exclude_water: bool = True
    require_hydrogen: bool = False
    # First-phase CPU/GPU interaction backend.  ``auto`` uses CUDA only when
    # an existing CUDA-enabled PyTorch is available and the pair table is
    # large enough to amortize transfers.  No dependency is installed by the
    # extractor itself.
    interaction_backend: str = "auto"  # auto, cpu, cuda
    gpu_min_pairs: int = 250_000
    gpu_batch_pairs: int = 500_000
    gpu_device: str = "cuda:0"
    interaction_batch_size: int = 16
    support_workers: int = 4
    support_gpu_min_paths: int = 64
    support_gpu_batch_paths: int = 4096
    path_backend: str = "auto"  # auto, compact_csr, python_stream
    surface_backend: str = "auto"  # auto, none
    hodge_backend: str = "auto"  # auto, cpu, cuda
    # Intramolecular adapter (H3): when >0, atom pairs whose covalent-graph
    # distance is strictly below this many bonds are excluded from event
    # generation.  Covalently linked neighbours are real chemistry, not
    # microcontacts; 0 keeps the legacy intermolecular behaviour unchanged.
    intra_min_bond_sep: int = 0
    # Steric-clash floor: pairs closer than this fraction of the summed
    # van der Waals radii are modelling artifacts, not contacts.  0 disables
    # the filter (legacy behaviour).  Observed legitimate nonbonded contacts
    # stay above ~0.65 of the vdW sum; deep overlaps sit below ~0.4.
    contact_min_vdw_ratio: float = 0.0


@dataclass
class Atom:
    index: int
    chain: str
    name: str
    resname: str
    resid: int
    element: str
    xyz: np.ndarray
    serial: int = 0
    hetero: str = " "
    icode: str = ""


@dataclass
class Event:
    index: int
    donor: int
    acceptor: int
    event_type: str
    donor_role: str
    acceptor_role: str
    center: np.ndarray
    tau: float
    distance: float
    attributes: Dict[str, float]


def _resolve_config(config: GSPH1Config) -> Tuple[GSPH1Config, bool]:
    """Resolve the formal benchmark default without hiding bounded smoke runs.

    The historical defaults (32 events/64 paths/3000 local pairs/64 Hodge
    dimensions) are useful for debug examples but are not a complete
    benchmark.  When *all* historical defaults are untouched in benchmark
    mode, lift them to explicit unbounded values.  A caller that supplies a
    smaller bound (for a smoke test) retains it and receives the normal
    ``incomplete`` status instead of silently changing scientific scope.
    """
    if config.mode != "benchmark":
        return config, False
    historical = (config.max_events == 32 and config.max_paths_per_order == 64 and
                  config.max_local_pairs_per_event == 3000 and config.max_hodge_dimension == 64)
    if not historical:
        return config, False
    return replace(config, max_events=-1, max_paths_per_order=-1,
                   max_local_pairs_per_event=-1, max_hodge_dimension=-1), True


def _norm_element(value: str) -> str:
    value = (value or "").strip().upper()
    return value if value in ELEMENTS[:-1] else "X"


def _parse_atoms(path: str, exclude_water: bool = True) -> List[Atom]:
    if PDBParser is None:
        raise RuntimeError("Biopython is required for GSPH1 structure parsing")
    parser = MMCIFParser(QUIET=True) if str(path).lower().endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    structure = parser.get_structure("gsph1", path)
    atoms: List[Atom] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = str(residue.resname).strip().upper()
                hetero = str(residue.id[0] or " ")
                if exclude_water and resname in WATER_RESIDUES:
                    continue
                for atom in residue:
                    raw_element = (getattr(atom, "element", "") or "").strip().upper()
                    if raw_element == "H" or atom.get_name().upper().startswith("H"):
                        continue
                    element = _norm_element(raw_element)
                    atoms.append(Atom(
                        index=len(atoms), chain=str(chain.id), name=str(atom.get_name()).strip().upper(),
                        resname=resname, resid=int(residue.id[1]), element=element,
                        xyz=np.asarray(atom.get_coord(), dtype=float),
                        serial=int(getattr(atom, "serial_number", 0) or 0), hetero=hetero,
                        icode=str(residue.id[2] or ""),
                    ))
        break
    return atoms


def _pair_distance(a: Atom, b: Atom) -> float:
    return float(np.linalg.norm(a.xyz - b.xyz))


def _chain_ids(spec: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in str(spec).replace(";", ",").split(",") if x.strip())


def _entity_type(atoms: Sequence[Atom]) -> str:
    counts = {
        "protein": sum(a.resname in PROTEIN_RESIDUES for a in atoms),
        "dna": sum(a.resname in DNA_RESIDUES for a in atoms),
        "rna": sum(a.resname in RNA_RESIDUES for a in atoms),
    }
    label, count = max(counts.items(), key=lambda item: item[1])
    return label if count >= 2 else "other"


def _side_atoms(atoms: Sequence[Atom], spec: str) -> List[Atom]:
    ids = set(_chain_ids(spec))
    return [a for a in atoms if a.chain in ids]


def _interface_count(first: Sequence[Atom], second: Sequence[Atom], cutoff: float = 5.0) -> int:
    if not first or not second:
        return 0
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray([a.xyz for a in second], dtype=float))
        return int(sum(len(x) for x in tree.query_ball_point(np.asarray([a.xyz for a in first]), cutoff)))
    except Exception:  # pragma: no cover - deterministic fallback
        return sum(_pair_distance(a, b) <= cutoff for a in first for b in second)


def _validate_scenario_pair(atoms: Sequence[Atom], chain_a: str, chain_b: str, scenario: str) -> None:
    if scenario == "rna_quality":
        if set(_chain_ids(chain_a)) == set(_chain_ids(chain_b)) and \
                _entity_type(_side_atoms(atoms, chain_a)) == "rna":
            return
        raise ValueError(f"scenario/chain mismatch: rna_quality requires one "
                         f"shared rna side, got {chain_a},{chain_b}")
    if scenario == "protein_stability":
        if set(_chain_ids(chain_a)) == set(_chain_ids(chain_b)) and \
                _entity_type(_side_atoms(atoms, chain_a)) == "protein":
            return
        raise ValueError(f"scenario/chain mismatch: protein_stability requires "
                         f"one shared protein side, got {chain_a},{chain_b}")
    type_a = _entity_type(_side_atoms(atoms, chain_a)); type_b = _entity_type(_side_atoms(atoms, chain_b))
    pair = {type_a, type_b}
    valid = ((scenario == "protein_rna" and pair == {"protein", "rna"}) or
             (scenario == "rna_rna" and type_a == type_b == "rna") or
             (scenario == "protein_dna" and pair == {"protein", "dna"}))
    if not valid:
        raise ValueError(f"scenario/chain mismatch: {scenario} got {type_a},{type_b} for {chain_a},{chain_b}")


def _chain_pair(atoms: Sequence[Atom], scenario: str, chain_a: Optional[str], chain_b: Optional[str]) -> Tuple[str, str]:
    if chain_a and chain_b:
        _validate_scenario_pair(atoms, str(chain_a), str(chain_b), scenario)
        return str(chain_a), str(chain_b)
    chains = sorted({a.chain for a in atoms})
    by_chain = {chain: [a for a in atoms if a.chain == chain] for chain in chains}
    types = {chain: _entity_type(rows) for chain, rows in by_chain.items()}
    if scenario in ("rna_quality", "protein_stability"):
        want = "rna" if scenario == "rna_quality" else "protein"
        side = sorted(c for c in chains if types[c] == want)
        if not side:
            raise ValueError(f"no {want} chain for {scenario}")
        spec = ",".join(side)
        _validate_scenario_pair(atoms, spec, spec, scenario)
        return spec, spec
    if len(chains) < 2:
        raise ValueError("GSPH1 requires at least two chains")
    candidates = []
    if scenario == "protein_rna":
        proteins = [c for c in chains if types[c] == "protein"]
        rnas = [c for c in chains if types[c] == "rna"]
        # A multi-chain entity is represented as a comma-separated side.
        if proteins and rnas:
            candidates.append((",".join(proteins), ",".join(rnas)))
    elif scenario == "rna_rna":
        rnas = [c for c in chains if types[c] == "rna"]
        candidates.extend((a, b) for i, a in enumerate(rnas) for b in rnas[i + 1:])
    elif scenario == "protein_dna":
        proteins = [c for c in chains if types[c] == "protein"]
        dnas = [c for c in chains if types[c] == "dna"]
        candidates.extend((a, b) for a in proteins for b in dnas)
    if not candidates:
        raise ValueError(f"no scenario-compatible chain/entity pair for {scenario}")
    best = max(candidates, key=lambda pair: _interface_count(_side_atoms(atoms, pair[0]), _side_atoms(atoms, pair[1])))
    _validate_scenario_pair(atoms, best[0], best[1], scenario)
    return best


def _residue_class(resname: str) -> str:
    name = resname.upper()
    if name in NUCLEIC_RESIDUES:
        return "nucleic"
    if name in {"ARG", "ASP", "GLU", "LYS", "HIS", "HIP", "HID", "HIE"}:
        return "charged"
    if name in {"ASN", "GLN", "SER", "THR", "CYS", "TYR"}:
        return "polar"
    if name in {"ALA", "VAL", "ILE", "LEU", "MET", "PRO"}:
        return "hydrophobic"
    if name in {"PHE", "TRP", "TYR", "HIS", "HID", "HIE", "HIP"}:
        return "aromatic"
    return "other"


def _base_class(resname: str) -> str:
    name = resname.upper()
    # Keep thymine distinct from uracil.  PDB DNA residues are commonly
    # written as DT (and occasionally as bare T); collapsing them to U would
    # leak an RNA identity into the DNA base-edge/groove feature groups.
    if name in {"DT", "T"}:
        return "T"
    if name in DNA_RESIDUES:
        return BASE_MAP.get(name, "N")
    return BASE_MAP.get(name, "backbone" if name in RNA_RESIDUES else "other")


def _roles(atom: Atom) -> Tuple[bool, bool]:
    """Return template-based donor/acceptor candidate flags.

    This is a directional *candidate* annotation.  It deliberately excludes
    phosphorus as an H-bond acceptor, excludes water from the event graph, and
    does not claim a hydrogen bond when the input lacks explicit hydrogens.
    """
    name = atom.name.upper(); res = atom.resname.upper()
    if res in WATER_RESIDUES:
        return False, False
    donor = acceptor = False
    if res in PROTEIN_RESIDUES:
        donor_atoms = {
            "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"},
            "HIS": {"ND1", "NE2"}, "HID": {"ND1"}, "HIE": {"NE2"}, "HIP": {"ND1", "NE2"},
            "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
            "CYS": {"SG"}, "TRP": {"NE1"},
        }
        acceptor_atoms = {
            "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"},
            "GLN": {"OE1"}, "HIS": {"ND1", "NE2"}, "HID": {"ND1"}, "HIE": {"NE2"},
            "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "CYS": {"SG"}, "MET": {"SD"},
        }
        donor = name == "N" and res != "PRO" or name in donor_atoms.get(res, set())
        acceptor = name in {"O", "OXT"} or name in acceptor_atoms.get(res, set())
    elif res in NUCLEIC_RESIDUES:
        base = "T" if res in {"DT", "T"} else BASE_MAP.get(res, "N")
        donor_atoms = {"A": {"N6"}, "G": {"N1", "N2"}, "C": {"N3", "N4"}, "U": {"N3"}, "T": {"N3"}}
        acceptor_atoms = {"A": {"N1", "N7"}, "G": {"O6", "N3", "N7"}, "C": {"O2", "N3"}, "U": {"O2", "O4"}, "T": {"O2", "O4"}}
        donor = name in donor_atoms.get(base, set()) or name in {"O2'", "O3'", "O5'"}
        acceptor = name in acceptor_atoms.get(base, set()) or name in {"O2'", "O3'", "O4'", "O5'", "OP1", "OP2", "O1P", "O2P"}
    return bool(donor), bool(acceptor)


def _local_entry_query(entry, center: np.ndarray, radius: float):
    """Query a native or rigidly transformed local index.

    TargetGeometryCache may return ``(rows, coords, tree, R, t)`` for a chain
    that moved by a verified rigid transform.  The KD-tree remains in native
    coordinates; only the query point and returned neighbor coordinates are
    transformed.  The three-element legacy tuple stays fully supported.
    """
    if entry is None:
        return []
    rows, coords, tree = entry[:3]
    rotation = np.eye(3)
    translation = np.zeros(3)
    if len(entry) >= 5:
        rotation = np.asarray(entry[3], dtype=float)
        translation = np.asarray(entry[4], dtype=float)
    native_center = rotation.T @ (np.asarray(center, dtype=float) - translation)
    if tree is not None:
        indices = tree.query_ball_point(native_center, radius)
    else:
        indices = [i for i, xyz in enumerate(coords) if np.linalg.norm(np.asarray(xyz) - native_center) <= radius]
    transformed = [rotation @ np.asarray(coords[i], dtype=float) + translation for i in indices]
    return [(rows[i], xyz) for i, xyz in zip(indices, transformed)]


def _normal(atoms: Sequence[Atom], endpoint: Atom, other: Atom, radius: float,
            local_index: Optional[Mapping[str, Tuple[List[Atom], np.ndarray, object]]] = None) -> Tuple[np.ndarray, float]:
    if local_index is not None and endpoint.chain in local_index:
        neighbours = [xyz for _, xyz in _local_entry_query(local_index[endpoint.chain], endpoint.xyz, radius)]
    else:
        neighbours = [a.xyz for a in atoms if a.chain == endpoint.chain and np.linalg.norm(a.xyz - endpoint.xyz) <= radius]
    if len(neighbours) < 3:
        return np.zeros(3, dtype=float), 0.0
    x = np.asarray(neighbours, dtype=float) - np.mean(neighbours, axis=0)
    cov = (x.T @ x) / max(1, len(x))
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, int(np.argmin(vals))]
    direction = other.xyz - endpoint.xyz
    if float(np.dot(normal, direction)) < 0:
        normal = -normal
    curvature = float(vals[0] / max(float(np.sum(vals)), 1e-8))
    return normal, curvature


def _build_local_index(atoms: Sequence[Atom]) -> Dict[str, Tuple[List[Atom], np.ndarray, object]]:
    """Build reusable per-chain coordinate indexes for local event features."""
    grouped: Dict[str, List[Atom]] = defaultdict(list)
    for atom in atoms:
        grouped[atom.chain].append(atom)
    index: Dict[str, Tuple[List[Atom], np.ndarray, object]] = {}
    try:
        from scipy.spatial import cKDTree
    except Exception:  # pragma: no cover
        cKDTree = None
    for chain, rows in grouped.items():
        coords = np.asarray([atom.xyz for atom in rows], dtype=float)
        tree = cKDTree(coords) if cKDTree is not None and len(coords) else None
        index[chain] = (rows, coords, tree)
    return index


def _local_attributes(atoms: Sequence[Atom], donor: Atom, acceptor: Atom, config: GSPH1Config,
                      local_index: Optional[Mapping[str, Tuple[List[Atom], np.ndarray, object]]] = None,
                      pair_batch: Optional[InteractionPairBatch] = None,
                      surface_features: Optional[Mapping[str, object]] = None,
                      atom_map: Optional[Mapping[int, Atom]] = None) -> Dict[str, float]:
    """Compute event-local RDF/contact/LJ/shape from one global pair table.

    ``pair_batch`` is constructed once per candidate complex by ``_make_events``.
    The fallback construction keeps this helper independently usable in tests
    and notebooks, while avoiding the old per-event dense distance matrix in
    normal extraction.
    """
    if local_index is None:
        local_index = _build_local_index(atoms)

    def nearby(chain: str, center: np.ndarray) -> List[Atom]:
        return [row for row, _ in _local_entry_query(local_index.get(chain), center, config.local_radius)]

    donor_side = nearby(donor.chain, donor.xyz)
    acceptor_side = nearby(acceptor.chain, acceptor.xyz)
    if atom_map is None:
        atom_map = {atom.index: atom for atom in atoms}
    if pair_batch is None:
        left_atoms = [atom for atom in atoms if atom.chain == donor.chain]
        right_atoms = [atom for atom in atoms if atom.chain == acceptor.chain]
        pair_batch = build_interaction_pair_batch(
            atoms, left_atoms, right_atoms, max(config.event_cutoff + 2.0, 8.0),
            ELEMENTS, RESIDUE_CLASS_INDEX, RDF_EDGES, LJ_SIGMA, LJ_EPSILON,
            backend=config.interaction_backend, gpu_min_pairs=config.gpu_min_pairs,
            gpu_batch_pairs=config.gpu_batch_pairs, gpu_device=config.gpu_device,
        )
    rows, forward = pair_batch.rows_for_event(
        donor.index, acceptor.index,
        [atom.index for atom in donor_side], [atom.index for atom in acceptor_side],
        config.local_radius, atom_map,
    )
    if rows.size:
        if forward:
            donor_indices = pair_batch.left_indices[rows]
            acceptor_indices = pair_batch.right_indices[rows]
            donor_elements = pair_batch.left_element_codes[rows]
            acceptor_elements = pair_batch.right_element_codes[rows]
            donor_residues = pair_batch.left_residue_codes[rows]
            acceptor_residues = pair_batch.right_residue_codes[rows]
        else:
            donor_indices = pair_batch.right_indices[rows]
            acceptor_indices = pair_batch.left_indices[rows]
            donor_elements = pair_batch.right_element_codes[rows]
            acceptor_elements = pair_batch.left_element_codes[rows]
            donor_residues = pair_batch.right_residue_codes[rows]
            acceptor_residues = pair_batch.left_residue_codes[rows]
        order = np.lexsort((acceptor_indices, donor_indices, pair_batch.distances[rows]))
        rows = rows[order]
        distances = pair_batch.distances[rows]
        if forward:
            donor_elements = pair_batch.left_element_codes[rows]
            acceptor_elements = pair_batch.right_element_codes[rows]
            donor_residues = pair_batch.left_residue_codes[rows]
            acceptor_residues = pair_batch.right_residue_codes[rows]
        else:
            donor_elements = pair_batch.right_element_codes[rows]
            acceptor_elements = pair_batch.left_element_codes[rows]
            donor_residues = pair_batch.right_residue_codes[rows]
            acceptor_residues = pair_batch.left_residue_codes[rows]
    else:
        distances = np.zeros(0, dtype=float)
        donor_elements = acceptor_elements = np.zeros(0, dtype=np.int16)
        donor_residues = acceptor_residues = np.zeros(0, dtype=np.int16)
    truncated = config.max_local_pairs_per_event >= 0 and len(distances) > config.max_local_pairs_per_event
    keep = len(distances) if config.max_local_pairs_per_event < 0 else min(len(distances), config.max_local_pairs_per_event)
    rows = rows[:keep]
    distances = distances[:keep]
    donor_elements = donor_elements[:keep]
    acceptor_elements = acceptor_elements[:keep]
    donor_residues = donor_residues[:keep]
    acceptor_residues = acceptor_residues[:keep]

    rdf = np.zeros(len(ELEMENT_PAIRS) * RDF_BINS, dtype=float)
    ordered_rdf = np.zeros(len(ORDERED_ELEMENT_PAIRS) * RDF_BINS, dtype=float)
    contact_counts = Counter()
    lj_att = float(np.sum(pair_batch.lj_attraction[rows])) if rows.size else 0.0
    lj_rep = float(np.sum(pair_batch.lj_repulsion[rows])) if rows.size else 0.0
    for element_left, element_right, bin_index in zip(donor_elements, acceptor_elements, pair_batch.rdf_bins[rows]):
        bin_index = int(bin_index)
        if not 0 <= bin_index < RDF_BINS:
            continue
        left_name = ELEMENTS[int(element_left)]
        right_name = ELEMENTS[int(element_right)]
        symmetric_index = ELEMENT_PAIR_INDEX[tuple(sorted((left_name, right_name)))]
        ordered_index = ORDERED_ELEMENT_PAIR_INDEX[(left_name, right_name)]
        rdf[symmetric_index * RDF_BINS + bin_index] += 1.0
        ordered_rdf[ordered_index * RDF_BINS + bin_index] += 1.0
    for left_code, right_code in zip(donor_residues, acceptor_residues):
        left_class = RESIDUE_CLASSES[int(left_code)] if 0 <= int(left_code) < len(RESIDUE_CLASSES) else "other"
        right_class = RESIDUE_CLASSES[int(right_code)] if 0 <= int(right_code) < len(RESIDUE_CLASSES) else "other"
        contact_counts[f"{left_class}__{right_class}"] += 1
    d_normal, d_curvature = _normal(atoms, donor, acceptor, config.local_radius, local_index)
    a_normal, a_curvature = _normal(atoms, acceptor, donor, config.local_radius, local_index)
    facing = float(-np.dot(d_normal, a_normal)) if np.any(d_normal) and np.any(a_normal) else 0.0
    gap = max(0.0, _pair_distance(donor, acceptor) - 3.5)
    contact_mass = float(len(rows))
    probs = np.asarray(list(contact_counts.values()), dtype=float)
    entropy = 0.0
    if probs.size:
        probs = probs / probs.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    attrs: Dict[str, float] = {
        "rdf_mass": float(np.sum(rdf)),
        "rdf_short": float(np.sum(rdf.reshape(len(ELEMENT_PAIRS), RDF_BINS)[:, :3])),
        "rdf_mid": float(np.sum(rdf.reshape(len(ELEMENT_PAIRS), RDF_BINS)[:, 3:6])),
        "rdf_long": float(np.sum(rdf.reshape(len(ELEMENT_PAIRS), RDF_BINS)[:, 6:])),
        "contact_mass": contact_mass,
        "contact_entropy": entropy,
        "contact_raw_count": contact_mass,
        "contact_normalized_mass": 1.0 if contact_mass else 0.0,
        "lj_att": float(lj_att),
        "lj_rep": float(lj_rep),
        "lj_net": float(lj_att + lj_rep),
        "lj_proxy_att": float(lj_att),
        "lj_proxy_rep": float(lj_rep),
        "lj_proxy_net": float(lj_att + lj_rep),
        "shape_facing": facing,
        "shape_gap": gap,
        "shape_curvature": float(0.5 * (d_curvature + a_curvature)),
        "event_distance": _pair_distance(donor, acceptor),
        "local_pair_count": float(len(rows)),
        "local_pairs_truncated": float(truncated),
        # Both CUDA implementations (the fused extension and the explicit
        # PyTorch fallback) use GPU arithmetic.  Keep this field independent
        # of the implementation name so downstream audits do not report a
        # fused-kernel run as CPU work.
        "interaction_gpu_used": float(pair_batch.backend in {"cuda_extension", "torch_cuda"}),
    }
    if surface_features:
        # Keep the legacy event-local channels for backwards-compatible
        # invariance tests; expose the actual accessible-surface quantities as
        # separate fields rather than replacing a historical feature in place.
        attrs["surface_nearest_mean"] = float(surface_features.get("surface_nearest_mean", 0.0))
        attrs["surface_normal_dot_mean"] = float(surface_features.get("surface_normal_dot_mean", 0.0))
        attrs["surface_normal_antialignment"] = float(surface_features.get("surface_normal_antialignment", 0.0))
        attrs["surface_contact_fraction"] = float(surface_features.get("surface_contact_fraction", 0.0))
    
    for i, value in enumerate(rdf):
        attrs[f"rdf_{i:03d}"] = float(value)
    for i, value in enumerate(ordered_rdf):
        attrs[f"rdf_ordered_{i:03d}"] = float(value)
    for key, value in sorted(contact_counts.items()):
        attrs[f"cp_{key}"] = float(value)
        attrs[f"cp_norm_{key}"] = float(value / contact_mass) if contact_mass else 0.0
    return attrs


def _bond_close_pairs(bond_graph: Mapping[int, set], max_depth: int) -> set:
    """Ordered-independent atom index pairs within ``max_depth`` bonds.

    Used by the intramolecular adapter to keep covalently linked neighbours
    out of the microcontact set.  Depth semantics: depth-1 neighbours of an
    atom are excluded when ``max_depth`` >= 2, etc.
    """
    close: set = set()
    for src in bond_graph:
        frontier = {int(src)}
        seen = {int(src)}
        for _ in range(max_depth - 1):
            nxt = set()
            for node in frontier:
                nxt.update(int(n) for n in bond_graph.get(node, ()))
            nxt -= seen
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        for node in seen - {int(src)}:
            close.add((min(int(src), node), max(int(src), node)))
    return close


def _make_events(atoms: Sequence[Atom], chain_a: str, chain_b: str, config: GSPH1Config,
                 event_stats: Optional[Dict[str, object]] = None,
                 pair_batch: Optional[object] = None,
                 local_index: Optional[Mapping[str, Tuple[List[Atom], np.ndarray, object]]] = None,
                 bond_graph: Optional[Mapping[int, set]] = None) -> Tuple[List[Event], bool]:
    """Build typed events and one reusable fused interaction table.

    Candidate pairs are generated once for the complete interface.  Event
    typing and local aggregation then index that table instead of rebuilding a
    local dense distance matrix for every event.
    """
    left = _side_atoms(atoms, chain_a)
    right = _side_atoms(atoms, chain_b)
    # The pair table must cover the local-geometry radius: a smaller cutoff
    # would silently drop neighbors that _local_attributes queries within
    # local_radius, making local features depend on table truncation.
    pair_cutoff = max(config.event_cutoff + 2.0, 8.0, float(config.local_radius))
    if pair_batch is None:
        pair_batch = build_interaction_pair_batch(
            atoms, left, right, pair_cutoff, ELEMENTS, RESIDUE_CLASS_INDEX,
            RDF_EDGES, LJ_SIGMA, LJ_EPSILON, backend=config.interaction_backend,
            gpu_min_pairs=config.gpu_min_pairs, gpu_batch_pairs=config.gpu_batch_pairs,
            gpu_device=config.gpu_device,
        )
    surface_features: Dict[str, object] = {}
    if config.surface_backend != "none":
        try:
            from cspph.chemistry.surface_geometry import surface_complementarity
            surface_features = surface_complementarity(
                left, right, backend=config.interaction_backend,
                device_name=config.gpu_device,
            )
        except Exception as exc:
            surface_features = {
                "surface_backend": "surface_error",
                "surface_error": f"{type(exc).__name__}: {exc}",
            }
    if event_stats is not None:
        event_stats["global_pair_count"] = pair_batch.size
        event_stats["global_pair_cutoff"] = float(pair_cutoff)
        event_stats["interaction_backend"] = pair_batch.backend
        event_stats["interaction_backend_reason"] = pair_batch.backend_reason
        event_stats["interaction_gpu_available"] = pair_batch.gpu_available
        event_stats["interaction_cpu_build_seconds"] = pair_batch.cpu_build_seconds
        event_stats["interaction_gpu_compute_seconds"] = pair_batch.gpu_compute_seconds
        event_stats["interaction_transfer_seconds"] = pair_batch.transfer_seconds
        event_stats["interaction_gpu_max_distance_error"] = pair_batch.gpu_max_distance_error
        event_stats["interaction_threshold_rechecks"] = pair_batch.threshold_rechecks
        event_stats["interaction_same_side"] = bool(getattr(pair_batch, "same_side", False))
        event_stats["intra_min_bond_sep"] = int(getattr(config, "intra_min_bond_sep", 0) or 0)
        for key, value in surface_features.items():
            event_stats[key] = value
    atom_map = {atom.index: atom for atom in atoms}
    # Intramolecular adapter: atom pairs linked by fewer than
    # ``intra_min_bond_sep`` covalent bonds are real chemistry (backbone,
    # side-chain links, base/sugar links) -- not microcontacts.
    close_pairs: Optional[set] = None
    intra_sep = int(getattr(config, "intra_min_bond_sep", 0) or 0)
    if intra_sep > 0:
        if bond_graph is None:
            bond_graph = _build_bond_graph(atoms)
        close_pairs = _bond_close_pairs(bond_graph, intra_sep)
    # Donor/acceptor template flags are per-atom invariants; evaluate them
    # once instead of once per pair row.
    role_flags = {atom.index: _roles(atom) for atom in atoms}
    raw_events: List[Tuple[Atom, Atom, str, float]] = []
    seen = set()
    local_index = local_index if local_index is not None else _build_local_index(atoms)
    for row in range(pair_batch.size):
        a = atom_map[int(pair_batch.left_indices[row])]
        b = atom_map[int(pair_batch.right_indices[row])]
        if a.index == b.index:
            continue
        distance = float(pair_batch.distances[row])
        if distance > config.event_cutoff:
            continue
        clash_floor = float(getattr(config, "contact_min_vdw_ratio", 0.0) or 0.0)
        if clash_floor > 0.0 and distance < clash_floor * (
                _vdw_radius(a.element) + _vdw_radius(b.element)):
            if event_stats is not None:
                event_stats["clash_filtered_count"] = int(
                    event_stats.get("clash_filtered_count", 0)) + 1
            continue
        if close_pairs is not None and \
                (min(a.index, b.index), max(a.index, b.index)) in close_pairs:
            continue
        a_donor, a_acceptor = role_flags[a.index]
        b_donor, b_acceptor = role_flags[b.index]
        orientations = []
        if a_donor and b_acceptor:
            orientations.append((a, b, "A_to_B"))
        if b_donor and a_acceptor:
            orientations.append((b, a, "B_to_A"))
        for donor, acceptor, direction in orientations:
            key = (donor.index, acceptor.index)
            if key in seen:
                continue
            seen.add(key)
            raw_events.append((donor, acceptor, direction, distance))
    vdw = {atom.element: _vdw_radius(atom.element) for atom in atoms}
    raw_events.sort(key=lambda item: (
        max(0.0, 0.5 * (item[3] - vdw[item[0].element] - vdw[item[1].element])),
        item[0].index, item[1].index,
    ))
    if event_stats is not None:
        event_stats["raw_event_count"] = len(raw_events)
        event_stats["selected_event_count"] = min(len(raw_events), config.max_events) if config.max_events >= 0 else len(raw_events)
    truncated = config.max_events >= 0 and len(raw_events) > config.max_events
    selected = raw_events if config.max_events < 0 else raw_events[:config.max_events]
    events: List[Event] = []
    for donor, acceptor, direction, distance in selected:
        attrs = _local_attributes(atoms, donor, acceptor, config, local_index, pair_batch,
                                  surface_features, atom_map=atom_map)
        attrs["chem_donor_element"] = float(ELEMENTS.index(donor.element) if donor.element in ELEMENTS else ELEMENTS.index("X"))
        attrs["chem_acceptor_element"] = float(ELEMENTS.index(acceptor.element) if acceptor.element in ELEMENTS else ELEMENTS.index("X"))
        attrs["chem_direction_a_to_b"] = float(direction == "A_to_B")
        attrs["chem_donor_residue_class"] = float(RESIDUE_CLASS_INDEX[_residue_class(donor.resname)])
        attrs["chem_acceptor_residue_class"] = float(RESIDUE_CLASS_INDEX[_residue_class(acceptor.resname)])
        tau = max(0.0, 0.5 * (distance - _vdw_radius(donor.element) - _vdw_radius(acceptor.element)))
        events.append(Event(
            index=len(events), donor=donor.index, acceptor=acceptor.index,
            event_type=f"{donor.element}->{acceptor.element}", donor_role=direction,
            acceptor_role=_base_class(acceptor.resname), center=0.5 * (donor.xyz + acceptor.xyz),
            tau=float(tau), distance=float(distance), attributes=attrs,
        ))
    return events, truncated


def _build_bond_graph(atoms: Sequence[Atom], path: Optional[str] = None) -> Dict[int, set]:
    """Build a conservative intra-entity covalent graph from templates/distances.

    PDB CONECT records are honored when present; the distance fallback is
    restricted to one residue neighborhood and covalent radii, so an interface
    proximity alone cannot create a chemical transfer edge.
    """
    graph: Dict[int, set] = defaultdict(set)
    by_residue: Dict[Tuple[str, int, str], List[Atom]] = defaultdict(list)
    for atom in atoms:
        by_residue[(atom.chain, atom.resid, atom.icode)].append(atom)
    # Distance-based covalent inference is only defined for elements with a
    # covalent radius.  Unknown elements (metals, exotic ions) get NO
    # inferred bonds -- they may still be linked through explicit CONECT
    # records below, but proximity alone must never fabricate chemistry.
    def _cr(atom: Atom) -> Optional[float]:
        return COVALENT_RADII.get(str(atom.element))
    for key, rows in by_residue.items():
        for first, second in combinations(rows, 2):
            ra, rb = _cr(first), _cr(second)
            if ra is None or rb is None:
                continue
            if _pair_distance(first, second) <= ra + rb + 0.40:
                graph[first.index].add(second.index); graph[second.index].add(first.index)
    # Only compare neighboring residues within each chain.  The previous
    # all-pairs residue loop was mathematically equivalent but needlessly
    # quadratic for large docking candidates.
    by_chain: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
    for key in by_residue:
        by_chain[key[0]].append(key)
    for chain, keys in by_chain.items():
        keys.sort(key=lambda key: (key[1], key[2]))
        for pos, first_key in enumerate(keys):
            for second_key in keys[pos + 1:]:
                if second_key[1] - first_key[1] > 1:
                    break
                if second_key[1] - first_key[1] < 0:
                    continue
                for first in by_residue[first_key]:
                    for second in by_residue[second_key]:
                        ra, rb = _cr(first), _cr(second)
                        if ra is None or rb is None:
                            continue
                        if _pair_distance(first, second) <= ra + rb + 0.35:
                            graph[first.index].add(second.index); graph[second.index].add(first.index)
    if path and os.path.exists(path):
        serial_to_index = {atom.serial: atom.index for atom in atoms if atom.serial}
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("CONECT"):
                        numbers = [int(line[i:i + 5]) for i in range(6, len(line), 5) if line[i:i + 5].strip()]
                        if numbers:
                            source = serial_to_index.get(numbers[0])
                            for target_serial in numbers[1:]:
                                target = serial_to_index.get(target_serial)
                                if source is not None and target is not None:
                                    graph[source].add(target); graph[target].add(source)
        except OSError:
            pass
    return graph


def _bond_distance(graph: Mapping[int, set], source: int, target: int, limit: int) -> Optional[int]:
    if source == target:
        return 0
    frontier = {source}
    visited = {source}
    for distance in range(1, limit + 1):
        next_frontier = {node for node in frontier for node in graph.get(node, set()) if node not in visited}
        if target in next_frontier:
            return distance
        visited.update(next_frontier); frontier = next_frontier
        if not frontier:
            break
    return None


def _bond_neighborhood(graph: Mapping[int, set], source: int, limit: int) -> Dict[int, int]:
    """Return all atoms reachable within ``limit`` covalent steps."""
    distances = {source: 0}
    frontier = {source}
    for distance in range(1, limit + 1):
        next_frontier = {node for current in frontier for node in graph.get(current, set())
                         if node not in distances}
        for node in next_frontier:
            distances[node] = distance
        frontier = next_frontier
        if not frontier:
            break
    return distances


def _transfer_edges(events: Sequence[Event], atoms: Sequence[Atom], config: GSPH1Config,
                    path: Optional[str] = None,
                    bond_graph: Optional[Mapping[int, set]] = None) -> List[Tuple[int, int, float]]:
    by_index = {a.index: a for a in atoms}
    graph = bond_graph if bond_graph is not None else _build_bond_graph(atoms, path)
    events_by_donor: Dict[int, List[Event]] = defaultdict(list)
    for event in events:
        events_by_donor[event.donor].append(event)
    edges = []
    for first in events:
        acceptor = by_index[first.acceptor]
        for donor_index, bond_distance in _bond_neighborhood(graph, acceptor.index, config.max_bond_transfer_distance).items():
            for second in events_by_donor.get(donor_index, []):
                if first.index == second.index or bond_distance <= 0:
                    continue
                donor = by_index[second.donor]
                if acceptor.chain == donor.chain:
                    edges.append((first.index, second.index, float(bond_distance)))
    return sorted(set(edges))


def _support_certificate(point: np.ndarray, radius: float, centers: np.ndarray,
                         radii: np.ndarray, tol: float = 1e-6) -> Tuple[bool, int, float]:
    """Return a small KKT/Helly certificate for the 3-D support epigraph.

    For a positive optimum, the active sphere gradients must have a convex
    combination equal to zero.  In R^3 Caratheodory/Helly reduces this to at
    most four active constraints.  At radius zero the feasible intersection
    is already certified by the non-positive residuals and the lower bound on
    the epigraph variable.
    """
    residual = max(
        max(float(np.linalg.norm(center - point) - atom_radius - radius), 0.0)
        for center, atom_radius in zip(centers, radii)
    ) if len(centers) else 0.0
    if residual > tol:
        return False, 0, float(residual)
    if radius <= tol:
        return True, 0, float(residual)
    active = []
    for index, (center, atom_radius) in enumerate(zip(centers, radii)):
        distance = float(np.linalg.norm(point - center))
        gap = abs(distance - atom_radius - radius)
        if gap <= 5.0 * tol:
            if distance <= tol:
                # A zero-length gradient cannot participate in the stationarity
                # certificate; keep the constraint in the feasibility audit.
                continue
            active.append((index, (point - center) / distance))
    for size in range(1, min(4, len(active)) + 1):
        for subset in combinations(active, size):
            gradients = np.asarray([item[1] for item in subset], dtype=float).T
            system = np.vstack((gradients, np.ones((1, size), dtype=float)))
            rhs = np.r_[np.zeros(3, dtype=float), 1.0]
            weights, *_ = np.linalg.lstsq(system, rhs, rcond=None)
            stationarity = float(np.linalg.norm(system @ weights - rhs, ord=np.inf))
            if stationarity <= 5e-5 and float(np.min(weights)) >= -5e-5:
                return True, int(size), float(max(residual, stationarity))
    return False, int(min(4, len(active))), float(max(residual, 1.0))


def _active_set_support(centers: np.ndarray, radii: np.ndarray,
                        tol: float = 2e-6) -> Optional[Tuple[np.ndarray, float]]:
    """Solve the 3-D weighted 1-center problem by active-set enumeration.

    In three dimensions a Helly/KKT certificate needs at most four active
    balls. For each subset of size 2--4 we eliminate the linearized
    equal-distance equations, obtain ``x(t)=p+t*v`` in the active centers'
    affine hull, and solve one scalar quadratic. Every candidate is checked
    against all balls and by the same float64 KKT certificate used by the
    extractor. SLSQP remains only as a marked degeneracy fallback.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = int(len(centers))
    if n == 0:
        return np.zeros(3, dtype=float), 0.0
    if n == 1:
        return np.array(centers[0], copy=True), 0.0
    # A zero-radius feasible intersection is a valid optimum.
    trial_points = [np.mean(centers, axis=0)] + [np.array(c, copy=True) for c in centers]
    for point in trial_points:
        violation = max(float(np.linalg.norm(point - c) - r) for c, r in zip(centers, radii))
        if violation <= tol:
            return point, 0.0
    # Detect a non-empty intersection at tau=0 without invoking a generic
    # optimizer. Cyclic projections are only used for this zero-radius branch;
    # positive-radius optima are certified by the active equations below.
    point = np.mean(centers, axis=0)
    for _ in range(256):
        for center, radius in zip(centers, radii):
            delta = point - center
            distance = float(np.linalg.norm(delta))
            if distance > radius and distance > 1e-14:
                point = center + delta * (float(radius) / distance)
        violation = max(float(np.linalg.norm(point - c) - r) for c, r in zip(centers, radii))
        if violation <= tol:
            return point, 0.0
    best: Optional[Tuple[np.ndarray, float]] = None
    for size in range(2, min(4, n) + 1):
        for subset in combinations(range(n), size):
            c0 = centers[subset[0]]; r0 = float(radii[subset[0]])
            A = centers[list(subset[1:])] - c0
            dr = radii[list(subset[1:])] - r0
            q = 0.5 * (
                np.sum(centers[list(subset[1:])] ** 2, axis=1) - np.sum(c0 ** 2)
                - (radii[list(subset[1:])] ** 2 - r0 ** 2)
            )
            # Work in y=x-c0 so the least-norm solution lies in the active
            # centers' affine hull, as required by KKT stationarity.
            p_y, *_ = np.linalg.lstsq(A, q - A @ c0, rcond=None)
            v_y, *_ = np.linalg.lstsq(A, -np.asarray(dr, dtype=float), rcond=None)
            p = c0 + p_y
            v = v_y
            rel = p - c0
            qa = float(np.dot(v, v) - 1.0)
            qb = float(2.0 * np.dot(v, rel) - 2.0 * r0)
            qc = float(np.dot(rel, rel) - r0 * r0)
            roots: List[float] = []
            if abs(qa) <= 1e-12:
                if abs(qb) > 1e-12:
                    roots.append(-qc / qb)
            else:
                disc = qb * qb - 4.0 * qa * qc
                if disc >= -1e-9:
                    sqrt_disc = math.sqrt(max(0.0, disc))
                    roots.extend(((-qb - sqrt_disc) / (2.0 * qa),
                                  (-qb + sqrt_disc) / (2.0 * qa)))
            for tau in roots:
                if tau < -tol or not math.isfinite(tau):
                    continue
                tau = max(0.0, float(tau))
                point = p + tau * v
                residual = max(float(np.linalg.norm(point - c) - r - tau)
                               for c, r in zip(centers, radii))
                if residual > tol:
                    continue
                if tau > 0.0:
                    certified, _, _ = _support_certificate(point, tau, centers, radii)
                    if not certified:
                        continue
                if best is None or tau < best[1] - 1e-9:
                    best = (point, tau)
    return best


def _support_birth(path: Sequence[int], events: Mapping[int, Event], atoms: Mapping[int, Atom],
                   allow_fallback: bool = True,
                   support_stats: Optional[Dict[str, object]] = None,
                   initial_center: Optional[np.ndarray] = None,
                   initial_tau: Optional[float] = None) -> Tuple[float, float]:
    """Solve the exact common-support epigraph problem for event atoms.

    For a path with involved atoms ``v`` and van-der-Waals radii ``r_v`` this
    computes ``min_x max_v (||x-x_v||-r_v)_+``.  The normal path enumerates
    fixed-dimensional active subsets (at most four balls) and solves scalar
    quadratics; SLSQP is retained only for explicitly marked numerical
    degeneracies.
    """
    vertex_ids = sorted({events[i].donor for i in path} | {events[i].acceptor for i in path})
    centers = np.asarray([atoms[i].xyz for i in vertex_ids], dtype=float)
    radii = np.asarray([_vdw_radius(atoms[i].element) for i in vertex_ids], dtype=float)
    if not len(vertex_ids):
        return 0.0, 0.0
    # The one-event case is a weighted two-ball intersection and has a closed
    # form.  Avoid invoking SLSQP for every isolated interface event while
    # retaining the same certificate/residual accounting as the general path.
    if len(vertex_ids) == 2:
        delta = centers[1] - centers[0]
        distance = float(np.linalg.norm(delta))
        best = max(0.0, 0.5 * (distance - radii[0] - radii[1]))
        if distance > 1e-12 and best > 1e-12:
            point = centers[0] + delta * ((radii[0] + best) / distance)
            certificate_ok, certificate_size, certificate_residual = _support_certificate(
                point, best, centers, radii
            )
        else:
            certificate_ok, certificate_size, certificate_residual = True, 0, 0.0
        if support_stats is not None:
            support_stats["certificate_checks"] = int(support_stats.get("certificate_checks", 0)) + 1
            support_stats["certificate_verified"] = int(support_stats.get("certificate_verified", 0)) + int(certificate_ok)
            support_stats["certificate_failures"] = int(support_stats.get("certificate_failures", 0)) + int(not certificate_ok)
            support_stats["certificate_max_size"] = max(int(support_stats.get("certificate_max_size", 0)), certificate_size)
            support_stats["certificate_max_residual"] = max(float(support_stats.get("certificate_max_residual", 0.0)), certificate_residual)
        if not certificate_ok:
            if not allow_fallback:
                if support_stats is not None:
                    support_stats["solver_failures"] = int(support_stats.get("solver_failures", 0)) + 1
                raise RuntimeError("two-ball common-support certificate failed")
            if support_stats is not None:
                support_stats["fallbacks"] = int(support_stats.get("fallbacks", 0)) + 1
        else:
            if support_stats is not None:
                support_stats["solver_successes"] = int(support_stats.get("solver_successes", 0)) + 1
        tau_ord = max(events[i].tau for i in path) if path else 0.0
        return float(best), float(max(0.0, best - tau_ord))
    # The optional C++ active-set backend supplies a float64 candidate.  If it
    # already satisfies the exact feasibility + Helly/KKT certificate, it is
    # the answer and no generic optimizer is started.  Otherwise it remains a
    # warm start for the deterministic fallback solver.
    x0 = np.mean(centers, axis=0)
    if initial_center is not None:
        candidate_center = np.asarray(initial_center, dtype=float).reshape(-1)
        if candidate_center.size == 3 and np.all(np.isfinite(candidate_center)):
            x0 = candidate_center
    t0 = max(0.0, max(float(np.linalg.norm(center - x0) - radius) for center, radius in zip(centers, radii)))
    if initial_center is not None and initial_tau is not None:
        candidate_tau = max(0.0, float(initial_tau))
        candidate = np.asarray(initial_center, dtype=float).reshape(-1)
        if candidate.size == 3 and np.all(np.isfinite(candidate)) and np.isfinite(candidate_tau):
            certified, cert_size, cert_residual = _support_certificate(candidate, candidate_tau, centers, radii)
            if support_stats is not None:
                support_stats["certificate_checks"] = int(support_stats.get("certificate_checks", 0)) + 1
                support_stats["certificate_verified"] = int(support_stats.get("certificate_verified", 0)) + int(certified)
                support_stats["certificate_failures"] = int(support_stats.get("certificate_failures", 0)) + int(not certified)
                support_stats["certificate_max_size"] = max(int(support_stats.get("certificate_max_size", 0)), cert_size)
                support_stats["certificate_max_residual"] = max(float(support_stats.get("certificate_max_residual", 0.0)), cert_residual)
            if certified:
                if support_stats is not None:
                    support_stats["solver_successes"] = int(support_stats.get("solver_successes", 0)) + 1
                    support_stats["active_set_certified"] = int(support_stats.get("active_set_certified", 0)) + 1
                tau_ord = max(events[i].tau for i in path) if path else 0.0
                return candidate_tau, float(max(0.0, candidate_tau - tau_ord))
    # Exact fixed-dimensional active-set path.  This enumerates only the
    # Helly/KKT support subsets (at most four balls), so ordinary paths do not
    # launch a generic optimizer.
    active_candidate = _active_set_support(centers, radii)
    if active_candidate is not None:
        active_center, active_tau = active_candidate
        certified, cert_size, cert_residual = _support_certificate(
            active_center, active_tau, centers, radii
        )
        if support_stats is not None:
            support_stats["certificate_checks"] = int(support_stats.get("certificate_checks", 0)) + 1
            support_stats["certificate_verified"] = int(support_stats.get("certificate_verified", 0)) + int(certified)
            support_stats["certificate_failures"] = int(support_stats.get("certificate_failures", 0)) + int(not certified)
            support_stats["certificate_max_size"] = max(int(support_stats.get("certificate_max_size", 0)), cert_size)
            support_stats["certificate_max_residual"] = max(float(support_stats.get("certificate_max_residual", 0.0)), cert_residual)
        if certified:
            if support_stats is not None:
                support_stats["solver_successes"] = int(support_stats.get("solver_successes", 0)) + 1
                support_stats["active_set_certified"] = int(support_stats.get("active_set_certified", 0)) + 1
            tau_ord = max(events[i].tau for i in path) if path else 0.0
            return float(active_tau), float(max(0.0, active_tau - tau_ord))
    best = None
    solver_ok = False
    try:
        from scipy.optimize import minimize
        constraints = [{"type": "ineq", "fun": lambda y, c=center, r=radius: y[3] + r - np.linalg.norm(y[:3] - c)}
                       for center, radius in zip(centers, radii)]
        result = minimize(lambda y: float(y[3]), np.r_[x0, t0], method="SLSQP",
                          bounds=((None, None), (None, None), (None, None), (0.0, None)),
                          constraints=constraints, options={"maxiter": 100, "ftol": 1e-9})
        if result.success and np.isfinite(result.fun):
            candidate = max(0.0, float(result.fun))
            residual = max(
                max(float(np.linalg.norm(center - result.x[:3]) - radius - candidate), 0.0)
                for center, radius in zip(centers, radii)
            )
            if residual <= 1e-6:
                best = candidate; solver_ok = True
    except Exception:
        best = None
    certificate_ok = False
    certificate_size = 0
    certificate_residual = float("inf")
    if solver_ok:
        certificate_ok, certificate_size, certificate_residual = _support_certificate(
            result.x[:3], best, centers, radii
        )
        if support_stats is not None:
            support_stats["certificate_checks"] = int(support_stats.get("certificate_checks", 0)) + 1
            support_stats["certificate_verified"] = int(support_stats.get("certificate_verified", 0)) + int(certificate_ok)
            support_stats["certificate_failures"] = int(support_stats.get("certificate_failures", 0)) + int(not certificate_ok)
            support_stats["certificate_max_size"] = max(int(support_stats.get("certificate_max_size", 0)), certificate_size)
            support_stats["certificate_max_residual"] = max(float(support_stats.get("certificate_max_residual", 0.0)), certificate_residual)
        if not certificate_ok and not allow_fallback:
            if support_stats is not None:
                support_stats["solver_failures"] = int(support_stats.get("solver_failures", 0)) + 1
            raise RuntimeError("common-support Helly/KKT certificate failed")
        if not certificate_ok:
            # Debug extraction may continue, but only through the explicitly
            # marked deterministic fallback; never return an unverified SLSQP
            # point as if it were a certified support birth.
            solver_ok = False
            best = None
    if solver_ok:
        if support_stats is not None:
            support_stats["solver_successes"] = int(support_stats.get("solver_successes", 0)) + 1
    elif not allow_fallback:
        if support_stats is not None:
            support_stats["solver_failures"] = int(support_stats.get("solver_failures", 0)) + 1
        raise RuntimeError("common-support convex solver failed or residual exceeded tolerance")
    else:
        candidates = [x0] + [center for center in centers]
        candidates += [0.5 * (a + b) for a, b in combinations(centers, 2)]
        best = min(max(0.0, max(float(np.linalg.norm(center - candidate) - radius)
                                for center, radius in zip(centers, radii))) for candidate in candidates)
        if support_stats is not None:
            support_stats["fallbacks"] = int(support_stats.get("fallbacks", 0)) + 1
    tau_ord = max(events[i].tau for i in path) if path else 0.0
    return float(best), float(max(0.0, best - tau_ord))


def _enumerate_paths(events: Sequence[Event], edges: Sequence[Tuple[int, int, float]], atoms: Mapping[int, Atom], config: GSPH1Config,
                     support_cache: Optional[Dict[Tuple[int, ...], float]] = None,
                     support_stats: Optional[Dict[str, object]] = None) -> Tuple[Dict[int, List[Tuple[int, ...]]], Dict[Tuple[int, ...], Tuple[float, float]], bool]:
    from cspph.graph.path_stream import PathStreamStats, stream_directed_paths
    event_map = {e.index: e for e in events}
    support_cache = support_cache if support_cache is not None else {}
    support_stats = support_stats if support_stats is not None else {"hits": 0, "misses": 0}
    stream_stats = PathStreamStats()
    paths: Dict[int, List[Tuple[int, ...]]] = {order: [] for order in range(4)}
    births: Dict[Tuple[int, ...], Tuple[float, float]] = {}
    path_engine_name = "python_stream"
    if (config.path_backend in {"auto", "compact_csr"}
            and config.mode == "benchmark" and config.max_paths_per_order < 0):
        try:
            from cspph.graph.path_enumeration import enumerate_csr_paths
            compact, counts = enumerate_csr_paths([event.index for event in events], edges, max_order=3)
            streamed = [(order, tuple(int(value) for value in row))
                        for order in range(4) for row in compact.get(order, np.empty((0, order + 1), dtype=np.int32))]
            stream_stats.generated = dict(counts)
            stream_stats.retained = dict(counts)
            path_engine_name = "compact_numpy_csr"
            if os.environ.get("GSPH_ENABLE_CPP_PATH", "0") == "1":
                try:
                    from cspph.graph.path_engine_backend import load_cpp_path_engine
                    load_cpp_path_engine()
                    path_engine_name = "compact_cpp17_csr"
                except Exception:
                    pass
        except Exception as exc:
            # A compact backend failure must be visible but cannot change the
            # exact path set; use the deterministic streaming implementation.
            stream_stats.backend_error = f"{type(exc).__name__}: {exc}"  # type: ignore[attr-defined]
            streamed = list(stream_directed_paths(
                events, edges, max_order=3,
                max_paths_per_order=config.max_paths_per_order,
                stats=stream_stats,
            ))
    else:
        streamed = list(stream_directed_paths(
            events, edges, max_order=3,
            max_paths_per_order=config.max_paths_per_order,
            stats=stream_stats,
        ))
    representatives: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    path_keys: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    for order, path in streamed:
        paths[order].append(path)
        atom_key = tuple(sorted({event_map[i].donor for i in path} | {event_map[i].acceptor for i in path}))
        path_keys[path] = atom_key
        representatives.setdefault(atom_key, path)

    pending = [key for key in representatives if key not in support_cache]
    support_stats["hits"] = int(support_stats.get("hits", 0)) + len(representatives) - len(pending)
    support_stats["misses"] = int(support_stats.get("misses", 0)) + len(pending)

    # The C++ support backend supplies only certified candidates.  A rejected
    # candidate cannot alter benchmark values and falls through to SLSQP only
    # for the exceptional degenerate cases.
    support_candidates: Dict[Tuple[int, ...], Tuple[np.ndarray, float]] = {}
    if (pending and os.environ.get("GSPH_ENABLE_CPP_SUPPORT", "0") == "1"
            and len(pending) >= max(1, int(config.support_gpu_min_paths))):
        try:
            from cspph.geometry.support_active_backend import build_support_candidates
        except ImportError:  # pragma: no cover - direct script execution
            from cspph.geometry.support_active_backend import build_support_candidates
        support_candidates, gpu_stats = build_support_candidates(
            pending[:max(1, int(config.support_gpu_batch_paths))], atoms
        )
        support_stats.update(gpu_stats)
        support_stats["support_cpp_candidate_count"] = len(support_candidates)

    def solve_one(atom_key: Tuple[int, ...]) -> Tuple[Tuple[int, ...], float, Dict[str, object]]:
        local_stats: Dict[str, object] = {
            "solver_successes": 0, "solver_failures": 0, "fallbacks": 0,
            "certificate_checks": 0, "certificate_verified": 0,
            "certificate_failures": 0, "certificate_max_size": 0,
            "certificate_max_residual": 0.0,
        }
        initial_center = None
        initial_tau = None
        if atom_key in support_candidates:
            initial_center, initial_tau = support_candidates[atom_key]
            local_stats["support_cpp_candidate_used"] = 1
        tau_g, _ = _support_birth(
            representatives[atom_key], event_map, atoms,
            allow_fallback=(config.mode != "benchmark"),
            support_stats=local_stats,
            initial_center=initial_center, initial_tau=initial_tau,
        )
        return atom_key, float(tau_g), local_stats

    solved: List[Tuple[Tuple[int, ...], float, Dict[str, object]]] = []
    worker_count = max(1, int(config.support_workers))
    if len(pending) >= 32 and worker_count > 1:
        # SciPy's SLSQP releases the GIL around its numerical work.  Solving
        # independent unique atom sets concurrently keeps the exact CPU
        # certificate path intact while removing the serial optimizer loop.
        with ThreadPoolExecutor(max_workers=min(worker_count, len(pending))) as executor:
            solved = list(executor.map(solve_one, pending))
    else:
        solved = [solve_one(key) for key in pending]
    for atom_key, tau_g, local_stats in solved:
        support_cache[atom_key] = tau_g
        for key in ("solver_successes", "solver_failures", "fallbacks", "certificate_checks", "certificate_verified", "certificate_failures", "active_set_certified"):
            support_stats[key] = int(support_stats.get(key, 0)) + int(local_stats.get(key, 0))
        support_stats["certificate_max_size"] = max(int(support_stats.get("certificate_max_size", 0)), int(local_stats.get("certificate_max_size", 0)))
        support_stats["certificate_max_residual"] = max(float(support_stats.get("certificate_max_residual", 0.0)), float(local_stats.get("certificate_max_residual", 0.0)))
    for _, path in streamed:
        tau_g = float(support_cache[path_keys[path]])
        tau_ord = max((event_map[i].tau for i in path), default=0.0)
        births[path] = (tau_g, float(max(0.0, tau_g - tau_ord)))
    support_stats["path_counts_generated"] = {str(k): int(v) for k, v in stream_stats.generated.items()}
    support_stats["path_counts_retained"] = {str(k): int(v) for k, v in stream_stats.retained.items()}
    support_stats["path_stream_stopped_order"] = int(stream_stats.stopped_order)
    support_stats["path_engine"] = path_engine_name
    support_stats["path_engine_error"] = str(getattr(stream_stats, "backend_error", ""))
    return paths, births, bool(stream_stats.truncated)


def _rank(matrix: np.ndarray, tol: float) -> int:
    if matrix.size == 0:
        return 0
    return int(np.linalg.matrix_rank(matrix, tol=tol))


def _nullspace(matrix: np.ndarray, tol: float, max_dimension: int) -> np.ndarray:
    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=float)
    u, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(singular > tol * max(1.0, float(singular[0]) if singular.size else 1.0)))
    basis = vh[rank:].T
    if max_dimension >= 0 and basis.shape[1] > max_dimension:
        return basis[:, :max_dimension]
    return basis


def _boundary(paths_k: Sequence[Tuple[int, ...]], paths_prev: Sequence[Tuple[int, ...]], config: GSPH1Config) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, ...]]]:
    previous_index = {path: i for i, path in enumerate(paths_prev)}
    allowed = np.zeros((len(paths_prev), len(paths_k)), dtype=float)
    forbidden = []
    forbidden_index = {}
    raw_terms: List[List[Tuple[float, Tuple[int, ...]]]] = []
    for path in paths_k:
        terms = []
        for i in range(len(path)):
            face = path[:i] + path[i + 1:]
            # Regular path homology quotients non-regular faces (adjacent
            # repeated vertices) to zero; they do not impose an Omega
            # constraint.  Only regular but absent faces are forbidden.
            if any(face[j] == face[j + 1] for j in range(len(face) - 1)):
                continue
            if face in previous_index:
                terms.append(((-1.0) ** i, face))
            else:
                # GLMY's regular/allowed path space is strict: deleting a
                # vertex is admissible only when the resulting ordered face
                # is itself present in A_{k-1}.  This covers both adjacent
                # repeats and faces absent from the scale-specific complex.
                if face not in forbidden_index:
                    forbidden_index[face] = len(forbidden); forbidden.append(face)
                terms.append(((-1.0) ** i, face))
        raw_terms.append(terms)
    bad = np.zeros((len(forbidden), len(paths_k)), dtype=float)
    for col, terms in enumerate(raw_terms):
        for sign, face in terms:
            if face in forbidden_index:
                bad[forbidden_index[face], col] += sign
            else:
                allowed[previous_index[face], col] += sign
    return allowed, bad, forbidden


def _chain_algebra(active: Dict[int, List[Tuple[int, ...]]], config: GSPH1Config) -> Tuple[Dict[str, float], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    omega_basis: Dict[int, np.ndarray] = {}
    allowed_boundary: Dict[int, np.ndarray] = {}
    omega_dims: Dict[int, int] = {}
    boundary_ranks: Dict[int, int] = {}
    original_omega_dims: Dict[int, int] = {}
    basis_truncated: Dict[int, bool] = {}
    omega_basis[0] = np.eye(len(active.get(0, [])), dtype=float)
    omega_dims[0] = len(active.get(0, []))
    original_omega_dims[0] = omega_dims[0]
    basis_truncated[0] = False
    allowed_boundary[0] = np.zeros((0, len(active.get(0, []))), dtype=float)
    for order in (1, 2, 3):
        b_allowed, b_bad, _ = _boundary(active.get(order, []), active.get(order - 1, []), config)
        basis = _nullspace(b_bad, config.svd_tol, config.max_hodge_dimension)
        omega_basis[order] = basis
        omega_dims[order] = int(basis.shape[1])
        original_omega_dims[order] = len(active.get(order, [])) - _rank(b_bad, config.svd_tol)
        basis_truncated[order] = bool(omega_dims[order] < original_omega_dims[order])
        allowed_boundary[order] = b_allowed
    for order in range(4):
        image = allowed_boundary[order] @ omega_basis[order]
        boundary_ranks[order] = _rank(image, config.svd_tol)
    features: Dict[str, float] = {}
    for order in (1, 2, 3):
        composition = allowed_boundary[order - 1] @ allowed_boundary[order] @ omega_basis[order]
        features[f"glmy_boundary_square_residual_{order}"] = float(np.linalg.norm(composition))
    for order in range(4):
        dim_a = float(len(active.get(order, [])))
        dim_o = float(omega_dims[order])
        rank_d = float(boundary_ranks[order])
        rank_next = float(boundary_ranks[order + 1] if order < 3 else 0.0)
        kernel = max(0.0, dim_o - rank_d)
        beta = max(0.0, dim_o - rank_d - rank_next)
        features[f"glmy_dim_A_{order}"] = dim_a
        features[f"glmy_dim_Omega_{order}"] = dim_o
        features[f"glmy_dim_gap_{order}"] = max(0.0, dim_a - dim_o)
        features[f"glmy_rank_boundary_{order}"] = rank_d
        features[f"glmy_kernel_{order}"] = kernel
        features[f"glmy_admissibility_{order}"] = dim_o / max(dim_a, 1.0)
        valid_filling = not basis_truncated.get(order, False) and not basis_truncated.get(order + 1, False)
        features[f"glmy_filling_{order}"] = rank_next / max(kernel, 1e-8) if valid_filling else 0.0
        features[f"glmy_filling_valid_{order}"] = float(valid_filling)
        features[f"glmy_omega_original_dimension_{order}"] = float(original_omega_dims[order])
        features[f"glmy_omega_truncated_{order}"] = float(basis_truncated.get(order, False))
        if order <= 2:
            features[f"glmy_betti_{order}"] = beta
    features["glmy_omega_truncated"] = float(any(basis_truncated.values()))
    features["glmy_hodge_dimension_limit"] = float(config.max_hodge_dimension)
    return features, omega_basis, allowed_boundary, active


def _chain_algebra_sparse(active: Dict[int, List[Tuple[int, ...]]]):
    """Return exact sparse GLMY features and the shared chain object.

    The chain is deliberately returned to the caller.  Hodge and persistence
    must consume the same Omega bases and boundary operators; reconstructing a
    second or third ``SparseChain`` for the same filtration level was both
    wasteful and a source of inconsistent persistence objects.
    """
    from cspph.topology.glmy_sparse import sparse_chain
    chain = sparse_chain(active)
    features: Dict[str, float] = {}
    for order in range(4):
        dim_a = float(len(chain.paths.get(order, [])))
        dim_o = float(chain.omega_dims[order])
        rank_d = float(chain.boundary_ranks[order])
        rank_next = float(chain.boundary_ranks[order + 1] if order < 3 else 0)
        kernel = max(0.0, dim_o - rank_d)
        features[f"glmy_dim_A_{order}"] = dim_a
        features[f"glmy_dim_Omega_{order}"] = dim_o
        features[f"glmy_dim_gap_{order}"] = max(0.0, dim_a - dim_o)
        features[f"glmy_rank_boundary_{order}"] = rank_d
        features[f"glmy_kernel_{order}"] = kernel
        features[f"glmy_admissibility_{order}"] = dim_o / max(dim_a, 1.0)
        features[f"glmy_filling_{order}"] = rank_next / max(kernel, 1e-8)
        features[f"glmy_filling_valid_{order}"] = 1.0
        features[f"glmy_omega_original_dimension_{order}"] = dim_o
        features[f"glmy_omega_truncated_{order}"] = 0.0
        if order <= 2:
            features[f"glmy_betti_{order}"] = float(chain.betti[order])
    for order, residual in chain.boundary_square_residual.items():
        features[f"glmy_boundary_square_residual_{order}"] = float(residual)
    features["glmy_omega_truncated"] = 0.0
    features["glmy_hodge_dimension_limit"] = 0.0
    features["glmy_engine_sparse_exact"] = 1.0
    return features, chain


def _path_weight(path: Tuple[int, ...], events: Mapping[int, Event], channel: str) -> float:
    values = []
    for index in path:
        attrs = events[index].attributes
        if channel == "rdf":
            value = attrs.get("rdf_mass", 0.0)
        elif channel == "contact":
            value = attrs.get("contact_mass", 0.0)
        elif channel == "lj":
            value = abs(attrs.get("lj_att", 0.0)) + 0.1 * attrs.get("lj_rep", 0.0)
        else:
            value = max(0.0, attrs.get("shape_facing", 0.0)) + 1.0 / (1.0 + attrs.get("shape_gap", 0.0))
        values.append(math.log(1e-6 + max(0.0, float(value))))
    return float(math.exp(sum(values) / max(1, len(values))))


def _hodge_stats(active: Dict[int, List[Tuple[int, ...]]], events: Mapping[int, Event], config: GSPH1Config, channel: str,
                 chain: Optional[object] = None,
                 weights_by_order: Optional[Mapping[int, np.ndarray]] = None) -> Dict[str, float]:
    # Reconstruct the Omega bases and boundary matrices for the weighted inner
    # product.  Small dimensions are intentional: Hodge is a signature, while
    # the complete path counts remain in the GLMY algebra features.
    if chain is None:
        _, bases, boundaries, _ = _chain_algebra(active, config)
    else:
        bases = chain.omega_basis
        boundaries = chain.boundaries
    if config.hodge_backend in {"auto", "cuda"}:
        try:
            from cspph.spectral.hodge_matrix_free import hodge_spectrum_matrix_free
            if weights_by_order is None:
                weights_by_order = {
                    order: np.asarray([_path_weight(path, events, channel) for path in active.get(order, [])], dtype=float)
                    for order in range(4)
                }
            accelerated: Dict[str, float] = {"hodge_backend_cuda": 1.0}
            for order in range(3):
                result = hodge_spectrum_matrix_free(
                    bases, boundaries, weights_by_order, order,
                    k=min(config.max_hodge_dimension, 8) if config.max_hodge_dimension > 0 else 8,
                    backend=config.hodge_backend, device_name=config.gpu_device,
                )
                prefix = f"hodge_{channel}_k{order}_"
                accelerated[prefix + "spectral_gap"] = float(result.get("spectral_gap", 0.0))
                accelerated[prefix + "spectral_entropy"] = float(result.get("spectral_entropy", 0.0))
                accelerated[prefix + "effective_rank"] = float(result.get("effective_rank", 0.0))
                accelerated[prefix + "smallest_positive"] = float(result.get("spectral_gap", 0.0))
                accelerated[prefix + "residual"] = float(result.get("residual_max", float("inf")))
                accelerated[prefix + "converged"] = float(bool(result.get("converged", False)))
                heat = result.get("heat_trace", {})
                for time in HEAT_TIMES:
                    accelerated[prefix + f"heat_{time:g}"] = float(heat.get(str(time), 0.0))
            return accelerated
        except Exception:
            if config.hodge_backend == "cuda":
                raise
    gram = {}
    coords = {}
    for order in range(4):
        paths = active.get(order, [])
        weights = (np.asarray(weights_by_order.get(order, np.zeros(len(paths))), dtype=float)
                   if weights_by_order is not None else
                   np.asarray([_path_weight(path, events, channel) for path in paths], dtype=float))
        if len(weights) == 0:
            gram[order] = np.zeros((0, 0)); coords[order] = np.zeros((0, 0)); continue
        w = np.diag(np.maximum(weights, 1e-8))
        basis = bases[order]
        gram[order] = basis.T @ w @ basis + np.eye(basis.shape[1]) * 1e-10
    for order in (1, 2, 3):
        if bases[order].shape[1] == 0 or bases[order - 1].shape[1] == 0:
            coords[order] = np.zeros((bases[order - 1].shape[1], bases[order].shape[1]))
        else:
            image = boundaries[order] @ bases[order]
            # Omega bases come from an SVD nullspace and have orthonormal
            # columns.  Therefore the coordinate projection is N^T image;
            # using pinv here needlessly repeated a dense SVD for every
            # channel/scale.
            coords[order] = bases[order - 1].T @ image
    out: Dict[str, float] = {}
    for order in range(3):
        dimension = bases[order].shape[1]
        if dimension == 0:
            out[f"hodge_{channel}_k{order}_spectral_gap"] = 0.0
            out[f"hodge_{channel}_k{order}_spectral_entropy"] = 0.0
            out[f"hodge_{channel}_k{order}_effective_rank"] = 0.0
            out[f"hodge_{channel}_k{order}_smallest_positive"] = 0.0
            for time in HEAT_TIMES:
                out[f"hodge_{channel}_k{order}_heat_{time:g}"] = 0.0
            continue
        laplacian = np.zeros((dimension, dimension), dtype=float)
        if order >= 1 and coords[order].size:
            gk = gram[order]
            gprev = gram[order - 1]
            adjoint = np.linalg.solve(gk, coords[order].T @ gprev)
            laplacian += adjoint @ coords[order]
        if order < 3 and coords.get(order + 1, np.zeros((0, 0))).size:
            next_map = coords[order + 1]
            gnext = gram[order + 1]
            gcur = gram[order]
            next_adjoint = np.linalg.solve(gnext, next_map.T @ gcur)
            laplacian += next_map @ next_adjoint
        laplacian = 0.5 * (laplacian + laplacian.T)
        # Round tiny numerical negatives away before spectral summaries.  The
        # weighted construction is PSD in exact arithmetic, but SVD/pinv
        # round-off can otherwise contaminate heat traces.
        eigenvalues = np.maximum(np.linalg.eigvalsh(laplacian), 0.0)
        positive = eigenvalues[eigenvalues > 1e-8]
        if positive.size:
            probs = positive / np.sum(positive)
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            gap = float(np.min(positive)); smallest = gap
            effective = float(math.exp(entropy))
        else:
            entropy = gap = smallest = effective = 0.0
        out[f"hodge_{channel}_k{order}_spectral_gap"] = gap
        out[f"hodge_{channel}_k{order}_spectral_entropy"] = entropy
        out[f"hodge_{channel}_k{order}_effective_rank"] = effective
        out[f"hodge_{channel}_k{order}_smallest_positive"] = smallest
        for time in HEAT_TIMES:
            out[f"hodge_{channel}_k{order}_heat_{time:g}"] = float(np.sum(np.exp(-time * eigenvalues)))
    return out


def _stat_names() -> Tuple[str, ...]:
    return ("sum", "mean", "std", "min", "max", "median", "q25", "q75", "variation", "endpoint_delta")


def _build_path_lifting_cache(paths_by_order: Mapping[int, Sequence[Tuple[int, ...]]], events: Mapping[int, Event],
                              births: Mapping[Tuple[int, ...], Tuple[float, float]]) -> Dict[int, Dict[str, object]]:
    """Precompute path-local reductions once for reuse across scales."""
    event_ids = sorted(events)
    event_pos = {event_id: index for index, event_id in enumerate(event_ids)}
    # Event-level RDF keeps the complete ordered histogram.  At path level we
    # retain compact contact/chemical channels and physical geometry rather
    # than multiplying every RDF bin by every path statistic.
    typed_keys = sorted({
        key for event in events.values() for key in event.attributes
        if key.startswith("cp_")
        or key in {"lj_att", "lj_rep", "lj_net", "shape_facing", "shape_gap", "shape_curvature"}
    })
    values: Dict[str, np.ndarray] = {
        key: np.asarray([float(events[event_id].attributes.get(key, 0.0)) for event_id in event_ids], dtype=float)
        for key in typed_keys
    }
    values.update({
        "rdf_mass": np.asarray([float(events[event_id].attributes.get("rdf_mass", 0.0)) for event_id in event_ids], dtype=float),
        "contact_mass": np.asarray([float(events[event_id].attributes.get("contact_mass", 0.0)) for event_id in event_ids], dtype=float),
        "lj_abs": np.asarray([abs(float(events[event_id].attributes.get("lj_att", 0.0))) + float(events[event_id].attributes.get("lj_rep", 0.0)) for event_id in event_ids], dtype=float),
        "shape_facing": np.asarray([float(events[event_id].attributes.get("shape_facing", 0.0)) for event_id in event_ids], dtype=float),
        "shape_gap": np.asarray([float(events[event_id].attributes.get("shape_gap", 0.0)) for event_id in event_ids], dtype=float),
    })
    channels = ("rdf_mass", "contact_mass", "lj_abs", "shape_facing", "shape_gap")
    cache: Dict[int, Dict[str, object]] = {}
    for order, paths in paths_by_order.items():
        if not paths:
            continue
        path_indices = np.asarray([[event_pos[index] for index in path] for path in paths], dtype=np.int64)
        channel_rows: Dict[str, np.ndarray] = {}
        for channel in channels:
            matrix = values[channel][path_indices]
            q25, median, q75 = np.quantile(matrix, (0.25, 0.5, 0.75), axis=1)
            channel_rows[channel] = np.column_stack((
                np.sum(matrix, axis=1), np.mean(matrix, axis=1), np.std(matrix, axis=1),
                np.min(matrix, axis=1), np.max(matrix, axis=1), median, q25, q75,
                np.sum(np.abs(np.diff(matrix, axis=1)), axis=1) if matrix.shape[1] > 1 else np.zeros(len(matrix)),
                matrix[:, -1] - matrix[:, 0],
            ))
        typed_rows: Dict[str, np.ndarray] = {}
        for key in typed_keys:
            matrix = values[key][path_indices]
            typed_rows[key] = np.column_stack((
                np.mean(matrix, axis=1), np.std(matrix, axis=1),
                np.sum(np.abs(np.diff(matrix, axis=1)), axis=1) if matrix.shape[1] > 1 else np.zeros(len(matrix)),
                matrix[:, -1] - matrix[:, 0],
            ))
        cache[order] = {
            "path_index": {path: index for index, path in enumerate(paths)},
            "channels": channel_rows,
            "typed": typed_rows,
            "supports": np.asarray([births[path][0] for path in paths], dtype=float),
            "delays": np.asarray([births[path][1] for path in paths], dtype=float),
        }
    return cache


def _path_lifting(active: Dict[int, List[Tuple[int, ...]]], events: Mapping[int, Event], births: Mapping[Tuple[int, ...], Tuple[float, float]], prefix: str,
                  attribute_cache: Optional[Mapping[str, np.ndarray]] = None,
                  path_cache: Optional[Mapping[int, Mapping[str, object]]] = None) -> Dict[str, float]:
    """Lift event attributes to ordered paths using vectorized reductions.

    The previous implementation called ``std``/``quantile`` once per path and
    once per attribute.  That made complete 4RWP/1ASY runs spend most of their
    time in millions of tiny NumPy calls.  Paths are fixed-length integer
    arrays, so gathering them into a dense ``(n_paths, order+1)`` view is both
    exact and much cheaper; it does not change the path order or topology.
    """
    out: Dict[str, float] = {}
    channels = ("rdf_mass", "contact_mass", "lj_abs", "shape_facing", "shape_gap")
    event_ids = sorted(events)
    event_pos = {event_id: index for index, event_id in enumerate(event_ids)}
    if attribute_cache is None:
        typed_keys = sorted({
            key for event in events.values() for key in event.attributes
            if key.startswith("cp_")
            or key in {"lj_att", "lj_rep", "lj_net", "shape_facing", "shape_gap", "shape_curvature"}
        })
        attribute_cache = {
            key: np.asarray([float(events[event_id].attributes.get(key, 0.0)) for event_id in event_ids], dtype=float)
            for key in typed_keys
        }
        attribute_cache = dict(attribute_cache)
        attribute_cache["rdf_mass"] = np.asarray([float(events[event_id].attributes.get("rdf_mass", 0.0)) for event_id in event_ids], dtype=float)
        attribute_cache["contact_mass"] = np.asarray([float(events[event_id].attributes.get("contact_mass", 0.0)) for event_id in event_ids], dtype=float)
        attribute_cache["lj_abs"] = np.asarray([abs(float(events[event_id].attributes.get("lj_att", 0.0))) + float(events[event_id].attributes.get("lj_rep", 0.0)) for event_id in event_ids], dtype=float)
        attribute_cache["shape_facing"] = np.asarray([float(events[event_id].attributes.get("shape_facing", 0.0)) for event_id in event_ids], dtype=float)
        attribute_cache["shape_gap"] = np.asarray([float(events[event_id].attributes.get("shape_gap", 0.0)) for event_id in event_ids], dtype=float)
    typed_keys = sorted(key for key in attribute_cache if key.startswith("cp_")
                        or key in {"lj_att", "lj_rep", "lj_net", "shape_facing", "shape_gap", "shape_curvature"})
    for order, paths in active.items():
        if order == 0 or not paths:
            continue
        out[f"{prefix}_path_count_k{order}"] = float(len(paths))
        cached = path_cache.get(order) if path_cache is not None else None
        cached_indices = np.asarray([cached["path_index"][path] for path in paths], dtype=np.int64) if cached else None
        path_indices = np.asarray([[event_pos[index] for index in path] for path in paths], dtype=np.int64) if cached is None else None
        for channel in channels:
            rows = (np.asarray(cached["channels"][channel])[cached_indices]
                    if cached is not None else np.asarray(attribute_cache[channel], dtype=float)[path_indices])
            for index, name in enumerate(_stat_names()):
                out[f"{prefix}_path_{channel}_{name}_k{order}"] = float(np.mean(rows[:, index]))
        for key in typed_keys:
            typed = (np.asarray(cached["typed"][key])[cached_indices]
                     if cached is not None else None)
            if typed is None:
                matrix = np.asarray(attribute_cache[key], dtype=float)[path_indices]
                typed = np.column_stack((
                    np.mean(matrix, axis=1), np.std(matrix, axis=1),
                    np.sum(np.abs(np.diff(matrix, axis=1)), axis=1) if matrix.shape[1] > 1 else np.zeros(len(matrix)),
                    matrix[:, -1] - matrix[:, 0],
                ))
            safe_key = key.replace("__", "_x_")
            for index, metric in enumerate(("mean", "std", "variation", "endpoint_delta")):
                arr = typed[:, index]
                stats = (float(np.sum(arr)), float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)),
                         float(np.max(arr)), float(np.median(arr)), float(np.quantile(arr, 0.25)),
                         float(np.quantile(arr, 0.75)))
                for stat_name, stat_value in zip(("sum", "mean", "std", "min", "max", "median", "q25", "q75"), stats):
                    out[f"{prefix}_path_attr_{safe_key}_{metric}_{stat_name}_k{order}"] = stat_value
        supports = (np.asarray(cached["supports"])[cached_indices]
                    if cached is not None else np.asarray([births[path][0] for path in paths], dtype=float))
        delays = (np.asarray(cached["delays"])[cached_indices]
                  if cached is not None else np.asarray([births[path][1] for path in paths], dtype=float))
        out[f"{prefix}_path_support_mean_k{order}"] = float(np.mean(supports))
        out[f"{prefix}_path_support_max_k{order}"] = float(np.max(supports))
        out[f"{prefix}_path_delay_mean_k{order}"] = float(np.mean(delays))
        out[f"{prefix}_path_delay_max_k{order}"] = float(np.max(delays))
    return out


def _flatten_event_summary(events: Sequence[Event]) -> Dict[str, float]:
    out = {"event_count": float(len(events))}
    for key in ("rdf_mass", "rdf_short", "rdf_mid", "rdf_long", "contact_mass", "contact_raw_count", "contact_normalized_mass", "contact_entropy", "lj_att", "lj_rep", "lj_net", "lj_proxy_att", "lj_proxy_rep", "lj_proxy_net", "shape_facing", "shape_gap", "shape_curvature", "surface_nearest_mean", "surface_normal_dot_mean", "surface_normal_antialignment", "surface_contact_fraction", "event_distance", "local_pair_count"):
        values = np.asarray([e.attributes.get(key, 0.0) for e in events], dtype=float)
        out[f"event_{key}_mean"] = float(np.mean(values)) if values.size else 0.0
        out[f"event_{key}_std"] = float(np.std(values)) if values.size else 0.0
        out[f"event_{key}_max"] = float(np.max(values)) if values.size else 0.0
    # Preserve ordered atom-type RDF and normalized contact channels in the
    # flat feature row.  The legacy symmetric summaries above remain for
    # backwards compatibility, while these channels distinguish A->B from
    # B->A chemistry.
    ordered_keys = sorted({key for event in events for key in event.attributes if key.startswith("rdf_ordered_")})
    ordered_keys += sorted({key for event in events for key in event.attributes if key.startswith("cp_norm_")})
    for key in ordered_keys:
        values = np.asarray([event.attributes.get(key, 0.0) for event in events], dtype=float)
        out[f"event_{key}_mean"] = float(np.mean(values)) if values.size else 0.0
        out[f"event_{key}_std"] = float(np.std(values)) if values.size else 0.0
        out[f"event_{key}_max"] = float(np.max(values)) if values.size else 0.0
    for event_type, count in sorted(Counter(e.event_type for e in events).items()):
        out[f"event_type_{event_type.replace('>', '_to_')}"] = float(count)
    return out


def _extract_from_atoms(atoms: Sequence[Atom], scenario: str, chain_a: Optional[str],
                        chain_b: Optional[str], config: GSPH1Config,
                        source_path: Optional[str] = None,
                        static_bond_graph: Optional[Mapping[int, set]] = None,
                        interaction_pair_batch: Optional[object] = None) -> Dict[str, object]:
    """Extract one bounded GSPH1 record from already parsed heavy atoms."""
    config = config or GSPH1Config()
    config, benchmark_limits_resolved = _resolve_config(config)
    if config.mode not in {"debug", "benchmark"}:
        raise ValueError("GSPH1Config.mode must be 'debug' or 'benchmark'")
    if not atoms:
        raise ValueError("no heavy atoms parsed")
    selected_a, selected_b = _chain_pair(atoms, scenario, chain_a, chain_b)
    event_stats: Dict[str, object] = {}
    events, events_truncated = _make_events(
        atoms, selected_a, selected_b, config, event_stats,
        pair_batch=interaction_pair_batch,
    )
    local_pairs_truncated = any(
        float(event.attributes.get("local_pairs_truncated", 0.0)) > 0.0
        for event in events
    )
    edges = _transfer_edges(events, atoms, config, source_path, static_bond_graph)
    atom_map = {a.index: a for a in atoms}
    support_cache: Dict[Tuple[int, ...], float] = {}
    support_stats: Dict[str, object] = {
        "hits": 0, "misses": 0, "solver_successes": 0, "solver_failures": 0,
        "fallbacks": 0, "certificate_checks": 0, "certificate_verified": 0,
        "certificate_failures": 0, "certificate_max_size": 0,
        "certificate_max_residual": 0.0,
    }
    paths, births, paths_truncated = _enumerate_paths(events, edges, atom_map, config, support_cache, support_stats)
    event_map = {e.index: e for e in events}
    path_lifting_cache = _build_path_lifting_cache(paths, event_map, births)
    features: Dict[str, float] = {
        "schema_version": 1.0,
        "atom_count": float(len(atoms)),
        "chain_a_atom_count": float(sum(a.chain in set(_chain_ids(selected_a)) for a in atoms)),
        "chain_b_atom_count": float(sum(a.chain in set(_chain_ids(selected_b)) for a in atoms)),
        "directed_edge_count": float(len(edges)),
        "events_truncated": float(events_truncated),
        "paths_truncated": float(paths_truncated),
    }
    features.update(_flatten_event_summary(events))
    hodge_unavailable = False
    active_by_scale: List[Dict[int, List[Tuple[int, ...]]]] = []
    chains = []
    for scale in config.scales:
        active = {order: [p for p in paths.get(order, []) if births[p][0] <= scale + 1e-9] for order in range(4)}
        active_by_scale.append(active)
        prefix = f"scale_{scale:g}"
        if config.mode == "benchmark":
            try:
                algebra, hodge_chain = _chain_algebra_sparse(active)
            except Exception:
                algebra, hodge_chain = {}, None
        else:
            algebra, _, _, _ = _chain_algebra(active, config)
            try:
                from cspph.topology.glmy_sparse import sparse_chain
                hodge_chain = sparse_chain(active)
            except Exception:
                hodge_chain = None
        chains.append(hodge_chain)
        for key, value in algebra.items():
            features[f"{prefix}_{key}"] = float(value)
        features.update(_path_lifting(active, event_map, births, prefix, path_cache=path_lifting_cache))
        if config.mode == "benchmark":
            # Hodge is an approximate spectral descriptor, but it is now
            # evaluated from the same sparse GLMY chain and induced Gram
            # products.  Exact Betti/rank values remain the sparse CPU core.
            if hodge_chain is None:
                hodge_unavailable = True
                features[f"{prefix}_hodge_unavailable"] = 1.0
            else:
                try:
                    hodge_weight_cache = {
                        channel: {
                            order: np.asarray(
                                [_path_weight(path, event_map, channel) for path in active.get(order, [])],
                                dtype=float,
                            )
                            for order in range(4)
                        }
                        for channel in ("rdf", "contact", "lj", "shape")
                    }
                    for channel in ("rdf", "contact", "lj", "shape"):
                        features.update({
                            f"{prefix}_{key}": value
                            for key, value in _hodge_stats(
                                active, event_map, config, channel, hodge_chain,
                                hodge_weight_cache[channel],
                            ).items()
                        })
                except Exception as exc:
                    hodge_unavailable = True
                    features[f"{prefix}_hodge_unavailable"] = 1.0
        else:
            for channel in ("rdf", "contact", "lj", "shape"):
                features.update({f"{prefix}_{key}": value for key, value in _hodge_stats(active, event_map, config, channel).items()})
        for order in range(4):
            features[f"{prefix}_path_birth_count_k{order}"] = float(len(active.get(order, [])))
    persistence_meta: Dict[str, object] = {"persistence_valid": False, "persistence_error": ""}
    try:
        from cspph.topology.persistence import (build_omega_boundary_filtration,
                                  build_omega_inclusion_chain_maps,
                                  persistent_intervals_from_filtration,
                                  validate_chain_maps)
        if any(chain is None for chain in chains):
            raise ValueError("sparse chain construction unavailable for persistence")
        boundary_filtration = build_omega_boundary_filtration(chains)
        inclusion_maps = build_omega_inclusion_chain_maps(chains, active_by_scale)
        chain_check = validate_chain_maps(boundary_filtration, inclusion_maps)
        if not chain_check.get("valid", False):
            raise ValueError(str(chain_check.get("error", "chain-map validation failed")))
        interval_counts: Dict[str, int] = {}
        right_censored: Dict[str, int] = {}
        for order in (0, 1, 2):
            intervals = persistent_intervals_from_filtration(boundary_filtration, inclusion_maps, order)
            interval_counts[str(order)] = len(intervals)
            right_censored[str(order)] = sum(int(item.get("right_censored", False)) for item in intervals)
            features[f"persistence_interval_count_k{order}"] = float(len(intervals))
            features[f"persistence_right_censored_k{order}"] = float(right_censored[str(order)])
        persistence_meta = {
            "persistence_valid": True,
            "persistence_max_chain_residual": float(chain_check.get("max_residual", 0.0)),
            "persistence_interval_counts": interval_counts,
            "persistence_right_censored": right_censored,
        }
    except Exception as exc:
        persistence_meta = {"persistence_valid": False, "persistence_error": f"{type(exc).__name__}: {exc}"}
    omega_truncated = any(key.endswith("glmy_omega_truncated") and float(value) > 0.0
                          for key, value in features.items())
    # Rejected warm-start candidates are expected: the exact active-set solver
    # then supplies the certified answer. Only an unverified final fallback or
    # solver failure makes a benchmark record incomplete.
    support_unverified = bool(
        int(support_stats.get("solver_failures", 0)) > 0
        or int(support_stats.get("fallbacks", 0)) > 0
    )
    # Hodge spectra are an optional P3 diagnostic and are deliberately kept
    # separate from the exact sparse GLMY algebra.  Their absence must not
    # erase valid Betti/boundary features from a complete P0 record.
    incomplete = bool(events_truncated or paths_truncated or omega_truncated or local_pairs_truncated or support_unverified)
    if config.mode == "benchmark" and incomplete:
        # A benchmark record may retain auditable event/count metadata, but it
        # must not expose bounded Betti/Hodge values as if they were complete.
        features = {key: value for key, value in features.items() if not key.startswith("scale_")}
        features["incomplete"] = 1.0
        features["topology_valid"] = 0.0
    metadata = {
        "schema_version": "gsph1.v1",
        "path": os.path.abspath(source_path) if source_path else "<in_memory>",
        "scenario": scenario,
        "chain_a": selected_a,
        "chain_b": selected_b,
        "entity_type_a": _entity_type(_side_atoms(atoms, selected_a)),
        "entity_type_b": _entity_type(_side_atoms(atoms, selected_b)),
        "config": asdict(config),
        "benchmark_limits_resolved": bool(benchmark_limits_resolved),
        "event_count": len(events),
        "raw_event_count": int(event_stats.get("raw_event_count", len(events))),
        "selected_event_count": int(event_stats.get("selected_event_count", len(events))),
        "global_pair_count": int(event_stats.get("global_pair_count", 0)),
        "global_pair_cutoff": float(event_stats.get("global_pair_cutoff", 0.0)),
        "interaction_backend": str(event_stats.get("interaction_backend", "unknown")),
        "interaction_backend_reason": str(event_stats.get("interaction_backend_reason", "unknown")),
        "interaction_gpu_available": bool(event_stats.get("interaction_gpu_available", False)),
        "interaction_cpu_build_seconds": float(event_stats.get("interaction_cpu_build_seconds", 0.0)),
        "interaction_gpu_compute_seconds": float(event_stats.get("interaction_gpu_compute_seconds", 0.0)),
        "interaction_transfer_seconds": float(event_stats.get("interaction_transfer_seconds", 0.0)),
        "interaction_gpu_max_distance_error": float(event_stats.get("interaction_gpu_max_distance_error", 0.0)),
        "interaction_threshold_rechecks": int(event_stats.get("interaction_threshold_rechecks", 0)),
        "directed_edge_count": len(edges),
        "path_counts": {str(k): len(v) for k, v in paths.items()},
        "path_counts_generated": support_stats.get("path_counts_generated", {}),
        "path_counts_retained": support_stats.get("path_counts_retained", {}),
        "path_engine": str(support_stats.get("path_engine", "python_stream")),
        "path_engine_error": str(support_stats.get("path_engine_error", "")),
        "path_stream_stopped_order": int(support_stats.get("path_stream_stopped_order", -1)),
        "events_truncated": events_truncated,
        "paths_truncated": paths_truncated,
        "local_pairs_truncated": local_pairs_truncated,
        "mode": config.mode,
        "status": "incomplete" if incomplete else ("empty_events" if not events else "complete"),
        "topology_valid": not incomplete,
        "annotation_backend": "geometry_only; FR3D/DSSR not required by gsph1 core",
        "lj_backend": "fixed_element_class_constants; not Amber/OpenMM energy",
        "shape_backend": str(event_stats.get("surface_backend", "local_neighbor_covariance_proxy")),
        "shape_surface_point_count_a": int(event_stats.get("surface_point_count_a", 0)),
        "shape_surface_point_count_b": int(event_stats.get("surface_point_count_b", 0)),
        "shape_surface_seconds": float(event_stats.get("surface_seconds", 0.0)),
        "shape_surface_error": str(event_stats.get("surface_error", "")),
        "support_birth_method": "cpp_candidate_or_exact_float64_active_set_with_helly_kkt_certificate_and_slsqp_degeneracy_fallback",
        "support_cache_hits": int(support_stats.get("hits", 0)),
        "support_cache_misses": int(support_stats.get("misses", 0)),
        "support_solver_successes": int(support_stats.get("solver_successes", 0)),
        "support_solver_failures": int(support_stats.get("solver_failures", 0)),
        "support_fallbacks": int(support_stats.get("fallbacks", 0)),
        "support_certificate_checks": int(support_stats.get("certificate_checks", 0)),
        "support_certificate_verified": int(support_stats.get("certificate_verified", 0)),
        "support_certificate_failures": int(support_stats.get("certificate_failures", 0)),
        "support_certificate_max_size": int(support_stats.get("certificate_max_size", 0)),
        "support_certificate_max_residual": float(support_stats.get("certificate_max_residual", 0.0)),
        "support_gpu_attempted": bool(support_stats.get("support_gpu_attempted", False)),
        "support_gpu_available": bool(support_stats.get("support_gpu_available", False)),
        "support_gpu_candidate_count": int(support_stats.get("support_gpu_candidate_count", 0)),
        "support_gpu_batch": int(support_stats.get("support_gpu_batch", 0)),
        "support_gpu_seconds": float(support_stats.get("support_gpu_seconds", 0.0)),
        "support_gpu_error": str(support_stats.get("support_gpu_error", "")),
        "support_cpp_attempted": bool(support_stats.get("support_cpp_attempted", False)),
        "support_cpp_available": bool(support_stats.get("support_cpp_available", False)),
        "support_cpp_candidate_count": int(support_stats.get("support_cpp_candidate_count", 0)),
        "support_cpp_seconds": float(support_stats.get("support_cpp_seconds", 0.0)),
        "support_cpp_error": str(support_stats.get("support_cpp_error", "")),
        "support_active_set_certified": int(support_stats.get("active_set_certified", 0)),
        "support_active_set_solver": "enumerated_affine_quadratics_max4_active_balls",
        "support_unverified": support_unverified,
        "omega_truncated": omega_truncated,
        "hodge_dimension_limit": config.max_hodge_dimension,
        "glmy_engine": "sparse_structured_kernel_modular_rank_verified" if config.mode == "benchmark" else "dense_reference_debug",
        "glmy_coefficient_field": "Fp_multi_prime_rank_consensus" if config.mode == "benchmark" else "R_float64_reference",
        "glmy_rank_primes": [1000003, 1000033, 1000037] if config.mode == "benchmark" else [],
        "glmy_rank_status": "probabilistic_consensus" if config.mode == "benchmark" else "dense_reference",
        "glmy_rank_backend": "python_modular",
        "hodge_status": ("unavailable" if hodge_unavailable else
                          ("cuda_matrix_free" if config.hodge_backend in {"auto", "cuda"} else "cpu_matrix_free")),
        "hodge_operator_mode": "gpu_matrix_free_with_exact_small_audit" if config.hodge_backend in {"auto", "cuda"} else "cpu_matrix_free",
        "hodge_incomplete": bool(hodge_unavailable),
        **persistence_meta,
    }
    try:
        from cspph.topology.glmy_sparse import rank_backend_status
        metadata["glmy_rank_backend"] = rank_backend_status()
    except Exception:
        pass
    event_records = []
    by_atom = {a.index: a for a in atoms}
    for event in events:
        donor = by_atom[event.donor]; acceptor = by_atom[event.acceptor]
        event_records.append({
            "index": event.index, "donor": f"{donor.chain}:{donor.resname}{donor.resid}:{donor.name}",
            "acceptor": f"{acceptor.chain}:{acceptor.resname}{acceptor.resid}:{acceptor.name}",
            "event_type": event.event_type, "distance": event.distance, "tau": event.tau,
        })
    return {"metadata": metadata, "features": features, "events": event_records}


def extract_gsph1_atoms(atoms: Sequence[Atom], scenario: str = "unknown",
                        chain_a: Optional[str] = None, chain_b: Optional[str] = None,
                        config: Optional[GSPH1Config] = None,
                        static_bond_graph: Optional[Mapping[int, set]] = None,
                        interaction_pair_batch: Optional[object] = None) -> Dict[str, object]:
    """Public in-memory extractor used by streamed decoy workers."""
    config = config or GSPH1Config()
    return _extract_from_atoms(list(atoms), scenario, chain_a, chain_b, config, None,
                               static_bond_graph, interaction_pair_batch)


def extract_gsph1(path: str, scenario: str = "unknown", chain_a: Optional[str] = None,
                  chain_b: Optional[str] = None, config: Optional[GSPH1Config] = None) -> Dict[str, object]:
    """Extract one bounded GSPH1 feature record from a PDB/mmCIF candidate."""
    config = config or GSPH1Config()
    atoms = _parse_atoms(path, exclude_water=config.exclude_water)
    return _extract_from_atoms(atoms, scenario, chain_a, chain_b, config, path)


def flatten_features(record: Mapping[str, object]) -> Dict[str, object]:
    """Return a flat row suitable for Parquet/CSV while preserving metadata."""
    metadata = record.get("metadata", {})
    row = dict(record.get("features", {}))
    if isinstance(metadata, Mapping):
        for key in ("schema_version", "scenario", "chain_a", "chain_b", "entity_type_a", "entity_type_b", "path", "event_count", "directed_edge_count", "global_pair_count", "interaction_backend", "interaction_backend_reason", "interaction_gpu_available", "interaction_cpu_build_seconds", "interaction_gpu_compute_seconds", "interaction_transfer_seconds", "interaction_gpu_max_distance_error", "interaction_threshold_rechecks", "benchmark_limits_resolved", "events_truncated", "paths_truncated", "local_pairs_truncated", "status"):
            value = metadata.get(key)
            if isinstance(value, (str, int, float, bool)):
                row[f"metadata_{key}"] = value
    return row


def _main() -> None:
    parser = argparse.ArgumentParser(description="Extract one GSPH1 feature record")
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--scenario", default="unknown")
    parser.add_argument("--chain-a", default=None)
    parser.add_argument("--chain-b", default=None)
    parser.add_argument("--mode", choices=("debug", "benchmark"), default="debug")
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--max-paths-per-order", type=int, default=64)
    parser.add_argument("--max-local-pairs-per-event", type=int, default=3000)
    parser.add_argument("--interaction-backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-min-pairs", type=int, default=250000)
    parser.add_argument("--gpu-batch-pairs", type=int, default=500000)
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--support-workers", type=int, default=4)
    parser.add_argument("--path-backend", choices=("auto", "compact_csr", "python_stream"), default="auto")
    parser.add_argument("--surface-backend", choices=("auto", "none"), default="auto")
    parser.add_argument("--hodge-backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default=None, help="JSON output path; stdout if omitted")
    args = parser.parse_args()
    config = GSPH1Config(
        mode=args.mode, max_events=args.max_events,
        max_paths_per_order=args.max_paths_per_order,
        max_local_pairs_per_event=args.max_local_pairs_per_event,
        interaction_backend=args.interaction_backend,
        gpu_min_pairs=args.gpu_min_pairs, gpu_batch_pairs=args.gpu_batch_pairs,
        gpu_device=args.gpu_device, support_workers=args.support_workers,
        path_backend=args.path_backend, surface_backend=args.surface_backend,
        hodge_backend=args.hodge_backend,
    )
    record = extract_gsph1(args.pdb, args.scenario, args.chain_a, args.chain_b, config)
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":  # pragma: no cover
    _main()
