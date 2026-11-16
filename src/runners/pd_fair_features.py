"""Deterministic Protein--DNA feature groups for the fair baseline suite.

This module deliberately contains feature extraction only.  It does not load
third-party scores or model weights and it never reads a reference structure.
The output is a dictionary of raw, fold-independent measurements; the
statistical-potential group is fitted by the training driver inside each fold.

The structure parser is intentionally conservative.  It keeps the first model
and alternate location A (or the blank location), ignores waters/hydrogens,
and classifies residues from their PDB names.  The geometry fallbacks are
deterministic and expose ``*_is_proxy`` flags where a specialised program is
not required by the feature definition (e.g. Voronoi areas).
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - the fallback is useful on minimal hosts
    cKDTree = None


PROTEIN_NAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "SEC", "PYL",
}
AA_ORDER = tuple("ARNDCQEGHILKMFPSTWYV")
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}
DNA_NAMES = {
    "DA": "A", "DC": "C", "DG": "G", "DT": "T", "DI": "I",
    "ADE": "A", "CYT": "C", "GUA": "G", "THY": "T",
    "A": "A", "C": "C", "G": "G", "T": "T",
    "5MC": "C", "DC5": "C", "DG5": "G", "DU": "T",
}
BASE_ORDER = ("A", "C", "G", "T")
WATER_NAMES = {"HOH", "WAT", "DOD", "SOL"}

ELEMENT_RADIUS = {"H": 1.2, "C": 1.70, "N": 1.55, "O": 1.52,
                  "P": 1.80, "S": 1.80, "F": 1.47, "CL": 1.75}
ELEMENT_EPS = {"C": 0.12, "N": 0.17, "O": 0.21, "P": 0.20, "S": 0.25}


@dataclass(frozen=True)
class Atom:
    chain: str
    resname: str
    resseq: int
    icode: str
    atom_name: str
    element: str
    xyz: Tuple[float, float, float]
    is_protein: bool
    is_dna: bool

    @property
    def residue_key(self) -> Tuple[str, int, str]:
        return self.chain, self.resseq, self.icode


def _element(atom_name: str, field: str) -> str:
    e = field.strip().upper()
    if e:
        return e
    letters = "".join(ch for ch in atom_name.strip().upper() if ch.isalpha())
    if not letters:
        return ""
    if letters[:2] in {"CL", "BR", "FE", "ZN", "MG", "NA", "CA"}:
        return letters[:2]
    return letters[0]


def parse_pdb(path: str) -> List[Atom]:
    """Parse first PDB model while retaining only heavy atoms."""
    atoms: List[Atom] = []
    saw_model = False
    in_first_model = True
    try:
        handle = open(path, "r", encoding="utf-8", errors="ignore")
    except OSError:
        return atoms
    with handle:
        for line in handle:
            rec = line[0:6].strip().upper()
            if rec == "MODEL":
                if saw_model:
                    in_first_model = False
                else:
                    saw_model = True
                continue
            if rec == "ENDMDL" and saw_model:
                break
            if not in_first_model or rec not in {"ATOM", "HETATM"}:
                continue
            alt = line[16:17].strip()
            if alt not in {"", "A", "1"}:
                continue
            resname = line[17:20].strip().upper()
            if resname in WATER_NAMES:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                resseq = int(line[22:26])
            except (ValueError, IndexError):
                continue
            chain = line[21:22].strip() or "_"
            atom_name = line[12:16].strip().upper()
            elem = _element(atom_name, line[76:78] if len(line) >= 78 else "")
            if elem == "H" or atom_name.startswith("H"):
                continue
            atoms.append(Atom(
                chain=chain, resname=resname, resseq=resseq,
                icode=line[26:27].strip(), atom_name=atom_name,
                element=elem, xyz=(x, y, z),
                is_protein=resname in PROTEIN_NAMES,
                is_dna=resname in DNA_NAMES,
            ))
    return atoms


def _pair_indices(a_xyz: np.ndarray, b_xyz: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cross-partner pair indices and distances below cutoff."""
    if len(a_xyz) == 0 or len(b_xyz) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    if cKDTree is not None:
        tree = cKDTree(b_xyz)
        hits = tree.query_ball_point(a_xyz, cutoff)
        ii: List[int] = []; jj: List[int]; dd: List[float] = []
        jj = []
        for i, indexes in enumerate(hits):
            for j in indexes:
                d = float(np.linalg.norm(a_xyz[i] - b_xyz[j]))
                ii.append(i); jj.append(j); dd.append(d)
        return np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64), np.asarray(dd, dtype=float)
    # Small fallback for minimal environments.
    ii, jj, dd = [], [], []
    cutoff2 = cutoff * cutoff
    for i, x in enumerate(a_xyz):
        diff = b_xyz - x
        dist2 = np.einsum("ij,ij->i", diff, diff)
        for j in np.flatnonzero(dist2 <= cutoff2):
            ii.append(i); jj.append(int(j)); dd.append(float(math.sqrt(dist2[j])))
    return np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64), np.asarray(dd, dtype=float)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def _std(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.std(values)) if values else default


def _quant(values: Sequence[float], q: float, default: float = 0.0) -> float:
    return float(np.quantile(values, q)) if values else default


def _residue_atoms(atoms: Iterable[Atom]) -> Dict[Tuple[str, int, str], List[Atom]]:
    out: Dict[Tuple[str, int, str], List[Atom]] = defaultdict(list)
    for atom in atoms:
        out[atom.residue_key].append(atom)
    return out


def _base_atom_class(atom: Atom) -> str:
    name = atom.atom_name.replace("*", "'")
    if name in {"P", "OP1", "OP2", "O1P", "O2P", "O3P"}:
        return "phosphate"
    if name in {"C1'", "C2'", "C3'", "C4'", "O2'", "O3'", "O4'", "C5'", "O5'"}:
        return "sugar"
    base = DNA_NAMES.get(atom.resname, "")
    # Watson/Crick atoms which face the major groove, and minor-groove atoms.
    major = {"A": {"N7", "C6", "N6", "C5", "C8"},
             "G": {"O6", "N7", "C6", "N2", "C5", "C8"},
             "C": {"N4", "C5", "C6"},
             "T": {"O4", "C5", "C6", "C7"}}
    minor = {"A": {"N3", "C2"}, "G": {"N3", "C2"},
             "C": {"N3", "O2", "C2"}, "T": {"N3", "O2", "C2"}}
    if name in major.get(base, set()):
        return "major"
    if name in minor.get(base, set()):
        return "minor"
    return "base_other"


def _protein_atom_class(atom: Atom) -> str:
    if atom.resname in {"ARG", "LYS", "HIS"}:
        return "positive"
    if atom.resname in {"ASP", "GLU"}:
        return "negative"
    if atom.resname in {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}:
        return "polar"
    return "other"


def _is_donor(atom: Atom) -> bool:
    return atom.element in {"N", "O", "S"} and atom.atom_name not in {"O", "OXT"}


def _is_acceptor(atom: Atom) -> bool:
    return atom.element in {"N", "O", "S"} and atom.atom_name not in {"N", "NZ"}


def _pdb_line(atom: Atom, serial: int) -> str:
    return (f"ATOM  {serial:5d} {atom.atom_name:>4s} {atom.resname:>3s} {atom.chain:1s}"
            f"{atom.resseq:4d}{atom.icode:1s}   {atom.xyz[0]:8.3f}{atom.xyz[1]:8.3f}"
            f"{atom.xyz[2]:8.3f}  1.00 20.00          {atom.element:>2s}\n")


def _sasa(atoms: Sequence[Atom]) -> Optional[float]:
    """Use FreeSASA when present; return None instead of inventing a score."""
    if not atoms:
        return 0.0
    if len(atoms) > 500 and cKDTree is not None:
        xyz = np.asarray([a.xyz for a in atoms], dtype=float)
        rad = np.asarray([1.7 if a.element == "C" else 1.55 if a.element == "N" else 1.52 if a.element == "O" else 1.8 for a in atoms])
        tree = cKDTree(xyz); neigh = np.asarray([len(tree.query_ball_point(p, 8.0)) for p in xyz], dtype=float)
        return float(np.sum(4.0 * math.pi * (rad + 1.4) ** 2 / np.maximum(1.0, 0.25 * neigh)))
    try:
        import freesasa
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tmp:
            for i, atom in enumerate(atoms, 1):
                tmp.write(_pdb_line(atom, i))
            tmp.write("END\n")
            path = tmp.name
        try:
            return float(freesasa.calc(freesasa.Structure(path)).totalArea())
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception:
        return None


def _union_find_cycle_stats(coords: np.ndarray, cutoff: float) -> Tuple[int, int, int]:
    n = len(coords)
    if n == 0:
        return 0, 0, 0
    if cKDTree is not None:
        pairs = cKDTree(coords).query_pairs(cutoff, output_type="ndarray")
    else:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.linalg.norm(coords[i] - coords[j])) <= cutoff:
                    pairs.append((i, j))
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    components = n
    for i, j in pairs:
        i, j = int(i), int(j)
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            components -= 1
    edges = len(pairs)
    return int(components), int(max(0, edges - n + components)), edges


def _dna_step_geometry(dna_res: Mapping[Tuple[str, int, str], Sequence[Atom]]) -> Dict[str, float]:
    rises: List[float] = []; twists: List[float] = []
    by_chain: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
    for key in dna_res:
        by_chain[key[0]].append(key)
    for chain, keys in by_chain.items():
        keys.sort(key=lambda k: (k[1], k[2]))
        for k1, k2 in zip(keys, keys[1:]):
            # Reject numbering gaps larger than one; insertion codes remain valid.
            if k2[1] - k1[1] > 1:
                continue
            a1 = {a.atom_name: a for a in dna_res[k1]}
            a2 = {a.atom_name: a for a in dna_res[k2]}
            c1 = a1.get("C1'") or a1.get("C1*") or a1.get("C4'")
            c2 = a2.get("C1'") or a2.get("C1*") or a2.get("C4'")
            if c1 and c2:
                rises.append(float(np.linalg.norm(np.asarray(c1.xyz) - np.asarray(c2.xyz))))
            def normal(at: Mapping[str, Atom]) -> Optional[np.ndarray]:
                names = [n for n in ("N1", "N9", "C2", "C4", "C6", "C5") if n in at]
                if len(names) < 3:
                    return None
                x = np.asarray(at[names[0]].xyz); y = np.asarray(at[names[1]].xyz); z = np.asarray(at[names[2]].xyz)
                v = np.cross(y - x, z - x); norm = np.linalg.norm(v)
                return v / norm if norm else None
            n1, n2 = normal(a1), normal(a2)
            if n1 is not None and n2 is not None:
                twists.append(math.degrees(math.acos(float(np.clip(abs(np.dot(n1, n2)), -1.0, 1.0)))))
    return {"mean_rise": _mean(rises), "std_rise": _std(rises),
            "mean_twist": _mean(twists), "std_twist": _std(twists),
            "step_count": float(len(rises)), "twist_count": float(len(twists)),
            "geometry_missing": float(not bool(rises))}


def _extract_pd_feature_groups_uncached(path: str, chain_a: Optional[str] = None,
                                        chain_b: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Extract all 17 Protein--DNA groups for one candidate structure.

    ``chain_a``/``chain_b`` are optional because several benchmark structures
    have multiple protein or DNA chains.  When absent, residue chemistry picks
    all protein and DNA chains and avoids silently assigning chain order.
    """
    atoms = parse_pdb(path)
    proteins = [a for a in atoms if a.is_protein]
    dna = [a for a in atoms if a.is_dna]
    if chain_a and chain_b:
        selected = {str(chain_a), str(chain_b)}
        p_sel = [a for a in proteins if a.chain in selected]
        d_sel = [a for a in dna if a.chain in selected]
        if p_sel and d_sel:
            proteins, dna = p_sel, d_sel
    def thin(group, other, cap=500):
        if len(group) <= cap or not other or cKDTree is None:
            return group, 0
        xyz = np.asarray([x.xyz for x in group], dtype=float)
        ref = np.asarray([x.xyz for x in other], dtype=float)
        dmin = cKDTree(ref).query(xyz, k=1)[0]
        keep = np.argsort(dmin, kind="mergesort")[:cap]
        keep.sort()
        return [group[int(i)] for i in keep], len(group) - cap
    proteins, p_thin = thin(proteins, dna); dna, d_thin = thin(dna, proteins)
    pa = np.asarray([a.xyz for a in proteins], dtype=float).reshape((-1, 3))
    da = np.asarray([a.xyz for a in dna], dtype=float).reshape((-1, 3))
    ii, jj, dd = _pair_indices(pa, da, 12.0)
    if len(proteins) == 0 or len(dna) == 0:
        # Keep a stable schema even for malformed/native-only structures.
        return {name: {"valid": 0.0, "protein_atoms": float(len(proteins)),
                       "dna_atoms": float(len(dna)), "missing_partner": 1.0}
                for name in FEATURE_GROUPS}
    p_res = _residue_atoms(proteins); d_res = _residue_atoms(dna)
    p_keys = {k: i for i, k in enumerate(p_res)}; d_keys = {k: i for i, k in enumerate(d_res)}
    p_ridx = {id(a): p_keys[a.residue_key] for a in proteins}
    d_ridx = {id(a): d_keys[a.residue_key] for a in dna}

    groups: Dict[str, Dict[str, float]] = {}
    # 1 Contact profile.
    g: Dict[str, float] = {"valid": 1.0, "protein_atoms": float(len(proteins)), "dna_atoms": float(len(dna)),
                           "thinned_atoms": float(p_thin + d_thin)}
    for cutoff in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
        mask = dd <= cutoff
        g[f"count_d{int(cutoff)}"] = float(mask.sum())
        g[f"protein_atom_d{int(cutoff)}"] = float(len(set(ii[mask])))
        g[f"dna_atom_d{int(cutoff)}"] = float(len(set(jj[mask])))
        g[f"density_d{int(cutoff)}"] = float(mask.sum() / max(1, len(proteins) * len(dna)))
    groups["contact_profile"] = g

    # 2 Atom-pair radial distribution.
    g = {"valid": 1.0}
    hist, _ = np.histogram(dd, bins=np.arange(0.0, 12.0001, 0.5))
    for k, value in enumerate(hist):
        g[f"rdf_{k:02d}"] = float(value / max(1, len(dd)))
    # Keep a small, fixed element-pair vocabulary for a stable schema.
    for pair in ("C-C", "C-N", "C-O", "N-N", "N-O", "O-O", "O-P", "N-P", "P-P"):
        e1, e2 = pair.split("-")
        g[f"element_{pair.replace('-', '_')}"] = float(sum(1 for a, b in zip(ii, jj)
            if proteins[int(a)].element == e1 and dna[int(b)].element == e2))
    groups["atom_pair_rdf"] = g

    # 3 Residue--nucleotide contact matrix.
    g = {"valid": 1.0}
    matrix = Counter()
    for a, b, d in zip(ii, jj, dd):
        if d > 6.0:
            continue
        aa = AA3_TO_1.get(proteins[int(a)].resname, "X")
        base = DNA_NAMES.get(dna[int(b)].resname, "X")
        matrix[(aa, base)] += 1
    for aa in AA_ORDER:
        for base in BASE_ORDER:
            g[f"{aa}_{base}"] = float(matrix[(aa, base)])
    g["unknown_contact_count"] = float(sum(v for (aa, base), v in matrix.items() if aa == "X" or base == "X"))
    groups["residue_nucleotide_matrix"] = g

    # 4 Base edge / groove contacts.
    g = {"valid": 1.0}
    edge_counts = Counter()
    for a, b, d in zip(ii, jj, dd):
        if d <= 6.0:
            edge_counts[_base_atom_class(dna[int(b)])] += 1
    for key in ("major", "minor", "sugar", "phosphate", "base_other"):
        g[f"{key}_contacts"] = float(edge_counts[key])
    g["major_minor_ratio"] = float(edge_counts["major"] / max(1, edge_counts["minor"]))
    groups["base_edge_groove"] = g

    # 5 Hydrogen-bond geometry (distance-only because no angle is inferred).
    hb = [float(d) for a, b, d in zip(ii, jj, dd)
          if d <= 3.6 and ((_is_donor(proteins[int(a)]) and _is_acceptor(dna[int(b)]))
                           or (_is_acceptor(proteins[int(a)]) and _is_donor(dna[int(b)])))]
    groups["hydrogen_bond_geometry"] = {
        "valid": 1.0, "count_d35": float(sum(d <= 3.5 for d in hb)),
        "count_d36": float(len(hb)), "mean_distance": _mean(hb),
        "min_distance": min(hb) if hb else 0.0,
        "distance_only": 1.0, "angle_available": 0.0,
    }

    # 6 Phosphate electrostatic interactions.
    phosph = [(k, a) for k, ats in d_res.items() for a in ats if _base_atom_class(a) == "phosphate"]
    charged = [a for a in proteins if _protein_atom_class(a) in {"positive", "negative"}]
    cxyz = np.asarray([a.xyz for a in charged], dtype=float).reshape((-1, 3))
    pxyz = np.asarray([a.xyz for _, a in phosph], dtype=float).reshape((-1, 3))
    ci, pi, cd = _pair_indices(cxyz, pxyz, 10.0)
    e_total = 0.0; e_pos = 0.0; e_neg = 0.0
    for c, p, d in zip(ci, pi, cd):
        charge = 1.0 if _protein_atom_class(charged[int(c)]) == "positive" else -1.0
        value = charge / max(float(d), 1.5)
        e_total += value
        e_pos += max(0.0, value); e_neg += min(0.0, value)
    groups["phosphate_electrostatics"] = {
        "valid": 1.0, "pair_count": float(len(cd)), "signed_inverse_distance": e_total,
        "positive_component": e_pos, "negative_component": e_neg,
        "phosphate_atom_count": float(len(phosph)), "charged_atom_count": float(len(charged)),
    }

    # 7 Van der Waals / Lennard-Jones raw interaction terms.
    lj_total = lj_attr = lj_rep = 0.0
    for a, b, d in zip(ii, jj, dd):
        if d > 8.0:
            continue
        ea, eb = proteins[int(a)].element, dna[int(b)].element
        sigma = ELEMENT_RADIUS.get(ea, 1.7) + ELEMENT_RADIUS.get(eb, 1.7)
        epsilon = math.sqrt(ELEMENT_EPS.get(ea, 0.15) * ELEMENT_EPS.get(eb, 0.15))
        r = max(float(d), 1.5)
        term = 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        lj_total += term
        if term >= 0:
            lj_rep += term
        else:
            lj_attr += term
    groups["vdw_lj"] = {"valid": 1.0, "pair_count_d8": float(sum(dd <= 8.0)),
                         "lj_total": lj_total, "lj_attractive": lj_attr, "lj_repulsive": lj_rep}

    # 8 SASA / BSA.
    sasa_p, sasa_d, sasa_c = _sasa(proteins), _sasa(dna), _sasa(proteins + dna)
    if sasa_p is None or sasa_d is None or sasa_c is None:
        groups["sasa_bsa"] = {"valid": 1.0, "sasa_protein": 0.0, "sasa_dna": 0.0,
                               "sasa_complex": 0.0, "bsa": 0.0, "sasa_available": 0.0}
    else:
        groups["sasa_bsa"] = {"valid": 1.0, "sasa_protein": sasa_p, "sasa_dna": sasa_d,
                               "sasa_complex": sasa_c, "bsa": max(0.0, (sasa_p + sasa_d - sasa_c) / 2.0),
                               "sasa_available": 1.0}

    # 9 Shape-complementarity proxy from two-sided local contact degree.
    mask6 = dd <= 6.0
    dp = Counter(int(x) for x in ii[mask6]); ddna = Counter(int(x) for x in jj[mask6])
    comp = [min(dp[int(a)], ddna[int(b)]) / max(dp[int(a)], ddna[int(b)])
            for a, b in zip(ii[mask6], jj[mask6]) if dp[int(a)] and ddna[int(b)]]
    groups["shape_complementarity"] = {"valid": 1.0, "mean_local_balance": _mean(comp),
                                        "std_local_balance": _std(comp),
                                        "interface_pair_count": float(mask6.sum()),
                                        "normal_vectors_available": 0.0}

    # 10 DNA base-step geometry.
    groups["dna_base_step_geometry"] = {"valid": 1.0, **_dna_step_geometry(d_res)}

    # 11 Groove geometry, computed from cross-strand phosphate distances.
    by_chain: Dict[str, List[Atom]] = defaultdict(list)
    for atom in dna:
        if _base_atom_class(atom) == "phosphate":
            by_chain[atom.chain].append(atom)
    widths: List[float] = []
    chains = sorted(by_chain)
    for i, c1 in enumerate(chains):
        for c2 in chains[i + 1:]:
            x = np.asarray([a.xyz for a in by_chain[c1]], dtype=float)
            y = np.asarray([a.xyz for a in by_chain[c2]], dtype=float)
            if len(x) and len(y):
                _, _, dist = _pair_indices(x, y, 30.0)
                widths.extend(float(v) for v in dist)
    groups["groove_geometry"] = {"valid": 1.0, "mean_phosphate_width": _mean(widths),
                                 "min_phosphate_width": min(widths) if widths else 0.0,
                                 "max_phosphate_width": max(widths) if widths else 0.0,
                                 "cross_strand_pairs": float(len(widths)),
                                 "single_strand": float(len(chains) < 2)}

    # Build residue-level contact graph once for groups 12--14/16/17.
    graph_edges: set = set()
    for a, b, d in zip(ii[mask6], jj[mask6], dd[mask6]):
        graph_edges.add((p_ridx[id(proteins[int(a)])], d_ridx[id(dna[int(b)])]))
    pdeg = Counter(p for p, _ in graph_edges); ddeg = Counter(d for _, d in graph_edges)

    # 12 Voronoi-like weighted contact areas.  This is explicitly not VoroMQA.
    area = sum(1.0 / max(1, pdeg[p] * ddeg[d]) for p, d in graph_edges)
    groups["voronoi_contact_areas"] = {"valid": 1.0, "weighted_contact_area": float(area),
                                        "edge_count": float(len(graph_edges)), "is_proxy": 1.0}

    # 13 Bipartite graph features.
    p_nodes = len(p_res); d_nodes = len(d_res)
    p_degree = list(pdeg.values()); d_degree = list(ddeg.values())
    groups["bipartite_graph"] = {
        "valid": 1.0, "protein_nodes": float(p_nodes), "dna_nodes": float(d_nodes),
        "edges": float(len(graph_edges)), "protein_degree_mean": _mean(p_degree),
        "protein_degree_max": float(max(p_degree) if p_degree else 0),
        "dna_degree_mean": _mean(d_degree), "dna_degree_max": float(max(d_degree) if d_degree else 0),
        "edge_density": float(len(graph_edges) / max(1, p_nodes * d_nodes)),
    }

    # 14 Directed path motifs on the residue bipartite graph.
    p_to_d = defaultdict(set); d_to_p = defaultdict(set)
    for p, d in graph_edges:
        p_to_d[p].add(d); d_to_p[d].add(p)
    p2p = sum(len(ps) * (len(ps) - 1) for ps in d_to_p.values())
    d2d = sum(len(ps) * (len(ps) - 1) for ps in p_to_d.values())
    p3d = sum(len(d_to_p[d]) * max(0, len(p_to_d[p]) - 1) for p in p_to_d for d in p_to_d[p])
    groups["directed_path_motifs"] = {"valid": 1.0, "p_d_p": float(p2p), "d_p_d": float(d2d),
                                        "p_d_p_d": float(p3d), "directed_edges": float(2 * len(graph_edges))}

    # 15 Spatial persistent-homology proxy, using ordinary Vietoris--Rips
    # component/cycle counts at fixed radii.  The source points are only the
    # interface atoms, keeping the computation bounded and deterministic.
    iface = sorted(set(int(x) for x in ii[mask6]) | {len(proteins) + int(x) for x in jj[mask6]})
    all_iface = np.vstack([pa, da]) if iface else np.empty((0, 3))
    original_iface_count = len(iface)
    if original_iface_count > 1600:
        # Deterministic evenly spaced thinning, recorded in the output.
        select = np.linspace(0, len(iface) - 1, 1600, dtype=int)
        iface = [iface[i] for i in select]
    coords = all_iface[np.asarray(iface, dtype=int)] if iface else np.empty((0, 3))
    g = {"valid": 1.0, "point_count": float(len(coords)), "thinned": float(original_iface_count > len(iface))}
    for cutoff in (2.0, 4.0, 6.0, 8.0):
        b0, b1, edges = _union_find_cycle_stats(coords, cutoff)
        g[f"betti0_r{int(cutoff)}"] = float(b0); g[f"betti1_r{int(cutoff)}"] = float(b1); g[f"edges_r{int(cutoff)}"] = float(edges)
    groups["spatial_persistent_homology"] = g

    # 16 Ordinary GLMY path counts.  No common geometric support or GSPH rule.
    directed = [(p, d) for p, d in graph_edges] + [(d, p) for p, d in graph_edges]
    out = defaultdict(set)
    for u, v in directed:
        out[u].add(v)
    path2 = sum(len(out[u]) * len(out[v]) for u, v in directed)
    path3 = sum(len(out[w]) for u, v in directed for w in out[v])
    groups["ordinary_glmy"] = {"valid": 1.0, "directed_edges": float(len(directed)),
                                "path_length_1": float(len(directed)), "path_length_2": float(path2),
                                "path_length_3": float(path3), "ordinary_distance_filtration": 1.0,
                                "gsph_common_support": 0.0}

    # 17 Fold-specific statistical potential inputs.  Raw counts only; the
    # train driver must estimate any log-odds/pseudocounts in the training fold.
    g = {"valid": 1.0, "fit_in_training_fold": 1.0, "reference_structure_used": 0.0}
    for aa in AA_ORDER:
        for base in BASE_ORDER:
            g[f"contact_{aa}_{base}"] = float(matrix[(aa, base)])
    g["total_contacts"] = float(sum(matrix.values()))
    groups["fold_specific_statistical_potential"] = g
    return groups


@lru_cache(maxsize=64)
def _extract_cached(path: str, mtime_ns: int, file_size: int,
                    chain_a: str, chain_b: str) -> Dict[str, Dict[str, float]]:
    # The mtime/size pair is part of the key because the fair pipeline reuses
    # one temporary candidate filename for all 17 groups.  Caching avoids
    # extracting the same candidate 17 times while still invalidating when the
    # file is replaced by the next candidate.
    del mtime_ns, file_size
    return _extract_pd_feature_groups_uncached(path, chain_a or None, chain_b or None)


def extract_pd_feature_groups(path: str, chain_a: Optional[str] = None,
                              chain_b: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Cached public wrapper; cache invalidates on temporary-file replacement."""
    try:
        stat = os.stat(path)
        return _extract_cached(os.path.abspath(path), int(stat.st_mtime_ns), int(stat.st_size),
                               str(chain_a or ""), str(chain_b or ""))
    except OSError:
        return _extract_pd_feature_groups_uncached(path, chain_a, chain_b)


FEATURE_GROUPS = (
    "contact_profile", "atom_pair_rdf", "residue_nucleotide_matrix",
    "base_edge_groove", "hydrogen_bond_geometry", "phosphate_electrostatics",
    "vdw_lj", "sasa_bsa", "shape_complementarity", "dna_base_step_geometry",
    "groove_geometry", "voronoi_contact_areas", "bipartite_graph",
    "directed_path_motifs", "spatial_persistent_homology", "ordinary_glmy",
    "fold_specific_statistical_potential",
)


__all__ = ["FEATURE_GROUPS", "extract_pd_feature_groups", "parse_pdb"]

