"""Protein--RNA fair feature baselines.

This module contains deterministic feature extraction only.  It deliberately
does not call CARP, DRPScore, FTDMP, PDIScore, an online service, or a
pre-trained model.  Each extractor sees the candidate coordinates and the
candidate's protein/RNA chain labels only.  The returned vectors can therefore
be used with the common target-grouped XGBoost head used by the fair baseline
runner.

The last two groups are intentionally ordinary baselines: ``spatial_ph`` is a
0-dimensional distance-filtration signature and ``ordinary_glmy`` is the
single-event directed path signature on the corresponding contact graph.  No
GSPH geometry-sharing rule is present here.

The implementation is conservative about malformed input.  Hydrogens, water,
ions and alternate conformers are ignored; missing or unsupported residues do
not abort the target.  Every feature group returns a fixed set of numeric
columns, including all-zero columns for an empty interface, so a batch can be
concatenated without schema-dependent imputation.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback keeps parsing usable
    cKDTree = None


PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "MSE", "SEC", "PYL",
}
RNA_RESIDUES = {
    "A", "C", "G", "U", "RA", "RC", "RG", "RU", "ADE", "CYT",
    "GUA", "URA", "PSU", "H2U", "1MA", "5MC", "OMG", "M2G", "7MG",
    "OMC", "YYG", "I", "INO",
}
WATER_IONS = {
    "HOH", "WAT", "DOD", "SOL", "SO4", "PO4", "CL", "NA", "K", "MG",
    "CA", "ZN", "MN", "FE", "F", "BR", "IOD",
}
AA3 = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
BASES = ["A", "C", "G", "U"]
ELEM = ["C", "N", "O", "S", "P", "OTHER"]
ELEM_INDEX = {x: i for i, x in enumerate(ELEM)}


@dataclass(frozen=True)
class Atom:
    chain: str
    resseq: int
    icode: str
    resname: str
    name: str
    element: str
    xyz: Tuple[float, float, float]
    serial: int

    @property
    def residue_key(self) -> Tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode)


@dataclass
class PairTable:
    ia: np.ndarray
    ib: np.ndarray
    d: np.ndarray


def _zero(prefix: str, names: Iterable[str]) -> Dict[str, float]:
    return {prefix + str(n): 0.0 for n in names}


def _norm_name(name: str) -> str:
    return (name or "").strip().upper()


def _is_hydrogen(atom_name: str, element: str) -> bool:
    e = _norm_name(element)
    n = _norm_name(atom_name)
    return e == "H" or n.startswith("H") or n.startswith("D")


def _canonical_base(resname: str) -> Optional[str]:
    r = _norm_name(resname)
    aliases = {
        "A": "A", "RA": "A", "ADE": "A", "1MA": "A", "M2A": "A",
        "C": "C", "RC": "C", "CYT": "C", "5MC": "C", "OMC": "C",
        "G": "G", "RG": "G", "GUA": "G", "M2G": "G", "7MG": "G",
        "U": "U", "RU": "U", "URA": "U", "PSU": "U", "H2U": "U",
    }
    if r in aliases:
        return aliases[r]
    # Some files use a leading ribose marker, e.g. "rG".
    if len(r) >= 1 and r[-1] in BASES:
        return r[-1]
    return None


def _parse_pdb(path: str) -> List[Atom]:
    """Parse the first model using Bio.PDB, dropping waters/ions/altlocs."""
    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Biopython is required for PR fair features") from exc
    parser = MMCIFParser(QUIET=True) if path.lower().endswith((".cif", ".mmcif")) else PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("candidate", path)
    except Exception as exc:
        raise ValueError("cannot parse structure %s: %s" % (path, exc)) from exc
    atoms: List[Atom] = []
    model = next(structure.get_models())
    seen_res_alt: Set[Tuple[str, int, str, str]] = set()
    for chain in model:
        chain_id = str(chain.id).strip() or "_"
        for residue in chain:
            resname = _norm_name(residue.resname)
            if resname in WATER_IONS:
                continue
            rid = residue.id
            resseq = int(rid[1])
            icode = str(rid[2]).strip()
            # A PDB residue may have a disordered alternate wrapper.  Keep the
            # first selected conformer; feature extraction must be deterministic.
            for atom in residue:
                alt = getattr(atom, "altloc", " ")
                if alt not in (" ", "", "A", "1"):
                    continue
                aname = _norm_name(atom.get_name())
                elem = _norm_name(getattr(atom, "element", ""))
                if not elem:
                    elem = aname[0] if aname else ""
                if _is_hydrogen(aname, elem):
                    continue
                key = (chain_id, resseq, icode, aname)
                if key in seen_res_alt:
                    continue
                seen_res_alt.add(key)
                coord = tuple(float(x) for x in atom.get_coord())
                if not all(math.isfinite(x) for x in coord):
                    continue
                atoms.append(Atom(chain_id, resseq, icode, resname, aname, elem, coord, int(atom.serial_number or 0)))
    return atoms


def _chain_types(atoms: Sequence[Atom]) -> Tuple[List[str], List[str]]:
    by: Dict[str, List[str]] = defaultdict(list)
    for a in atoms:
        by[a.chain].append(a.resname)
    proteins: List[str] = []
    rnas: List[str] = []
    for chain, names in sorted(by.items()):
        p = sum(n in PROTEIN_RESIDUES for n in names)
        r = sum((_canonical_base(n) is not None) or n in RNA_RESIDUES for n in names)
        if p and p >= r:
            proteins.append(chain)
        elif r:
            rnas.append(chain)
    # Fallback for files with non-standard residue names: a chain containing P
    # and N/O-rich base atoms is more likely to be RNA than a protein chain.
    if not proteins or not rnas:
        for chain, names in sorted(by.items()):
            if chain in proteins or chain in rnas:
                continue
            rows = [a for a in atoms if a.chain == chain]
            p_atoms = sum(a.element == "P" for a in rows)
            if p_atoms:
                rnas.append(chain)
            else:
                proteins.append(chain)
    return sorted(set(proteins)), sorted(set(rnas))


def _select_partners(atoms: Sequence[Atom], protein_chains: Optional[Sequence[str]], rna_chains: Optional[Sequence[str]]) -> Tuple[List[Atom], List[Atom], List[str], List[str]]:
    inferred_p, inferred_r = _chain_types(atoms)
    pchains = sorted(set(str(x) for x in (protein_chains or inferred_p)))
    rchains = sorted(set(str(x) for x in (rna_chains or inferred_r)))
    pa = [a for a in atoms if a.chain in pchains]
    rb = [a for a in atoms if a.chain in rchains]
    return pa, rb, pchains, rchains


def _pairs(a: Sequence[Atom], b: Sequence[Atom], cutoff: float = 12.0) -> PairTable:
    if not a or not b:
        return PairTable(np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=float))
    xyz_a = np.asarray([x.xyz for x in a], dtype=float)
    xyz_b = np.asarray([x.xyz for x in b], dtype=float)
    ia: List[int] = []
    ib: List[int] = []
    dd: List[float] = []
    if cKDTree is not None:
        tree = cKDTree(xyz_b)
        for i, hits in enumerate(tree.query_ball_point(xyz_a, float(cutoff))):
            if not hits:
                continue
            delta = xyz_b[np.asarray(hits, dtype=np.int32)] - xyz_a[i]
            dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            ia.extend([i] * len(hits))
            ib.extend(int(j) for j in hits)
            dd.extend(float(x) for x in dist)
    else:  # pragma: no cover - scipy is part of the remote environment
        for i, xa in enumerate(xyz_a):
            delta = xyz_b - xa
            dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            for j in np.where(dist <= cutoff)[0]:
                ia.append(i); ib.append(int(j)); dd.append(float(dist[j]))
    return PairTable(np.asarray(ia, dtype=np.int32), np.asarray(ib, dtype=np.int32), np.asarray(dd, dtype=float))


def _element(a: Atom) -> str:
    e = _norm_name(a.element)
    return e if e in ELEM_INDEX and e != "OTHER" else "OTHER"


def _atom_charge(a: Atom) -> float:
    """Fixed, transparent charges used only for the fair electrostatic feature."""
    n, r, e = a.name, a.resname, _element(a)
    if r in ("ARG", "LYS") and e == "N":
        return 0.45
    if r in ("ASP", "GLU") and e == "O":
        return -0.45
    if _canonical_base(r) is not None:
        if e == "P" or n in {"OP1", "OP2", "O1P", "O2P"}:
            return -0.55
        if n in {"N1", "N2", "N3", "N4", "N6", "N7", "N9"}:
            return 0.12
        if e == "O":
            return -0.18
    return {"N": -0.10, "O": -0.20, "S": -0.05, "P": 0.20, "C": 0.05}.get(e, 0.0)


def _residue_class(a: Atom) -> str:
    return a.resname if a.resname in AA3 else "OTHER"


def _base_edge(atom_name: str) -> str:
    n = _norm_name(atom_name)
    # FR3D-compatible edge atom classes.  The mapping is a geometric atom
    # classification, not a pretrained FR3D score.
    wc = {"N1", "N2", "N3", "N4", "N6", "O2", "O4", "O6"}
    hoog = {"N7", "N6", "O6", "C8"}
    sugar = {"N9", "N3", "C2", "C4", "C5", "C6"}
    if n in wc: return "wc"
    if n in hoog: return "hoogsteen"
    if n in sugar: return "sugar"
    return "other"


def _is_aromatic_protein(a: Atom) -> bool:
    return a.resname in {"PHE", "TYR", "TRP", "HIS"} and a.element == "C"


def _is_base_atom(a: Atom) -> bool:
    return _canonical_base(a.resname) is not None and a.name not in {"P", "OP1", "OP2", "O1P", "O2P", "O3P", "O5'", "C3'", "C4'", "C5'", "O4'"}


def _contact_edges(a: Sequence[Atom], b: Sequence[Atom], pairs: PairTable, cutoff: float = 6.0) -> Set[Tuple[Tuple[str, int, str], Tuple[str, int, str]]]:
    edges: Set[Tuple[Tuple[str, int, str], Tuple[str, int, str]]] = set()
    for i, j, d in zip(pairs.ia, pairs.ib, pairs.d):
        if d <= cutoff:
            edges.add((a[int(i)].residue_key, b[int(j)].residue_key))
    return edges


def _residue_graph_features(a: Sequence[Atom], b: Sequence[Atom], pairs: PairTable) -> Dict[str, float]:
    out = {"pr_bip_nodes_protein": 0.0, "pr_bip_nodes_rna": 0.0, "pr_bip_edges": 0.0,
           "pr_bip_degree_mean": 0.0, "pr_bip_degree_max": 0.0, "pr_bip_density": 0.0,
           "pr_bip_components": 0.0, "pr_bip_spectral_radius": 0.0}
    pnodes = sorted(set(x.residue_key for x in a))
    rnodes = sorted(set(x.residue_key for x in b))
    edges = _contact_edges(a, b, pairs)
    out["pr_bip_nodes_protein"] = float(len(pnodes)); out["pr_bip_nodes_rna"] = float(len(rnodes)); out["pr_bip_edges"] = float(len(edges))
    if not edges:
        return out
    adj: Dict[Tuple[str, int, str], Set[Tuple[str, int, str]]] = defaultdict(set)
    for p, r in edges:
        adj[p].add(r); adj[r].add(p)
    deg = [len(adj[x]) for x in adj]
    out["pr_bip_degree_mean"] = float(np.mean(deg)); out["pr_bip_degree_max"] = float(np.max(deg))
    out["pr_bip_density"] = float(len(edges) / max(1, len(pnodes) * len(rnodes)))
    seen: Set[Tuple[str, int, str]] = set(); comps = 0
    for node in adj:
        if node in seen: continue
        comps += 1; q = [node]; seen.add(node)
        while q:
            cur = q.pop()
            for nxt in adj[cur]:
                if nxt not in seen: seen.add(nxt); q.append(nxt)
    out["pr_bip_components"] = float(comps)
    nodes = list(adj); index = {n: i for i, n in enumerate(nodes)}
    mat = np.zeros((len(nodes), len(nodes)), dtype=float)
    for x, ns in adj.items():
        for y in ns: mat[index[x], index[y]] = 1.0
    try: out["pr_bip_spectral_radius"] = float(np.max(np.abs(np.linalg.eigvalsh(mat))))
    except Exception: pass
    return out


def _directed_graph(a: Sequence[Atom], b: Sequence[Atom], pairs: PairTable, threshold: float = 6.0) -> Dict[Tuple[str, int, str], Set[Tuple[str, int, str]]]:
    """Build a directed contact graph with donor->acceptor orientation.

    Chain-neighbour edges are added in both directions, which gives ordinary
    path homology a meaningful backbone without using a native/reference
    structure.  Cross-partner edges use fixed donor/acceptor atom chemistry.
    """
    g: Dict[Tuple[str, int, str], Set[Tuple[str, int, str]]] = defaultdict(set)
    pa = sorted(set(x.residue_key for x in a), key=lambda x: (x[0], x[1], x[2]))
    rb = sorted(set(x.residue_key for x in b), key=lambda x: (x[0], x[1], x[2]))
    for seq in (pa, rb):
        for u, v in zip(seq, seq[1:]):
            if u[0] == v[0] and abs(u[1] - v[1]) <= 2:
                g[u].add(v); g[v].add(u)
    for i, j, d in zip(pairs.ia, pairs.ib, pairs.d):
        if d > threshold: continue
        x, y = a[int(i)], b[int(j)]
        if x.element not in {"N", "O", "S"} or y.element not in {"N", "O", "S"}: continue
        # Sign of the fixed charge is a reproducible donor/acceptor proxy.
        if _atom_charge(x) >= _atom_charge(y): g[x.residue_key].add(y.residue_key)
        else: g[y.residue_key].add(x.residue_key)
    return g


def _path_counts(g: Mapping[Tuple[str, int, str], Set[Tuple[str, int, str]]], max_len: int = 3) -> List[float]:
    out = [0.0] * max_len
    for start in g:
        frontier = {start}
        for k in range(max_len):
            nxt: Set[Tuple[str, int, str]] = set()
            for u in frontier:
                nxt.update(g.get(u, ()))
            out[k] += float(len(nxt))
            frontier = nxt
            if not frontier: break
    return out


def _all_interface_xyz(a: Sequence[Atom], b: Sequence[Atom], pairs: PairTable, cutoff: float = 6.0) -> np.ndarray:
    ai = set(int(x) for x in pairs.ia[pairs.d <= cutoff]); bi = set(int(x) for x in pairs.ib[pairs.d <= cutoff])
    xyz = [a[i].xyz for i in sorted(ai)] + [b[i].xyz for i in sorted(bi)]
    return np.asarray(xyz, dtype=float) if xyz else np.zeros((0, 3), dtype=float)


def _components_at(xyz: np.ndarray, threshold: float) -> int:
    n = len(xyz)
    if n == 0: return 0
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x: int, y: int) -> None:
        x, y = find(x), find(y)
        if x != y: parent[y] = x
    if cKDTree is not None:
        for i, j in cKDTree(xyz).query_pairs(float(threshold)):
            union(int(i), int(j))
    else:  # pragma: no cover
        for i in range(n):
            for j in range(i):
                if np.linalg.norm(xyz[i] - xyz[j]) <= threshold: union(i, j)
    return len({find(i) for i in range(n)})


def _sasa_bsa(path: str, a: Sequence[Atom], b: Sequence[Atom]) -> Tuple[float, float, float]:
    """Return protein SASA, RNA SASA and conventional half-delta BSA."""
    if len(a) + len(b) > 500 and cKDTree is not None:
        # FreeSASA is cubic-ish on very large complexes.  Use a deterministic
        # rolling-neighbour solvent exposure for oversized candidates; this is
        # a geometry-only SASA approximation, not a learned/proxy score.
        def approx(rows):
            if not rows:
                return 0.0
            xyz = np.asarray([x.xyz for x in rows], dtype=float)
            rad = np.asarray([1.7 if x.element == "C" else 1.55 if x.element == "N" else 1.52 if x.element == "O" else 1.8 for x in rows])
            tree = cKDTree(xyz); neigh = np.asarray([len(tree.query_ball_point(p, 8.0)) for p in xyz], dtype=float)
            return float(np.sum(4.0 * math.pi * (rad + 1.4) ** 2 / np.maximum(1.0, 0.25 * neigh)))
        p_sasa, r_sasa = approx(a), approx(b); comp = approx(list(a) + list(b))
        return p_sasa, r_sasa, max(0.0, (p_sasa + r_sasa - comp) / 2.0)
    try:
        import freesasa
    except Exception:
        return (0.0, 0.0, 0.0)
    # Build the selected complex through FreeSASA's native API.  This avoids
    # fragile fixed-column PDB formatting for atom names such as O5' while
    # preventing solvent/third chains from changing pairwise BSA.
    try:
        freesasa.setVerbosity(freesasa.silent)
    except Exception:
        pass
    try:
        def make(rows: Sequence[Atom], chain: str):
            s = freesasa.Structure()
            for i, x in enumerate(rows, 1):
                res = x.resname[:3] if x.resname in PROTEIN_RESIDUES else (_canonical_base(x.resname) or "A")
                try:
                    s.addAtom(x.name[:4], res, str(i), chain, float(x.xyz[0]), float(x.xyz[1]), float(x.xyz[2]))
                except Exception:
                    # Unsupported modified atoms are still represented with a
                    # generic carbon radius rather than dropping the target.
                    s.addAtom("C", "GLY" if chain == "P" else "A", str(i), chain, float(x.xyz[0]), float(x.xyz[1]), float(x.xyz[2]))
            return s
        p_sasa = float(freesasa.calc(make(a, "P")).totalArea())
        r_sasa = float(freesasa.calc(make(b, "R")).totalArea())
        comp = float(freesasa.calc(make(list(a) + list(b), "X")).totalArea())
        return p_sasa, r_sasa, max(0.0, (p_sasa + r_sasa - comp) / 2.0)
    except Exception:
        return (0.0, 0.0, 0.0)


def _base_dict() -> Dict[str, float]:
    """Initialize the complete fixed PR schema."""
    d: Dict[str, float] = {}
    d.update(_zero("pr_contact_bin_", ["2p5", "3", "3p5", "4", "5", "6", "8", "10", "12"]))
    d.update(_zero("pr_contact_", ["count_norm", "atom_a_frac", "atom_b_frac", "residue_edges", "min_dist", "mean_dist", "std_dist"]))
    for x in ELEM:
        for y in ELEM[ELEM_INDEX[x]:]:
            for bi in range(24): d["pr_rdf_%s_%s_b%02d" % (x.lower(), y.lower(), bi)] = 0.0
    for aa in AA3 + ["OTHER"]:
        for base in BASES + ["OTHER"]: d["pr_resnu_%s_%s" % (aa.lower(), base.lower())] = 0.0
    d.update(_zero("pr_hbond_", ["count", "close", "mean_dist", "std_dist", "n_o", "n_n", "o_o", "s_any"]))
    d.update(_zero("pr_baseedge_", ["wc", "hoogsteen", "sugar", "other", "wc_frac", "hoogsteen_frac", "sugar_frac"]))
    d.update(_zero("pr_stack_", ["count", "3p5_4", "4_5", "mean_dist", "phe", "tyr", "trp", "his"]))
    d.update(_zero("pr_sasa_", ["protein", "rna", "bsa", "bsa_frac"]))
    d.update(_zero("pr_electro_", ["sum", "abs_sum", "attractive", "repulsive", "mean", "min"]))
    d.update(_zero("pr_vdw_", ["lj_sum", "lj_mean", "lj_min", "lj_repulsive", "lj_attractive"]))
    d.update(_zero("pr_shape_", ["nn_mean", "nn_std", "nn_min", "nn_frac4", "nn_frac6", "reciprocal_mean", "pair_dist_mean"]))
    d.update(_zero("pr_voro_", ["area_total", "area_mean", "area_max", "area_protein", "area_rna", "area_balanced", "weighted_contacts"]))
    d.update(_zero("pr_bip_", ["nodes_protein", "nodes_rna", "edges", "degree_mean", "degree_max", "density", "components", "spectral_radius"]))
    d.update(_zero("pr_path_", ["edges", "paths1", "paths2", "paths3", "backbone_edges", "cross_edges", "cross_path2", "cross_path3"]))
    for t in ("2", "3", "4", "5", "6", "8", "10"):
        d["pr_ph_b0_t" + t] = 0.0
    d.update(_zero("pr_ph_", ["b0_auc", "b0_initial", "b0_final", "b0_persistence", "interface_atoms"]))
    for t in ("3", "4", "5", "6", "8"):
        for k in (1, 2, 3): d["pr_glmy_p%d_t%s" % (k, t)] = 0.0
    d.update(_zero("pr_glmy_", ["euler_mean", "nodes", "edges", "nonempty_thresholds"]))
    return d


def extract_pr_features(path: str, protein_chains: Optional[Sequence[str]] = None, rna_chains: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Extract all 15 Protein--RNA feature groups from one candidate PDB/mmCIF."""
    atoms = _parse_pdb(path)
    a, b, pchains, rchains = _select_partners(atoms, protein_chains, rna_chains)
    # Large PRDB structures can contain >10k heavy atoms.  Keep the complete
    # interface shell and deterministically thin only distant interior atoms
    # for bounded pair calculations; the raw structure remains unchanged.
    def thin(group, other, cap=500):
        if len(group) <= cap or not other or cKDTree is None:
            return group, 0
        xyz = np.asarray([x.xyz for x in group], dtype=float)
        ref = np.asarray([x.xyz for x in other], dtype=float)
        dmin = cKDTree(ref).query(xyz, k=1)[0]
        keep = np.argsort(dmin, kind="mergesort")[:cap]
        keep.sort()
        return [group[int(i)] for i in keep], len(group) - cap
    a, ta = thin(a, b); b, tb = thin(b, a)
    out = _base_dict()
    out["pr_meta_n_protein_atoms"] = float(len(a)); out["pr_meta_n_rna_atoms"] = float(len(b))
    out["pr_meta_n_protein_chains"] = float(len(pchains)); out["pr_meta_n_rna_chains"] = float(len(rchains))
    out["pr_meta_thinned_atoms"] = float(ta + tb)
    if not a or not b:
        return out
    pt = _pairs(a, b, 12.0)
    d = pt.d
    bins = [("2p5", 2.5), ("3", 3.0), ("3p5", 3.5), ("4", 4.0), ("5", 5.0), ("6", 6.0), ("8", 8.0), ("10", 10.0), ("12", 12.0)]
    prev = 0.0
    for name, hi in bins:
        out["pr_contact_bin_" + name] = float(np.sum((d > prev) & (d <= hi))) / max(1.0, math.sqrt(len(a) * len(b))); prev = hi
    close = d[d <= 6.0]
    edges = _contact_edges(a, b, pt)
    out.update({"pr_contact_count_norm": float(len(close) / max(1.0, math.sqrt(len(a) * len(b)))),
                "pr_contact_atom_a_frac": float(len(set(pt.ia[d <= 6])) / max(1, len(a))),
                "pr_contact_atom_b_frac": float(len(set(pt.ib[d <= 6])) / max(1, len(b))),
                "pr_contact_residue_edges": float(len(edges)),
                "pr_contact_min_dist": float(np.min(d)) if len(d) else 0.0,
                "pr_contact_mean_dist": float(np.mean(close)) if len(close) else 0.0,
                "pr_contact_std_dist": float(np.std(close)) if len(close) else 0.0})
    # Atom-pair RDF, upper triangular element classes and fixed 0.5A bins.
    pair_labels: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for i, j, x in zip(pt.ia, pt.ib, d):
        e1, e2 = _element(a[int(i)]), _element(b[int(j)])
        if ELEM_INDEX[e1] > ELEM_INDEX[e2]: e1, e2 = e2, e1
        pair_labels[(e1, e2)].append(float(x))
    for (e1, e2), vals in pair_labels.items():
        hist, _ = np.histogram(vals, bins=np.arange(0.0, 12.5, 0.5))
        for k, v in enumerate(hist): out["pr_rdf_%s_%s_b%02d" % (e1.lower(), e2.lower(), k)] = float(v / max(1.0, math.sqrt(len(a) * len(b))))
    # Residue/nucleotide contact matrix.
    rn_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for i, j, x in zip(pt.ia, pt.ib, d):
        if x > 6.0: continue
        aa = _residue_class(a[int(i)]); base = _canonical_base(b[int(j)].resname) or "OTHER"
        rn_counts[(aa, base)] += 1
    den = max(1.0, float(sum(rn_counts.values())))
    for (aa, base), val in rn_counts.items(): out["pr_resnu_%s_%s" % (aa.lower(), base.lower())] = float(val / den)
    # H-bond geometry.
    hb = [(i, j, x) for i, j, x in zip(pt.ia, pt.ib, d) if 2.3 <= x <= 3.6 and a[int(i)].element in {"N", "O", "S"} and b[int(j)].element in {"N", "O", "S"}]
    if hb:
        hd = np.asarray([x[2] for x in hb]); out["pr_hbond_count"] = float(len(hb)); out["pr_hbond_close"] = float(np.sum(hd <= 3.2)); out["pr_hbond_mean_dist"] = float(np.mean(hd)); out["pr_hbond_std_dist"] = float(np.std(hd))
        for i, j, _ in hb:
            x, y = a[int(i)].element, b[int(j)].element
            key = "n_o" if {x, y} == {"N", "O"} else "n_n" if x == y == "N" else "o_o" if x == y == "O" else "s_any" if "S" in {x, y} else None
            if key: out["pr_hbond_" + key] += 1.0
    # Base edge recognition.
    edge_counts = defaultdict(int)
    for i, j, x in zip(pt.ia, pt.ib, d):
        if x <= 4.0 and _is_base_atom(b[int(j)]): edge_counts[_base_edge(b[int(j)].name)] += 1
    for key, val in edge_counts.items(): out["pr_baseedge_" + key] = float(val)
    ed = float(sum(edge_counts.values()))
    if ed:
        for key in ("wc", "hoogsteen", "sugar"): out["pr_baseedge_" + key + "_frac"] = out["pr_baseedge_" + key] / ed
    # Aromatic stacking geometry proxy (distance geometry only; no learned score).
    stack = [(a[int(i)], b[int(j)], x) for i, j, x in zip(pt.ia, pt.ib, d) if 3.5 <= x <= 5.0 and _is_aromatic_protein(a[int(i)]) and _is_base_atom(b[int(j)])]
    if stack:
        sd = np.asarray([x[2] for x in stack]); out["pr_stack_count"] = float(len(stack)); out["pr_stack_3p5_4"] = float(np.sum(sd <= 4.0)); out["pr_stack_4_5"] = float(np.sum(sd > 4.0)); out["pr_stack_mean_dist"] = float(np.mean(sd))
        for x, _, _ in stack: out["pr_stack_" + x.resname.lower()] += 1.0
    # SASA/BSA uses the same selected protein/RNA atom sets for every candidate.
    ps, rs, bsa = _sasa_bsa(path, a, b); out.update({"pr_sasa_protein": ps, "pr_sasa_rna": rs, "pr_sasa_bsa": bsa, "pr_sasa_bsa_frac": bsa / max(1e-6, ps + rs)})
    # Fixed-charge electrostatics.
    if len(d):
        q = np.asarray([_atom_charge(a[int(i)]) * _atom_charge(b[int(j)]) / max(float(x), 1.5) for i, j, x in zip(pt.ia, pt.ib, d) if x <= 8.0], dtype=float)
        if len(q): out.update({"pr_electro_sum": float(np.sum(q)), "pr_electro_abs_sum": float(np.sum(np.abs(q))), "pr_electro_attractive": float(np.sum(q[q < 0])), "pr_electro_repulsive": float(np.sum(q[q > 0])), "pr_electro_mean": float(np.mean(q)), "pr_electro_min": float(np.min(q))})
    # Lennard-Jones energy with fixed element parameters.
    ljpar = {"C": (1.70, 0.12), "N": (1.55, 0.17), "O": (1.52, 0.20), "S": (1.80, 0.25), "P": (1.80, 0.20), "OTHER": (1.70, 0.12)}
    lv = []
    for i, j, x in zip(pt.ia, pt.ib, d):
        if x > 8.0: continue
        si, ei = ljpar[_element(a[int(i)])]; sj, ej = ljpar[_element(b[int(j)])]; sigma = (si + sj) / 2.0; epsilon = math.sqrt(ei * ej); r = max(float(x), 1.5); z = sigma / r; z6 = z ** 6; lv.append(4.0 * epsilon * (z6 * z6 - z6))
    if lv:
        lv = np.asarray(lv); out.update({"pr_vdw_lj_sum": float(np.sum(lv)), "pr_vdw_lj_mean": float(np.mean(lv)), "pr_vdw_lj_min": float(np.min(lv)), "pr_vdw_lj_repulsive": float(np.sum(lv[lv > 0])), "pr_vdw_lj_attractive": float(np.sum(lv[lv < 0]))})
    # Shape complementarity from nearest-neighbour distances (no native input).
    xa = np.asarray([x.xyz for x in a]); xb = np.asarray([x.xyz for x in b])
    if len(xa) and len(xb):
        if cKDTree is not None:
            nda = cKDTree(xb).query(xa, k=1)[0]; ndb = cKDTree(xa).query(xb, k=1)[0]
        else:  # pragma: no cover
            nda = np.min(np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=2), axis=1); ndb = np.min(np.linalg.norm(xa[None, :, :] - xb[:, None, :], axis=2), axis=1)
        nn = np.concatenate([nda, ndb]); out.update({"pr_shape_nn_mean": float(np.mean(nn)), "pr_shape_nn_std": float(np.std(nn)), "pr_shape_nn_min": float(np.min(nn)), "pr_shape_nn_frac4": float(np.mean(nn <= 4.0)), "pr_shape_nn_frac6": float(np.mean(nn <= 6.0)), "pr_shape_reciprocal_mean": float(np.mean(1.0 / (nn + 0.5))), "pr_shape_pair_dist_mean": float(np.mean(close)) if len(close) else 0.0})
    # Voronoi-style contact area allocation: each interface atom owns a
    # spherical Voronoi patch; each patch is shared by its cross neighbours.
    pairs6 = [(int(i), int(j), float(x)) for i, j, x in zip(pt.ia, pt.ib, d) if x <= 6.0]
    deg_a = defaultdict(int); deg_b = defaultdict(int)
    for i, j, _ in pairs6: deg_a[i] += 1; deg_b[j] += 1
    areas = []
    for i, j, x in pairs6:
        ra = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}.get(_element(a[i]), 1.7); rb = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}.get(_element(b[j]), 1.7)
        area = 0.5 * (4.0 * math.pi * ra * ra / max(1, deg_a[i]) + 4.0 * math.pi * rb * rb / max(1, deg_b[j])) * math.exp(-max(0.0, x - ra - rb))
        areas.append((area, i, j))
    if areas:
        av = np.asarray([x[0] for x in areas]); out.update({"pr_voro_area_total": float(np.sum(av)), "pr_voro_area_mean": float(np.mean(av)), "pr_voro_area_max": float(np.max(av)), "pr_voro_area_protein": float(np.sum(av) / max(1, len(set(x[1] for x in areas)))), "pr_voro_area_rna": float(np.sum(av) / max(1, len(set(x[2] for x in areas)))), "pr_voro_area_balanced": float(np.sum(av) / max(1, len(set(x[1] for x in areas)) + len(set(x[2] for x in areas)))), "pr_voro_weighted_contacts": float(len(areas))})
    out.update(_residue_graph_features(a, b, pt))
    g = _directed_graph(a, b, pt, 6.0)
    pcounts = _path_counts(g, 3); cross_nodes = set(x.residue_key for x in a) | set(x.residue_key for x in b)
    cross_edges = sum(1 for x, ys in g.items() for y in ys if x[0] != y[0]); backbone_edges = sum(1 for x, ys in g.items() for y in ys if x[0] == y[0])
    out.update({"pr_path_edges": float(sum(len(x) for x in g.values())), "pr_path_paths1": pcounts[0], "pr_path_paths2": pcounts[1], "pr_path_paths3": pcounts[2], "pr_path_backbone_edges": float(backbone_edges), "pr_path_cross_edges": float(cross_edges), "pr_path_cross_path2": float(sum(1 for x, ys in g.items() for y in ys for z in g.get(y, ()) if x[0] != z[0])), "pr_path_cross_path3": float(sum(1 for x, ys in g.items() for y in ys for z in g.get(y, ()) for w in g.get(z, ()) if x[0] != w[0]))})
    xyz = _all_interface_xyz(a, b, pt, 6.0); thresholds = (2, 3, 4, 5, 6, 8, 10); cvals = []
    for t in thresholds:
        c = _components_at(xyz, float(t)); cvals.append(float(c)); out["pr_ph_b0_t" + str(t)] = float(c)
    if cvals:
        out.update({"pr_ph_b0_auc": float(np.trapz(cvals, thresholds)), "pr_ph_b0_initial": cvals[0], "pr_ph_b0_final": cvals[-1], "pr_ph_b0_persistence": float(max(cvals) - min(cvals)), "pr_ph_interface_atoms": float(len(xyz))})
    glmy_nonempty = 0; euler = []
    for t in (3, 4, 5, 6, 8):
        gt = _directed_graph(a, b, pt, float(t)); pc = _path_counts(gt, 3); glmy_nonempty += int(any(pc)); euler.append(float(len(gt) - pc[0] + pc[1] - pc[2]))
        for k in (1, 2, 3): out["pr_glmy_p%d_t%s" % (k, t)] = pc[k - 1]
    out.update({"pr_glmy_euler_mean": float(np.mean(euler)), "pr_glmy_nodes": float(len(g)), "pr_glmy_edges": float(sum(len(x) for x in g.values())), "pr_glmy_nonempty_thresholds": float(glmy_nonempty)})
    return out


PR_FEATURE_GROUPS = {
    "contact_profile": [k for k in _base_dict() if k.startswith("pr_contact_")],
    "atom_pair_rdf": [k for k in _base_dict() if k.startswith("pr_rdf_")],
    "residue_nucleotide_matrix": [k for k in _base_dict() if k.startswith("pr_resnu_")],
    "hydrogen_bond_geometry": [k for k in _base_dict() if k.startswith("pr_hbond_")],
    "base_edge_recognition": [k for k in _base_dict() if k.startswith("pr_baseedge_")],
    "aromatic_stacking": [k for k in _base_dict() if k.startswith("pr_stack_")],
    "sasa_bsa": [k for k in _base_dict() if k.startswith("pr_sasa_")],
    "electrostatic": [k for k in _base_dict() if k.startswith("pr_electro_")],
    "vdw_lj": [k for k in _base_dict() if k.startswith("pr_vdw_")],
    "shape_complementarity": [k for k in _base_dict() if k.startswith("pr_shape_")],
    "voronoi_contact_areas": [k for k in _base_dict() if k.startswith("pr_voro_")],
    "bipartite_graph": [k for k in _base_dict() if k.startswith("pr_bip_")],
    "directed_path_motifs": [k for k in _base_dict() if k.startswith("pr_path_")],
    "spatial_persistent_homology": [k for k in _base_dict() if k.startswith("pr_ph_")],
    "ordinary_glmy": [k for k in _base_dict() if k.startswith("pr_glmy_")],
}


def feature_group_vector(features: Mapping[str, float], group: str) -> Dict[str, float]:
    """Return a fixed group vector; useful to write one Parquet row per group."""
    if group not in PR_FEATURE_GROUPS:
        raise KeyError("unknown Protein-RNA feature group: %s" % group)
    return {name: float(features.get(name, 0.0)) for name in PR_FEATURE_GROUPS[group]}


__all__ = ["PR_FEATURE_GROUPS", "extract_pr_features", "feature_group_vector"]
