"""Target-level immutable/shared cache for rigid-body decoys.

The cache deliberately stores only coordinate-independent objects (atom
identity, covalent connectivity, site-transfer adjacency and path templates)
plus validated candidate warm starts.  Candidate coordinates are never treated
as immutable.  A rigid transform is accepted only after an explicit Kabsch
fit/residual check; arbitrary flexible candidates therefore take the safe
rebuild path.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Sequence, Mapping, Optional, Tuple

import numpy as np

from cspph.core.gsph1 import _build_bond_graph, _build_local_index, _roles, _base_class
from cspph.chemistry.patch_builder import _site_key


def _rigid_fit(native_xyz: np.ndarray, candidate_xyz: np.ndarray,
               tolerance: float = 1e-5) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Return ``(R, t, rmsd)`` for candidate ~= R @ native + t.

    The fit is deterministic and rejects reflections, rank-deficient fits and
    any non-rigid deformation above ``tolerance``.  It is intentionally kept
    in the cache module so every reuse site shares the same acceptance rule.
    """
    x = np.asarray(native_xyz, dtype=float)
    y = np.asarray(candidate_xyz, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        return None
    if len(x) == 0:
        return np.eye(3), np.zeros(3), 0.0
    if len(x) == 1:
        return np.eye(3), y[0] - x[0], 0.0
    xc = x.mean(axis=0); yc = y.mean(axis=0)
    x0 = x - xc; y0 = y - yc
    # Kabsch with column-vector convention y ~= R x + t.
    covariance = x0.T @ y0
    u, singular, vt = np.linalg.svd(covariance)
    if singular.size < 2 or float(singular[0]) <= 1e-12:
        # Collinear/degenerate sets cannot identify a unique rotation.  Only
        # accept a pure translation when all displacement vectors agree.
        delta = y - x
        if np.max(np.linalg.norm(delta - delta[0], axis=1)) <= tolerance:
            return np.eye(3), delta[0], float(np.sqrt(np.mean(np.sum((x + delta[0] - y) ** 2, axis=1))))
        return None
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = yc - r @ xc
    residual = np.asarray(x @ r.T + t - y, dtype=float)
    rmsd = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if not np.isfinite(rmsd) or rmsd > float(tolerance):
        return None
    return np.asarray(r, dtype=float), np.asarray(t, dtype=float), rmsd


class PatchCatalog:
    """Stable target-level IDs for functional-site pair patches.

    The historical implementation materialized the full donor-site x
    acceptor-site Cartesian product and sorted it in memory.  Large complexes
    can exceed the worker memory limit before a single candidate is computed.
    IDs below reproduce exactly that sorted Cartesian rank using per-side site
    order maps, so no pair list/set is allocated.
    """

    def __init__(self, donor_keys, acceptor_keys):
        self._donor_keys = tuple(sorted(set(map(tuple, donor_keys))))
        self._acceptor_keys = tuple(sorted(set(map(tuple, acceptor_keys))))
        self._acceptor_rank = {
            key: index for index, key in enumerate(self._acceptor_keys)}
        acceptor_chain_count = defaultdict(int)
        acceptor_chain_seen = defaultdict(int)
        self._acceptor_chain_pos = {}
        for key in self._acceptor_keys:
            chain = key[0]
            self._acceptor_chain_pos[key] = acceptor_chain_seen[chain]
            acceptor_chain_seen[chain] += 1
            acceptor_chain_count[chain] += 1
        self._donor_prefix = {}
        prefix = 0
        for key in self._donor_keys:
            self._donor_prefix[key] = prefix
            prefix += len(self._acceptor_keys) - acceptor_chain_count[key[0]]
        self._catalog_size = prefix
        self._extra_ids = {}

    def id_for_key(self, key):
        donor, acceptor = tuple(key[0]), tuple(key[1])
        if (donor in self._donor_prefix and acceptor in self._acceptor_rank
                and donor[0] != acceptor[0]):
            return int(self._donor_prefix[donor] + self._acceptor_rank[acceptor]
                       - self._acceptor_chain_pos[acceptor])
        # Rigid candidates should not introduce new sites.  If a non-rigid
        # input ever does, assign deterministic overflow IDs without changing
        # the rank of any native catalog pair.
        key = (donor, acceptor)
        if key not in self._extra_ids:
            self._extra_ids[key] = self._catalog_size + len(self._extra_ids)
        return int(self._extra_ids[key])

    def __len__(self):
        return self._catalog_size + len(self._extra_ids)

    @property
    def keys(self):
        native = ((d, a) for d in self._donor_keys
                  for a in self._acceptor_keys if d[0] != a[0])
        return tuple(native) + tuple(self._extra_ids)


@dataclass
class TargetGeometryCache:
    atoms: Sequence[object]
    atom_signature: Tuple[Tuple[object, ...], ...]
    bond_graph: Dict[int, set]
    local_index: Dict[str, object]
    atom_count: int
    local_index_is_candidate: bool = False
    patch_catalog: Optional[PatchCatalog] = None
    site_transfer_catalog: Optional[Mapping[Tuple[Tuple[str, int, str], Tuple[str, int, str]], Mapping[str, float]]] = None
    site_transfer_distance: Optional[int] = None
    path_template_cache: Dict[Tuple[Tuple[int, ...], Tuple[Tuple[int, int, float], ...], int, bool], object] = field(default_factory=dict)
    last_path_state: Optional[Tuple[Tuple[int, ...], Tuple[Tuple[int, int, float], ...], object]] = None
    chain_template_cache: Dict[Tuple[Tuple[Tuple[int, ...], ...], ...], object] = field(default_factory=dict)
    hodge_spectrum_cache: Dict[Tuple[Tuple[Tuple[int, ...], ...], ...], Dict[int, object]] = field(default_factory=dict)
    persistence_cache: Dict[Tuple[Tuple[Tuple[int, ...], ...], ...], object] = field(default_factory=dict)
    max_hodge_spectra: int = 64
    max_persistence_maps: int = 64
    max_path_templates: int = 128
    max_chain_templates: int = 128
    max_support_results: int = 8192
    previous_support: Dict[Tuple[int, ...], object] = field(default_factory=dict)
    reuse_stats: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_atoms(cls, atoms: Sequence[object], structure_path: Optional[str] = None) -> "TargetGeometryCache":
        atoms = tuple(atoms)
        cache = cls(atoms=atoms, atom_signature=cls._atom_signature(atoms),
                    bond_graph=_build_bond_graph(atoms, path=structure_path or None),
                    local_index=_build_local_index(atoms), atom_count=len(atoms), local_index_is_candidate=False,
                    path_template_cache={}, previous_support={}, reuse_stats={})
        donor_keys, acceptor_keys = cache._candidate_site_keys(atoms)
        cache.patch_catalog = PatchCatalog(donor_keys, acceptor_keys)
        return cache

    @staticmethod
    def _atom_signature(atoms: Sequence[object]):
        return tuple(sorted((int(getattr(atom, "index", -1)), str(getattr(atom, "chain", "")),
                             str(getattr(atom, "name", "")), str(getattr(atom, "resname", "")),
                             int(getattr(atom, "resid", -1)), str(getattr(atom, "element", "")),
                             int(getattr(atom, "serial", 0)), str(getattr(atom, "hetero", "")),
                             str(getattr(atom, "icode", ""))) for atom in atoms))

    def identity_matches(self, atoms: Sequence[object]) -> bool:
        matches = self._atom_signature(atoms) == self.atom_signature
        if not matches:
            self.reuse_stats["identity_mismatches"] = int(self.reuse_stats.get("identity_mismatches", 0)) + 1
        return bool(matches)

    @staticmethod
    def _candidate_site_keys(atoms: Sequence[object]):
        donors = set(); acceptors = set()
        for atom in atoms:
            donor, acceptor = _roles(atom)
            base = _base_class(str(getattr(atom, "resname", "")))
            if donor:
                donors.add(_site_key(atom, "donor", base))
            if acceptor:
                acceptors.add(_site_key(atom, "acceptor", base))
        return donors, acceptors

    def build_site_transfer_catalog(self, max_distance: int = 3):
        """Build site-level covalent transfer witnesses once per target."""
        if self.site_transfer_catalog is not None and self.site_transfer_distance == int(max_distance):
            self.reuse_stats["site_transfer_hits"] = int(self.reuse_stats.get("site_transfer_hits", 0)) + 1
            return self.site_transfer_catalog
        atoms = {int(atom.index): atom for atom in self.atoms}
        site_atoms = {}
        donor_sites = set(); acceptor_sites = set()
        for atom in self.atoms:
            donor, acceptor = _roles(atom)
            base = _base_class(str(getattr(atom, "resname", "")))
            if donor:
                key = _site_key(atom, "donor", base); donor_sites.add(key); site_atoms.setdefault(key, []).append(int(atom.index))
            if acceptor:
                key = _site_key(atom, "acceptor", base); acceptor_sites.add(key); site_atoms.setdefault(key, []).append(int(atom.index))
        graph = self.bond_graph
        catalog = {}
        # One bounded BFS per acceptor atom replaces repeated patch-pair BFS.
        for acceptor_site in sorted(acceptor_sites):
            distances = {}
            for source in site_atoms.get(acceptor_site, ()):
                seen = {int(source)}; queue = deque([(int(source), 0)])
                while queue:
                    node, depth = queue.popleft()
                    if depth >= int(max_distance):
                        continue
                    for nxt in graph.get(node, ()):
                        nxt = int(nxt)
                        if nxt in seen:
                            continue
                        seen.add(nxt); queue.append((nxt, depth + 1))
                        distances[nxt] = min(int(depth + 1), int(distances.get(nxt, 10**9)))
            for donor_site in sorted(donor_sites):
                values = [distances[atom_id] for atom_id in site_atoms.get(donor_site, ()) if atom_id in distances and distances[atom_id] > 0]
                if values:
                    catalog[(acceptor_site, donor_site)] = {
                        "witness_count": float(len(values)),
                        "min_bond_distance": float(min(values)),
                        "mean_bond_distance": float(np.mean(values)),
                    }
        self.site_transfer_catalog = catalog
        self.site_transfer_distance = int(max_distance)
        self.reuse_stats["site_transfer_builds"] = int(self.reuse_stats.get("site_transfer_builds", 0)) + 1
        return catalog

    def cached_paths(self, patch_ids, edges, max_order: int, use_cpp: bool):
        key = (tuple(sorted(int(x) for x in patch_ids)), tuple(sorted((int(a), int(b), float(w)) for a, b, w in edges)), int(max_order), bool(use_cpp))
        value = self.path_template_cache.get(key)
        if value is not None:
            self.reuse_stats["path_template_hits"] = int(self.reuse_stats.get("path_template_hits", 0)) + 1
            return value
        self.reuse_stats["path_template_misses"] = int(self.reuse_stats.get("path_template_misses", 0)) + 1
        return None

    def store_paths(self, patch_ids, edges, max_order: int, use_cpp: bool, value):
        key = (tuple(sorted(int(x) for x in patch_ids)), tuple(sorted((int(a), int(b), float(w)) for a, b, w in edges)), int(max_order), bool(use_cpp))
        if key not in self.path_template_cache and len(self.path_template_cache) >= int(self.max_path_templates):
            self.path_template_cache.pop(next(iter(self.path_template_cache)))
        self.path_template_cache[key] = value

    def incremental_paths(self, patch_ids, edges, max_order: int = 3):
        """Return a small-delta path update or ``None`` for full rebuild."""
        from cspph.graph.patch_digraph import incremental_patch_paths
        current_ids = tuple(sorted(int(x) for x in patch_ids))
        current_edges = tuple(sorted((int(a), int(b), float(w)) for a, b, w in edges))
        if self.last_path_state is None:
            return None
        previous_ids, previous_edges, previous_value = self.last_path_state
        value = incremental_patch_paths(previous_value[0], previous_ids, previous_edges, current_ids, current_edges, max_order=max_order)
        if value is not None:
            self.reuse_stats["incremental_path_updates"] = int(self.reuse_stats.get("incremental_path_updates", 0)) + 1
        return value

    def remember_paths(self, patch_ids, edges, value):
        self.last_path_state = (tuple(sorted(int(x) for x in patch_ids)), tuple(sorted((int(a), int(b), float(w)) for a, b, w in edges)), value)

    def cached_chain(self, active):
        key = tuple(tuple(sorted(tuple(int(x) for x in path) for path in active.get(order, ()))) for order in range(4))
        value = self.chain_template_cache.get(key)
        if value is not None:
            self.reuse_stats["glmy_structure_hits"] = int(self.reuse_stats.get("glmy_structure_hits", 0)) + 1
        return value

    def remember_chain(self, active, chain):
        key = tuple(tuple(sorted(tuple(int(x) for x in path) for path in active.get(order, ()))) for order in range(4))
        if key not in self.chain_template_cache and len(self.chain_template_cache) >= int(self.max_chain_templates):
            self.chain_template_cache.pop(next(iter(self.chain_template_cache)))
        self.chain_template_cache[key] = chain

    def cached_hodge(self, key):
        return self.hodge_spectrum_cache.get(key)

    def remember_hodge(self, key, spectra):
        if key not in self.hodge_spectrum_cache and len(self.hodge_spectrum_cache) >= int(self.max_hodge_spectra):
            self.hodge_spectrum_cache.pop(next(iter(self.hodge_spectrum_cache)))
        self.hodge_spectrum_cache[key] = spectra

    def cached_persistence(self, key):
        return self.persistence_cache.get(key)

    def remember_persistence(self, key, value):
        if key not in self.persistence_cache and len(self.persistence_cache) >= int(self.max_persistence_maps):
            self.persistence_cache.pop(next(iter(self.persistence_cache)))
        self.persistence_cache[key] = value

    def warm_support(self, key):
        return self.previous_support.get(tuple(sorted(int(x) for x in key)))

    def remember_support(self, key, value):
        cache_key = tuple(sorted(int(x) for x in key))
        if cache_key not in self.previous_support and len(self.previous_support) >= int(self.max_support_results):
            self.previous_support.pop(next(iter(self.previous_support)))
        self.previous_support[cache_key] = value

    def note_active_patches(self, patch_ids):
        current = frozenset(int(x) for x in patch_ids)
        previous = getattr(self, "_previous_active_patches", None)
        if previous is not None:
            union = len(previous | current)
            overlap = len(previous & current)
            self.reuse_stats["patch_overlap_sum"] = float(self.reuse_stats.get("patch_overlap_sum", 0.0)) + (overlap / union if union else 1.0)
            self.reuse_stats["patch_overlap_observations"] = int(self.reuse_stats.get("patch_overlap_observations", 0)) + 1
        self._previous_active_patches = current

    def local_index_for_candidate(self, atoms: Sequence[object]) -> Dict[str, object]:
        """Reuse per-chain KD-trees for rigid-body-static chains.

        Cross-chain distances and moved-chain coordinates are candidate-specific,
        so they are never reused.  A chain whose coordinates are byte-for-byte
        unchanged can safely share the immutable native index; only changed
        chains pay the local-index construction cost.
        """
        if not self.identity_matches(atoms):
            self.reuse_stats["identity_rebuilds"] = int(self.reuse_stats.get("identity_rebuilds", 0)) + 1
            return _build_local_index(atoms)
        grouped = defaultdict(list)
        for atom in atoms:
            grouped[str(atom.chain)].append(atom)
        result = {}
        for chain, rows in grouped.items():
            cached = self.local_index.get(chain)
            if cached is not None:
                native_rows = cached[0]
                same = len(native_rows) == len(rows) and all(
                    int(getattr(old, "index", -1)) == int(getattr(new, "index", -2))
                    and np.array_equal(np.asarray(old.xyz, dtype=float), np.asarray(new.xyz, dtype=float))
                    for old, new in zip(native_rows, rows)
                )
                if same:
                    result[chain] = cached
                    continue
                fit = _rigid_fit(
                    np.asarray([old.xyz for old in native_rows], dtype=float),
                    np.asarray([new.xyz for new in rows], dtype=float),
                )
                if fit is not None:
                    rotation, translation, _ = fit
                    # Keep the native tree and native coordinates.  gsph1's
                    # query helper understands this optional transform and
                    # transforms query points/neighbor coordinates lazily.
                    result[chain] = (cached[0], cached[1], cached[2], rotation, translation)
                    self.reuse_stats["rigid_kdtree_reuses"] = int(self.reuse_stats.get("rigid_kdtree_reuses", 0)) + 1
                    continue
                self.reuse_stats["flexible_kdtree_rebuilds"] = int(self.reuse_stats.get("flexible_kdtree_rebuilds", 0)) + 1
            result.update(_build_local_index(rows))
        return result


__all__ = ["TargetGeometryCache"]
