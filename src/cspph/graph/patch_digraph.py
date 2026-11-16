"""Patch transfer graph and compact CSR path interface."""
from __future__ import annotations

from collections import Counter, deque
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from cspph.chemistry.patch_builder import InteractionPatch


def _bond_distance(graph: Mapping[int, Iterable[int]], source: int, target: int, limit: int) -> int | None:
    if source == target: return 0
    seen = {source}; q = deque([(source, 0)])
    while q:
        node, dist = q.popleft()
        if dist >= limit: continue
        for nxt in graph.get(node, ()):
            if nxt == target: return dist + 1
            if nxt not in seen:
                seen.add(nxt); q.append((nxt, dist + 1))
    return None


def build_patch_transfer_graph(patches: Sequence[InteractionPatch], bond_graph: Mapping[int, Iterable[int]], max_distance: int = 3,
                              site_transfer_catalog: Mapping[Tuple[Tuple[str, int, str], Tuple[str, int, str]], Mapping[str, float]] | None = None) -> Tuple[List[Tuple[int, int, float]], Dict[Tuple[int, int], Dict[str, float]]]:
    edges: List[Tuple[int, int, float]] = []; witnesses: Dict[Tuple[int, int], Dict[str, float]] = {}
    for left in patches:
        for right in patches:
            if left.patch_id == right.patch_id: continue
            if site_transfer_catalog is not None:
                witness = site_transfer_catalog.get((left.acceptor_site, right.donor_site))
                if witness is not None:
                    key = (left.patch_id, right.patch_id)
                    witnesses[key] = dict(witness)
                    edges.append((left.patch_id, right.patch_id, float(witness["min_bond_distance"])))
                continue
            lengths: List[int] = []; count = 0
            for acceptor in left.acceptor_members:
                for donor in right.donor_members:
                    distance = _bond_distance(bond_graph, int(acceptor), int(donor), int(max_distance))
                    if distance is not None and distance > 0:
                        lengths.append(distance); count += 1
            if lengths:
                key = (left.patch_id, right.patch_id); minimum = min(lengths)
                witnesses[key] = {"witness_count": float(count), "min_bond_distance": float(minimum), "mean_bond_distance": float(np.mean(lengths))}
                edges.append((left.patch_id, right.patch_id, float(minimum)))
    return sorted(set(edges)), witnesses


def enumerate_patch_paths(patch_ids: Sequence[int], edges: Sequence[Tuple[int, int, float]], max_order: int = 3, use_cpp: bool = True) -> Tuple[Dict[int, np.ndarray], Dict[int, int], str]:
    """Use the existing C++17 CSR engine when enabled, with exact CSR fallback."""
    from cspph.graph.path_enumeration import enumerate_csr_paths, path_engine_status
    if use_cpp:
        import os
        old = os.environ.get("GSPH_ENABLE_CPP_PATH")
        os.environ["GSPH_ENABLE_CPP_PATH"] = "1"
        try:
            paths, counts = enumerate_csr_paths(patch_ids, edges, max_order=max_order)
            backend, error = path_engine_status()
            return paths, counts, backend if not error else f"{backend}:{error}"
        finally:
            if old is None: os.environ.pop("GSPH_ENABLE_CPP_PATH", None)
            else: os.environ["GSPH_ENABLE_CPP_PATH"] = old
    import os
    old = os.environ.get("GSPH_ENABLE_CPP_PATH")
    os.environ["GSPH_ENABLE_CPP_PATH"] = "0"
    try:
        paths, counts = enumerate_csr_paths(patch_ids, edges, max_order=max_order)
    finally:
        if old is None: os.environ.pop("GSPH_ENABLE_CPP_PATH", None)
        else: os.environ["GSPH_ENABLE_CPP_PATH"] = old
    backend, error = path_engine_status()
    return paths, counts, backend if not error else f"{backend}:{error}"


def incremental_patch_paths(previous_paths: Mapping[int, np.ndarray], previous_patch_ids: Sequence[int],
                            previous_edges: Sequence[Tuple[int, int, float]], patch_ids: Sequence[int],
                            edges: Sequence[Tuple[int, int, float]], max_order: int = 3,
                            max_delta_fraction: float = 0.35) -> Tuple[Dict[int, np.ndarray], Dict[int, int], str] | None:
    """Update a path template when only a small active graph delta changed.

    Paths not touching an added/removed node or changed edge are retained.  New
    paths are generated from every node that can reach the affected region in
    at most ``max_order`` steps, which is sufficient to cover all newly
    possible paths.  ``None`` requests a full CSR rebuild when the delta is too
    large.  The routine is exact; it is a structural cache, never a numerical
    approximation.
    """
    old_nodes = set(int(x) for x in previous_patch_ids); new_nodes = set(int(x) for x in patch_ids)
    old_edge_set = {(int(a), int(b)) for a, b, _ in previous_edges}
    new_edge_set = {(int(a), int(b)) for a, b, _ in edges}
    changed_edges = old_edge_set ^ new_edge_set
    affected = (old_nodes ^ new_nodes) | {node for edge in changed_edges for node in edge}
    denominator = max(1, len(old_nodes | new_nodes) + len(old_edge_set | new_edge_set))
    if len(affected) + len(changed_edges) > float(max_delta_fraction) * denominator:
        return None
    if not affected and old_nodes == new_nodes and old_edge_set == new_edge_set:
        return {int(k): np.array(v, copy=True) for k, v in previous_paths.items()}, {int(k): int(v.shape[0]) for k, v in previous_paths.items()}, "cached_exact"
    adjacency = {int(node): [] for node in new_nodes}
    reverse = {int(node): [] for node in new_nodes}
    for source, target, _ in edges:
        source = int(source); target = int(target)
        if source in new_nodes and target in new_nodes and source != target:
            adjacency.setdefault(source, []).append(target); reverse.setdefault(target, []).append(source)
    adjacency = {node: tuple(sorted(set(values))) for node, values in adjacency.items()}
    reverse = {node: tuple(sorted(set(values))) for node, values in reverse.items()}
    # Any path that can be newly created must start within max_order steps of
    # an affected node in the reverse graph.
    starts = set(affected) & new_nodes
    frontier = set(starts)
    for _ in range(int(max_order)):
        frontier = {parent for node in frontier for parent in reverse.get(node, ())}
        starts.update(frontier)
    edge_set = new_edge_set
    retained = {order: [] for order in range(int(max_order) + 1)}
    for order, rows in previous_paths.items():
        for row in np.asarray(rows, dtype=np.int32):
            path = tuple(int(x) for x in row.tolist())
            if all(node in new_nodes for node in path) and all((path[i], path[i + 1]) in edge_set for i in range(len(path) - 1)) and not any(node in affected for node in path):
                retained[int(order)].append(path)
    generated = {order: set(rows) for order, rows in retained.items()}
    for start in sorted(starts):
        generated[0].add((int(start),))
        frontier_paths = [(int(start),)]
        for order in range(1, int(max_order) + 1):
            next_frontier = []
            for path in frontier_paths:
                for target in adjacency.get(path[-1], ()):
                    if target == path[-1]:
                        continue
                    candidate = path + (int(target),)
                    if candidate not in generated[order]:
                        generated[order].add(candidate)
                    # A prefix may already have been retained from the old
                    # graph, but its descendants can still reach an affected
                    # edge/node and must be expanded.
                    next_frontier.append(candidate)
            frontier_paths = next_frontier
    output = {order: np.asarray(sorted(generated[order]), dtype=np.int32).reshape((-1, order + 1)) if generated[order] else np.empty((0, order + 1), dtype=np.int32) for order in range(int(max_order) + 1)}
    return output, {order: int(rows.shape[0]) for order, rows in output.items()}, "incremental_numpy_csr"


def summarize_path_tokens(path_arrays: Mapping[int, np.ndarray], patches: Mapping[int, InteractionPatch]) -> Dict[str, float]:
    """Compute fixed-dimensional chemistry/path tokens once per candidate.

    These tokens are independent of the filtration scale.  The scale loop only
    filters already-enumerated rows by birth time and never recomputes motif or
    direction statistics.
    """
    out: Dict[str, float] = {}
    for order in range(4):
        rows = path_arrays.get(order, np.empty((0, order + 1), dtype=np.int32))
        transitions: list[float] = []; role_changes: list[float] = []; span: list[float] = []
        role_counts: Counter[str] = Counter()
        for row in rows:
            ids = [int(x) for x in np.asarray(row).tolist()]
            if not ids: continue
            roles = [f"{patches[i].donor_site[2]}->{patches[i].acceptor_site[2]}" for i in ids]
            role_counts.update(roles)
            transitions.append(float(sum(a != b for a, b in zip(roles, roles[1:]))))
            role_changes.append(float(len(set(roles))))
            centers = np.asarray([[patches[i].geometry.get("center_x", 0.0), patches[i].geometry.get("center_y", 0.0), patches[i].geometry.get("center_z", 0.0)] for i in ids], dtype=float)
            span.append(float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1))) if len(centers) else 0.0)
        total = float(sum(role_counts.values()))
        probs = np.asarray([count / total for count in role_counts.values()], dtype=float) if total else np.zeros(0)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12))) if len(probs) else 0.0
        out[f"pathchem_count_k{order}"] = float(len(rows))
        out[f"pathchem_transition_mean_k{order}"] = float(np.mean(transitions)) if transitions else 0.0
        out[f"pathchem_transition_std_k{order}"] = float(np.std(transitions)) if transitions else 0.0
        out[f"pathchem_role_diversity_k{order}"] = float(np.mean(role_changes)) if role_changes else 0.0
        out[f"pathchem_role_entropy_k{order}"] = entropy
        out[f"path_geom_span_mean_k{order}"] = float(np.mean(span)) if span else 0.0
    return out


__all__ = ["build_patch_transfer_graph", "enumerate_patch_paths", "summarize_path_tokens"]
