"""Exact sparse GLMY regular-path operators for orders 0--3.

The module keeps the standard regular-path convention: an adjacent repeated
vertex is a zero face, while a regular face absent from ``A_{k-1}`` is a
forbidden face constraint.  For k=2 and k=3 the forbidden-face incidence has
columns of degree at most one and two respectively.  Its exact kernel is
constructed with grouped constraints and a spanning-forest fundamental-cycle
construction; no dense forbidden-face matrix or SVD is required.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import sparse
except Exception as exc:  # pragma: no cover
    sparse = None


PathTuple = Tuple[int, ...]


@dataclass
class SparseChain:
    paths: Dict[int, List[PathTuple]]
    omega_basis: Dict[int, "sparse.csr_matrix"]
    boundaries: Dict[int, "sparse.csr_matrix"]
    forbidden: Dict[int, "sparse.csr_matrix"]
    omega_dims: Dict[int, int]
    boundary_ranks: Dict[int, int]
    betti: Dict[int, int]
    boundary_square_residual: Dict[int, float]
    rank_primes: Tuple[int, ...]
    kernel_backend: str = "structured_sparse"


RANK_PRIMES = (1000003, 1000033, 1000037)
_LAST_RANK_BACKEND = "python_modular"
_KERNEL_BACKENDS: List[str] = []


def _require_sparse() -> None:
    if sparse is None:
        raise RuntimeError("scipy.sparse is required for sparse GLMY")


def _is_regular(path: PathTuple) -> bool:
    return all(path[i] != path[i + 1] for i in range(len(path) - 1))


def boundary_sparse(paths_k: Sequence[PathTuple], paths_prev: Sequence[PathTuple]):
    """Return sparse allowed boundary, forbidden incidence and face list."""
    _require_sparse()
    previous_index = {path: i for i, path in enumerate(paths_prev)}
    forbidden_index: Dict[PathTuple, int] = {}
    allowed_rows: List[int] = []; allowed_cols: List[int] = []; allowed_data: List[int] = []
    forbidden_rows: List[int] = []; forbidden_cols: List[int] = []; forbidden_data: List[int] = []
    forbidden_faces: List[PathTuple] = []
    for col, path in enumerate(paths_k):
        for i in range(len(path)):
            face = path[:i] + path[i + 1:]
            sign = -1 if i % 2 else 1
            # Non-regular faces are zero in the regular path complex.
            if not _is_regular(face):
                continue
            if face in previous_index:
                allowed_rows.append(previous_index[face]); allowed_cols.append(col); allowed_data.append(sign)
            else:
                row = forbidden_index.get(face)
                if row is None:
                    row = len(forbidden_faces); forbidden_index[face] = row; forbidden_faces.append(face)
                forbidden_rows.append(row); forbidden_cols.append(col); forbidden_data.append(sign)
    allowed = sparse.csr_matrix((allowed_data, (allowed_rows, allowed_cols)),
                                shape=(len(paths_prev), len(paths_k)), dtype=np.int64)
    forbidden = sparse.csr_matrix((forbidden_data, (forbidden_rows, forbidden_cols)),
                                  shape=(len(forbidden_faces), len(paths_k)), dtype=np.int64)
    return allowed, forbidden, forbidden_faces


def _connected_components(n_rows: int, columns: Sequence[Sequence[Tuple[int, int]]]):
    adjacency: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    constrained_rows = set()
    for col, entries in enumerate(columns):
        rows = [row for row, _ in entries]
        constrained_rows.update(rows)
        if len(rows) == 2 and rows[0] != rows[1]:
            adjacency[rows[0]].append((rows[1], col)); adjacency[rows[1]].append((rows[0], col))
    seen = set()
    components = []
    for row in range(n_rows):
        if row in seen or row not in adjacency:
            continue
        queue = [row]; seen.add(row); rows = []
        while queue:
            current = queue.pop(); rows.append(current)
            for other, _ in adjacency.get(current, []):
                if other not in seen:
                    seen.add(other); queue.append(other)
        components.append(sorted(rows))
    # Single-row constraints with no two-row edge are their own components.
    for row in sorted(constrained_rows):
        if row not in seen:
            components.append([row]); seen.add(row)
    return components, adjacency


def _solve_tree(rows: Sequence[int], pivot_columns: Sequence[int], columns: Sequence[Sequence[Tuple[int, int]]], rhs: Mapping[int, int], has_root: bool) -> Dict[int, int]:
    """Solve a tree incidence system by leaf elimination."""
    row_set = set(rows)
    work_rhs = {row: int(rhs.get(row, 0)) for row in rows}
    edge_rows: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for col in pivot_columns:
        for row, coeff in columns[col]:
            edge_rows[row].append((col, coeff))
    active_edges = set(pivot_columns)
    active_degree = {
        row: sum(1 for edge, _ in edge_rows[row] if edge in active_edges)
        for row in row_set
    }
    solution: Dict[int, int] = {}
    # A heap gives deterministic leaf elimination while avoiding the previous
    # O(|rows|*|edges|) rescan at every elimination step.
    import heapq
    leaf_queue = [row for row in row_set if active_degree[row] <= 1]
    heapq.heapify(leaf_queue)
    # Tree edges are guaranteed by the caller; repeatedly eliminate a leaf.
    while row_set:
        row = None
        while leaf_queue:
            candidate = heapq.heappop(leaf_queue)
            if candidate in row_set and active_degree[candidate] <= 1:
                row = candidate
                break
        if row is None:
            raise ValueError("pivot incidence graph is not a forest")
        incident = [(edge, coeff) for edge, coeff in edge_rows[row] if edge in active_edges]
        if not incident:
            if work_rhs[row] != 0:
                raise ValueError("inconsistent sparse GLMY kernel system")
            row_set.remove(row)
            continue
        edge, coeff = incident[0]
        value = int(work_rhs[row] // coeff)
        if coeff * value != work_rhs[row]:
            raise ValueError("non-integral sparse GLMY kernel solution")
        solution[edge] = value
        active_edges.remove(edge)
        work_rhs[row] = 0
        for other, other_coeff in columns[edge]:
            if other != row and other in row_set:
                work_rhs[other] -= other_coeff * value
                active_degree[other] -= 1
                if active_degree[other] <= 1:
                    heapq.heappush(leaf_queue, other)
        row_set.remove(row)
    if any(value != 0 for value in work_rhs.values()):
        raise ValueError("sparse GLMY kernel residual is nonzero")
    return solution


def _structured_kernel_impl(forbidden):
    """Construct an exact sparse basis for ker(forbidden)."""
    _require_sparse()
    forbidden = forbidden.tocsc()
    n_rows, n_cols = forbidden.shape
    columns: List[List[Tuple[int, int]]] = []
    for col in range(n_cols):
        start, end = forbidden.indptr[col], forbidden.indptr[col + 1]
        columns.append([(int(row), int(value)) for row, value in zip(forbidden.indices[start:end], forbidden.data[start:end])])
    components, adjacency = _connected_components(n_rows, columns)
    row_to_component = {row: component for component in components for row in component}
    component_columns: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for col, entries in enumerate(columns):
        if entries:
            component_columns[tuple(row_to_component[entries[0][0]])].append(col)
    basis_vectors: List[Dict[int, int]] = []
    used = set()
    # Columns without constraints are independent kernel vectors.
    for col, entries in enumerate(columns):
        if not entries:
            basis_vectors.append({col: 1}); used.add(col)
    for component in components:
        key = tuple(component)
        cols = component_columns.get(key, [])
        if not cols:
            continue
        row_set = set(component)
        # Build a spanning forest over forbidden rows using two-row columns.
        parent = {row: row for row in component}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        tree_cols: List[int] = []
        for col in cols:
            entries = columns[col]
            if len(entries) != 2:
                continue
            u, v = entries[0][0], entries[1][0]
            if find(u) != find(v):
                parent[find(u)] = find(v); tree_cols.append(col)
        one_row_cols = [col for col in cols if len(columns[col]) == 1]
        root_col = one_row_cols[0] if one_row_cols else None
        pivots = list(tree_cols)
        if root_col is not None:
            pivots.append(root_col)
        pivot_set = set(pivots)
        # Fundamental cycles / single-row differences give the full kernel.
        for col in cols:
            if col in pivot_set:
                continue
            rhs = {row: -coeff for row, coeff in columns[col]}
            solved = _solve_tree(component, pivots, columns, rhs, root_col is not None)
            vector = {col: 1}
            vector.update({pivot: value for pivot, value in solved.items() if value})
            basis_vectors.append(vector)
            used.add(col)
    # A defensive check catches malformed incidence patterns rather than
    # silently returning a deficient basis.
    basis = sparse.lil_matrix((n_cols, len(basis_vectors)), dtype=np.int64)
    for basis_col, vector in enumerate(basis_vectors):
        for row, value in vector.items():
            basis[row, basis_col] = value
    basis = basis.tocsr()
    residual = forbidden @ basis
    if residual.nnz and np.any(residual.data != 0):
        raise ValueError("structured GLMY kernel residual is nonzero")
    expected_dim = n_cols - _sparse_rank_verified(forbidden)
    if basis.shape[1] != expected_dim:
        raise ValueError(f"structured GLMY kernel dimension mismatch: {basis.shape[1]} != {expected_dim}")
    return basis


def _dense_kernel_fallback(forbidden):
    """Dense SVD kernel for rare non-forest incidence patterns.

    The structured elimination is preferred because it is exact and sparse.
    When a real forbidden-face graph violates that specialization, a bounded
    dense SVD gives the same mathematical Omega basis and is explicitly
    recorded in ``SparseChain.kernel_backend`` instead of being hidden.
    """
    import os
    rows, cols = forbidden.shape
    limit = int(os.environ.get("GSPH_GLMY_DENSE_MAX_ELEMENTS", "4000000"))
    if rows * cols > limit:
        raise ValueError("dense GLMY kernel fallback dimension limit exceeded")
    dense = forbidden.toarray().astype(float)
    if dense.size:
        _, singular, vh = np.linalg.svd(dense, full_matrices=True)
        tol = 1e-10 * max(1.0, float(singular[0]) if len(singular) else 1.0)
        rank = int(np.sum(singular > tol))
        basis = vh[rank:].T
    else:
        basis = np.eye(cols)
    result = sparse.csr_matrix(basis)
    residual = forbidden @ result
    if residual.nnz and float(np.max(np.abs(residual.data))) > 1e-7:
        raise ValueError("dense GLMY kernel residual is nonzero")
    expected_dim = cols - _sparse_rank_verified(forbidden)
    if result.shape[1] != expected_dim:
        raise ValueError(f"dense GLMY kernel dimension mismatch: {result.shape[1]} != {expected_dim}")
    return result


def structured_kernel(forbidden):
    """Return an Omega basis; sparse construction first, dense only when safe."""
    try:
        _KERNEL_BACKENDS.append("structured_sparse")
        return _structured_kernel_impl(forbidden)
    except ValueError:
        _KERNEL_BACKENDS[-1] = "dense_svd_fallback"
        return _dense_kernel_fallback(forbidden)


def _sparse_rank_mod(matrix, modulus: int = 1000003) -> int:
    """Sparse exact rank over a prime field for integer incidence matrices."""
    global _LAST_RANK_BACKEND
    matrix = matrix.tocsr()
    if matrix.nnz >= 1024:
        try:
            import os
            if os.environ.get("GSPH_ENABLE_CPP_GLMY", "0") == "1":
                import torch
                from cspph.topology.glmy_rank_backend import load_glmy_rank_engine
                indptr = torch.as_tensor(matrix.indptr.astype(np.int64, copy=False), dtype=torch.int64)
                indices = torch.as_tensor(matrix.indices.astype(np.int64, copy=False), dtype=torch.int64)
                data = torch.as_tensor(matrix.data.astype(np.int64, copy=False), dtype=torch.int64)
                result = int(load_glmy_rank_engine().sparse_rank_mod(
                    indptr, indices, data, int(matrix.shape[0]), int(matrix.shape[1]), int(modulus)))
                _LAST_RANK_BACKEND = "cpp_sparse_modular"
                return result
        except Exception:
            # The Python modular eliminator remains exact and is retained as a
            # transparent fallback when the optional compiler is unavailable.
            pass
    rows: List[Dict[int, int]] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        rows.append({int(col): int(value) % modulus for col, value in zip(matrix.indices[start:end], matrix.data[start:end]) if int(value) % modulus})
    pivots: Dict[int, Dict[int, int]] = {}
    rank = 0
    for row in rows:
        while row:
            pivot = min(row)
            value = row[pivot] % modulus
            if pivot not in pivots:
                inv = pow(value, modulus - 2, modulus)
                normalized = {col: (entry * inv) % modulus for col, entry in row.items()}
                pivots[pivot] = normalized; rank += 1; break
            factor = value
            base = pivots[pivot]
            for col, entry in base.items():
                updated = (row.get(col, 0) - factor * entry) % modulus
                if updated:
                    row[col] = updated
                elif col in row:
                    del row[col]
    return rank


def rank_backend_status() -> str:
    return _LAST_RANK_BACKEND


def _sparse_rank_verified(matrix) -> int:
    """Return a modular rank only when independent primes agree."""
    ranks = [_sparse_rank_mod(matrix, modulus=prime) for prime in RANK_PRIMES]
    if len(set(ranks)) != 1:
        raise ValueError(f"modular sparse-rank disagreement: {ranks}")
    return ranks[0]


def sparse_chain(active: Mapping[int, Sequence[PathTuple]]) -> SparseChain:
    """Build exact sparse Omega/boundary operators and H0--H2 Betti numbers."""
    _require_sparse()
    _KERNEL_BACKENDS.clear()
    paths = {order: list(active.get(order, [])) for order in range(4)}
    bases: Dict[int, sparse.csr_matrix] = {0: sparse.identity(len(paths[0]), dtype=np.int64, format="csr")}
    boundaries: Dict[int, sparse.csr_matrix] = {0: sparse.csr_matrix((0, len(paths[0])), dtype=np.int64)}
    forbidden: Dict[int, sparse.csr_matrix] = {}
    for order in (1, 2, 3):
        allowed, bad, _ = boundary_sparse(paths[order], paths[order - 1])
        boundaries[order] = allowed
        forbidden[order] = bad
        bases[order] = structured_kernel(bad)
    omega_dims = {order: int(bases[order].shape[1]) for order in range(4)}
    boundary_ranks: Dict[int, int] = {}
    projected = {}
    import os
    dense_rank_limit = int(os.environ.get("GSPH_DENSE_RANK_MAX_ELEMENTS", "4000000"))
    for order in range(4):
        image = boundaries[order] @ bases[order]
        projected[order] = image
        if image.dtype.kind == "f":
            # Dense-SVD fallback bases are floating point: modular rank over
            # a prime field is undefined for them (int() truncation would
            # silently corrupt the rank).  Use a real-arithmetic rank instead.
            rows, cols = image.shape
            if rows * cols > dense_rank_limit:
                raise ValueError("floating Omega image too large for dense rank")
            dense = image.toarray() if image.nnz or rows * cols else np.zeros((rows, cols))
            boundary_ranks[order] = int(np.linalg.matrix_rank(dense)) if dense.size else 0
        else:
            boundary_ranks[order] = _sparse_rank_verified(image)
    betti = {}
    for order in range(3):
        betti[order] = omega_dims[order] - boundary_ranks[order] - boundary_ranks[order + 1]
    residuals = {}
    for order in (1, 2, 3):
        residuals[order] = float((boundaries[order - 1] @ boundaries[order] @ bases[order]).nnz)
    kernel_backend = "+".join(sorted(set(_KERNEL_BACKENDS))) or "structured_sparse"
    return SparseChain(paths, bases, boundaries, forbidden, omega_dims, boundary_ranks, betti, residuals, RANK_PRIMES, kernel_backend)


def dense_reference(active: Mapping[int, Sequence[PathTuple]]) -> Dict[str, object]:
    """Small-matrix reference used only for P0 equivalence tests."""
    paths = {order: list(active.get(order, [])) for order in range(4)}
    bases = {0: np.eye(len(paths[0]))}
    boundaries = {0: np.zeros((0, len(paths[0])))}
    for order in (1, 2, 3):
        allowed, bad, _ = boundary_sparse(paths[order], paths[order - 1])
        bad_dense = bad.toarray().astype(float)
        if bad_dense.size:
            _, singular, vh = np.linalg.svd(bad_dense, full_matrices=True)
            tol = 1e-10 * max(1.0, float(singular[0]) if len(singular) else 1.0)
            rank = int(np.sum(singular > tol))
            basis = vh[rank:].T
        else:
            basis = np.eye(len(paths[order]))
        bases[order] = basis
        boundaries[order] = allowed.toarray().astype(float)
    dims = {k: int(v.shape[1]) for k, v in bases.items()}
    ranks = {}
    for k in range(4):
        matrix = boundaries[k] @ bases[k]
        ranks[k] = int(np.linalg.matrix_rank(matrix)) if matrix.size else 0
    betti = {k: dims[k] - ranks[k] - (ranks[k + 1] if k < 3 else 0) for k in range(3)}
    return {"omega_dims": dims, "boundary_ranks": ranks, "betti": betti, "bases": bases, "boundaries": boundaries}

