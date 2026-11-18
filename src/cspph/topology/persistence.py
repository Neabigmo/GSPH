"""True persistence utilities that require explicit chain maps.

Multiscale Betti snapshots are not persistence.  This module therefore never
guesses an inclusion between GSPH complexes.  Callers must provide chain maps
and validate the chain-map equation first; interval recovery then operates on
the induced homology rank invariant supplied by a verified downstream stage.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

try:
    from scipy import sparse as _sparse
except Exception:  # pragma: no cover
    _sparse = None


def _array(matrix):
    """Return a sparse CSR object when possible, never densifying it."""
    if hasattr(matrix, "toarray"):
        return matrix.tocsr().astype(float) if hasattr(matrix, "tocsr") else matrix
    return np.asarray(matrix, dtype=float)


def validate_chain_maps(boundaries_by_time: Sequence[Mapping[int, object]],
                        chain_maps_by_time: Sequence[Mapping[int, object]],
                        tol: float = 1e-8) -> Dict[str, object]:
    """Verify ``d_{t+1} F_{t+1} = F_t d_{t+1}`` at every adjacent time.

    Boundary matrices use the convention ``boundary[k]: C_k -> C_{k-1}``.
    A map dictionary for step ``t`` contains one matrix per chain order.
    """
    if len(boundaries_by_time) < 1:
        raise ValueError("at least one filtered complex is required")
    if len(chain_maps_by_time) != max(0, len(boundaries_by_time) - 1):
        raise ValueError("one chain-map dictionary is required per adjacent time pair")
    residuals: Dict[str, float] = {}
    for time, maps in enumerate(chain_maps_by_time):
        left = boundaries_by_time[time + 1]
        right = boundaries_by_time[time]
        for order in sorted(order for order in (set(left) & set(right) & set(maps)) if order - 1 in maps):
            boundary_next = _array(left[order])
            boundary_prev = _array(right[order])
            map_k = _array(maps[order])
            map_prev = _array(maps[order - 1])
            residual = map_prev @ boundary_prev - boundary_next @ map_k
            if hasattr(residual, "data") and hasattr(residual, "tocsr"):
                value = float(np.sqrt(np.dot(residual.data, residual.data)))
            else:
                value = float(np.linalg.norm(residual))
            residuals[f"t{time}_k{order}"] = value
            if value > tol:
                return {"valid": False, "max_residual": value, "residuals": residuals,
                        "error": f"chain-map equation failed at time {time}, order {order}"}
    return {"valid": True, "max_residual": float(max(residuals.values()) if residuals else 0.0),
            "residuals": residuals}


def intervals_from_rank_invariant(rank_invariant: Sequence[Sequence[int]]) -> List[Dict[str, int]]:
    """Recover half-open intervals from a verified homology rank invariant.

    ``rank_invariant[i][j]`` is the rank of the induced map H_i -> H_j for
    ``i <= j``.  The diagonal is the homology dimension and death ``T`` means
    right-censored at the end of the supplied filtration.
    """
    ranks = np.asarray(rank_invariant, dtype=int)
    if ranks.ndim != 2 or ranks.shape[0] != ranks.shape[1]:
        raise ValueError("rank invariant must be a square matrix")
    times = int(ranks.shape[0])
    if np.any(ranks < 0):
        raise ValueError("ranks must be non-negative")
    if np.any(ranks[np.tril_indices(times, -1)] != 0):
        raise ValueError("rank invariant lower triangle must be zero")
    for i in range(times):
        if int(ranks[i, i]) != ranks[i, i]:
            raise ValueError("diagonal homology dimensions must be integers")
        for j in range(i, times):
            if ranks[i, j] > ranks[i, i] or ranks[i, j] > ranks[j, j]:
                raise ValueError("map rank exceeds a source or target homology dimension")
            if j > i and ranks[i, j] > ranks[i, j - 1]:
                raise ValueError("rank invariant must be non-increasing in the target time")
        for j in range(i + 1, times):
            if ranks[i, j] > ranks[i + 1, j]:
                raise ValueError("rank invariant must be non-decreasing in the source time")
    intervals: List[Dict[str, int]] = []

    def rank(i: int, j: int) -> int:
        if i < 0 or j >= times or i > j:
            return 0
        return int(ranks[i, j])

    for birth in range(times):
        for death in range(birth + 1, times + 1):
            multiplicity = (rank(birth, death - 1) - rank(birth - 1, death - 1)
                            - rank(birth, death) + rank(birth - 1, death))
            if multiplicity < 0:
                raise ValueError(f"invalid rank invariant: negative multiplicity at [{birth}, {death})")
            for _ in range(int(multiplicity)):
                intervals.append({"birth": birth, "death": death})
    return intervals


__all__ = ["validate_chain_maps", "intervals_from_rank_invariant",
           "build_inclusion_chain_maps", "build_omega_boundary_filtration",
           "build_omega_inclusion_chain_maps", "rank_invariant_from_filtration",
           "persistent_intervals_from_filtration"]


def build_inclusion_chain_maps(paths_by_time: Sequence[Mapping[int, Sequence[tuple]]],
                               tol: float = 0.0) -> List[Dict[int, object]]:
    """Build sparse inclusion maps for nested regular-path complexes.

    Each scale must contain a subset of the next scale.  Maps are identity on
    matching ordered paths and zero otherwise; missing paths are rejected
    rather than silently projected.
    """
    if len(paths_by_time) < 1:
        raise ValueError("at least one filtration level is required")
    if _sparse is None:
        raise RuntimeError("scipy.sparse is required for filtration maps")
    maps: List[Dict[int, object]] = []
    for time in range(len(paths_by_time) - 1):
        current = paths_by_time[time]
        following = paths_by_time[time + 1]
        step: Dict[int, object] = {}
        # GSPH stores orders 0--3 explicitly, including empty levels.  Emit a
        # map for every order so downstream Omega-coordinate code never has to
        # infer a missing zero-dimensional block.
        for order in range(4):
            source = list(current.get(order, [])); target = list(following.get(order, []))
            target_index = {path: index for index, path in enumerate(target)}
            rows = []; cols = []; data = []
            for col, path in enumerate(source):
                row = target_index.get(path)
                if row is None:
                    raise ValueError(f"filtration is not nested at time {time}, order {order}")
                rows.append(row); cols.append(col); data.append(1.0)
            step[order] = _sparse.csr_matrix((data, (rows, cols)), shape=(len(target), len(source)), dtype=float)
        maps.append(step)
    return maps


def _solve_sparse_coordinates(basis, rhs, tol: float = 1e-9):
    """Solve ``basis @ X = rhs`` without materialising a dense basis.

    Omega bases are full-column-rank sparse matrices.  We solve the normal
    equations with a sparse factorisation and verify the residual.  This is a
    coordinate change only; it does not alter the chain complex or introduce a
    scoring approximation.
    """
    if _sparse is None:
        raise RuntimeError("scipy.sparse is required for Omega coordinates")
    basis = basis.tocsr().astype(float)
    rhs = rhs.tocsc().astype(float) if hasattr(rhs, "tocsc") else _sparse.csc_matrix(rhs, dtype=float)
    rows, cols = basis.shape[0], basis.shape[1]
    if cols == 0:
        return _sparse.csr_matrix((0, rhs.shape[1]), dtype=float)
    if rhs.shape[0] != rows:
        raise ValueError(f"Omega coordinate shape mismatch: {basis.shape} vs {rhs.shape}")
    gram = (basis.T @ basis).tocsc()
    # A mathematically full-rank basis should give SPD Gram.  The tiny diagonal
    # is only a numerical safeguard for nearly dependent floating conversions.
    gram = gram + _sparse.eye(cols, format="csc") * 1e-12
    from scipy.sparse.linalg import splu
    factor = splu(gram)
    projected = basis.T @ rhs
    out = np.empty((cols, rhs.shape[1]), dtype=float)
    for j in range(rhs.shape[1]):
        out[:, j] = factor.solve(np.asarray(projected[:, j].toarray()).ravel())
    residual = basis @ _sparse.csc_matrix(out) - rhs
    if residual.nnz:
        max_residual = float(np.max(np.abs(residual.data)))
        if max_residual > tol:
            raise ValueError(f"Omega coordinate residual {max_residual:g} exceeds tolerance")
    return _sparse.csr_matrix(out)


def build_omega_boundary_filtration(chains: Sequence[object]) -> List[Dict[int, object]]:
    """Build boundary operators in the Omega coordinates of each chain.

    If ``B_k`` embeds Omega_k in the regular path space A_k, this constructs
    ``d~_k`` from ``B_{k-1} d~_k = d_k B_k``.  Persistence must use these
    induced operators rather than the raw A-space boundaries.
    """
    if _sparse is None:
        raise RuntimeError("scipy.sparse is required for Omega boundaries")
    result: List[Dict[int, object]] = []
    for chain in chains:
        boundaries: Dict[int, object] = {
            0: _sparse.csr_matrix((0, int(chain.omega_dims[0])), dtype=float)
        }
        for order in (1, 2, 3):
            rhs = chain.boundaries[order] @ chain.omega_basis[order]
            boundaries[order] = _solve_sparse_coordinates(chain.omega_basis[order - 1], rhs)
        result.append(boundaries)
    return result


def build_omega_inclusion_chain_maps(chains: Sequence[object],
                                     paths_by_time: Sequence[Mapping[int, Sequence[tuple]]]) -> List[Dict[int, object]]:
    """Construct adjacent-scale inclusion maps in Omega coordinates."""
    if _sparse is None:
        raise RuntimeError("scipy.sparse is required for Omega maps")
    raw_maps = build_inclusion_chain_maps(paths_by_time)
    maps: List[Dict[int, object]] = []
    for time, raw in enumerate(raw_maps):
        step: Dict[int, object] = {}
        current = chains[time]
        following = chains[time + 1]
        for order in range(4):
            rhs = raw[order] @ current.omega_basis[order]
            step[order] = _solve_sparse_coordinates(following.omega_basis[order], rhs)
        maps.append(step)
    return maps


def _integer_csr(matrix, tol: float = 1e-7):
    """Convert an exact incidence-like sparse matrix to modular integers."""
    if _sparse is None:
        raise RuntimeError("scipy.sparse is required for modular persistence")
    csr = matrix.tocsr() if hasattr(matrix, "tocsr") else _sparse.csr_matrix(matrix)
    if csr.nnz:
        rounded = np.rint(csr.data)
        if float(np.max(np.abs(csr.data - rounded))) > tol:
            raise ValueError("Omega persistence map is not integer-valued within tolerance")
        data = rounded.astype(np.int64)
    else:
        data = np.zeros(0, dtype=np.int64)
    return _sparse.csr_matrix((data, csr.indices.copy(), csr.indptr.copy()), shape=csr.shape)


def _modular_rref_nullspace(matrix, modulus: int = 1000003):
    """Sparse row-echelon nullspace over a prime field."""
    from cspph.topology.glmy_sparse import _sparse_rank_mod
    matrix = _integer_csr(matrix)
    m, n = matrix.shape
    rows: List[Dict[int, int]] = []
    for i in range(m):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        rows.append({int(c): int(v) % modulus for c, v in zip(matrix.indices[start:end], matrix.data[start:end]) if int(v) % modulus})
    pivots: List[int] = []
    pivot_rows: List[int] = []
    row = 0
    for col in range(n):
        pivot = next((r for r in range(row, m) if rows[r].get(col, 0)), None)
        if pivot is None:
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        inv = pow(rows[row][col] % modulus, modulus - 2, modulus)
        rows[row] = {c: (v * inv) % modulus for c, v in rows[row].items() if (v * inv) % modulus}
        for r in range(row + 1, m):
            factor = rows[r].get(col, 0)
            if not factor:
                continue
            for c, v in rows[row].items():
                value = (rows[r].get(c, 0) - factor * v) % modulus
                if value:
                    rows[r][c] = value
                else:
                    rows[r].pop(c, None)
        pivots.append(col); pivot_rows.append(row); row += 1
        if row == m:
            break
    free = [col for col in range(n) if col not in set(pivots)]
    triplets = []
    for free_col in free:
        vector = {free_col: 1}
        for prow, pcol in reversed(list(zip(pivot_rows, pivots))):
            value = -sum(rows[prow].get(col, 0) * coeff for col, coeff in vector.items()) % modulus
            if value:
                vector[pcol] = value
        for col, value in vector.items():
            triplets.append((col, len(triplets), value))
    if not free:
        return _sparse.csc_matrix((n, 0), dtype=np.int64), len(pivots)
    # Rebuild with a stable basis-column index rather than relying on the
    # append order above.
    rows_out = []; cols_out = []; data_out = []
    for basis_col, free_col in enumerate(free):
        vector = {free_col: 1}
        for prow, pcol in reversed(list(zip(pivot_rows, pivots))):
            value = -sum(rows[prow].get(col, 0) * coeff for col, coeff in vector.items()) % modulus
            if value:
                vector[pcol] = value
        for col, value in vector.items():
            rows_out.append(col); cols_out.append(basis_col); data_out.append(value)
    return _sparse.csc_matrix((data_out, (rows_out, cols_out)), shape=(n, len(free)), dtype=np.int64), len(pivots)


def rank_invariant_from_filtration(boundaries_by_time: Sequence[Mapping[int, object]],
                                   chain_maps_by_time: Sequence[Mapping[int, object]],
                                   order: int, tol: float = 1e-8) -> np.ndarray:
    """Compute the rank invariant of the induced homology maps.

    The returned upper-triangular matrix has ``rank[i,j]`` equal to the rank
    of ``H_order(i) -> H_order(j)``.  It uses nullspaces of the source
    boundary and quotients by the target boundary image, never treating
    independent Betti snapshots as persistence.
    """
    if len(boundaries_by_time) != len(chain_maps_by_time) + 1:
        raise ValueError("boundary/map filtration lengths disagree")
    times = len(boundaries_by_time)
    ranks = np.zeros((times, times), dtype=int)

    from cspph.topology.glmy_sparse import _sparse_rank_mod

    def modular(matrix):
        return _integer_csr(matrix)

    # Pre-convert boundaries/maps to modular integers once
    # (was O(S^2) reconversions of the same matrices).
    _empty = _sparse.csr_matrix((0, 0))
    mod_bounds = [modular(b.get(order, _empty)) for b in boundaries_by_time]
    mod_bounds_next = [modular(b.get(order + 1, _empty)) for b in boundaries_by_time]
    mod_maps = [modular(m.get(order, _empty)) for m in chain_maps_by_time]
    base_ranks = [int(_sparse_rank_mod(mb)) if mb.shape[1] else 0
                  for mb in mod_bounds_next]

    for start in range(times):
        z_basis, _ = _modular_rref_nullspace(mod_bounds[start])
        # induced(stop) = compose_maps(start, stop) @ z_basis; roll the
        # running product instead of recomputing O(S) matmuls per pair
        # (mathematically identical: product order is preserved).
        induced = z_basis
        for stop in range(start, times):
            if stop > start:
                induced = mod_maps[stop - 1] @ induced
            d_next = mod_bounds_next[stop]
            combined = _sparse.hstack((d_next, induced), format="csr")
            combined_rank = int(_sparse_rank_mod(combined)) if combined.shape[1] else 0
            ranks[start, stop] = combined_rank - base_ranks[stop]
    return ranks


def persistent_intervals_from_filtration(boundaries_by_time: Sequence[Mapping[int, object]],
                                          chain_maps_by_time: Sequence[Mapping[int, object]],
                                          order: int, tol: float = 1e-8) -> List[Dict[str, object]]:
    """Recover verified intervals and right-censor intervals at the final scale."""
    invariant = rank_invariant_from_filtration(boundaries_by_time, chain_maps_by_time, order, tol)
    intervals = intervals_from_rank_invariant(invariant)
    final_time = invariant.shape[0]
    for interval in intervals:
        interval["order"] = int(order)
        interval["right_censored"] = bool(interval["death"] == final_time)
    return intervals


__all__ += ["build_inclusion_chain_maps", "rank_invariant_from_filtration",
            "persistent_intervals_from_filtration"]
