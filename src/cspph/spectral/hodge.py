"""Shared Hodge readout on the exact GLMY boundary operators.

The base spectrum is computed once per order from the unweighted GLMY Hodge
operator.  RDF, contact, LJ and shape are then read out through that same
eigenspace; they do not trigger four independent eigensolves.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple
import numpy as np


def _gram_and_couplings(chain, order: int):
    from scipy import sparse
    basis = chain.omega_basis[order].tocsr()
    dim = int(basis.shape[1])
    gram = basis.T @ basis + sparse.eye(dim, format="csr") * 1e-10
    lower = None
    if order >= 1 and order in chain.boundaries and order - 1 in chain.omega_basis:
        prev = chain.omega_basis[order - 1].tocsr()
        boundary = chain.boundaries[order].tocsr()
        lower = (prev.T @ boundary @ basis, prev.T @ prev + sparse.eye(prev.shape[1], format="csr") * 1e-10)
    upper = None
    if order + 1 in chain.boundaries and order + 1 in chain.omega_basis:
        nxt = chain.omega_basis[order + 1].tocsr()
        boundary = chain.boundaries[order + 1].tocsr()
        upper = (basis.T @ boundary @ nxt, nxt.T @ nxt + sparse.eye(nxt.shape[1], format="csr") * 1e-10)
    return basis, gram.tocsc(), lower, upper


def _dense_hodge(chain, order: int):
    from scipy.linalg import eigh
    basis, gram, lower, upper = _gram_and_couplings(chain, order)
    dim = int(basis.shape[1])
    H = np.zeros((dim, dim), dtype=float)
    if lower is not None:
        coupling, gram_prev = lower
        H += np.asarray(coupling.T @ sparse_solve(gram_prev, coupling))
    if upper is not None:
        coupling, gram_next = upper
        H += np.asarray(coupling @ sparse_solve(gram_next, coupling.T))
    values, vectors = eigh((H + H.T) * 0.5, gram.toarray())
    return basis, np.asarray(values, dtype=float), np.asarray(vectors, dtype=float), 0.0, True, "cpu_dense_glmy_hodge"


def sparse_solve(matrix, rhs):
    from scipy.sparse.linalg import factorized
    if hasattr(rhs, "toarray"):
        rhs = rhs.toarray()
    return factorized(matrix.tocsc())(np.asarray(rhs, dtype=float))


def _matrix_free_hodge(chain, order: int, max_eigenvalues: int, tol: float):
    from scipy.sparse.linalg import LinearOperator, eigsh, factorized
    basis, gram, lower, upper = _gram_and_couplings(chain, order)
    dim = int(basis.shape[1])
    solve_prev = factorized(lower[1].tocsc()) if lower is not None else None
    solve_next = factorized(upper[1].tocsc()) if upper is not None else None
    def apply_h(x):
        out = np.zeros(dim, dtype=float)
        if lower is not None:
            coupling, gram_prev = lower
            out += np.asarray(coupling.T @ solve_prev(coupling @ x)).ravel()
        if upper is not None:
            coupling, gram_next = upper
            out += np.asarray(coupling @ solve_next(coupling.T @ x)).ravel()
        return out
    def apply(x):
        # Keep the generalized symmetric problem H v = lambda G v.  Applying
        # G^-1 before eigsh turns this into a non-symmetric Euclidean
        # operator, which invalidates eigsh's symmetric Lanczos assumptions.
        return apply_h(x)
    # GLMY supplies the dimension of the harmonic kernel (the Betti number).
    # Ask Lanczos for those zero modes plus the requested non-zero modes; using
    # only ``k=max_eigenvalues`` can otherwise return a block of zeros and
    # report a meaningless spectral gap.
    known_zero_modes = int(max(0, getattr(chain, "betti", {}).get(order, 0)))
    k = max(1, min(int(max_eigenvalues) + known_zero_modes, dim - 1))
    op = LinearOperator((dim, dim), matvec=apply, dtype=float)
    # Deterministic starting vector: eigsh's default random v0 makes the
    # returned Ritz values non-reproducible under near-degenerate spectra,
    # which would break rigid-motion invariance of the downstream summaries.
    v0 = np.random.default_rng(20240711).standard_normal(dim)
    values, vectors = eigsh(op, M=gram, k=k, which="SM", tol=tol, maxiter=max(100, 20 * dim), v0=v0)
    order_idx = np.argsort(values); values = np.asarray(values)[order_idx]; vectors = np.asarray(vectors)[:, order_idx]
    residual = max((float(np.linalg.norm(apply(vectors[:, i]) - values[i] * (gram @ vectors[:, i]))) for i in range(len(values))), default=0.0)
    return basis, np.maximum(values, 0.0), vectors, residual, bool(residual <= max(1e-6, 10 * tol)), "matrix_free_glmy_hodge"


def _summary(values: np.ndarray, estimation_error: float = 0.0) -> Tuple[float, float, float, float, float, float]:
    positive = np.asarray(values)[np.asarray(values) > 1e-10]
    if not len(positive): return 0.0, 0.0, 0.0, 0.0, 0.0, float(estimation_error)
    probs = positive / max(float(positive.sum()), 1e-12)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return float(positive.min()), entropy, float(np.exp(entropy)), float(np.exp(-0.1 * values).sum()), float(np.exp(-0.5 * values).sum()), float(estimation_error)


def hodge_base_spectrum(chain, order: int, dense_limit: int = 384,
                        max_eigenvalues: int = 32, tol: float = 1e-8) -> Dict[str, object]:
    """Compute the shared base Hodge spectrum for one order ONCE.

    Returns a cacheable dict ``{basis, values, vectors, residual, converged,
    method, dim, estimation_error}``.  The spectrum depends only on the GLMY
    chain, never on candidate-specific channel weights, so rigid decoys with
    identical active complexes reuse it through the target cache.
    """
    dim = int(chain.omega_dims.get(order, 0))
    if dim == 0:
        return {"basis": chain.omega_basis[order], "values": np.zeros(0),
                "vectors": np.zeros((0, 0)), "residual": 0.0, "converged": True,
                "method": "empty", "dim": 0, "estimation_error": 0.0}
    if dim <= dense_limit:
        basis, base, vectors, residual, converged, method = _dense_hodge(chain, order)
        est = 0.0
    else:
        try:
            basis, base, vectors, residual, converged, method = _matrix_free_hodge(chain, order, max_eigenvalues, tol)
        except Exception:
            basis = chain.omega_basis[order]
            base = np.zeros(0); vectors = np.zeros((dim, 0))
            residual = float("inf"); converged = False
            method = "matrix_free_glmy_hodge_failed"
        if method == "matrix_free_glmy_hodge" and len(base) < dim and len(base):
            est = float((dim - len(base)) * np.exp(-0.1 * float(np.max(base))))
        else:
            est = 0.0 if method != "matrix_free_glmy_hodge_failed" else 1.0
    return {"basis": basis, "values": np.asarray(base, dtype=float),
            "vectors": np.asarray(vectors, dtype=float), "residual": float(residual),
            "converged": bool(converged), "method": method, "dim": dim,
            "estimation_error": float(est)}


def hodge_channel_readout(chain, channel_weights: Mapping[str, Mapping[int, Sequence[float]]],
                          spectra: Mapping[int, Dict[str, object]],
                          dense_limit: int = 384) -> Dict[str, float]:
    """Read the four physical channels through cached base eigenspaces."""
    out: Dict[str, float] = {"hodge_dimension": float(max(chain.omega_dims.values(), default=0)), "hodge_dense_limit": float(dense_limit)}
    channel_values: Dict[Tuple[str, int], np.ndarray] = {}
    for order in range(3):
        spec = spectra.get(order) or {}
        dim = int(spec.get("dim", chain.omega_dims.get(order, 0)))
        basis = spec.get("basis", chain.omega_basis[order])
        base = np.asarray(spec.get("values", np.zeros(0)), dtype=float)
        vectors = np.asarray(spec.get("vectors", np.zeros((0, 0))), dtype=float)
        residual = float(spec.get("residual", 0.0))
        converged = bool(spec.get("converged", False))
        method = str(spec.get("method", "empty"))
        base_estimation_error = float(spec.get("estimation_error", 0.0))
        base_gap, base_entropy, base_rank, heat01, heat05, _ = _summary(base, base_estimation_error)
        spectrum_complete = float(len(base) >= dim and dim > 0 or dim == 0)
        prefix_b = f"shared_hodge_base_k{order}_"
        out[prefix_b + "spectral_gap"] = base_gap
        out[prefix_b + "spectral_entropy"] = base_entropy
        out[prefix_b + "effective_rank"] = base_rank
        out[prefix_b + "heat_trace_0.1"] = heat01
        out[prefix_b + "heat_trace_0.5"] = heat05
        out[prefix_b + "residual"] = residual
        out[prefix_b + "converged"] = float(converged)
        out[prefix_b + "estimation_error"] = base_estimation_error
        out[prefix_b + "spectrum_complete"] = spectrum_complete
        out[prefix_b + "omitted_modes"] = float(max(0, dim - len(base)))
        out[prefix_b + "n_modes"] = float(len(base))
        # Explicit low-mode statistics: on a truncated spectrum the canonical
        # entropy/rank/heat values ARE low-mode values, and these keys make
        # that interpretation unambiguous instead of implied.
        out[prefix_b + "lowmode_spectral_entropy"] = base_entropy
        out[prefix_b + "lowmode_effective_rank"] = base_rank
        out[prefix_b + "lowmode_heat_trace_0.1"] = heat01
        out[prefix_b + "lowmode_heat_trace_0.5"] = heat05
        out[prefix_b + "lowmode_n_modes"] = float(len(base))
        out[prefix_b + "method_code"] = float({"cpu_dense_glmy_hodge": 1, "matrix_free_glmy_hodge": 2, "matrix_free_glmy_hodge_failed": 3, "empty": 0}.get(method, 9))
        projected = basis @ vectors if vectors.size else np.zeros((basis.shape[0], 0))
        for channel in ("rdf", "contact", "lj", "shape"):
            weights = np.asarray(channel_weights.get(channel, {}).get(order, np.ones(len(chain.paths.get(order, [])))), dtype=float)
            if projected.shape[0] != len(weights): weights = np.ones(projected.shape[0], dtype=float)
            # The quadratic form must be positive semidefinite, so weights are
            # taken by magnitude.  The SIGNED mass fraction is reported
            # separately so directional channels (e.g. shape_facing) keep
            # their sign information instead of being silently clipped.
            w_abs = np.abs(weights) + 1e-12
            response = projected.T @ (w_abs[:, None] * projected) if projected.size else np.zeros((0, 0))
            response_values = np.linalg.eigvalsh((response + response.T) * 0.5) if response.size else np.zeros(0)
            channel_values[(channel, order)] = response_values
            # Channel projection error: omitted base modes contribute at most
            # max|w| times the base omitted-mode bound.
            chan_err = base_estimation_error * (float(np.max(np.abs(weights))) if len(weights) else 1.0) if not converged or spectrum_complete < 1.0 else 0.0
            gap, entropy, rank, tr01, tr05, error = _summary(response_values, float(chan_err))
            prefix = f"shared_hodge_{channel}_k{order}_"
            out[prefix + "spectral_gap"] = gap; out[prefix + "spectral_entropy"] = entropy; out[prefix + "effective_rank"] = rank
            out[prefix + "heat_trace_0.1"] = tr01; out[prefix + "heat_trace_0.5"] = tr05; out[prefix + "heat_trace_1.0"] = float(np.exp(-response_values).sum()) if len(response_values) else 0.0
            out[prefix + "lowmode_spectral_entropy"] = entropy
            out[prefix + "lowmode_effective_rank"] = rank
            out[prefix + "lowmode_heat_trace_0.1"] = tr01
            out[prefix + "lowmode_n_modes"] = float(len(response_values))
            denom = float(np.sum(w_abs[:, None] * (projected * projected))) if projected.size else 0.0
            numer = float(np.sum(weights[:, None] * (projected * projected))) if projected.size else 0.0
            out[prefix + "signed_weight_mass"] = numer / denom if denom > 0 else 0.0
            out[prefix + "hodge_method_code"] = float({"cpu_dense_glmy_hodge": 1, "matrix_free_glmy_hodge": 2, "matrix_free_glmy_hodge_failed": 3, "empty": 0}.get(method, 9))
            out[prefix + "converged"] = float(converged); out[prefix + "residual"] = float(residual); out[prefix + "estimation_error"] = error
            out[prefix + "spectrum_complete"] = spectrum_complete
            out[prefix + "omitted_modes"] = float(max(0, dim - len(base)))
    for order in range(3):
        for left, right in (("rdf", "contact"), ("rdf", "lj"), ("rdf", "shape"), ("contact", "lj"), ("contact", "shape"), ("lj", "shape")):
            a = channel_values.get((left, order), np.zeros(0)); b = channel_values.get((right, order), np.zeros(0)); n = min(len(a), len(b))
            out[f"shared_hodge_cross_{left}_{right}_k{order}"] = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 1 and np.std(a[:n]) > 0 and np.std(b[:n]) > 0 else 0.0
    return out


def shared_hodge_readout(chain, channel_weights: Mapping[str, Mapping[int, Sequence[float]]], dense_limit: int = 384, max_eigenvalues: int = 32, tol: float = 1e-8) -> Dict[str, float]:
    spectra = {order: hodge_base_spectrum(chain, order, dense_limit=dense_limit, max_eigenvalues=max_eigenvalues, tol=tol) for order in range(3)}
    return hodge_channel_readout(chain, channel_weights, spectra=spectra, dense_limit=dense_limit)


__all__ = ["shared_hodge_readout", "hodge_base_spectrum", "hodge_channel_readout"]
