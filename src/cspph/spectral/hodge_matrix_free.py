"""Matrix-free weighted Hodge readout for sparse GLMY bases.

Betti numbers remain the exact algebraic output.  This module only provides
approximate spectral descriptors: the Hodge operator is exposed as a
``LinearOperator`` and eigensolver residuals/convergence are returned with the
descriptor, so an approximate spectrum cannot be mistaken for exact homology.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import LinearOperator, eigsh, factorized
except Exception:  # pragma: no cover
    sparse = None
    LinearOperator = eigsh = factorized = None


def _as_csr(matrix, shape=None):
    if sparse is None:
        raise RuntimeError("scipy.sparse is required for matrix-free Hodge readout")
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(float)
    array = np.asarray(matrix, dtype=float)
    return sparse.csr_matrix(array if shape is None else array.reshape(shape))


def _dense_array(matrix):
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _gram(basis, weights):
    basis = _as_csr(basis)
    weights = np.asarray(weights, dtype=float)
    if weights.size != basis.shape[0]:
        raise ValueError(f"weight length {weights.size} != basis rows {basis.shape[0]}")
    weighted = basis.multiply(np.maximum(weights, 1e-12)[:, None])
    gram = basis.T @ weighted
    gram = gram.tocsc() + sparse.eye(basis.shape[1], format="csc") * 1e-10
    return gram


def _spectrum_summary(eigenvalues, residuals, status: str, dimension: int, matrix_free: bool,
                      backend: str = "cpu") -> Dict[str, object]:
    values = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    positive = values[values > 1e-8]
    if positive.size:
        probs = positive / np.sum(positive)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        gap = float(np.min(positive))
        effective = float(np.exp(entropy))
    else:
        entropy = gap = effective = 0.0
    return {
        "eigenvalues": [float(value) for value in values],
        "residual_max": float(max(residuals) if residuals else 0.0),
        "converged": bool(all(residual <= 1e-4 for residual in residuals)),
        "status": status, "dimension": int(dimension), "matrix_free": bool(matrix_free),
        "backend": backend, "spectral_gap": gap, "spectral_entropy": entropy,
        "effective_rank": effective,
        "heat_trace": {str(time): float(np.sum(np.exp(-time * values))) for time in (0.1, 1.0, 5.0)},
    }


def _cuda_hodge_spectrum(bases: Mapping[int, object], boundaries: Mapping[int, object],
                         weights_by_order: Mapping[int, Sequence[float]], order: int,
                         k: int, tol: float, maxiter: int, device_name: str) -> Dict[str, object]:
    """Torch-CUDA matrix-free Lanczos/power readout.

    Only Gram/coupling operators are materialized; the Hodge operator itself
    is applied as a callable.  CPU and CUDA use the same induced Gram inner
    products, and the returned residual is computed on the selected device.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(device_name)
    basis_k = _dense_array(bases[order])
    dim = int(basis_k.shape[1])
    if dim == 0:
        return _spectrum_summary([], [], "empty", 0, True, "torch_cuda")
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device)
    bk = tensor(basis_k)
    wk = tensor(np.maximum(np.asarray(weights_by_order.get(order, np.ones(basis_k.shape[0])), dtype=np.float32), 1e-12))
    gram_k = bk.T @ (wk[:, None] * bk) + torch.eye(dim, device=device) * 1e-6
    lower = None
    if order >= 1 and order in boundaries and (order - 1) in bases:
        bp_np = _dense_array(bases[order - 1])
        boundary_np = _dense_array(boundaries[order])
        bp = tensor(bp_np)
        wp = tensor(np.maximum(np.asarray(weights_by_order.get(order - 1, np.ones(bp_np.shape[0])), dtype=np.float32), 1e-12))
        gram_prev = bp.T @ (wp[:, None] * bp) + torch.eye(bp.shape[1], device=device) * 1e-6
        coords = bp.T @ (tensor(boundary_np) @ bk)
        lower = (gram_prev, coords)
    upper = None
    if order + 1 in boundaries and order + 1 in bases:
        bn_np = _dense_array(bases[order + 1])
        boundary_np = _dense_array(boundaries[order + 1])
        bn = tensor(bn_np)
        wn = tensor(np.maximum(np.asarray(weights_by_order.get(order + 1, np.ones(bn_np.shape[0])), dtype=np.float32), 1e-12))
        gram_next = bn.T @ (wn[:, None] * bn) + torch.eye(bn.shape[1], device=device) * 1e-6
        coords_next = bk.T @ (tensor(boundary_np) @ bn)
        upper = (gram_next, coords_next)
    def apply(vector):
        out = torch.zeros_like(vector)
        if lower is not None:
            gp, c = lower
            out = out + torch.linalg.solve(gram_k, c.T @ (gp @ (c @ vector)))
        if upper is not None:
            gn, c = upper
            out = out + c @ torch.linalg.solve(gn, c.T @ (gram_k @ vector))
        return out
    def apply_transpose(vector):
        out = torch.zeros_like(vector)
        if lower is not None:
            gp, c = lower
            out = out + c.T @ (gp @ (c @ torch.linalg.solve(gram_k.T, vector)))
        if upper is not None:
            gn, c = upper
            out = out + gram_k.T @ (c @ torch.linalg.solve(gn.T, c.T @ vector))
        return out

    def apply_symmetric(vector):
        return 0.5 * (apply(vector) + apply_transpose(vector))

    # Acceptance-sized Omega spaces use a deterministic GPU eigensolve for
    # residual auditing. Larger production spaces stay on the matrix-free
    # Lanczos path below and never allocate a full Hodge matrix.
    if dim <= 512:
        identity = torch.eye(dim, device=device)
        matrix = torch.stack([apply_symmetric(identity[:, i]) for i in range(dim)], dim=1)
        matrix = 0.5 * (matrix + matrix.T)
        values, vectors = torch.linalg.eigh(matrix)
        take = min(max(1, int(k)), dim)
        values = values[:take]
        vectors = vectors[:, :take]
        residuals = [float(torch.linalg.vector_norm(matrix @ vectors[:, i] - values[i] * vectors[:, i]).detach().cpu()) for i in range(take)]
        return _spectrum_summary(values.detach().cpu().numpy(), residuals, "torch_cuda_exact_small", dim, False, "torch_cuda")

    # A small projected Lanczos basis avoids constructing the full Hodge matrix.
    # Enough Krylov vectors for stable residuals without ever constructing the
    # full dense Hodge matrix.  The cap is still bounded by the Omega
    # dimension, and full paths remain represented by the operator callback.
    steps = min(max(2, 8 * int(k), 64), max(2, dim))
    q = torch.randn(dim, device=device)
    q = q / torch.linalg.vector_norm(q).clamp_min(1e-12)
    basis_vectors = []
    alpha = []
    beta = []
    previous = torch.zeros_like(q)
    for step in range(steps):
        basis_vectors.append(q)
        z = apply_symmetric(q)
        if step:
            z = z - beta[-1] * previous
        a = torch.dot(q, z)
        z = z - a * q
        if basis_vectors:
            for old in basis_vectors[:-1]:
                z = z - torch.dot(old, z) * old
        b = torch.linalg.vector_norm(z)
        alpha.append(float(a.detach().cpu()))
        if step + 1 >= steps or float(b.detach().cpu()) <= tol:
            break
        beta.append(float(b.detach().cpu()))
        previous = q
        q = z / b.clamp_min(1e-12)
    tri = torch.zeros((len(alpha), len(alpha)), device=device)
    for i, value in enumerate(alpha):
        tri[i, i] = value
        if i < len(beta):
            tri[i, i + 1] = tri[i + 1, i] = beta[i]
    values, vectors = torch.linalg.eigh(tri)
    take = min(max(1, int(k)), len(values))
    values = values[:take]
    qmat = torch.stack(basis_vectors, dim=1)
    full_vectors = qmat @ vectors[:, :take]
    residuals = [float(torch.linalg.vector_norm(apply_symmetric(full_vectors[:, i]) - values[i] * full_vectors[:, i]).detach().cpu()) for i in range(take)]
    return _spectrum_summary(values.detach().cpu().numpy(), residuals, "torch_cuda_lanczos", dim, True, "torch_cuda")


def hodge_spectrum_matrix_free(bases: Mapping[int, object], boundaries: Mapping[int, object],
                               weights_by_order: Mapping[int, Sequence[float]], order: int,
                               k: int = 8, tol: float = 1e-7, maxiter: int = 300,
                               backend: str = "auto", device_name: str = "cuda:0") -> Dict[str, object]:
    """Approximate the weighted order-``order`` Hodge spectrum.

    ``bases[k]`` maps Omega coordinates to regular-path coordinates and
    ``boundaries[k]`` is the allowed path boundary into order ``k-1``.  The
    implementation factorizes only weighted Gram matrices; it never builds a
    dense Hodge matrix for production-sized dimensions.
    """
    if backend in {"auto", "cuda"}:
        try:
            return _cuda_hodge_spectrum(bases, boundaries, weights_by_order, order, k, tol, maxiter, device_name)
        except Exception:
            if backend == "cuda":
                raise
    if sparse is None or LinearOperator is None:
        raise RuntimeError("scipy.sparse is required for matrix-free Hodge readout")
    if order not in bases:
        raise KeyError(f"missing basis for order {order}")
    basis_k = _as_csr(bases[order])
    dim = int(basis_k.shape[1])
    if dim == 0:
        return {"eigenvalues": [], "residual_max": 0.0, "converged": True, "status": "empty"}
    gram_k = _gram(basis_k, weights_by_order.get(order, np.ones(basis_k.shape[0])))
    solve_k = factorized(gram_k)

    lower = None
    if order >= 1 and order in boundaries and (order - 1) in bases:
        basis_prev = _as_csr(bases[order - 1])
        boundary = _as_csr(boundaries[order])
        if boundary.shape[1] != basis_k.shape[0] or boundary.shape[0] != basis_prev.shape[0]:
            raise ValueError("boundary and basis shapes are incompatible")
        gram_prev = _gram(basis_prev, weights_by_order.get(order - 1, np.ones(basis_prev.shape[0])))
        solve_prev = factorized(gram_prev)
        coupling = basis_prev.T @ sparse.diags(np.maximum(np.asarray(weights_by_order.get(order - 1, np.ones(basis_prev.shape[0])), dtype=float), 1e-12)) @ boundary @ basis_k

        def lower_d(x):
            return solve_prev(coupling @ x)

        def lower_delta(y):
            return solve_k(coupling.T @ y)
        lower = (lower_d, lower_delta, int(basis_prev.shape[1]))

    upper = None
    if (order + 1) in boundaries and (order + 1) in bases:
        basis_next = _as_csr(bases[order + 1])
        boundary_next = _as_csr(boundaries[order + 1])
        gram_next = _gram(basis_next, weights_by_order.get(order + 1, np.ones(basis_next.shape[0])))
        solve_next = factorized(gram_next)
        coupling_next = basis_k.T @ sparse.diags(np.maximum(np.asarray(weights_by_order.get(order, np.ones(basis_k.shape[0])), dtype=float), 1e-12)) @ boundary_next @ basis_next

        def upper_delta(y):
            # Adjoint d_{k+1}^*: Omega_k -> Omega_{k+1}.
            return solve_next(coupling_next.T @ y)

        def upper_d(z):
            # d_{k+1}: Omega_{k+1} -> Omega_k.
            return solve_k(coupling_next @ z)
        upper = (upper_d, upper_delta, int(basis_next.shape[1]))

    def apply(vector):
        vector = np.asarray(vector, dtype=float)
        output = np.zeros_like(vector)
        if lower is not None:
            d, delta, _ = lower
            output += delta(d(vector))
        if upper is not None:
            d, delta, _ = upper
            output += d(delta(vector))
        return output

    operator = LinearOperator((dim, dim), matvec=apply, dtype=float)
    materialized_small = False
    if dim <= 2 or k >= dim:
        # Only tiny debug dimensions are materialized; large jobs stay fully
        # matrix-free. This path makes deterministic unit tests possible.
        matrix = np.column_stack([apply(np.eye(dim)[:, i]) for i in range(dim)])
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
        materialized_small = True
    else:
        requested = min(max(1, int(k)), dim - 1)
        try:
            eigenvalues, eigenvectors = eigsh(operator, k=requested, which="SM", tol=tol, maxiter=maxiter)
            order_idx = np.argsort(eigenvalues)
            eigenvalues, eigenvectors = eigenvalues[order_idx], eigenvectors[:, order_idx]
        except Exception as exc:
            return {"eigenvalues": [], "residual_max": float("inf"), "converged": False,
                    "status": "eigensolver_failed", "error": str(exc), "dimension": dim,
                    "matrix_free": True}
    residuals = [float(np.linalg.norm(apply(eigenvectors[:, i]) - eigenvalues[i] * eigenvectors[:, i]))
                 for i in range(len(eigenvalues))]
    return {
        "eigenvalues": [float(value) for value in eigenvalues],
        "residual_max": float(max(residuals) if residuals else 0.0),
        "converged": bool(all(residual <= max(1e-5, 10.0 * tol) for residual in residuals)),
        "status": "materialized_small_debug" if materialized_small else "matrix_free_eigsh",
        "dimension": dim,
        "matrix_free": not materialized_small,
    }


__all__ = ["hodge_spectrum_matrix_free"]
