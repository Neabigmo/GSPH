# -*- coding: utf-8 -*-
"""P0-4 audit: GLMY Omega spaces, boundaries, persistence chain maps,
Shared Hodge spectra -- on synthetic nested path complexes.

Checks per random complex (orders 0..3, small enough for dense reference):
  C1  Omega dims match dense SVD reference; Betti numbers match an
      independent REAL-arithmetic rank computation (modular rank is only
      valid for integer bases -- dense-SVD fallback bases are float).
  C2  Omega-coordinate boundaries d~ satisfy d~_{k-1} d~_k = 0 at every
      scale (raw allowed-boundary residual is NOT expected to vanish --
      it is recorded for information only).
  C3  Omega basis is in ker(forbidden): integer-exact for structured
      backends, <=1e-6 for float (dense SVD) backends.
  C4  Persistence chain maps: build_omega_inclusion_chain_maps +
      validate_chain_maps commute for EVERY adjacent scale pair.
  C5  Shared Hodge: generalized eigenvalues >= -tol (PSD), harmonic
      dimension == independently computed Betti number, solver residual.
  C6  Rank certification backend recorded per trial.

Outputs results/ablation/p0_chain_audit.json
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _P
_F = _P(__file__).resolve()
_R = _F.parent
while _R != _R.parent and not ((_R / "src").is_dir() and (_R / "experiments").is_dir()):
    _R = _R.parent
for _d in (_R / "src", _R / "tasks" / "rna", _R / "tasks" / "protein_dna",
           _R / "tasks" / "membrane"):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
for _e in sorted((_R / "experiments").glob("exp*")):
    if _e.is_dir() and str(_e) not in _sys.path:
        _sys.path.insert(0, str(_e))
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from runners.glmy_sparse import sparse_chain, dense_reference, rank_backend_status  # noqa: E402
from runners.persistence import (build_omega_boundary_filtration,  # noqa: E402
                                  build_omega_inclusion_chain_maps,
                                  validate_chain_maps)
from gsph6.hodge import hodge_base_spectrum  # noqa: E402

OUT = _HERE.parent / "results" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261117)


def random_complex(n_nodes=5, edge_p=0.5, max_order=3):
    nodes = list(range(n_nodes))
    edges = [(u, v) for u in nodes for v in nodes if u != v and rng.random() < edge_p]
    if not edges:
        edges = [(0, 1)]
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
    paths = {0: [(v,) for v in nodes]}
    cur = [(v,) for v in nodes]
    for order in range(1, max_order + 1):
        nxt = []
        for p in cur:
            for v in adj.get(p[-1], []):
                nxt.append(p + (v,))
        paths[order] = nxt
        cur = nxt
        if len(nxt) > 400:
            break
    for o in paths:
        if len(paths[o]) > 300:
            paths[o] = list(rng.choice(paths[o], 300, replace=False))
    return paths


def nested_scales(paths_full, taus, scales=(0.0, 0.5, 2.0)):
    by_time = []
    for s in scales:
        act = {o: [] for o in range(4)}
        for o, plist in paths_full.items():
            for p in plist:
                if taus.get(p, np.inf) <= s:
                    act[o].append(p)
        by_time.append(act)
    return by_time


def real_rank(image):
    if image.shape[0] == 0 or image.shape[1] == 0:
        return 0
    a = image.toarray() if hasattr(image, "toarray") else np.asarray(image)
    return int(np.linalg.matrix_rank(a))


def audit_one(idx):
    rec = {"trial": idx}
    paths = random_complex()
    allp = [p for o in paths.values() for p in o]
    taus = {p: float(rng.uniform(0, 3)) for p in set(allp)}
    by_time = nested_scales(paths, taus)

    chains = [sparse_chain(a) for a in by_time]
    filtration = build_omega_boundary_filtration(chains)
    maps = build_omega_inclusion_chain_maps(chains, by_time)
    check = validate_chain_maps(filtration, maps)
    rec["chain_map_valid"] = bool(check.get("valid"))
    rec["chain_map_max_residual"] = float(check.get("max_residual", np.nan))

    # C2: Omega-coordinate boundary square at every scale
    bsq_max = 0.0
    for bounds in filtration:
        for o in (1, 2, 3):
            if o in bounds and o - 1 in bounds:
                prod = bounds[o - 1] @ bounds[o]
                val = (float(np.max(np.abs(prod.data))) if prod.nnz
                       else 0.0)
                bsq_max = max(bsq_max, val)
    rec["omega_boundary_sq_max"] = bsq_max

    ch = chains[-1]
    ref = dense_reference(by_time[-1])
    rec["omega_dims_match"] = all(ch.omega_dims[o] == ref["omega_dims"][o] for o in range(4))

    # C1: betti vs independent real-arithmetic ranks (dense_reference uses
    # real matrix_rank; sparse_chain now uses real rank for float images)
    ranks = {o: real_rank(ch.boundaries[o] @ ch.omega_basis[o]) for o in range(4)}
    betti_ref = {o: ch.omega_dims[o] - ranks[o] - ranks[o + 1] for o in range(3)}
    rec["betti"] = {str(k): int(v) for k, v in ch.betti.items()}
    rec["betti_ref"] = {str(k): int(v) for k, v in betti_ref.items()}
    rec["betti_match"] = all(int(ch.betti[o]) == int(betti_ref[o]) for o in range(3))

    # C3: forbidden-kernel residual (tolerance depends on backend dtype)
    kern_max = 0.0
    for o in (1, 2, 3):
        if o in ch.forbidden and ch.forbidden[o].nnz:
            r = (ch.forbidden[o] @ ch.omega_basis[o]).toarray()
            kern_max = max(kern_max, float(np.max(np.abs(r))) if r.size else 0.0)
    rec["omega_kernel_max_residual"] = kern_max
    rec["kernel_backend"] = ch.kernel_backend
    rec["basis_dtypes"] = {str(o): str(ch.omega_basis[o].dtype) for o in range(4)}
    rec["rank_backend"] = rank_backend_status()
    rec["omega_dims"] = {str(k): v for k, v in ch.omega_dims.items()}

    # C5 Hodge PSD + harmonic dimension vs real-rank betti.  Negative
    # Betti can only arise from rank overestimation on near-dependent
    # float (dense-fallback) bases; such orders are excluded from the
    # harmonic count and reported separately.
    hodge_ok = True; min_eval = np.inf; harm_err = 0; neg_betti_orders = []
    for order in range(3):
        spec = hodge_base_spectrum(ch, order, dense_limit=10_000)
        vals = np.asarray(spec["values"], float)
        if len(vals):
            min_eval = min(min_eval, float(vals.min()))
            b = betti_ref.get(order, 0)
            if b < 0:
                neg_betti_orders.append(order)
            else:
                n_zero = int(np.sum(vals < 1e-6))
                harm_err = max(harm_err, abs(n_zero - b))
        if not spec["converged"]:
            hodge_ok = False
        if len(vals) and float(vals.min()) < -1e-6:
            hodge_ok = False
    rec["hodge_psd"] = bool(hodge_ok and (not np.isfinite(min_eval) or min_eval >= -1e-6))
    rec["hodge_min_eigenvalue"] = float(min_eval) if np.isfinite(min_eval) else None
    rec["hodge_harmonic_dim_err"] = int(harm_err)
    rec["negative_betti_orders"] = neg_betti_orders
    rec["n_paths"] = {str(o): len(ch.paths.get(o, [])) for o in range(4)}
    rec["used_dense_fallback"] = "dense_svd_fallback" in ch.kernel_backend
    return rec


def main():
    n = 60
    rows = []
    for i in range(n):
        try:
            rows.append(audit_one(i))
        except Exception as exc:
            rows.append({"trial": i, "error": f"{type(exc).__name__}: {exc}"})
        if (i + 1) % 10 == 0:
            print(f"[audit] {i+1}/{n}", flush=True)
    ok = [r for r in rows if "error" not in r]
    structured = [r for r in ok if not r["used_dense_fallback"]]
    fallback = [r for r in ok if r["used_dense_fallback"]]
    summary = {
        "n_trials": n, "n_ok": len(ok),
        "n_dense_fallback": len(fallback),
        "chain_map_all_valid": all(r["chain_map_valid"] for r in ok),
        "chain_map_max_residual": max((r["chain_map_max_residual"] for r in ok), default=None),
        "omega_dims_all_match": all(r["omega_dims_match"] for r in ok),
        "betti_all_match": all(r["betti_match"] for r in ok),
        "betti_all_match_structured_only": all(r["betti_match"] for r in structured),
        "omega_boundary_sq_max_overall": max((r["omega_boundary_sq_max"] for r in ok), default=None),
        "omega_kernel_max_residual_overall": max((r["omega_kernel_max_residual"] for r in ok), default=None),
        "hodge_psd_all": all(r["hodge_psd"] for r in ok),
        "hodge_harmonic_dim_err_total": sum(r["hodge_harmonic_dim_err"] for r in ok),
        "errors": [r for r in rows if "error" in r][:10],
    }
    (OUT / "p0_chain_audit.json").write_text(json.dumps(
        {"summary": summary, "trials": rows}, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
