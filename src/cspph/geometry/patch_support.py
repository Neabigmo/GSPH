"""Exact set-valued patch support with branch-and-bound.

Patch support is the minimum over one real microcontact selected from each
patch.  Helly/KKT certification is applied to the selected atom balls only;
no Helly claim is made for a union of lenses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from cspph.core.gsph1 import VDW_RADII, _active_set_support, _support_certificate
from cspph.chemistry.patch_builder import InteractionPatch, MicroContact


def _support_from_cpp_candidate(atom_ids: Sequence[int], atoms: Mapping[int, object], candidate) -> Tuple[float, np.ndarray, bool, str]:
    ids = tuple(sorted(set(int(x) for x in atom_ids)))
    centers = np.asarray([atoms[i].xyz for i in ids], dtype=float)
    radii = np.asarray([VDW_RADII.get(str(atoms[i].element), VDW_RADII["X"]) for i in ids], dtype=float)
    if candidate is None:
        return float("inf"), np.zeros(3), False, "cpp_no_candidate"
    if isinstance(candidate, dict):
        point = np.asarray(candidate.get("center", candidate.get("point", np.zeros(3))), dtype=float)
        tau = float(candidate.get("tau", float("inf")))
    else:
        point, tau = candidate
        point = np.asarray(point, dtype=float); tau = float(tau)
    ok, _, _ = _support_certificate(point, tau, centers, radii, tol=4e-6)
    return max(0.0, tau), point, bool(ok), "cpp_exact_certified" if ok else "cpp_candidate_rejected"


def _certify_cpp_batch(keys: Sequence[Tuple[int, ...]], atoms: Mapping[int, object], raw: Mapping[Tuple[int, ...], object],
                       tol: float = 4e-6) -> Dict[Tuple[int, ...], Tuple[float, np.ndarray, bool, str]]:
    """Batched float64 KKT/Helly certificate for C++ support candidates.

    The acceptance test is exactly ``_support_certificate``: per-key
    feasibility residual, radius-zero acceptance, active gradient subsets of
    size 1--4 solved by ``np.linalg.lstsq`` with the same stationarity and
    nonnegativity tolerances.  Vectorization only removes per-key Python
    overhead; batched LAPACK solves each key's system independently, so the
    certified/rejected decision for every key is unchanged.
    """
    results: Dict[Tuple[int, ...], Tuple[float, np.ndarray, bool, str]] = {}
    rows: List[Tuple[int, ...]] = []
    n_keys = len(keys)
    centers = np.zeros((n_keys, 8, 3), dtype=float)
    radii = np.zeros((n_keys, 8), dtype=float)
    colmask = np.zeros((n_keys, 8), dtype=bool)
    points = np.zeros((n_keys, 3), dtype=float)
    taus = np.full(n_keys, np.inf)
    for row, key in enumerate(keys):
        candidate = raw.get(key)
        if candidate is None:
            results[key] = (float("inf"), np.zeros(3), False, "cpp_no_candidate")
            continue
        n = len(key)
        centers[row, :n] = np.asarray([atoms[i].xyz for i in key], dtype=float)
        radii[row, :n] = [VDW_RADII.get(str(atoms[i].element), VDW_RADII["X"]) for i in key]
        colmask[row, :n] = True
        if isinstance(candidate, dict):
            points[row] = np.asarray(candidate.get("center", candidate.get("point", np.zeros(3))), dtype=float)
            taus[row] = float(candidate.get("tau", float("inf")))
        else:
            point, tau = candidate
            points[row] = np.asarray(point, dtype=float); taus[row] = float(tau)
        rows.append(key)
    if not rows:
        return results
    key_pos = {key: i for i, key in enumerate(keys)}
    sub = np.asarray([key_pos[key] for key in rows], dtype=np.int64)
    dist = np.linalg.norm(centers[sub] - points[sub][:, None, :], axis=2)
    gaps = dist - radii[sub] - taus[sub][:, None]
    residual = np.maximum(np.where(colmask[sub], gaps, -np.inf).max(axis=1), 0.0)
    certified = np.zeros(len(rows), dtype=bool)
    certified |= residual <= tol
    certified &= taus[sub] <= tol
    # Rows failing feasibility are rejected outright; certified radius-zero
    # rows are done.  The rest need the KKT stationarity test.
    done = np.copy(certified)
    active = (np.abs(gaps) <= 5.0 * tol) & colmask[sub] & (dist > tol)
    grads = np.zeros((len(rows), 8, 3), dtype=float)
    safe = np.where(dist > tol, dist, 1.0)
    grads = np.where(active[:, :, None], (points[sub][:, None, :] - centers[sub]) / safe[:, :, None], 0.0)
    pending = ~done & (residual <= tol)
    rhs = np.r_[np.zeros(3, dtype=float), 1.0]
    # np.linalg.lstsq does not accept stacked systems; np.linalg.pinv does.
    # pinv(A) @ rhs yields the same minimum-norm least-squares solution that
    # LAPACK gelsd returns.  Any row whose pass/fail margin is inside a
    # safety band around the 5e-5 tolerances is re-certified with the
    # original per-key routine, so batched numerics can never flip a
    # boundary decision.
    edge = np.zeros(len(rows), dtype=bool)
    edge |= np.abs(residual - tol) <= 1e-7
    edge |= np.abs(taus[sub] - tol) <= 1e-7
    for size in range(1, 5):
        if not pending.any():
            break
        for combo in combinations(range(8), size):
            sel = pending.copy()
            for col in combo:
                sel &= active[:, col]
            if not sel.any():
                continue
            sel_idx = np.where(sel)[0]
            system = np.concatenate(
                (grads[sel_idx][:, list(combo), :].transpose(0, 2, 1),
                 np.ones((sel_idx.size, 1, size), dtype=float)),
                axis=1)
            weights = np.linalg.pinv(system) @ rhs
            stationarity = np.abs(system @ weights[:, :, None] - rhs[None, :, None]).max(axis=(1, 2))
            edge[sel_idx[np.abs(stationarity - 5e-5) <= 1e-5]] = True
            edge[sel_idx[np.abs(weights.min(axis=1) + 5e-5) <= 1e-5]] = True
            passed = (stationarity <= 5e-5) & (weights.min(axis=1) >= -5e-5)
            certified[sel_idx[passed]] = True
            pending[sel_idx[passed]] = False
    for i, key in enumerate(rows):
        ok = bool(certified[i])
        if edge[i]:
            # Boundary margin: repeat the authoritative per-key certificate.
            n = int(colmask[sub[i]].sum())
            centers_i = centers[sub[i], :n]; radii_i = radii[sub[i], :n]
            ok, _, _ = _support_certificate(points[sub[i]], float(taus[sub[i]]), centers_i, radii_i, tol=tol)
        tau_out = max(0.0, float(taus[sub[i]]))
        results[key] = (tau_out, points[sub[i]], bool(ok), "cpp_exact_certified" if ok else "cpp_candidate_rejected")
    return results


@dataclass
class PatchSupportResult:
    tau: float
    center: Tuple[float, float, float]
    selection: Tuple[Tuple[int, int], ...]
    certificate_passed: bool
    complete: bool
    explored: int
    pruned: int
    lower_bound: float
    route: str
    error: str = ""


def _atom_support(atom_ids: Sequence[int], atoms: Mapping[int, object], use_cpp: bool = True, support_stats: Optional[Dict[str, object]] = None) -> Tuple[float, np.ndarray, bool, str]:
    ids = tuple(sorted(set(int(x) for x in atom_ids)))
    if not ids:
        return 0.0, np.zeros(3), True, "empty"
    centers = np.asarray([atoms[i].xyz for i in ids], dtype=float)
    radii = np.asarray([VDW_RADII.get(str(atoms[i].element), VDW_RADII["X"]) for i in ids], dtype=float)
    if len(ids) == 1:
        return 0.0, centers[0], True, "single_ball"
    # Fast C++ exact active-set candidate.  A Python float64 certificate is
    # always applied before accepting it; a rejection enters the explicit
    # robust cascade below rather than becoming a silent failure.
    if use_cpp and len(ids) <= 8:
        try:
            from cspph.geometry.support_active_backend import build_support_candidates
            raw, _ = build_support_candidates([ids], atoms)
            tau, point, ok, route = _support_from_cpp_candidate(ids, atoms, raw.get(ids))
            if ok:
                return tau, point, True, route
        except Exception as exc:
            # The batch helper is optional; the explicit reference cascade
            # below remains authoritative.  Preserve the compact reason in
            # metadata so a fallback can be diagnosed after a long run.
            if support_stats is not None:
                reason = f"{type(exc).__name__}: {exc}"[:500]
                prior = support_stats.get("support_cpp_error")
                support_stats["support_cpp_error"] = reason if not prior else f"{prior}; {reason}"[:1000]
            pass
    result = _active_set_support(centers, radii)
    if result is not None:
        point, tau = result
        ok, _, _ = _support_certificate(point, float(tau), centers, radii, tol=4e-6)
        if ok:
            return float(max(0.0, tau)), np.asarray(point, dtype=float), True, "python_active_set_certified"
    # Explicit numerical fallback.  It is independently certified and is not a silent fallback.
    try:
        from scipy.optimize import minimize
        x0 = np.r_[centers.mean(axis=0), max(0.0, max(np.linalg.norm(centers - centers.mean(axis=0), axis=1) - radii))]
        cons = [{"type": "ineq", "fun": lambda y, c=c, r=float(r): y[3] + r - np.linalg.norm(y[:3] - c)} for c, r in zip(centers, radii)]
        opt = minimize(lambda y: float(y[3]), x0, method="SLSQP", bounds=[(None, None)] * 3 + [(0.0, None)], constraints=cons, options={"maxiter": 300, "ftol": 1e-10})
        if opt.success and np.isfinite(opt.fun):
            tau = max(0.0, float(opt.x[3])); ok, _, _ = _support_certificate(opt.x[:3], tau, centers, radii, tol=5e-5)
            if ok:
                return tau, np.asarray(opt.x[:3], dtype=float), True, "scipy_float64_certified"
    except Exception:
        pass
    return float("inf"), np.zeros(3), False, "no_certified_solution"


def _contact_support(contact: MicroContact, atoms: Mapping[int, object]) -> float:
    donor = atoms[contact.donor]; acceptor = atoms[contact.acceptor]
    distance = float(np.linalg.norm(np.asarray(donor.xyz) - np.asarray(acceptor.xyz)))
    return max(0.0, 0.5 * (distance - VDW_RADII.get(str(donor.element), VDW_RADII["X"]) - VDW_RADII.get(str(acceptor.element), VDW_RADII["X"])))


def _microcontact_pair_lower_bound(left: MicroContact, right: MicroContact, atoms: Mapping[int, object]) -> float:
    """Strict atom-separation lower bound for two contact lenses.

    It is deliberately cheaper than a four-ball solve.  The exact solver is
    reserved for surviving leaf combinations, so this stage can never change
    a support birth value or create a false prune.
    """
    left_ids = (left.donor, left.acceptor); right_ids = (right.donor, right.acceptor)
    value = 0.0
    for first in left_ids:
        for second in right_ids:
            a = atoms[int(first)]; b = atoms[int(second)]
            distance = float(np.linalg.norm(np.asarray(a.xyz) - np.asarray(b.xyz)))
            value = max(value, 0.5 * max(0.0, distance - VDW_RADII.get(str(a.element), VDW_RADII["X"]) - VDW_RADII.get(str(b.element), VDW_RADII["X"])))
    return value


def compute_patch_support(path: Sequence[InteractionPatch], atoms: Mapping[int, object], max_combinations: Optional[int] = None, support_cache: Optional[Dict[Tuple[int, ...], Tuple[float, np.ndarray, bool, str]]] = None, support_stats: Optional[Dict[str, object]] = None, pair_support_cache: Optional[Dict[Tuple[int, int], float]] = None, warm_start: Optional[PatchSupportResult] = None) -> PatchSupportResult:
    """Compute exact union-of-lenses support for a patch path.

    Branch-and-bound uses a strict atom-separation lower bound and exact pair
    supports only for small surviving products.  If a caller sets
    ``max_combinations`` and the limit is reached, ``complete`` is false and
    the result must not enter a formal benchmark.
    """
    patches = list(path)
    # GLMY paths may revisit a patch (e.g. P1,P2,P1).  The union support is
    # over the SET of patches, so duplicate positions are collapsed here.
    # This is exact -- Q_P appears once in the intersection either way --
    # and removes redundant microcontact combinations from the search.
    _seen = set(); _deduped = []
    for _p in patches:
        if _p.patch_id in _seen:
            continue
        _seen.add(_p.patch_id); _deduped.append(_p)
    patches = _deduped
    support_cache = support_cache if support_cache is not None else {}; pair_support_cache = pair_support_cache if pair_support_cache is not None else {}
    if max_combinations == 0:
        # Diagnostic/resume mode: do not spend time building pair bounds when
        # the caller explicitly requested zero support combinations.
        return PatchSupportResult(float("inf"), (0.0, 0.0, 0.0), (), False, False, 0, 0, float("inf"), "bounded_zero", "support search disabled by max_combinations=0")

    def cached_atom_support(atom_ids):
        key = tuple(sorted(set(int(x) for x in atom_ids)))
        if key not in support_cache:
            if support_stats is not None:
                support_stats["support_cache_misses"] = int(support_stats.get("support_cache_misses", 0)) + 1
                if len(key) <= 8:
                    support_stats["support_cpp_attempted"] = int(support_stats.get("support_cpp_attempted", 0)) + 1
            started = time.perf_counter()
            support_cache[key] = _atom_support(key, atoms, support_stats=support_stats)
            elapsed = time.perf_counter() - started
            if support_stats is not None:
                support_stats["support_solver_seconds"] = float(support_stats.get("support_solver_seconds", 0.0)) + float(elapsed)
                route = support_cache[key][3]
                if route == "cpp_exact_certified":
                    support_stats["support_cpp_successes"] = int(support_stats.get("support_cpp_successes", 0)) + 1
                    support_stats["support_cpp_seconds"] = float(support_stats.get("support_cpp_seconds", 0.0)) + float(elapsed)
                elif route in {"python_active_set_certified", "scipy_float64_certified"}:
                    support_stats["support_fallbacks"] = int(support_stats.get("support_fallbacks", 0)) + 1
                    support_stats["support_fallback_seconds"] = float(support_stats.get("support_fallback_seconds", 0.0)) + float(elapsed)
                elif route == "no_certified_solution":
                    support_stats["support_solver_failures"] = int(support_stats.get("support_solver_failures", 0)) + 1
        elif support_stats is not None:
            support_stats["support_cache_hits"] = int(support_stats.get("support_cache_hits", 0)) + 1
        return support_cache[key]

    def batch_prefill(keys: Sequence[Tuple[int, ...]]) -> None:
        normalized = sorted({tuple(sorted(set(int(x) for x in key))) for key in keys if tuple(sorted(set(int(x) for x in key))) not in support_cache and len(set(key)) <= 8})
        # A bounded diagnostic run must not spend an unbounded prefill budget
        # before branch-and-bound has a chance to honor max_combinations.
        if max_combinations is not None:
            normalized = normalized[:max(0, int(max_combinations))]
        missing = normalized
        if not missing:
            return
        try:
            from cspph.geometry.support_active_backend import build_support_candidates
            started = time.perf_counter()
            raw, backend_stats = build_support_candidates(missing, atoms)
            elapsed = time.perf_counter() - started
            if support_stats is not None:
                support_stats["support_cpp_attempted"] = int(support_stats.get("support_cpp_attempted", 0)) + len(missing)
                support_stats["support_cpp_batch_calls"] = int(support_stats.get("support_cpp_batch_calls", 0)) + 1
                support_stats["support_cpp_seconds"] = float(support_stats.get("support_cpp_seconds", 0.0)) + elapsed
                support_stats["support_cpp_batch_seconds"] = float(support_stats.get("support_cpp_batch_seconds", 0.0)) + elapsed
                support_stats["support_solver_seconds"] = float(support_stats.get("support_solver_seconds", 0.0)) + elapsed
            certified_rows = _certify_cpp_batch(missing, atoms, raw)
            for key in missing:
                tau, point, ok, route = certified_rows[key]
                if ok:
                    support_cache[key] = (tau, point, True, route)
                    if support_stats is not None:
                        support_stats["support_cpp_successes"] = int(support_stats.get("support_cpp_successes", 0)) + 1
                else:
                    # The candidate was rejected by the shared certificate;
                    # solve this key once through the explicit reference
                    # cascade rather than silently trying C++ again.
                    fallback_started = time.perf_counter()
                    support_cache[key] = _atom_support(key, atoms, use_cpp=False)
                    fallback_elapsed = time.perf_counter() - fallback_started
                    if support_stats is not None:
                        support_stats["support_cpp_rejected"] = int(support_stats.get("support_cpp_rejected", 0)) + 1
                        support_stats["support_solver_seconds"] = float(support_stats.get("support_solver_seconds", 0.0)) + float(fallback_elapsed)
                        if support_cache[key][3] in {"python_active_set_certified", "scipy_float64_certified"}:
                            support_stats["support_fallbacks"] = int(support_stats.get("support_fallbacks", 0)) + 1
                            support_stats["support_fallback_seconds"] = float(support_stats.get("support_fallback_seconds", 0.0)) + float(fallback_elapsed)
                        elif support_cache[key][3] == "no_certified_solution":
                            support_stats["support_solver_failures"] = int(support_stats.get("support_solver_failures", 0)) + 1
        except Exception as exc:
            if support_stats is not None:
                support_stats["support_cpp_error"] = f"{type(exc).__name__}: {exc}"
    if not patches:
        return PatchSupportResult(0.0, (0.0, 0.0, 0.0), (), True, True, 0, 0, 0.0, "empty")
    if len(patches) == 1:
        best = min(patches[0].contacts, key=lambda c: (_contact_support(c, atoms), c.donor, c.acceptor), default=None)
        if best is None:
            return PatchSupportResult(float("inf"), (0.0, 0.0, 0.0), (), False, False, 0, 0, float("inf"), "no_contacts", "patch has no microcontacts")
        tau = _contact_support(best, atoms); donor = atoms[best.donor]; acceptor = atoms[best.acceptor]
        # The witness point must satisfy the support certificate.  For two
        # balls of UNEQUAL radii the tangent point of the tau-inflated spheres
        # is NOT the midpoint: it divides the centre axis in the ratio of the
        # inflated radii.  Fall back to the certified atom solver, and finally
        # report an uncertified centre honestly instead of faking success.
        c1 = np.asarray(donor.xyz, dtype=float); c2 = np.asarray(acceptor.xyz, dtype=float)
        r1 = VDW_RADII.get(str(donor.element), VDW_RADII["X"])
        r2 = VDW_RADII.get(str(acceptor.element), VDW_RADII["X"])
        centers2 = np.array([c1, c2]); radii2 = np.array([r1, r2])
        center = None
        d = float(np.linalg.norm(c2 - c1))
        if d > 1e-12:
            # |p-c1| = r1+tau and |p-c2| = r2+tau with (r1+tau)+(r2+tau)=d.
            cand = c1 + ((d + r1 - r2) / (2.0 * d)) * (c2 - c1)
            ok, _, _ = _support_certificate(cand, tau, centers2, radii2, tol=4e-6)
            if ok:
                center = cand
        if center is None:
            tau_s, cen_s, ok_s, _route = _atom_support((best.donor, best.acceptor), atoms)
            if ok_s:
                ok2, _, _ = _support_certificate(np.asarray(cen_s, dtype=float), float(tau_s), centers2, radii2, tol=4e-6)
                if ok2:
                    tau = float(tau_s); center = np.asarray(cen_s, dtype=float)
        cert = center is not None
        if center is None:
            center = (c1 + c2) / 2.0
            ok, _, _ = _support_certificate(center, tau, centers2, radii2, tol=4e-6)
            cert = bool(ok)
        return PatchSupportResult(tau, tuple(map(float, center)), ((best.donor, best.acceptor),), cert, cert, 1, 0, tau, "closed_form_two_ball", "" if cert else "single_patch_center_uncertified")
    ordered = sorted(patches, key=lambda p: (len(p.contacts), p.minimum_tau, p.patch_id))
    total_combinations = 1
    for patch in ordered:
        total_combinations *= max(1, len(patch.contacts))
        if total_combinations > 512:
            break
    if total_combinations <= 512:
        batch_prefill([
            tuple(atom_id for contact in selection for atom_id in (contact.donor, contact.acceptor))
            for selection in product(*(p.contacts for p in ordered))
        ])
    pair_lb: Dict[Tuple[int, int], float] = {}
    for left, right in combinations(ordered, 2):
        key = (min(left.patch_id, right.patch_id), max(left.patch_id, right.patch_id))
        if key not in pair_support_cache:
            pair_product = len(left.contacts) * len(right.contacts)
            if pair_product <= 64:
                candidate_keys = [(c1.donor, c1.acceptor, c2.donor, c2.acceptor) for c1 in left.contacts for c2 in right.contacts]
                batch_prefill(candidate_keys)
                # A pair bound is only exact when EVERY microcontact
                # combination produced a certified solve.  If even one
                # combination is uncertified, its true support could be the
                # minimum -- using only the successful subset would be an
                # invalid (too-large) lower bound and could prune the optimum.
                exact = []
                all_certified = True
                for atom_key in candidate_keys:
                    nkey = tuple(sorted(set(int(x) for x in atom_key)))
                    entry = support_cache.get(nkey)
                    if entry is not None and entry[2] and np.isfinite(entry[0]):
                        exact.append(entry[0])
                    else:
                        all_certified = False
                # An unavailable exact sub-solve is not an infinite lower
                # bound.  Fall back to the strict atom-separation bound so a
                # numerical failure can never prune the true optimum.
                if all_certified and exact:
                    pair_support_cache[key] = float(min(exact))
                else:
                    pair_support_cache[key] = min((_microcontact_pair_lower_bound(c1, c2, atoms) for c1 in left.contacts for c2 in right.contacts), default=0.0)
                    if support_stats is not None:
                        support_stats["pair_bound_partial_exact"] = int(support_stats.get("pair_bound_partial_exact", 0)) + 1
                if support_stats is not None: support_stats["exact_pair_bound_calls"] = int(support_stats.get("exact_pair_bound_calls", 0)) + len(candidate_keys)
            else:
                pair_support_cache[key] = min((_microcontact_pair_lower_bound(c1, c2, atoms) for c1 in left.contacts for c2 in right.contacts), default=float("inf"))
                if support_stats is not None:
                    support_stats["cheap_pair_bound_calls"] = int(support_stats.get("cheap_pair_bound_calls", 0)) + pair_product
        pair_lb[key] = pair_support_cache[key]
    best_tau = float("inf"); best_center = np.zeros(3); best_sel: Tuple[Tuple[int, int], ...] = ()
    # A previous candidate may provide a valid feasible point for the same
    # patch path.  It is used only as an upper bound: every current candidate
    # still traverses the exact branch-and-bound search, so no stale value can
    # change the optimum.  This is safe for rigid and flexible candidates.
    if warm_start is not None and warm_start.certificate_passed and warm_start.selection:
        available = [set((int(contact.donor), int(contact.acceptor)) for contact in patch.contacts) for patch in patches]
        used = list(warm_start.selection)
        valid_selection = False
        if len(used) == len(available):
            # The warm selection must be assignable one-to-one to the current
            # patches; a loose ``any`` check could reuse one old contact for
            # two patches and create an invalid upper bound.
            from itertools import permutations
            valid_selection = any(all(used[index] in available[group] for index, group in enumerate(order)) for order in permutations(range(len(available))))
        if not valid_selection:
            warm_start = None
        if warm_start is None:
            pass
    if warm_start is not None and warm_start.certificate_passed and warm_start.selection:
        selected_ids = tuple(sorted({int(atom_id) for pair in warm_start.selection for atom_id in pair}))
        centers = np.asarray([atoms[i].xyz for i in selected_ids if i in atoms], dtype=float)
        if len(centers) == len(selected_ids):
            center = np.asarray(warm_start.center, dtype=float)
            radii = np.asarray([VDW_RADII.get(str(atoms[i].element), VDW_RADII["X"]) for i in selected_ids], dtype=float)
            tau = float(max(0.0, np.max(np.linalg.norm(centers - center, axis=1) - radii))) if len(centers) else 0.0
            if np.isfinite(tau):
                best_tau = tau; best_center = center; best_sel = tuple(warm_start.selection)
                if support_stats is not None:
                    support_stats["support_warm_start_attempts"] = int(support_stats.get("support_warm_start_attempts", 0)) + 1
                    support_stats["support_warm_start_passes"] = int(support_stats.get("support_warm_start_passes", 0)) + 1
    explored = 0; pruned = 0; incomplete = False; global_lb = 0.0

    def lower_bound(selected: Sequence[MicroContact], selected_patches: Sequence[InteractionPatch], remaining: Sequence[InteractionPatch]) -> float:
        values = [_contact_support(c, atoms) for c in selected]
        for a, b in combinations(selected, 2):
            values.append(_microcontact_pair_lower_bound(a, b, atoms))
        for patch in remaining:
            for chosen_patch in selected_patches:
                values.append(pair_lb.get((min(patch.patch_id, chosen_patch.patch_id), max(patch.patch_id, chosen_patch.patch_id)), 0.0))
        return max(values, default=0.0)

    def visit(index: int, selected: List[MicroContact]) -> None:
        nonlocal best_tau, best_center, best_sel, explored, pruned, incomplete, global_lb
        if max_combinations is not None and explored >= int(max_combinations):
            incomplete = True; return
        remaining = ordered[index:]
        lb = lower_bound(selected, ordered[:index], remaining)
        global_lb = max(global_lb, lb)
        if lb >= best_tau - 1e-10:
            pruned += 1; return
        if index == len(ordered):
            explored += 1
            ids = [x for c in selected for x in (c.donor, c.acceptor)]
            tau, center, ok, route = cached_atom_support(ids)
            if ok and tau < best_tau - 1e-10:
                best_tau, best_center, best_sel = tau, center, tuple((c.donor, c.acceptor) for c in selected)
            elif not ok:
                incomplete = True
            return
        for contact in sorted(ordered[index].contacts, key=lambda c: (_contact_support(c, atoms), c.distance, c.donor, c.acceptor)):
            visit(index + 1, selected + [contact])

    visit(0, [])
    if not np.isfinite(best_tau):
        return PatchSupportResult(float("inf"), (0.0, 0.0, 0.0), best_sel, False, not incomplete, explored, pruned, global_lb, "branch_and_bound", "no certified microcontact combination")
    return PatchSupportResult(float(best_tau), tuple(map(float, best_center)), best_sel, True, not incomplete, explored, pruned, global_lb, "branch_and_bound")


def higher_order_increments(supports: Mapping[int, float]) -> Dict[str, float]:
    tau2 = float(supports.get(2, 0.0)); tau3 = float(supports.get(3, 0.0)); tau4 = float(supports.get(4, 0.0))
    pair_max = max((float(v) for k, v in supports.items() if int(k) == 2), default=tau2)
    triple_max = max((float(v) for k, v in supports.items() if int(k) == 3), default=tau3)
    return {"tau_2": tau2, "tau_3": tau3, "tau_4": tau4,
            "eta_3": max(0.0, tau3 - pair_max), "eta_4": max(0.0, tau4 - triple_max)}


__all__ = ["PatchSupportResult", "compute_patch_support", "higher_order_increments"]
