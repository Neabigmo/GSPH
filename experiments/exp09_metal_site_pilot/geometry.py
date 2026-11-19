"""Signed union support using the repository's certified weighted 1-center."""
from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from cspph.core.gsph1 import _active_set_support, _support_certificate


def independent_support(centers, radii):
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    start = centers.mean(axis=0)
    y0 = np.r_[start, np.max(np.linalg.norm(centers - start, axis=1) - radii)]
    result = minimize(
        lambda y: y[3], y0, jac=lambda y: np.array([0., 0., 0., 1.]),
        constraints={"type": "ineq", "fun": lambda y: radii + y[3] - np.linalg.norm(centers - y[:3], axis=1)},
        method="SLSQP", options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"independent support failed: {result.message}")
    return result.x[:3], float(result.x[3])


def weighted_support(centers, radii):
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(centers) == 0 or len(centers) != len(radii):
        raise ValueError("nonempty matching centers and radii required")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
        raise ValueError("finite centers and radii required")
    shift = float(np.min(radii))
    local = centers - centers.mean(axis=0)
    result = _active_set_support(local, radii - shift)
    if result is None:
        raise RuntimeError("repository support solver returned no certified optimum")
    point, tau = result
    ok, _, residual = _support_certificate(point, tau, local, radii - shift, tol=4e-6)
    if not ok:
        raise RuntimeError(f"repository support certificate failed: {residual}")
    return point + centers.mean(axis=0), float(tau - shift)


def pair_support(left, right, xyz, radii):
    a, b = np.asarray(left), np.asarray(right)
    distance = np.linalg.norm(xyz[a, None] - xyz[b], axis=2)
    values = np.maximum((distance - radii[a, None] - radii[b]) / 2,
                        -np.minimum(radii[a, None], radii[b]))
    return float(values.min())


class UnionSolver:
    def __init__(self, xyz, radii):
        self.xyz, self.radii = xyz, radii
        self.cache = {}
        self.solves = 0
        self.failures = 0
        self.seconds = 0.
        self.calls = 0

    def solve(self, groups):
        self.calls += 1
        best = (None, float("inf"))
        for selection in product(*groups):
            key = tuple(sorted(set(selection)))
            if key not in self.cache:
                started = perf_counter()
                self.solves += 1
                try:
                    value = weighted_support(self.xyz[list(key)], self.radii[list(key)])
                except RuntimeError:
                    self.failures += 1
                    raise
                finally:
                    self.seconds += perf_counter() - started
                self.cache[key] = value
            value = self.cache[key]
            if value[1] < best[1]:
                best = value
        return best


def crosscheck(seed=0, cases=50):
    rng = np.random.default_rng(seed)
    errors = []
    start = perf_counter()
    for index in range(cases):
        n = 3 + index % 2
        centers = rng.normal(size=(n, 3)) * rng.uniform(.5, 3.)
        radii = rng.choice([2.03, 2.09, 2.31, 2.8, 2.07, 2.20], size=n)
        point, tau = weighted_support(centers, radii)
        _, reference = independent_support(centers, radii)
        errors.append(abs(tau - reference))
        if np.max(np.linalg.norm(centers - point, axis=1) - radii - tau) > 1e-5:
            raise RuntimeError("crosscheck primal residual")
    return {"cases": cases, "max_abs_delta_tau": max(errors), "seconds": perf_counter() - start,
            "certificate_pass_rate": 1., "reference": "SciPy SLSQP signed epigraph"}


def cliques(adjacency, order):
    def visit(prefix, available):
        if len(prefix) == order:
            yield tuple(prefix)
            return
        for vertex in sorted(available):
            yield from visit(prefix + [vertex], {v for v in available & adjacency[vertex] if v > vertex})
    yield from visit([], set(range(len(adjacency))))
