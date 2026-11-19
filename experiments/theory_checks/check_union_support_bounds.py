"""Reproducible numerical checks for support of unions of contact lenses."""
from itertools import combinations, product
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cspph.core.gsph1 import VDW_RADII, _active_set_support, _support_certificate
from cspph.geometry.patch_support import _atom_support, compute_patch_support


TOL = 1e-6
SEED = 20260709
COUNT = 2000
JUNG = np.sqrt(1.5)


def union_support(patches, centers, radii, solve):
    """Enumerate selections, pruning only by atom-pair separation bounds."""
    if not patches:
        return 0.0
    if any(not patch for patch in patches):
        return float("inf")
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    bounds = np.maximum(0.0, (distances - radii[:, None] - radii[None, :]) / 2)
    candidates = []
    for selection in product(*patches):
        ids = tuple(sorted({i for contact in selection for i in contact}))
        candidates.append((float(bounds[np.ix_(ids, ids)].max()), ids))
    best = float("inf")
    for lower, ids in sorted(candidates):
        if lower >= best:
            break
        best = min(best, solve(ids))
    return best


def evaluator(patches, centers, radii, elements=None):
    cache = {}
    atoms = None if elements is None else {
        i: SimpleNamespace(xyz=point, element=element)
        for i, (point, element) in enumerate(zip(centers, elements))
    }

    def solve(ids):
        if ids not in cache:
            if atoms is not None:
                tau, _, ok, route = _atom_support(ids, atoms)
            else:
                selected = list(ids)
                result = _active_set_support(centers[selected], radii[selected])
                if result is None:
                    raise RuntimeError("explicit-radius active-set solver failed")
                point, tau = result
                ok, _, _ = _support_certificate(point, tau, centers[selected], radii[selected])
                route = "explicit-radius active set"
            if not ok or not np.isfinite(tau):
                raise RuntimeError(f"uncertified support: {route}")
            cache[ids] = float(tau)
        return cache[ids]

    def support(family):
        return union_support(family, centers, radii, solve)

    tau = support(patches)
    pair = max(support(list(pair)) for pair in combinations(patches, 2))
    return tau, pair, support, atoms


def spread(patches, centers):
    return max(
        (float(np.linalg.norm(centers[c[side]] - centers[d[side]]))
         for patch in patches for c in patch for d in patch for side in (0, 1)),
        default=0.0,
    )


def tetrahedron(side):
    return side / np.sqrt(2) * np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]])


def main():
    rng = np.random.default_rng(SEED)
    violations = [0, 0]
    worst_gap = float("inf")
    max_ratio = -float("inf")
    witness = None
    comparisons = 0
    for index in range(COUNT):
        patches = []
        size = 0
        for _ in range(int(rng.integers(3, 6))):
            contacts = int(rng.integers(1, 4))
            patches.append([(size + 2 * j, size + 2 * j + 1) for j in range(contacts)])
            size += 2 * contacts
        centers = rng.uniform(0, 10, (size, 3))
        elements = rng.choice(["O", "N", "C", "S"], size=size)
        radii = np.array([VDW_RADII[e] for e in elements])
        tau, pair, _, atoms = evaluator(patches, centers, radii, elements)
        delta = spread(patches, centers)
        ratio = (radii.min() + tau) / (radii.max() + pair + delta)
        worst_gap = min(worst_gap, tau - pair)
        violations[0] += int(tau + TOL < pair)
        violations[1] += int(radii.min() + tau > JUNG * (radii.max() + pair + delta) + TOL)
        if ratio > max_ratio:
            max_ratio = ratio
            witness = dict(index=index, patches=patches, centers=centers.tolist(),
                           elements=elements.tolist(), tau=tau, tau_pw=pair, delta=delta,
                           r_min=float(radii.min()), r_max=float(radii.max()), ratio=ratio)
        if index < 3:
            objects = [SimpleNamespace(patch_id=i, contacts=[
                SimpleNamespace(donor=d, acceptor=a, distance=float(np.linalg.norm(centers[d] - centers[a])),
                                tau=0.0) for d, a in patch], minimum_tau=0.0)
                for i, patch in enumerate(patches)]
            result = compute_patch_support(objects, atoms)
            if not result.complete or not result.certificate_passed or abs(result.tau - tau) > TOL:
                raise RuntimeError("enumeration differs from compute_patch_support")
            comparisons += 1

    rows = [
        ("Monotonicity", violations[0] == 0, f"n={COUNT}; failures={violations[0]}; min(tau-tau_pw)={worst_gap:.9f}"),
        ("Affine Jung", violations[1] == 0, f"n={COUNT}; failures={violations[1]}; max ratio={max_ratio:.9f}; bound={JUNG:.9f}"),
    ]
    r = 1.6
    vertices = tetrahedron(2 * r)
    patches = [[(i, i)] for i in range(4)]
    tau, pair, _, _ = evaluator(patches, vertices, np.full(4, r))
    expected = (JUNG - 1) * r
    rows.append(("Convex tetrahedron", abs(pair) <= TOL and abs(tau - expected) <= TOL,
                 f"tau_pw={pair:.9f}; tau={tau:.9f}; analytic={expected:.9f}"))
    triangle = np.array([[0, 0, 0], [8, 0, 0], [4, 4 * np.sqrt(3), 0]])
    for name, vertices in [("Hollow triangle", triangle), ("Hollow tetrahedron", tetrahedron(8))]:
        n = len(vertices)
        patches = [[(j, j) for j in range(n) if j != i] for i in range(n)]
        tau, pair, support, _ = evaluator(patches, vertices, np.full(n, r))
        sub = max(support(list(family)) for family in combinations(patches, n - 1))
        expected = 8 / 2 - r
        rows.append((name, abs(pair) <= TOL and abs(sub) <= TOL and abs(tau - expected) <= TOL,
                     f"tau_pw={pair:.9f}; max proper subfamily={sub:.9f}; tau={tau:.9f}; analytic={expected:.9f}"))
    print(f"seed={SEED}; tolerance={TOL:g}; coordinates/radii in angstroms")
    print(f"Enumeration agrees with compute_patch_support on {comparisons} random families.")
    print(f"{'Check':<23} {'Result':<6} Details")
    for name, passed, detail in rows:
        print(f"{name:<23} {'PASS' if passed else 'FAIL':<6} {detail}")
    print("Best sampled random witness (not a claim of sharpness; atom IDs index centers/elements):")
    print(json.dumps(witness, sort_keys=True))
    return int(not all(passed for _, passed, _ in rows))


if __name__ == "__main__":
    sys.exit(main())
