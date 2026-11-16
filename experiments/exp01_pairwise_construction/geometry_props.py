# -*- coding: utf-8 -*-
"""Synthetic geometric validation of the GSPH common-support core.

All tests operate on ``gsph6.geometry.compute_patch_support`` — the exact
union-of-lenses support solver — with hand-built InteractionPatch objects.
No PDB parsing, no extraction pipeline: this isolates the MATH.

Tests
-----
T1 monotonicity      : pi subset pi'  ==>  tau(pi) <= tau(pi')
T2 rigid invariance  : SE(3) transform leaves every tau unchanged
T3 perturbation      : Gaussian atom jitter eps -> |Dtau| ~ O(eps)
T4 higher-order      : (a) pairwise-overlap booleans identical but triple
                       support differs (exact construction);
                       (b) eta_3 = tau_3 - max tau_2 > 0 prevalence;
                       (c) pairwise-tau vectors do NOT determine tau_3
T5 direction         : tau is set-valued (reversal invariant); chemical
                       direction enters only through edge enumeration
T6 filtration ctrl   : tau-filtration vs edge-weight-filtration birth
                       orderings differ; shuffled-tau destroys structure
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
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsph6.geometry import compute_patch_support  # noqa: E402
from gsph6.patches import InteractionPatch, MicroContact  # noqa: E402
import gsph6.geometry as _geo  # noqa: E402

# The C++ support backend is unavailable on this host (torch DLL load
# failure); each attempt costs ~0.4 s inside _atom_support.  Bypass it and
# use the certified python cascade directly — identical math, faster here.
_orig_atom_support = _geo._atom_support


def _fast_atom_support(atom_ids, atoms, use_cpp=True, support_stats=None):
    return _orig_atom_support(atom_ids, atoms, use_cpp=False,
                              support_stats=support_stats)


_geo._atom_support = _fast_atom_support

OUT = Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\ablation")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261115)
R_C = 1.70  # VDW_RADII['C']


# ---------------------------------------------------------------- atoms/patches
def _atom(i, xyz, element="C"):
    return SimpleNamespace(index=i, xyz=np.asarray(xyz, float), element=element,
                           name="X", resname="GLY", chain="A", resid=i, icode="")


class World:
    """Container mapping atom index -> atom, building patches on demand."""

    def __init__(self):
        self.atoms = {}
        self._next = 0

    def ball(self, p):
        """Register a coincident donor/acceptor pair acting as one ball."""
        i = self._next; j = self._next + 1
        self._next += 2
        p = np.asarray(p, float)
        self.atoms[i] = _atom(i, p + np.array([1e-3, 0, 0]))
        self.atoms[j] = _atom(j, p - np.array([1e-3, 0, 0]))
        return MicroContact(donor=i, acceptor=j, interaction_type="hb",
                            distance=2e-3, tau=0.0, attributes={})

    def patch(self, pid, points):
        contacts = [self.ball(p) for p in points]
        ids = sorted({a for c in contacts for a in c.atom_ids})
        return InteractionPatch(
            patch_id=pid,
            donor_site=("A", pid, "d"), acceptor_site=("A", pid, "a"),
            contacts=contacts, donor_members=tuple(ids[::2]),
            acceptor_members=tuple(ids[1::2]), geometry={}, physical={})

    def tau(self, patches):
        r = compute_patch_support(list(patches), self.atoms)
        return float(r.tau) if r.certificate_passed else float("nan")


def random_world(n_patches=4, max_opts=2, lo=0.0, hi=12.0):
    w = World()
    pts = rng.uniform(lo, hi, size=(n_patches, max_opts, 3))
    # keep options of one patch apart so unions are non-trivial
    patches = [w.patch(i, pts[i]) for i in range(n_patches)]
    return w, patches


# ---------------------------------------------------------------------- T1/T2
def t1_monotonicity(n_trials=60):
    viol = []
    for t in range(n_trials):
        w, patches = random_world(n_patches=4, max_opts=2)
        taus = {}
        for k in (1, 2, 3):
            for combo in combinations(range(4), k):
                taus[combo] = w.tau([patches[i] for i in combo])
        for small in taus:
            for big in taus:
                if set(small) < set(big) and taus[big] < taus[small] - 1e-9:
                    viol.append((t, small, big, taus[small], taus[big]))
    return dict(n_trials=n_trials, violations=len(viol), examples=viol[:5])


def t2_rigid(n_trials=40):
    max_dev = 0.0
    for t in range(n_trials):
        w, patches = random_world(n_patches=4, max_opts=2)
        paths = [list(c) for k in (2, 3) for c in combinations(range(4), k)]
        base = {tuple(p): w.tau([patches[i] for i in p]) for p in paths}
        # random SE(3): QR rotation + translation
        A = rng.normal(size=(3, 3)); Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        tv = rng.uniform(-50, 50, 3)
        for a in w.atoms.values():
            a.xyz = (Q @ a.xyz) + tv
        for p in paths:
            dev = abs(w.tau([patches[i] for i in p]) - base[tuple(p)])
            max_dev = max(max_dev, dev)
    return dict(n_trials=n_trials, max_abs_dtau=max_dev)


# ---------------------------------------------------------------------- T3
def t3_perturbation(n_trials=30):
    rows = []
    for eps in (0.1, 0.25, 0.5, 1.0):
        worst = 0.0
        for t in range(n_trials):
            w, patches = random_world(n_patches=4, max_opts=2)
            paths = [list(c) for k in (2, 3) for c in combinations(range(4), k)]
            base = {tuple(p): w.tau([patches[i] for i in p]) for p in paths}
            for a in w.atoms.values():
                a.xyz = a.xyz + rng.normal(0, eps / np.sqrt(3), 3)
            for p in paths:
                d = abs(w.tau([patches[i] for i in p]) - base[tuple(p)])
                worst = max(worst, d)
        rows.append(dict(eps_A=eps, max_abs_dtau=worst,
                         ratio=worst / eps))
    return rows


# ---------------------------------------------------------------------- T4
def _tau3_of_points(A1, A2, c, r=R_C):
    """tau of triple {A1,A2,c} for coincident-ball contacts (exact solver)."""
    w = World()
    p1 = w.patch(0, [A1]); p2 = w.patch(1, [A2]); p3 = w.patch(2, [c])
    return w.tau([p1, p2, p3])


def _pair_tau(p, q, r=R_C):
    w = World()
    return w.tau([w.patch(0, [p]), w.patch(1, [q])])


def t4a_threshold_construction():
    """Exact construction: identical pairwise tau, different triple tau.

    A1=(0,0,0), A2=(6,0,0).  Patch3 options:
      config A: {(3,4,0)}          -> g1=g2=0.8, tau3 = 1.425
      config B: {(-4,3,0),(10,3,0)} -> g1=g2=0.8, tau3 = 3.52
    Pairwise tau(P1,P3)=tau(P2,P3)=0.8 and tau(P1,P2)=1.3 in BOTH configs;
    only the triple common support differs.  Ordinary pairwise/edge-weight
    filtrations cannot express this difference.
    """
    A1 = np.array([0., 0., 0.]); A2 = np.array([6., 0., 0.])
    out = {}
    for name, opts in (("A", [np.array([3., 4., 0.])]),
                       ("B", [np.array([-4., 3., 0.]), np.array([10., 3., 0.])])):
        w = World()
        p1 = w.patch(0, [A1]); p2 = w.patch(1, [A2]); p3 = w.patch(2, opts)
        out[name] = dict(
            tau12=w.tau([p1, p2]), tau13=w.tau([p1, p3]),
            tau23=w.tau([p2, p3]), tau123=w.tau([p1, p2, p3]),
            n_options3=len(opts))
    return out


def t4b_eta_prevalence(n_trials=300):
    """Fraction of random triples with eta_3 = tau3 - max(pair tau) > tol."""
    pos = 0; vals = []
    for _ in range(n_trials):
        w, patches = random_world(n_patches=3, max_opts=2, hi=10.0)
        p = patches
        t12 = w.tau(p[:2]); t13 = w.tau([p[0], p[2]]); t23 = w.tau(p[1:])
        t123 = w.tau(p)
        if not all(np.isfinite([t12, t13, t23, t123])):
            continue
        eta = t123 - max(t12, t13, t23)
        vals.append(eta)
        if eta > 1e-6:
            pos += 1
    vals = np.asarray(vals)
    return dict(n=len(vals), frac_eta_pos=float(pos / max(len(vals), 1)),
                eta_mean=float(vals.mean()), eta_max=float(vals.max()),
                eta_q90=float(np.quantile(vals, 0.9)))


def t4c_pairwise_not_determinate(n_trials=200):
    """Show tau_3 residual variance after pairwise taus: fit is impossible
    in general — report correlation and residual spread instead."""
    rows = []
    for _ in range(n_trials):
        w, patches = random_world(n_patches=3, max_opts=2, hi=10.0)
        p = patches
        t12 = w.tau(p[:2]); t13 = w.tau([p[0], p[2]]); t23 = w.tau(p[1:])
        t123 = w.tau(p)
        if np.isfinite([t12, t13, t23, t123]).all():
            rows.append((t12, t13, t23, t123))
    a = np.asarray(rows)
    # partial info: correlation of t123 with max pairwise
    mp = a[:, :3].max(axis=1)
    return dict(n=len(a),
                corr_t3_vs_maxpair=float(np.corrcoef(mp, a[:, 3])[0, 1]),
                resid_std_after_maxpair=float(np.std(a[:, 3] - mp)),
                t3_range=[float(a[:, 3].min()), float(a[:, 3].max())])


# ---------------------------------------------------------------------- T5
def t5_direction(n_trials=30):
    """tau(path) is set-valued: reversal gives identical tau by construction.
    Chemical direction enters ONLY through edge enumeration (directed
    acceptor->donor transfer graph), not through the support value.
    """
    max_dev = 0.0
    for _ in range(n_trials):
        w, patches = random_world(n_patches=4, max_opts=2)
        for combo in combinations(range(4), 3):
            fwd = [patches[i] for i in combo]
            rev = [patches[i] for i in combo[::-1]]
            d = abs(w.tau(fwd) - w.tau(rev))
            max_dev = max(max_dev, d)
    # directed vs symmetrized path enumeration on a tiny synthetic graph
    edges_dir = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 2, 2.0)]
    try:
        from gsph6.graph import enumerate_patch_paths
        pd_, cd_, be = enumerate_patch_paths([0, 1, 2, 3], edges_dir, max_order=3)
        n_dir = int(sum(v.shape[0] for v in pd_.values()))
        und_edges = sorted({(a, b, w_) for a, b, w_ in edges_dir} |
                           {(b, a, w_) for a, b, w_ in edges_dir})
        pu, cu, _ = enumerate_patch_paths([0, 1, 2, 3], und_edges, max_order=3)
        n_und = int(sum(v.shape[0] for v in pu.values()))
        enum = dict(backend_dir=be, n_directed_paths=n_dir,
                    n_undirected_paths=n_und)
    except Exception as exc:  # pragma: no cover
        enum = dict(error=f"{type(exc).__name__}: {exc}")
    return dict(max_abs_dtau_reversal=max_dev, enumeration=enum)


# ---------------------------------------------------------------------- T6
def t6_filtration_controls(n_trials=60):
    """Compare path birth under tau-filtration vs edge-weight filtration.

    Edge-weight birth = bottleneck over consecutive patch pairs of the
    minimum donor/acceptor atom distance between the two patches -- the
    ordinary graph surrogate where each patch adjacency carries its closest
    contact distance.  Shuffled-tau control permutes tau across paths.
    """
    inversions = 0; pairs_checked = 0
    shuffle_viol = 0
    for _ in range(n_trials):
        w, patches = random_world(n_patches=4, max_opts=1, hi=10.0)
        paths = [list(c) for k in (2, 3) for c in combinations(range(4), k)]
        # contact-atom coordinates per patch (donor + acceptor atoms)
        patch_pts = []
        for p in patches:
            pts = []
            for c in p.contacts:
                pts.append(w.atoms[c.donor].xyz)
                pts.append(w.atoms[c.acceptor].xyz)
            patch_pts.append(np.asarray(pts) if pts else np.zeros((0, 3)))
        taus, ews = [], []
        for p in paths:
            order = sorted(p)
            taus.append(w.tau([patches[i] for i in p]))
            # bottleneck edge weight: max over adjacent patch pairs of the
            # closest inter-patch contact distance
            ew = 0.0
            for u, v in zip(order[:-1], order[1:]):
                pu, pv = patch_pts[u], patch_pts[v]
                if len(pu) == 0 or len(pv) == 0:
                    continue
                d = np.linalg.norm(pu[:, None, :] - pv[None, :, :], axis=2)
                ew = max(ew, float(d.min()))
            ews.append(ew)
        taus = np.asarray(taus); ews = np.asarray(ews)
        ok = np.isfinite(taus)
        taus, ews = taus[ok], ews[ok]
        # count discordant orderings between the two filtrations
        for i in range(len(taus)):
            for j in range(i + 1, len(taus)):
                pairs_checked += 1
                if (taus[i] - taus[j]) * (ews[i] - ews[j]) < 0:
                    inversions += 1
        # shuffled tau must NOT reproduce the tau ordering
        perm = rng.permutation(len(taus))
        if not np.allclose(np.argsort(taus), np.argsort(taus[perm])):
            shuffle_viol += 1
    return dict(n_trials=n_trials, pairs_checked=pairs_checked,
                discordant_fraction=float(inversions / max(pairs_checked, 1)),
                shuffle_breaks_ordering_frac=float(shuffle_viol / n_trials))


def main():
    results = {}
    print("[T1] monotonicity ...", flush=True)
    results["T1_monotonicity"] = t1_monotonicity()
    print("  ", results["T1_monotonicity"], flush=True)
    print("[T2] rigid invariance ...", flush=True)
    results["T2_rigid"] = t2_rigid()
    print("  ", results["T2_rigid"], flush=True)
    print("[T3] perturbation ...", flush=True)
    results["T3_perturbation"] = t3_perturbation()
    for r in results["T3_perturbation"]:
        print("  ", r, flush=True)
    print("[T4a] threshold construction ...", flush=True)
    results["T4a_pairwise_identical_triple_differs"] = t4a_threshold_construction()
    print("  ", json.dumps(results["T4a_pairwise_identical_triple_differs"], indent=1), flush=True)
    print("[T4b] eta_3 prevalence ...", flush=True)
    results["T4b_eta3"] = t4b_eta_prevalence()
    print("  ", results["T4b_eta3"], flush=True)
    print("[T4c] pairwise->tau3 determinacy ...", flush=True)
    results["T4c"] = t4c_pairwise_not_determinate()
    print("  ", results["T4c"], flush=True)
    print("[T5] direction ...", flush=True)
    results["T5_direction"] = t5_direction()
    print("  ", results["T5_direction"], flush=True)
    print("[T6] filtration controls ...", flush=True)
    results["T6_filtration"] = t6_filtration_controls()
    print("  ", results["T6_filtration"], flush=True)
    (OUT / "geometry_props.json").write_text(json.dumps(results, indent=1, default=float))
    print("[done] ->", OUT / "geometry_props.json")


if __name__ == "__main__":
    main()
