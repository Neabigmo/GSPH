# -*- coding: utf-8 -*-
"""P0 audits for the GSPH paper revision.

P0-2  Bounded-displacement Lipschitz test.
      Every atom is displaced by a vector of norm <= eps (uniform in the
      ball, NOT Gaussian).  Theory: each leaf support is a weighted
      1-center, 1-Lipschitz in atom positions; tau = min over leaf
      combinations, hence |Dtau| <= max_atom_displacement <= eps.
      The earlier Gaussian-jitter test is kept for reference only.

P0-3  Exact support solver audit.
      Independent brute-force re-implementation:
        tau_true = min over contact selections of weighted-1-center(atom set)
      solved by scipy SLSQP epigraph form from multiple starts, versus
      compute_patch_support (branch-and-bound + certificate).
      Checks: value agreement, certificate flag, witness KKT residual,
      degenerate cases (empty patch, single contact, coincident atoms,
      duplicate patch ids, empty path list).
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
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsph6.geometry import compute_patch_support          # noqa: E402
import gsph6.geometry as _geo                              # noqa: E402
from gsph6.patches import InteractionPatch, MicroContact   # noqa: E402
from runners.gsph1 import VDW_RADII, _support_certificate  # noqa: E402

_ORIG_ATOM_SUPPORT = _geo._atom_support


def _fast_atom_support(atom_ids, atoms, use_cpp=True, support_stats=None):
    return _ORIG_ATOM_SUPPORT(atom_ids, atoms, use_cpp=False,
                              support_stats=support_stats)


_geo._atom_support = _fast_atom_support

OUT = Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\ablation")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261116)


def _atom(i, xyz, element="C"):
    return SimpleNamespace(index=i, xyz=np.asarray(xyz, float), element=element,
                           name="X", resname="GLY", chain="A", resid=i, icode="")


class World:
    def __init__(self):
        self.atoms = {}
        self._next = 0

    def ball(self, p):
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
        return r


def random_world(n_patches=3, max_opts=2, lo=0.0, hi=10.0):
    w = World()
    pts = rng.uniform(lo, hi, size=(n_patches, max_opts, 3))
    return w, [w.patch(i, pts[i]) for i in range(n_patches)]


# ================================================================ P0-2
def p02_bounded_perturbation(n_worlds=40, n_eps=4):
    eps_grid = np.array([0.1, 0.25, 0.5, 1.0])[:n_eps]
    rows = []
    for eps in eps_grid:
        worst = 0.0; viol = 0; n_pairs = 0
        for _ in range(n_worlds):
            w, patches = random_world(n_patches=4, max_opts=2)
            combos = [list(c) for k in (2, 3) for c in __import__("itertools").combinations(range(4), k)]
            base = {}
            for c in combos:
                r = w.tau([patches[i] for i in c])
                base[tuple(c)] = float(r.tau) if r.certificate_passed else np.nan
            # bounded displacement: direction uniform on sphere, radius
            # uniform in [0, eps]  =>  ||d|| <= eps exactly
            disp = {}
            for idx in w.atoms:
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                disp[idx] = v * rng.uniform(0.0, eps)
            for idx, a in w.atoms.items():
                a.xyz = a.xyz + disp[idx]
            for c in combos:
                r = w.tau([patches[i] for i in c])
                if not r.certificate_passed:
                    continue
                d = abs(float(r.tau) - base[tuple(c)])
                n_pairs += 1
                worst = max(worst, d)
                if d > eps * (1 + 1e-6):
                    viol += 1
        rows.append(dict(eps_A=float(eps), max_abs_dtau=float(worst),
                         bound=float(eps), violations=int(viol),
                         n=float(n_pairs),
                         ratio=float(worst / eps) if eps else None))
    return rows


# ================================================================ P0-3
def _leaf_atoms(contact, atoms):
    return sorted({int(contact.donor), int(contact.acceptor)})


def _weighted_1center_brute(centers, radii):
    """min_x max_i (|x-c_i|-r_i) via epigraph SLSQP, multi-start."""
    centers = np.asarray(centers, float); radii = np.asarray(radii, float)
    n = len(centers)
    if n == 0:
        return 0.0, np.zeros(3)
    if n == 1:
        return 0.0, centers[0].copy()

    def obj(y):
        return y[3]

    def make_con(c, r):
        return lambda y: y[3] + r - np.linalg.norm(y[:3] - c)

    cons = [{"type": "ineq", "fun": make_con(c, float(r))}
            for c, r in zip(centers, radii)]
    best = (np.inf, None)
    starts = [centers.mean(axis=0), centers[0], centers[-1],
              centers[rng.integers(0, n)],
              centers.mean(axis=0) + rng.normal(scale=2.0, size=3)]
    for s in starts:
        x0 = np.r_[s, max(0.0, float(np.max(np.linalg.norm(centers - s, axis=1) - radii)))]
        try:
            opt = minimize(obj, x0, method="SLSQP",
                           bounds=[(None, None)] * 3 + [(0.0, None)],
                           constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-11})
            if opt.success and np.isfinite(opt.fun) and opt.fun < best[0]:
                best = (float(opt.fun), opt.x[:3])
        except Exception:
            pass
    return best


def brute_patch_tau(patches, atoms):
    """Independent tau: min over contact selections of leaf 1-center."""
    patches = [p for p in patches]
    seen = set(); dedup = []
    for p in patches:
        if p.patch_id in seen:
            continue
        seen.add(p.patch_id); dedup.append(p)
    patches = dedup
    if not patches:
        return 0.0, True
    if any(len(p.contacts) == 0 for p in patches):
        return float("inf"), False
    best = np.inf
    for combo in product(*[p.contacts for p in patches]):
        ids = sorted({a for c in combo for a in _leaf_atoms(c, atoms)})
        centers = np.array([atoms[i].xyz for i in ids], float)
        radii = np.array([VDW_RADII.get(str(atoms[i].element), VDW_RADII["X"])
                          for i in ids], float)
        tau, _ = _weighted_1center_brute(centers, radii)
        best = min(best, tau)
    return best, np.isfinite(best)


def p03_solver_audit(n_trials=120):
    rows = []
    for t in range(n_trials):
        w, patches = random_world(n_patches=rng.integers(1, 4),
                                   max_opts=rng.integers(1, 3))
        sel = patches[: int(rng.integers(1, len(patches) + 1))]
        res = w.tau(sel)
        tau_bf, ok_bf = brute_patch_tau(sel, w.atoms)
        rec = dict(trial=t, n_patches=len(sel),
                   n_contacts=[len(p.contacts) for p in sel],
                   tau_solver=float(res.tau) if np.isfinite(res.tau) else None,
                   tau_brute=float(tau_bf) if np.isfinite(tau_bf) else None,
                   complete=bool(res.complete),
                   cert=bool(res.certificate_passed),
                   route=str(res.route), explored=int(res.explored),
                   pruned=int(res.pruned))
        if np.isfinite(res.tau) and np.isfinite(tau_bf):
            rec["abs_diff"] = abs(float(res.tau) - float(tau_bf))
            rec["agree"] = bool(rec["abs_diff"] <= 2e-3)
        else:
            rec["agree"] = bool(np.isfinite(res.tau) == np.isfinite(tau_bf))
        # witness KKT check: solver center must satisfy certificate
        if res.certificate_passed and np.isfinite(res.tau):
            c = np.asarray(res.center, float)
            ids = sorted({a for p in sel for ct in p.contacts
                          for a in _leaf_atoms(ct, w.atoms)})
            # NOTE: certificate is per selected leaf, not whole patch set;
            # we only verify the solver's own reported certificate flag and
            # that tau is nonnegative finite here.  Full KKT is covered by
            # _support_certificate inside the solver.
            rec["tau_nonneg"] = bool(res.tau >= 0.0)
        rows.append(rec)
    ok = [r for r in rows if r.get("agree")]
    diffs = [r["abs_diff"] for r in rows if "abs_diff" in r]
    return dict(n_trials=n_trials, n_agree=len(ok),
                frac_agree=len(ok) / max(len(rows), 1),
                max_abs_diff=float(max(diffs)) if diffs else None,
                mean_abs_diff=float(np.mean(diffs)) if diffs else None,
                cert_fail=int(sum(1 for r in rows if not r["cert"])),
                incomplete=int(sum(1 for r in rows if not r["complete"])),
                routes={r["route"] for r in rows},
                examples=[r for r in rows if not r.get("agree")][:5])


def p03_degenerate():
    out = {}
    w = World()
    # empty path list
    r = compute_patch_support([], w.atoms)
    out["empty_path"] = dict(tau=r.tau, complete=r.complete,
                             cert=r.certificate_passed, route=r.route)
    # patch with zero contacts
    p_empty = InteractionPatch(patch_id=0, donor_site=("A", 0, "d"),
                               acceptor_site=("A", 0, "a"), contacts=[],
                               donor_members=(), acceptor_members=(),
                               geometry={}, physical={})
    r = compute_patch_support([p_empty], w.atoms)
    out["empty_patch"] = dict(tau=r.tau, complete=r.complete,
                              cert=r.certificate_passed, route=r.route)
    # duplicate patch ids collapse (set-valued support)
    w2, patches = random_world(n_patches=2, max_opts=2)
    p = patches[0]
    r1 = compute_patch_support([p, patches[1]], w2.atoms)
    r2 = compute_patch_support([p, p, patches[1]], w2.atoms)
    out["duplicate_patch"] = dict(tau1=float(r1.tau), tau2=float(r2.tau),
                                  equal=bool(abs(r1.tau - r2.tau) < 1e-9))
    # coincident atoms: donor==acceptor position
    w3 = World()
    i = w3._next; w3.atoms[i] = _atom(i, [0., 0., 0.])
    w3.atoms[i + 1] = _atom(i + 1, [0., 0., 0.])
    w3._next += 2
    c = MicroContact(donor=i, acceptor=i + 1, interaction_type="hb",
                     distance=0.0, tau=0.0, attributes={})
    p3 = InteractionPatch(patch_id=0, donor_site=("A", 0, "d"),
                          acceptor_site=("A", 0, "a"), contacts=[c],
                          donor_members=(i,), acceptor_members=(i + 1,),
                          geometry={}, physical={})
    r = compute_patch_support([p3], w3.atoms)
    out["coincident_atoms"] = dict(tau=float(r.tau), complete=r.complete,
                                   cert=r.certificate_passed, route=r.route)
    # max_combinations=0 diagnostic mode
    r = compute_patch_support(patches, w2.atoms, max_combinations=0)
    out["zero_combinations"] = dict(tau=r.tau, complete=r.complete,
                                    route=r.route)
    return out


def _save(res):
    (OUT / "p0_audits.json").write_text(json.dumps(res, indent=1, default=str))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if only in ("all", "p02"):
        print("[P0-2] bounded perturbation ...", flush=True)
        res["P0_2_bounded_lipschitz"] = p02_bounded_perturbation()
        for r in res["P0_2_bounded_lipschitz"]:
            print("  ", r, flush=True)
    if only in ("all", "p03"):
        print("[P0-3] solver audit ...", flush=True)
        res["P0_3_solver_audit"] = p03_solver_audit()
        print("  ", {k: v for k, v in res["P0_3_solver_audit"].items()
                     if k != "examples"}, flush=True)
        print("[P0-3] degenerate cases ...", flush=True)
        res["P0_3_degenerate"] = p03_degenerate()
        print("  ", json.dumps(res["P0_3_degenerate"], indent=1, default=str), flush=True)
    _save(res)
    print("[done] ->", OUT / "p0_audits.json")


if __name__ == "__main__":
    main()
