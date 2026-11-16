# property tests: tau monotonicity, rigid invariance, duplicate invariance
import numpy as np
import pytest

from cspph.core.gsph1 import Atom
from cspph.chemistry.patch_builder import InteractionPatch, MicroContact
from cspph.geometry.patch_support import compute_patch_support


def make_atom(i, xyz):
    return Atom(index=i, chain="A", name=f"A{i}", resname="ALA",
                resid=i, element="C", xyz=np.asarray(xyz, float))


def make_patch(pid, atom_pairs):
    donors = sorted({d for d, a in atom_pairs})
    acceptors = sorted({a for d, a in atom_pairs})
    contacts = [MicroContact(donor=d, acceptor=a, interaction_type="hbond", distance=1.0, tau=0.5)
                for d, a in atom_pairs]
    return InteractionPatch(pid, ("s", pid, "d"), ("s", pid, "a"),
                            contacts, tuple(donors), tuple(acceptors),
                            {}, {})


def random_world(seed):
    rng = np.random.default_rng(seed)
    atoms = {i: make_atom(i, rng.uniform(0, 20, 3)) for i in range(1, 13)}
    patches = []
    for pid in range(4):
        pairs = [(int(rng.integers(1, 7)), int(rng.integers(7, 13)))
                 for _ in range(int(rng.integers(1, 3)))]
        patches.append(make_patch(pid, pairs))
    return atoms, patches


def test_tau_monotone_adding_patch():
    atoms, patches = random_world(0)
    taus = []
    for k in (1, 2, 3, 4):
        r = compute_patch_support(patches[:k], atoms)
        if r.certificate_passed and np.isfinite(r.tau):
            taus.append(r.tau)
    # nondecreasing sequence wherever certified
    for a, b in zip(taus, taus[1:]):
        assert b >= a - 1e-6


def test_rigid_motion_invariance():
    atoms, patches = random_world(1)
    rng = np.random.default_rng(2)
    # random rotation + translation
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    t = rng.uniform(-5, 5, 3)
    atoms2 = {i: make_atom(i, Q @ a.xyz + t) for i, a in atoms.items()}
    r1 = compute_patch_support(patches[:3], atoms)
    r2 = compute_patch_support(patches[:3], atoms2)
    if r1.certificate_passed and r2.certificate_passed:
        assert r2.tau == pytest.approx(r1.tau, abs=1e-6)


def test_duplicate_patch_set_invariant():
    atoms, patches = random_world(3)
    r1 = compute_patch_support(patches[:3], atoms)
    r2 = compute_patch_support(patches[:3] + [patches[0]], atoms)
    assert r2.tau == pytest.approx(r1.tau)
