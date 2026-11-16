# cspph geometry: patch union support unit tests
import numpy as np
import pytest

from cspph.core.gsph1 import Atom
from cspph.chemistry.patch_builder import InteractionPatch, MicroContact
from cspph.geometry.patch_support import compute_patch_support


def make_atom(i, xyz, element="C"):
    return Atom(index=i, chain="A", name=f"A{i}", resname="ALA",
                resid=i, element=element, xyz=np.asarray(xyz, float))


def make_patch(pid, contacts, donors, acceptors):
    return InteractionPatch(
        patch_id=pid,
        donor_site=("site", pid, "d"),
        acceptor_site=("site", pid, "a"),
        contacts=[MicroContact(donor=d, acceptor=a, interaction_type="hbond", distance=dist, tau=tau)
                  for d, a, dist, tau in contacts],
        donor_members=tuple(donors),
        acceptor_members=tuple(acceptors),
        geometry={}, physical={})


@pytest.fixture
def atoms():
    # two clusters of atoms ~4A apart
    return {
        1: make_atom(1, [0, 0, 0]), 2: make_atom(2, [0.5, 0, 0]),
        3: make_atom(3, [4, 0, 0]), 4: make_atom(4, [4.5, 0, 0]),
        5: make_atom(5, [0, 4, 0]), 6: make_atom(6, [0.5, 4, 0]),
    }


def test_single_contact_support_certified(atoms):
    p = make_patch(0, [(1, 3, 4.0, 1.5)], [1], [3])
    r = compute_patch_support([p], atoms)
    assert r.certificate_passed
    assert np.isfinite(r.tau) and r.tau >= 0.0


def test_duplicate_patch_dedup(atoms):
    p = make_patch(0, [(1, 3, 4.0, 1.5)], [1], [3])
    r1 = compute_patch_support([p], atoms)
    r2 = compute_patch_support([p, p], atoms)
    assert r2.tau == pytest.approx(r1.tau)


def test_empty_contacts_patch(atoms):
    p = make_patch(0, [], [1], [3])
    r = compute_patch_support([p], atoms)
    # empty union -> no feasible witness or tau=inf; must not crash
    assert isinstance(r.tau, float)


def test_support_monotone_under_patch_addition(atoms):
    p1 = make_patch(0, [(1, 3, 4.0, 1.5)], [1], [3])
    p2 = make_patch(1, [(2, 4, 4.1, 1.6)], [2], [4])
    p3 = make_patch(2, [(5, 6, 4.2, 1.7)], [5], [6])
    r12 = compute_patch_support([p1, p2], atoms)
    r123 = compute_patch_support([p1, p2, p3], atoms)
    if r12.certificate_passed and r123.certificate_passed:
        assert r123.tau >= r12.tau - 1e-9
