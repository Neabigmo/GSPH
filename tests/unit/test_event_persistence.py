from collections import Counter
from itertools import combinations
from random import Random

import numpy as np
import pytest
from scipy import sparse

from cspph.topology.event_persistence import compare_event_barcodes
from cspph.topology.glmy_sparse import _sparse_rank_mod, boundary_sparse
from cspph.topology.persistence import (
    _modular_rref_nullspace,
    intervals_from_rank_invariant,
)


def _diamond(vertices, death):
    a, b, c, d = vertices
    return {
        (a,): 0.0, (b,): 0.0, (c,): 0.0, (d,): 0.0,
        (a, b): 0.4, (a, c): 0.4, (b, d): 0.4, (c, d): 0.4,
        (a, b, d): 0.8, (a, c, d): death,
    }


def test_shared_angstrom_values_keep_multiplicities_and_infinite_intervals():
    common = _diamond(range(4), 1.2) | _diamond(range(4, 8), 1.2)
    pairwise = _diamond(range(4), 0.9) | _diamond(range(4, 8), 0.9)
    scales, common_bars, pairwise_bars = compare_event_barcodes(common, pairwise)
    assert scales == [0.0, 0.4, 0.8, 0.9, 1.2]
    assert common_bars[1] == [(0.4, 1.2)] * 2
    assert pairwise_bars[1] == [(0.4, 0.9)] * 2
    assert common_bars[0] == [(0.0, 0.4)] * 6 + [(0.0, None)] * 2
    assert common_bars[2] == pairwise_bars[2] == []


def test_face_birth_can_release_an_existing_path_into_omega():
    births = _diamond(range(4), 0.8)
    del births[(0, 2, 3)]
    births[(0, 3)] = 1.2
    _, bars, _ = compare_event_barcodes(births, births)
    # Adding 0,3 would create a cycle, but simultaneously admitting
    # 0,1,3 to Omega makes this new cycle a boundary at the same scale.
    assert bars[1] == [(0.4, None)]


def test_tetrahedral_boundary_persists_until_the_order_three_path_enters():
    common = {path: 0.2 * (size - 1) for size in range(1, 4)
              for path in combinations(range(4), size)}
    common[(0, 1, 2, 3)] = 0.9
    pairwise = common | {(0, 1, 2, 3): 0.7}
    _, common_bars, pairwise_bars = compare_event_barcodes(common, pairwise)
    assert common_bars[2] == [(0.4, 0.9)]
    assert pairwise_bars[2] == [(0.4, 0.7)]


def _rank_reference(allowed, order):
    full = {p: i for i, p in enumerate(allowed[-1][order])}

    def embed(matrix, paths):
        coo = matrix.tocoo()
        rows = [full[paths[r]] for r in coo.row]
        return sparse.csr_matrix((coo.data, (rows, coo.col)), shape=(len(full), matrix.shape[1]))

    cycles, boundaries = [], []
    for paths in allowed:
        if order:
            regular, forbidden, _ = boundary_sparse(paths[order], paths[order - 1])
            z = _modular_rref_nullspace(sparse.vstack([regular, forbidden]))[0]
        else:
            z = sparse.eye(len(paths[0]), dtype=np.int64)
        regular, forbidden, _ = boundary_sparse(paths[order + 1], paths[order])
        omega = _modular_rref_nullspace(forbidden)[0]
        cycles.append(embed(z, paths[order]))
        boundaries.append(embed(regular @ omega, paths[order]))
    ranks = np.zeros((len(allowed), len(allowed)), dtype=int)
    for j, boundary in enumerate(boundaries):
        boundary_rank = _sparse_rank_mod(boundary)
        for i in range(j + 1):
            ranks[i, j] = _sparse_rank_mod(sparse.hstack([cycles[i], boundary]).tocsr()) - boundary_rank
    return intervals_from_rank_invariant(ranks)


@pytest.mark.parametrize("seed", range(12))
def test_event_reduction_agrees_with_snapshot_rank_invariant(seed):
    rng = Random(seed)
    births = {(i,): 0.0 for i in range(4)}
    for order in range(1, 4):
        for prefix, birth in list(births.items()):
            if len(prefix) != order:
                continue
            for vertex in range(4):
                if vertex != prefix[-1] and rng.random() < 0.65:
                    births[prefix + (vertex,)] = max(birth, rng.randrange(4) / 4)
    scales, actual, _ = compare_event_barcodes(births, births)
    allowed = [
        {order: [path for path, b in births.items() if len(path) == order + 1 and b <= t]
         for order in range(4)}
        for t in scales
    ]
    for order in range(3):
        expected = Counter(
            (scales[bar["birth"]], None if bar["death"] == len(scales) else scales[bar["death"]])
            for bar in _rank_reference(allowed, order)
        )
        assert Counter(actual[order]) == expected


def test_rejects_different_path_sets_and_nonfinite_scales():
    with pytest.raises(ValueError, match="same paths"):
        compare_event_barcodes({(0,): 0.0}, {(1,): 0.0})
    with pytest.raises(ValueError, match="finite births"):
        compare_event_barcodes({(0,): 0.0}, {(0,): float("nan")})
