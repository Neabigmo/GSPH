from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gudhi")
pytest.importorskip("miniball")

from experiments.external_topology_cech_rips.run import (
    alpha_bars, archived_bars, check_simplex, distance, endpoint_error, point_controls,
    rips_bars, simplex_radius,
)
from experiments.external_topology_cech_rips import run


@pytest.mark.parametrize("points,radius", [
    ([[0, 0, 0]], 0),
    ([[0, 0, 0], [2, 0, 0]], 1),
    ([[0, 0, 0], [2, 0, 0], [1, np.sqrt(3), 0]], 2 / np.sqrt(3)),
    ([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], np.sqrt(8 / 3)),
    ([[0, 0, 0], [0, 0, 0], [2, 0, 0], [1, 0, 0]], 1),
])
def test_zero_radius_support_matches_miniball(points, radius):
    assert simplex_radius(points) == pytest.approx(radius, abs=1e-8)
    _, actual, tau, error = check_simplex((tuple(range(len(points))), points))
    assert actual == pytest.approx(tau, abs=1e-8)
    assert error <= 1e-8


def test_cech_alpha_agree_but_rips_differs_on_equilateral_triangle():
    points = [[0, 0, 0], [2, 0, 0], [1, np.sqrt(3), 0]]
    rows, audit = point_controls("triangle", points, [1, 2, 3], cap=4, workers=1)
    cech_alpha = [r for r in rows if r["left"] == "cech" and r["right"] == "alpha"]
    assert all(r["identical_1e8"] for r in cech_alpha)
    h1 = next(r for r in rows if r["left"] == "cech" and r["right"] == "rips" and r["order"] == 1)
    assert not h1["identical_1e8"]
    assert h1["bottleneck"] == pytest.approx((2 / np.sqrt(3) - 1) / 2)
    assert audit["simplex_checks"]["2"]["n_simplices"] == 1


def test_empty_cloud_and_diagrams():
    assert alpha_bars([]) == {0: [], 1: [], 2: []}
    assert distance([], []) == (0, 0, 0., True)
    assert rips_bars([]) == {0: [], 1: [], 2: []}


def test_multiplicity_censoring_and_matching():
    assert distance([(0, None)], [(0, None)] + [(1, None)])[2] == np.inf
    assert not distance([(0, 1)] * 2, [(0, 1)])[3]
    assert distance([(0, 1), (0.1, 2)], [(0.1, 2), (0, 1)])[3]
    assert distance([(0, 1e-10)], [])[3]
    assert not distance([(0, 1.5e-8)], [(1, 1 + 1.5e-8)])[3]
    assert endpoint_error([(0, None)], [(0, 2)]) == np.inf
    arm = {"chain_map_valid": True, "intervals": {"intervals_k0": [[0, 2, False], [1, 3, True]]}}
    assert archived_bars(arm, [0, .5, 2], 0) == [(0, 2), (.5, None)]
    arm["intervals"]["intervals_k0"] = [[0, 3, False]]
    with pytest.raises(ValueError, match="censoring"):
        archived_bars(arm, [0, .5, 2], 0)


def test_reject_invalid_geometry_and_intervals():
    with pytest.raises(ValueError, match="nonempty"):
        simplex_radius([])
    with pytest.raises(ValueError, match="invalid atom"):
        point_controls("bad", [[0, 0, 0]], [], cap=4, workers=1)
    with pytest.raises(ValueError, match="invalid interval"):
        distance([(2, 1)], [])


def test_control_extraction_accepts_serialized_edges_and_caches(tmp_path, monkeypatch):
    atoms = [SimpleNamespace(index=i, xyz=(i, 0, 0), element="C") for i in range(2)]
    patches = [SimpleNamespace(patch_id=i, atom_ids=(i,)) for i in range(2)]
    result = dict(metadata={k: False for k in ("events_truncated", "paths_truncated", "support_incomplete")},
                  patches=patches, edges=[[0, 1, 1.]],
                  support_by_set={(0,): 0., (1,): 0., (0, 1): .5},
                  path_support={p: {"valid": True} for p in ((0,), (1,), (0, 1))})
    (tmp_path / "tiny.pdb").write_text("synthetic control")
    monkeypatch.setattr(run, "parse_structure", lambda path: atoms)
    monkeypatch.setattr(run, "rna_event_filter", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(run, "extract_intra", lambda *args, **kwargs: result)
    control = run.extract_control("tiny", tmp_path)
    assert control["n_dir_paths"] == 3
    assert control["n_und_paths"] >= 4
    assert control["births"]["dir_edgew"][-1] == [[0, 1], 1.]
    assert run.extract_control("tiny", tmp_path) == control
    result["metadata"]["support_incomplete"] = True
    (tmp_path / "tiny.patch_control.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        run.extract_control("tiny", tmp_path)
