import sys
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import UnionSolver, cliques, pair_support, weighted_support
from run import clustered_count, config_hash, crop_region, download, pair_graph, read_structure, release_is_test, reusable_result
from run import execute, load_config, search_entries, selected_records, terminal


def test_search_cache_keys_full_query(tmp_path):
    calls = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"result_set": [{"identifier": "TEST"}]}
    class Session:
        def post(self, url, json, timeout):
            calls.append(json)
            return Response()
    session = Session()
    query = terminal("resolution", "less_or_equal", 1.5)
    (tmp_path / "search_water.json").write_text(json.dumps({"result_set": [{"identifier": "OLD"}]}))
    assert search_entries(session, query, tmp_path, "water") == ["test"]
    assert search_entries(session, dict(reversed(list(query.items()))), tmp_path, "water") == ["test"]
    assert len(calls) == 1
    changed = terminal("resolution", "less_or_equal", 2.)
    assert search_entries(session, changed, tmp_path, "water") == ["test"]
    assert len(calls) == 2
    assert calls[-1]["query"] == changed
    assert len(list(tmp_path.glob("search_water_*.json"))) == 2


def test_search_empty_results_and_request_error(tmp_path):
    class Session:
        def post(self, *args, **kwargs):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    assert search_entries(Session(), {}, tmp_path, "empty") == []
    class ErrorSession:
        def post(self, *args, **kwargs):
            raise RuntimeError("request failed")
    with pytest.raises(RuntimeError, match="request failed"):
        search_entries(ErrorSession(), {}, tmp_path, "error")
    assert not list(tmp_path.glob("search_error_*.json"))


def test_run_retries_missing_download_and_result(tmp_path, monkeypatch):
    cfg = load_config()
    cfg["workers"] = 1
    (tmp_path / "selection.json").write_text(json.dumps({"samples": {"zn": ["test"]}}))
    structures = tmp_path / "structures"
    structures.mkdir()
    output = structures / "zn_test.json"
    output.write_text(json.dumps({"status": "download_missing", "config_hash": config_hash(cfg)}))
    calls = []
    def retry(data, config):
        calls.append("download")
        (data / "test.cif.gz").touch()
    def solve(command, **kwargs):
        calls.append("worker")
        output.write_text(json.dumps({"status": "complete", "config_hash": config_hash(cfg)}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("run.download", retry)
    monkeypatch.setattr("run.subprocess.run", solve)
    args = SimpleNamespace(data=tmp_path, limit=None, config=tmp_path / "config.yaml")
    execute(args, cfg)
    assert calls == ["download", "worker"]
    execute(args, cfg)
    assert calls == ["download", "worker", "download"]


def test_yaml_thresholds_are_effective():
    cfg = load_config()
    changed = {**cfg, "clustering_threshold": .1, "time_split_date": "2025-01-01"}
    points = [[0., 0., 0.], [.5, 0., 0.]]
    assert clustered_count(points, cfg) == 1
    assert clustered_count(points, changed) == 2
    assert release_is_test("2024-05-01", cfg)
    assert not release_is_test("2024-05-01", changed)
    for key in ("coordination_radii", "site_match_distances", "sample_per_order_stratum"):
        assert config_hash(cfg) != config_hash({**cfg, key: None})


def test_summary_ignores_historical_results(tmp_path):
    cfg = load_config()
    assert selected_records(tmp_path, {"zn": {"current"}}, cfg) == []
    rows = [{"cohort": "zn", "pdb_id": "current", "config_hash": config_hash(cfg)},
            {"cohort": "zn", "pdb_id": "old", "config_hash": config_hash(cfg)},
            {"cohort": "water", "pdb_id": "current", "config_hash": config_hash(cfg)}]
    for index, row in enumerate(rows):
        (tmp_path / f"{index}.json").write_text(json.dumps(row))
    assert selected_records(tmp_path, {"zn": {"current"}}, cfg) == rows[:1]


def test_summary_skips_and_counts_wrong_fingerprints(tmp_path, capsys):
    cfg = load_config()
    rows = [{"cohort": "zn", "pdb_id": "missing"},
            {"cohort": "zn", "pdb_id": "stale", "config_hash": config_hash({**cfg, "delta": .5})},
            {"cohort": "zn", "pdb_id": "current", "config_hash": config_hash(cfg)},
            {"cohort": "zn", "pdb_id": "region", "region_radius_A": 15., "config_hash": config_hash(cfg, 15.)}]
    for index, row in enumerate(rows):
        (tmp_path / f"{index}.json").write_text(json.dumps(row))
    samples = {"zn": {r["pdb_id"] for r in rows}}
    assert selected_records(tmp_path, samples, cfg) == rows[2:]
    assert "skipped 2 results" in capsys.readouterr().err
    assert selected_records(tmp_path, samples, {**cfg, "delta": .6}) == []
    assert "skipped 4 results" in capsys.readouterr().err


def test_resume_requires_matching_config(tmp_path):
    path = tmp_path / "result.json"
    cfg = {"seed": 0, "delta": .3, "radii": {"O": 2.09}}
    fingerprint = config_hash(cfg)
    assert reusable_result(path, fingerprint) is None
    for row in ({"status": "complete"}, {"status": "download_missing", "config_hash": fingerprint},
                {"status": "error", "config_hash": fingerprint}):
        path.write_text(json.dumps(row))
        assert reusable_result(path, fingerprint) is None
    row = {"status": "complete", "config_hash": fingerprint}
    path.write_text(json.dumps(row))
    assert reusable_result(path, fingerprint) == row
    assert reusable_result(path, config_hash({**cfg, "seed": 1})) is None
    assert config_hash(cfg, 15.) != fingerprint
    path.write_text("invalid")
    assert reusable_result(path, fingerprint) is None


def test_release_split_boundary():
    assert not release_is_test("2024-04-30")
    assert release_is_test("2024-05-01")
    assert release_is_test("2025-01-01")
    with pytest.raises(ValueError):
        release_is_test("?")


def test_region_retains_clash_shell_and_excludes_partial_patches():
    atoms = [{"xyz": [x, 0., 0.]} for x in [0., 1., 4., 5., 20.]]
    xyz = np.array([a["xyz"] for a in atoms])
    groups = [{"atoms": (0,), "residue": 0}, {"atoms": (1, 2), "residue": 1}]
    kept, local, radii, patches, center = crop_region(atoms, xyz, np.full(5, 2.07), groups, 2., 1)
    assert len(kept) == 4
    assert len(patches) == 1
    assert patches[0]["atoms"] == (0,)
    with pytest.raises(ValueError):
        crop_region(atoms, xyz, np.full(5, 2.07), [], 2., 1)


@pytest.mark.parametrize("known_length", [False, True])
def test_download_budget_never_consumes_excess(tmp_path, monkeypatch, known_length):
    class Response:
        headers = {"Content-Length": "20"} if known_length else {}
        def __enter__(self):
            self.payload = io.BytesIO(b"x" * 20)
            self.raw = self
            return self
        def __exit__(self, *args):
            pass
        def read(self, count, decode_content):
            return self.payload.read(count)
        def raise_for_status(self):
            pass
    class Session:
        def get(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr("run.requests.Session", Session)
    (tmp_path / "selection.json").write_text(json.dumps({"samples": {"zn": ["test"]}}))
    cfg = {"download_limit_bytes": 10, "download_reserve_bytes": 1}
    with pytest.raises(RuntimeError, match="budget"):
        download(tmp_path, cfg)
    ledger = json.loads((tmp_path / "downloads.json").read_text())
    assert ledger[-1]["status"] == "budget_stop"
    assert sum(row["bytes"] for row in ledger) <= 10
    assert not (tmp_path / "test.cif.gz").exists()


def test_signed_support_and_hard_triple():
    xyz = np.array([[0., 0., 0.], [4.6, 0., 0.], [2.3, np.sqrt(3) * 2.3, 0.]])
    radii = np.full(3, 2.1)
    _, tau = weighted_support(xyz, radii)
    assert tau == pytest.approx(4.6 / np.sqrt(3) - 2.1)
    assert pair_support([0], [1], xyz, radii) == pytest.approx(.2)
    assert tau > .3
    _, inner = weighted_support(xyz / 2, radii)
    assert inner == pytest.approx(2.3 / np.sqrt(3) - 2.1)


def test_tetrahedron():
    xyz = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]]) * 4.4 / np.sqrt(8)
    _, tau = weighted_support(xyz, np.full(4, 2.1))
    assert tau == pytest.approx(4.4 * np.sqrt(6) / 4 - 2.1)


def test_union_selects_alternative_and_cache():
    xyz = np.array([[0., 0., 0.], [20., 0., 0.], [2., 0., 0.], [1., 1., 0.]])
    radii = np.full(4, 2.)
    solver = UnionSolver(xyz, radii)
    _, tau = solver.solve([(0, 1), (2,), (3,)])
    assert tau < 0
    assert solver.solves == 2
    solver.solve([(0, 1), (2,), (3,)])
    assert solver.solves == 2
    assert solver.failures == 0


def test_containment_and_invalid_inputs():
    xyz = np.array([[0., 0., 0.], [.1, 0., 0.]])
    radii = np.array([1., 3.])
    _, tau = weighted_support(xyz, radii)
    assert tau == pytest.approx(-1.)
    assert pair_support([0], [1], xyz, radii) == pytest.approx(-1.)
    with pytest.raises(ValueError):
        weighted_support([], [])
    with pytest.raises(ValueError):
        weighted_support([[np.nan, 0., 0.]], [2.])


def test_graph_cliques_empty_and_clustering():
    assert list(cliques([], 3)) == []
    assert clustered_count([]) == 0
    assert clustered_count([[0., 0., 0.], [.5, 0., 0.], [3., 0., 0.]]) == 2
    xyz = np.array([[0., 0., 0.], [4.6, 0., 0.], [2.3, 3.98, 0.], [30., 0., 0.]])
    groups = [{"atoms": (i,)} for i in range(4)]
    adjacency, pairs = pair_graph(groups, xyz, np.full(4, 2.1), .3)
    assert list(cliques(adjacency, 3)) == [(0, 1, 2)]
    assert list(cliques(adjacency, 4)) == []


def test_parser_first_model_altloc_and_stripping(tmp_path):
    text = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 O OD1 A ASP A 1 1 0 0 0 0.3 10 1 A 1
ATOM 2 O OD1 B ASP A 1 1 1 0 0 0.7 10 1 A 1
ATOM 3 O OD2 . ASP A 1 1 2 0 0 1 10 1 A 1
ATOM 4 O O . ASP A 1 1 3 0 0 1 10 1 A 1
HETATM 5 O O . HOH B 2 . 4 0 0 1 10 2 B 1
HETATM 6 ZN ZN . ZN C 3 . 5 0 0 1 10 3 C 1
ATOM 7 O O . ASP A 1 1 100 0 0 1 10 1 A 2
"""
    path = tmp_path / "test.cif"
    path.write_text(text)
    atoms, xyz, radii, groups, sites = read_structure(path, "zn")
    assert len(atoms) == 3
    assert xyz[0, 0] == 1
    assert len(groups) == 2
    assert groups[0]["atoms"] == (0, 1)
    assert sites == []
