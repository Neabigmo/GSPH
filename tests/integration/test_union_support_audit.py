import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cspph.chemistry.patch_builder import InteractionPatch, MicroContact
from experiments.exp05_molecular_landscape.deep_complex_dump import dump_complex
from experiments.union_support_audit import (
    audit_dumps, check_archived_path_counts, count_cohort_paths, main, patch_spread,
)


def test_patch_spread_uses_only_same_role_endpoints():
    patch = SimpleNamespace(donor_members=(0, 1), acceptor_members=(2, 3))
    atoms = {
        i: SimpleNamespace(xyz=np.array([x, 0.0, 0.0]))
        for i, x in enumerate((0.0, 2.0, 10.0, 13.0))
    }
    assert patch_spread(patch, atoms) == 3.0


def test_four_patch_excess_is_not_assumed_for_larger_families():
    with pytest.raises(ValueError, match="explicit four-patch subset supports"):
        count_cohort_paths(pd.DataFrame({"order": [4]}))


def test_complex_export_contains_role_spreads_and_tight_radius_extrema(tmp_path):
    atoms = [
        SimpleNamespace(index=i, xyz=np.array([x, 0.0, 0.0]), element=element)
        for i, (x, element) in enumerate([(0, "N"), (2, "N"), (3, "O"), (6, "C")])
    ]
    patch = InteractionPatch(
        patch_id=0, donor_site=("A", 1, "N"), acceptor_site=("B", 1, "O"),
        contacts=[MicroContact(0, 2, "vdw", 3.0, 0.0),
                  MicroContact(1, 3, "vdw", 4.0, 0.375)],
        donor_members=(0, 1), acceptor_members=(2, 3), geometry={},
    )
    dump_complex({
        "patches": [patch], "support_by_set": {(0,): 0.0},
        "path_support": {(0,): {"valid": True}},
    }, atoms, tmp_path, "example")
    row = pd.read_parquet(tmp_path / "example.patch_geometry.parquet").iloc[0]
    assert (row.donor_spread_angstrom, row.acceptor_spread_angstrom, row.delta_angstrom) == (2, 3, 3)
    assert (row.r_min_angstrom, row.r_max_angstrom) == (1.52, 1.7)
    assert audit_dumps(tmp_path, "example", "dir")[0]["ratio"] == pytest.approx(1.52 / 4.7)


def test_full_dump_audit_uses_all_pairs_and_checks_the_archive_count(tmp_path, monkeypatch, capsys):
    sid = "example"
    pd.DataFrame([dict(graph="dir", path="2,0,1", tau=0.9, valid=True)]).to_parquet(
        tmp_path / f"{sid}.complex_paths.parquet"
    )
    pd.DataFrame([dict(canon=key, tau=tau) for key, tau in
                  [("0,1", 0.3), ("0,2", 0.4), ("1,2", 0.6)]]).to_parquet(
        tmp_path / f"{sid}.tau_table.parquet"
    )
    patches = pd.DataFrame([
        dict(patch_id=i, contact_count=1 if i < 2 else 2,
             r_min_angstrom=1.52 + 0.03 * i, r_max_angstrom=1.7 + 0.1 * i,
             delta_angstrom=0.1 * i)
        for i in range(3)
    ])
    geometry = tmp_path / f"{sid}.patch_geometry.parquet"
    patches.to_parquet(geometry)
    output = tmp_path / "path_metrics.parquet"
    monkeypatch.setattr("sys.argv", [
        "union_support_audit", "--dumps", str(tmp_path), "--sid", sid,
        "--path-output", str(output),
    ])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["higher_order"]["n_paths"] == 1
    row = pd.read_parquet(output).iloc[0]
    assert row.tau_pw_angstrom == 0.6
    assert row.delta_angstrom == 0.2
    assert not row.single_contact
    assert row.ratio == pytest.approx((1.52 + 0.9) / (1.9 + 0.6 + 0.2))
    census = pd.DataFrame([dict(structure_id=sid, graph="dir", order=2, n_paths=2)])
    with pytest.raises(ValueError, match="do not match archive"):
        check_archived_path_counts([row], census, sid, "dir")
    patches.iloc[:2].to_parquet(geometry)
    with pytest.raises(ValueError, match="missing geometry for patch 2"):
        audit_dumps(tmp_path, sid, "dir")


def test_committed_molecular_evidence(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["union_support_audit"])
    main()
    report = json.loads(capsys.readouterr().out)
    witnesses = report["witness_summary"]
    assert witnesses["n_paths"] == 21
    assert witnesses["n_bound_violations"] == 0
    assert witnesses["ratio_quantiles"]["max"] == pytest.approx(1.1185032)
    assert witnesses["n_positive_eta_at_zero_delta"] == 5
    assert sum(row["n_paths"] for row in report["cohort_paths"]) == 473_012
    assert sum(row["n_strict_four_patch_excess"] for row in report["cohort_paths"]) == 0
    barcodes = report["archived_barcodes"]
    assert sum(row["verified_from_intervals"] for row in barcodes) == 30
    assert sum(row["task"] == "rna" and row["different"] for row in barcodes) == 14
    assert sum(row["task"] == "pd" and row["different"] for row in barcodes) == 4
