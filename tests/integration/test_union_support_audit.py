import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.union_support_audit import count_cohort_paths, main, patch_spread


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
