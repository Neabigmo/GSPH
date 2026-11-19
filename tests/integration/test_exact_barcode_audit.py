import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.exp03_filtration_comparison.compare_exact import main
from experiments.exp03_filtration_comparison import dump_rna_example


def test_exact_barcode_audit_on_parquet_dumps(tmp_path, monkeypatch):
    paths = {
        "0": 0, "1": 0, "2": 0, "3": 0,
        "0,1": 0.4, "0,2": 0.4, "1,3": 0.4, "2,3": 0.4,
        "0,1,3": 0.8, "0,2,3": 1.2,
    }
    pd.DataFrame(
        [dict(graph="dir", path=path, tau=tau, valid=True) for path, tau in paths.items()]
    ).to_parquet(tmp_path / "diamond.complex_paths.parquet")
    supports = pd.DataFrame(
        [dict(canon=key, tau=tau) for key, tau in (paths | {"0,3": 0.6}).items()]
    )
    tau_file = tmp_path / "diamond.tau_table.parquet"
    supports.to_parquet(tau_file)
    output = tmp_path / "bars.json"
    monkeypatch.setattr(sys, "argv", [
        "compare_exact",
        "--dir", str(tmp_path), "--sid", "diamond", "--out", str(output),
    ])
    main()
    result = json.loads(output.read_text())
    assert result["scales_A"] == [0, 0.4, 0.6, 0.8, 1.2]
    assert result["common_A"]["1"] == [[0.4, 1.2]]
    assert result["pairwise_A"]["1"] == [[0.4, 0.6]]
    assert result["common_A"]["0"][-1] == [0, None]
    assert result["equal_by_order"] == {"0": True, "1": False, "2": True}

    supports.loc[supports["canon"] != "0,3"].to_parquet(tau_file)
    with pytest.raises(ValueError, match="missing certified support for 0,3"):
        main()


def test_rna_dump_reads_control_thresholds_from_config(tmp_path, monkeypatch):
    settings = tmp_path / "controls.yaml"
    settings.write_text(
        "rna_molecular_controls:\n"
        "  event_filter: {fraction: 0.5, per_nt_quota: 3}\n"
        "  geometry: {intra_min_bond_sep: 5, contact_min_vdw_ratio: 0.7}\n"
    )
    monkeypatch.setattr(dump_rna_example, "parse_structure", lambda _: [])

    def event_filter(atoms, fraction, per_nt_quota):
        assert (fraction, per_nt_quota) == (0.5, 3)
        return "selection", {}

    def extract(atoms, scenario, config, event_filter):
        assert (config.intra_min_bond_sep, config.contact_min_vdw_ratio) == (5, 0.7)
        assert event_filter == "selection"
        return {
            "metadata": dict(events_truncated=False, paths_truncated=False, support_incomplete=False),
            "path_support": {(0,): {"tau": 0.0, "valid": True}},
            "support_by_set": {(0,): 0.0},
            "patches": [SimpleNamespace(patch_id=0)],
        }

    monkeypatch.setattr(dump_rna_example, "rna_event_filter", event_filter)
    monkeypatch.setattr(dump_rna_example, "extract_intra", extract)
    monkeypatch.setattr(sys, "argv", [
        "dump_rna_example", "rna.pdb", "--out-dir", str(tmp_path), "--config", str(settings),
    ])
    dump_rna_example.main()
    assert (tmp_path / "rna.complex_paths.parquet").exists()
