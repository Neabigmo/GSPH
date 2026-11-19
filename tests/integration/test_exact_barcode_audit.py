import json
import sys

import pandas as pd
import pytest

from experiments.exp03_filtration_comparison.compare_exact import main


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
