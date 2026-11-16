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
# -*- coding: utf-8 -*-
"""Invariant tests required by the ablation protocol."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_groups import build_manifest, group_columns  # noqa: E402

MAN = Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\ablation\feature_group_manifest.csv")


def _load():
    return pd.read_csv(MAN)


def test_groups_disjoint():
    m = _load()
    g = {t: set(m.loc[m.assigned_group == t, "feature_name"]) for t in "PTHS"}
    for a in "PTHS":
        for b in "PTHS":
            if a < b:
                assert not (g[a] & g[b]), f"{a}∩{b} non-empty"


def test_no_diagnostics_or_coords_in_features():
    m = _load()
    inc = m[m.included]
    assert not inc.feature_name.str.contains(
        "converged|residual|method_code|spectrum_complete|omitted_modes|"
        "n_modes|dimension|dense_limit|omega_truncated|boundary_square").any()
    assert not inc.feature_name.str.contains("center_[xyz]").any()
    assert not inc.feature_name.str.match(r"^(label_|base_|structure_id|row|group|kind)").any()


def test_candidate_pools():
    import json
    cov = json.loads(Path(MAN).with_name("candidate_coverage.json").read_text())
    assert cov["rna"]["final_rows"] == 806 and cov["rna"]["groups"] == 20
    assert cov["membrane"]["final_rows"] == 223 and cov["membrane"]["groups"] == 7
    assert cov["pd_bound"]["final_rows"] == 9400 and cov["pd_bound"]["groups"] == 47
    assert cov["pd_unbound"]["final_rows"] == 9399 and cov["pd_unbound"]["groups"] == 47


def test_every_column_classified():
    m = _load()
    assert m.assigned_group.notna().all()
    assert (m[m.included == False].exclusion_reason != "").all()  # noqa: E712


if __name__ == "__main__":
    for f in dir():
        if f.startswith("test_"):
            globals()[f](); print(f, "OK")
    print("all ablation invariants pass")
