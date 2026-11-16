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
"""P0-1 per-structure runner: compare_structure for ONE dumped complex.

Usage: python compare_one.py --dir <dump_dir> --sid <structure_id> --out <json>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep/GSPH_H3/ablation")
sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep/GSPH_H3")
sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep")

import filtration_compare as FC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    r = FC.compare_structure(Path(a.dir), a.sid, rng)
    r["source_dir"] = Path(a.dir).name
    Path(a.out).write_text(json.dumps(r, indent=1, default=str))
    print(f"[cmp1] {a.sid} ok -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
