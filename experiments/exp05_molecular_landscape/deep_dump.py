# -*- coding: utf-8 -*-
"""Deep GSPH6 serialization — barcode + per-path support for FUTURE runs.

Engine change (applied 2026-11-15, gsph6/engine.py):
  * ``path_support`` : {path_tuple: {"tau": float, "valid": bool}}
                        for every certified-path-candidate (order 0..3)
  * ``path_eta``     : {"3": {path_tuple: eta}, "4": {path_tuple: eta}}
  * ``persistence``  : unchanged — orders[k].intervals = [birth_idx,
                        death_idx, right_censored] with INTEGER indices
                        into cfg.scales (discrete filtration barcode)

IMPORTANT: barcode birth/death are scale-INDICES of the discrete
multi-scale filtration (cfg.scales = [0, 0.5, 2] by default), NOT
continuous coordinates.  Lifespan is therefore in filtration-step units;
TP/entropy statistics are still well-defined and honest.
"""
from __future__ import annotations
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
import json
from pathlib import Path
import numpy as np
import pandas as pd


def dump_persistence(res: dict, out_dir: Path, structure_id: str,
                     scales=None) -> dict:
    pers = res.get("persistence") or {}
    rows = []
    for order, blk in (pers.get("orders") or {}).items():
        k = int(order)
        for iv in (blk.get("intervals") or []):
            b, d, rc = int(iv[0]), int(iv[1]), bool(iv[2])
            rows.append(dict(structure_id=structure_id, dim=k,
                             birth_idx=b, death_idx=d, right_censored=rc,
                             lifespan_idx=(d - b) if not rc else np.nan))
    manifest = {"structure_id": structure_id, "available": bool(rows),
                "n_intervals": len(rows),
                "filtration": "discrete_scale_indices",
                "scales": list(scales or pers.get("scales") or [])}
    if rows:
        pd.DataFrame(rows).to_parquet(out_dir / f"{structure_id}.barcode.parquet",
                                      index=False)
    return manifest


def dump_path_support(res: dict, out_dir: Path, structure_id: str) -> dict:
    sup = res.get("path_support") or {}
    eta = res.get("path_eta") or {}
    rows = []
    for p, s in sup.items():
        p = tuple(p)
        key = "|".join(map(str, p))
        e = eta.get("3", {}).get(p, eta.get("4", {}).get(p, np.nan))
        rows.append(dict(structure_id=structure_id, order=len(p) - 1,
                         path=key, tau=float(s["tau"]),
                         valid=bool(s["valid"]), eta=float(e)))
    manifest = {"structure_id": structure_id, "available": bool(rows),
                "n_paths": len(rows)}
    if rows:
        pd.DataFrame(rows).to_parquet(out_dir / f"{structure_id}.support.parquet",
                                      index=False)
    return manifest


def dump_deep(res: dict, out_dir, structure_id: str) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    m1 = dump_persistence(res, out_dir, structure_id)
    m2 = dump_path_support(res, out_dir, structure_id)
    m = {"structure_id": structure_id, **m1, **{f"support_{k}": v for k, v in m2.items()}}
    with (out_dir / f"{structure_id}.deep_manifest.json").open("w") as f:
        json.dump(m, f, indent=1, default=str)
    return m
