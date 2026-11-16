# -*- coding: utf-8 -*-
"""Feature health audit: missing/inf/zero rates, tails, constants."""
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
import numpy as np
import pandas as pd


def audit_frame(df: pd.DataFrame, feature_cols: list[str], task: str) -> pd.DataFrame:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    n = len(X)
    a = X.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(a)
    with np.errstate(invalid="ignore"):
        qs = np.nanpercentile(np.where(finite, a, np.nan), [1, 50, 99, 99.9], axis=0)
    out = pd.DataFrame({
        "task": task,
        "feature_name": feature_cols,
        "missing_rate": X.isna().mean().to_numpy(),
        "inf_rate": (~np.isfinite(X.to_numpy(dtype=np.float64, na_value=np.nan))).mean(axis=0),
        "zero_rate": (X.fillna(np.nan) == 0).mean().to_numpy(),
        "mean": np.nanmean(np.where(finite, a, np.nan), axis=0),
        "std": np.nanstd(np.where(finite, a, np.nan), axis=0),
        "p1": qs[0], "p50": qs[1], "p99": qs[2], "p999": qs[3],
        "max_abs": np.nanmax(np.abs(np.where(finite, a, np.nan)), axis=0),
        "unique_count": X.nunique(dropna=True).to_numpy(),
    })
    out["constant"] = out.unique_count <= 1
    out["extreme_flag"] = out["max_abs"] > 1e5
    return out
