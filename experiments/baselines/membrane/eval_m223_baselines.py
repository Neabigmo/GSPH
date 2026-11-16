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
#!/usr/bin/env python3
"""Evaluate published ddG predictors on M223 vs Experimental_ddG.

Table_S1 already contains predictions from published tools (rosetta, foldx,
mcsm, sdm, duet, imutant3, ddgm*, provean, elaspic, easemm). These are
official per-tool outputs from the mCSM-membrane paper -- no refit.

Metrics: Pearson, Spearman, RMSE vs Experimental_ddG (negative = destabilizing).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.normpath(os.path.join(ROOT, "..", "..", "data", "membrane",
                                      "m223_original", "Table_S1_ddg.csv.xls"))

BASELINES = ["rosetta", "roselow", "imutant3", "foldx", "mcsm", "sdm", "duet",
             "ddgm8", "ddgm47", "provean", "elaspic", "easemm"]


def main():
    df = pd.read_excel(TABLE)
    df.columns = [str(c).strip() for c in df.columns]
    y = pd.to_numeric(df["Experimental_ddG"], errors="coerce")
    rows = []
    for b in BASELINES:
        if b not in df.columns:
            rows.append({"method": b, "status": "missing_column"})
            continue
        x = pd.to_numeric(df[b], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 10:
            rows.append({"method": b, "status": "insufficient_data",
                         "n": int(m.sum())})
            continue
        rows.append({
            "method": b, "status": "ok", "n": int(m.sum()),
            "pearson": pearsonr(x[m], y[m])[0],
            "spearman": spearmanr(x[m], y[m])[0],
            "rmse": float(np.sqrt(((x[m] - y[m]) ** 2).mean())),
        })
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(ROOT, "m223_baseline_summary.csv"), index=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
