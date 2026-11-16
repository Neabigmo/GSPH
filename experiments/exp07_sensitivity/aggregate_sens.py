# -*- coding: utf-8 -*-
"""EXP-07 sensitivity aggregation (runs on HPC against gsph_h3_sens).

For each condition dir (clash_r*, radius_s*), aggregates per-structure:
  n_support_records, tau_mean/median/p90, n_barcode_intervals (H0-H2),
  n_paths (from deep_manifest n_dir_paths+n_und_paths).
Emits sensitivity_summary.csv + per-condition curves JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/work/share/acn8s56xc0/gsph_h3_sens")
OUT = Path("/work/share/acn8s56xc0/gsph_h3_deep/exp07_sensitivity")


def main():
    rows = []
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        cond = d.name
        for f in sorted(d.glob("*.support.parquet")):
            sid = f.name.split(".")[0]
            df = pd.read_parquet(f)
            tau = pd.to_numeric(df.get("tau", df.iloc[:, -1]),
                                errors="coerce")
            bc = d / f"{sid}.barcode.parquet"
            n_int = None
            if bc.exists():
                b = pd.read_parquet(bc)
                n_int = int(len(b))
            man = d / f"{sid}.deep_manifest.json"
            n_paths = None
            if man.exists():
                m = json.loads(man.read_text())
                n_paths = int(m.get("n_dir_paths", 0)) + \
                    int(m.get("n_und_paths", 0))
            rows.append(dict(condition=cond, structure_id=sid,
                             n_support=len(df),
                             tau_mean=float(np.nanmean(tau)),
                             tau_median=float(np.nanmedian(tau)),
                             tau_p90=float(np.nanquantile(tau, 0.9)),
                             n_barcode_intervals=n_int, n_paths=n_paths))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "sensitivity_summary.csv", index=False)
    # curves: median over structures per condition
    curves = {}
    for cond, g in df.groupby("condition"):
        curves[cond] = dict(
            n_structures=int(g["structure_id"].nunique()),
            tau_median_mean=float(g["tau_median"].mean()),
            tau_p90_mean=float(g["tau_p90"].mean()),
            n_paths_mean=float(g["n_paths"].mean())
            if g["n_paths"].notna().any() else None,
            n_intervals_mean=float(g["n_barcode_intervals"].mean())
            if g["n_barcode_intervals"].notna().any() else None)
    (OUT / "sensitivity_curves.json").write_text(
        json.dumps(curves, indent=1))
    print(json.dumps(curves, indent=1))


if __name__ == "__main__":
    main()
