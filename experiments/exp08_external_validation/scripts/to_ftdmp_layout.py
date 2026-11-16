# -*- coding: utf-8 -*-
"""Convert exp08 decoys/ + manifest.csv into FTDMP-compatible layout:

  exp08_ftdmp/Bound_docking/<PDB>/relaxed_top_complexes/model_*.pdb
  exp08_ftdmp/Bound_docking/<PDB>/rmsd_results.txt
  exp08_ftdmp/Bound_docking/<PDB>/scoring_results-ranks.txt

so pd_ftdmp_driver.enumerate_units picks them up unchanged.
"""
import csv, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "decoys"
DST = HERE / "exp08_ftdmp" / "Bound_docking"

rows = list(csv.DictReader(open(SRC / "manifest.csv")))
by_pdb = {}
for r in rows:
    by_pdb.setdefault(r["pdb"], []).append(r)

for pdb, rs in by_pdb.items():
    case = DST / pdb.upper()
    comp = case / "relaxed_top_complexes"
    comp.mkdir(parents=True, exist_ok=True)
    for r in rs:
        src = SRC / pdb / "models" / r["model"]
        shutil.copy(src, comp / r["model"])
    with open(case / "rmsd_results.txt", "w") as f:
        f.write("model rmsd\n")
        for r in rs:
            f.write(f"{r['model']} {r['rmsd']}\n")
    with open(case / "scoring_results-ranks.txt", "w") as f:
        f.write("model rank\n")
        for i, r in enumerate(sorted(rs, key=lambda x: float(x["rmsd"])), 1):
            f.write(f"{r['model']} {i}\n")
print("cases:", len(by_pdb), "models:", len(rows))
