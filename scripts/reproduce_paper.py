# -*- coding: utf-8 -*-
"""Print frozen paper numbers from results/tables (never recomputes).

Usage: python scripts/reproduce_paper.py --version v33
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v33")
    a = ap.parse_args()
    tbl = ROOT / "results" / "tables"
    rep = ROOT / "results" / "reports"

    print(f"=== frozen results ({a.version}) ===")
    bc = tbl / "barcode_comparison_16structure.csv"
    if bc.exists():
        rows = list(csv.DictReader(bc.open()))
        data = [r for r in rows if r.get("graph") in ("dir", "und")
                and r.get("tau_pairwise_identical") in ("0", "1")]
        same = sum(int(r["tau_pairwise_identical"]) for r in data)
        print(f"filtration barcode-identical: {same}/{len(data)}")
        for hint in ("rna", "pd", "protein"):
            pass
        rna = [r for r in data if any(k in r["structure_id"].lower()
                for k in ("simrna", "3drna", "farfar", "rnacomposer"))]
        pd_ = [r for r in data if r not in rna]
        rs = sum(int(r["tau_pairwise_identical"]) for r in rna)
        ps = sum(int(r["tau_pairwise_identical"]) for r in pd_)
        print(f"  RNA-like: {rs}/{len(rna)} identical "
              f"({(len(rna)-rs)}/{len(rna)} differ)")
        print(f"  PD-like:   {ps}/{len(pd_)} identical "
              f"({(len(pd_)-ps)}/{len(pd_)} differ)")
    for name in ("metrics_v2.json",):
        f = tbl / name
        if f.exists():
            d = json.loads(f.read_text())
            print(f"{name}: {len(json.dumps(d))} bytes of frozen metrics")
    corr = rep / "manuscript_v04_corrections.md"
    if corr.exists():
        print("manuscript corrections: see",
              "results/reports/manuscript_v04_corrections.md")
    print("=== end ===")


if __name__ == "__main__":
    main()
