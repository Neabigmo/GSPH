# -*- coding: utf-8 -*-
"""Validate a manifest CSV against the frozen schema.

Usage: python scripts/validate_manifest.py --manifest manifests/deep_subset_v33.csv
"""
import argparse
import csv
import sys
from pathlib import Path

REQUIRED = {
    "deep_subset_v33.csv": ["task", "structure_id"],
    "targets_rna.csv": ["target_id"],
    "targets_pd.csv": ["target_id"],
    "candidate_manifest.csv": ["task", "target_id", "candidate_id"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()
    p = Path(a.manifest)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / a.manifest
    if not p.exists():
        sys.exit(f"missing manifest: {p}")
    name = p.name
    with open(p, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    if not rows:
        sys.exit("empty manifest")
    header = rows[0]
    req = REQUIRED.get(name, [])
    missing = [c for c in req if c not in header]
    n = len(rows) - 1
    status = "OK" if not missing else f"MISSING COLUMNS {missing}"
    print(f"{name}: {n} rows, {len(header)} cols -> {status}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
