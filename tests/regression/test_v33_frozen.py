# regression: frozen V33 numbers must match shipped tables
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_barcode_comparison_table_present():
    p = ROOT / "results" / "tables" / "barcode_comparison_16structure.csv"
    assert p.exists()
    rows = [r for r in csv.DictReader(p.open())
            if r.get("graph") in ("dir", "und")]
    assert len(rows) == 32


def test_frozen_counts():
    p = ROOT / "results" / "tables" / "barcode_comparison_16structure.csv"
    rows = [r for r in csv.DictReader(p.open())
            if r.get("graph") in ("dir", "und")]
    same = sum(int(r["tau_pairwise_identical"]) for r in rows)
    assert same == 14  # V33 frozen: 14/32 barcode-identical
