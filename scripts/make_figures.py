# -*- coding: utf-8 -*-
"""Regenerate paper figures from frozen tables only.

Currently supports the certified common-witness figures from the
witness JSON dumps (EXP-02). Additional figures are added as their
frozen tables land in results/tables/.

Usage:
  python scripts/make_figures.py --figure witness --witness-json <path>
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "exp02_molecular_witnesses"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default="witness",
                    choices=["witness"])
    ap.add_argument("--witness-json", default="")
    a = ap.parse_args()
    if a.figure == "witness":
        if not a.witness_json:
            # use frozen copies under results/ if present
            cands = sorted((ROOT / "results" / "witness").glob(
                "*.witness.json"))
            if not cands:
                sys.exit("no witness JSON found; pass --witness-json")
            a.witness_json = str(cands[0])
        import render_witness
        src = Path(a.witness_json)
        out = ROOT / "results" / "figures" / (
            src.name.replace(".witness.json", ""))
        render_witness.render(src, out)
        print("wrote", out.with_suffix(".png"), "and .pdf")


if __name__ == "__main__":
    main()
