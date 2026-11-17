"""Aggregate EXP-03 filtration comparisons.

Compares fine-grid (cmp_out_exp03_grid) and event-value
(cmp_out_exp03_event) outputs, and summarizes tau-vs-pairwise barcode
agreement per structure / per graph / per filtration rule.

Usage:
    python aggregate_exp03.py --grid DIR --event DIR --out results/tables/exp03_summary.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

RULES = ("tau", "pairwise", "facetmax", "edgew", "shuffled")


def _interval_set(res, order):
    iv = res.get(f"intervals_k{order}", []) if isinstance(res, dict) else []
    return {(int(b), int(d)) for b, d, _ in iv}


def _barcode_key(res):
    """Signature: multiset of (order, birth, death) intervals."""
    out = set()
    for order in range(4):
        for b, d, _ in res.get(f"intervals_k{order}", []):
            out.add((order, int(b), int(d)))
    return out


def _load(d: Path):
    out = {}
    for f in sorted(d.glob("*.json")):
        try:
            j = json.load(open(f))
            out[j.get("structure_id", f.stem)] = j
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--event", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grid = _load(Path(args.grid))
    event = _load(Path(args.event))
    rows = []
    sids = sorted(set(grid) | set(event))
    for sid in sids:
        g, e = grid.get(sid, {}), event.get(sid, {})
        for graph in ("dir", "und"):
            # tau vs pairwise barcode agreement (each mode)
            for mode, rec in (("grid", g), ("event", e)):
                t, p = rec.get(f"{graph}_tau"), rec.get(f"{graph}_pairwise")
                if not isinstance(t, dict) or not isinstance(p, dict):
                    continue
                same = _barcode_key(t) == _barcode_key(p)
                rows.append(dict(
                    structure_id=sid, graph=graph, mode=mode,
                    tau_n_scales=t.get("n_scales"),
                    pw_n_scales=p.get("n_scales"),
                    tau_intervals=t.get("total_intervals"),
                    pw_intervals=p.get("total_intervals"),
                    barcode_same=bool(same),
                    tau_chain_valid=t.get("chain_map_valid"),
                    pw_chain_valid=p.get("chain_map_valid"),
                    tau_max_res=t.get("chain_map_max_residual"),
                    subsampled=bool(rec.get(f"{graph}_tau_subsampled"))))
            # event-value vs grid barcode agreement for the same rule
            for rule in RULES:
                ev, gr = e.get(f"{graph}_{rule}"), g.get(f"{graph}_{rule}")
                if not isinstance(ev, dict) or not isinstance(gr, dict):
                    continue
                # normalized barcodes: event uses real birth values,
                # grid uses scale indices -- compare interval counts +
                # per-order counts rather than raw endpoints
                counts_same = all(
                    len(ev.get(f"intervals_k{o}", [])) ==
                    len(gr.get(f"intervals_k{o}", [])) for o in range(4))
                rows.append(dict(
                    structure_id=sid, graph=graph, mode="event_vs_grid",
                    rule=rule,
                    event_intervals=ev.get("total_intervals"),
                    grid_intervals=gr.get("total_intervals"),
                    counts_same=bool(counts_same),
                    event_n_scales=ev.get("n_scales"),
                    grid_n_scales=gr.get("n_scales")))
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    taupw = df[df["mode"].isin(["grid", "event"])]
    print(f"structures: {len(sids)}")
    print("tau-vs-pairwise barcode agreement by mode/graph:")
    print(taupw.groupby(["mode", "graph"])["barcode_same"]
          .agg(["sum", "count"]).to_string())
    evg = df[df["mode"] == "event_vs_grid"]
    if len(evg):
        print("event-value vs grid interval-count agreement:")
        print(evg.groupby(["graph", "rule"])["counts_same"]
              .agg(["sum", "count"]).to_string())
    print("->", args.out)


if __name__ == "__main__":
    main()
