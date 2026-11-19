"""Re-audit same-graph H0–H2 at the full union of birth values in Å."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import isfinite
from pathlib import Path
from time import perf_counter

import pandas as pd

from cspph.topology.event_persistence import compare_event_barcodes


def load_births(directory: Path, structure: str, graph: str):
    paths = pd.read_parquet(directory / f"{structure}.complex_paths.parquet")
    supports = pd.read_parquet(directory / f"{structure}.tau_table.parquet")
    paths = paths.loc[paths["graph"] == graph]
    if paths.empty or not paths["valid"].fillna(False).all():
        raise ValueError(f"{structure}: missing graph or uncertified {graph} paths")
    if supports["canon"].duplicated().any():
        raise ValueError(f"{structure}: duplicate support keys")
    tau = supports.set_index("canon")["tau"].to_dict()

    common, pairwise = {}, {}
    for row in paths.itertuples(index=False):
        path = tuple(map(int, row.path.split(",")))
        if path in common:
            raise ValueError(f"{structure}: duplicate path {path}")
        canon = tuple(sorted(set(path)))
        subsets = (canon,) if len(canon) <= 2 else combinations(canon, 2)
        values = []
        for subset in subsets:
            key = ",".join(map(str, subset))
            value = tau.get(key)
            if value is None or not isfinite(value):
                raise ValueError(f"{structure}: missing certified support for {key}")
            values.append(float(value))
        common[path] = float(row.tau)
        pairwise[path] = max(values)
    return common, pairwise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path, help="Directory of per-structure dumps")
    parser.add_argument("--sid", required=True, help="Structure identifier in the dump filenames")
    parser.add_argument("--graph", choices=("dir", "und"), default="dir")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    common, pairwise = load_births(args.dir, args.sid, args.graph)
    start = perf_counter()
    scales, common_bars, pairwise_bars = compare_event_barcodes(common, pairwise)
    result = dict(
        structure_id=args.sid,
        graph=args.graph,
        paths=len(common),
        scales_A=scales,
        common_A=common_bars,
        pairwise_A=pairwise_bars,
        equal_by_order={order: bars == pairwise_bars[order] for order, bars in common_bars.items()},
        persistence_seconds=perf_counter() - start,
    )
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
