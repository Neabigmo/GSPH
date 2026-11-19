"""Export a small RNA example's directed paths and certified pair supports."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter

import pandas as pd

from cspph.core.engine import GSPH6Config
from cspph.geometry.patch_support import compute_patch_support
from tasks.rna.intramolecular import extract_intra, parse_structure
from tasks.rna.sparse import rna_event_filter


def _key(path):
    return ",".join(map(str, path))


def dump_rna_example(structure: Path, directory: Path):
    start = perf_counter()
    atoms = parse_structure(structure)
    select_events, _ = rna_event_filter(atoms, fraction=0.25, per_nt_quota=8)
    result = extract_intra(
        atoms, "rna_quality_sparse",
        config=GSPH6Config(interaction_backend="cpu", intra_min_bond_sep=4,
                           contact_min_vdw_ratio=0.6, use_cpp_path=False),
        event_filter=select_events,
    )
    meta = result["metadata"]
    paths, supports = result["path_support"], result["support_by_set"]
    if (any(meta[key] for key in ("events_truncated", "paths_truncated", "support_incomplete"))
            or not paths or not all(info["valid"] for info in paths.values())):
        raise ValueError("expected a nonempty, complete and certified path set")

    canons = {path: tuple(sorted(set(path))) for path in paths}
    required_pairs = {pair for canon in canons.values() for pair in combinations(canon, 2)}
    missing_pairs = sorted(required_pairs - supports.keys())
    patches = {p.patch_id: p for p in result["patches"]}
    atom_map = {a.index: a for a in atoms}
    support_cache, pair_cache = {}, {}
    for pair in missing_pairs:
        support = compute_patch_support(
            [patches[i] for i in pair], atom_map, support_cache=support_cache,
            pair_support_cache=pair_cache,
        )
        if not support.complete or not support.certificate_passed:
            raise ValueError(f"uncertified pair support for {pair}")
        supports[pair] = support.tau

    directory.mkdir(parents=True, exist_ok=True)
    stem = structure.stem
    pd.DataFrame([
        dict(graph="dir", order=len(path) - 1, path=_key(path),
             canon=_key(canons[path]), tau=info["tau"], valid=True)
        for path, info in paths.items()
    ]).to_parquet(directory / f"{stem}.complex_paths.parquet", index=False)
    pd.DataFrame([dict(canon=_key(p), tau=tau) for p, tau in sorted(supports.items())]
                 ).to_parquet(directory / f"{stem}.tau_table.parquet", index=False)
    return dict(structure_id=stem, atoms=len(atoms), patches=len(patches), paths=len(paths),
                additional_pairs=len(missing_pairs), extraction_seconds=perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path, help="RNA coordinate file (first model)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dump_rna_example(args.structure, args.out_dir), allow_nan=False))


if __name__ == "__main__":
    main()
