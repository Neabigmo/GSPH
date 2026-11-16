"""Protein-DNA FTDMP decoy extraction driver (intermolecular scenario).

Enumerates relaxed_top_complexes per benchmark case under both
Bound_docking/ and Unbound_docking/, runs the frozen gsph6 intermolecular
pipeline (scenario=protein_dna), and joins the official FTDMP labels:
iface CAD-score, ligand RMSD, FTDMP rank.

Shard layout mirrors the RNA driver; audit JSONL keeps failures visible and
NaN features are never zero-filled.
"""
from __future__ import annotations
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

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WS = os.environ.get("GSPH_WORKSPACE",
                     str(_HERE.parents[2] if len(_HERE.parents) > 2 else _HERE))
for _p in (_WS, str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runners.gsph1 import _parse_atoms  # noqa: E402


def _labels(case_dir: Path) -> dict:
    labels = {}
    cad = case_dir / "cad_scores.txt"
    if cad.is_file():
        lines = cad.read_text().splitlines()
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    labels.setdefault(parts[0], {})["label_cadscore"] = float(parts[1])
                    labels.setdefault(parts[0], {})["label_site_cadscore"] = float(parts[2])
                except ValueError:
                    pass
    rmsd = case_dir / "rmsd_results.txt"
    if rmsd.is_file():
        for line in rmsd.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    labels.setdefault(parts[0], {})["label_ligand_rmsd"] = float(parts[1])
                except ValueError:
                    pass
    ranks = case_dir / "scoring_results-ranks.txt"
    if ranks.is_file():
        lines = ranks.read_text().splitlines()
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    labels.setdefault(parts[0], {})["label_ftdmp_rank"] = float(parts[1])
                except ValueError:
                    pass
    return labels


def enumerate_units(root: Path) -> list:
    units = []
    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir() or "docking" not in mode_dir.name.lower():
            continue
        bound = "bound" in mode_dir.name.lower() and "unbound" not in mode_dir.name.lower()
        for case_dir in sorted(mode_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            target = case_dir.name.upper()
            labels = _labels(case_dir)
            comp_dir = case_dir / "relaxed_top_complexes"
            if not comp_dir.is_dir():
                continue
            for pdb in sorted(comp_dir.glob("*.pdb")):
                entry = {"structure_id": pdb.stem, "path": str(pdb),
                         "group": target, "kind": "decoy",
                         "bound_protocol": bound}
                entry.update(labels.get(pdb.name, {}))
                units.append(entry)
    return units


def run_shard(units, out_dir: Path, shard: int, nshards: int,
              interaction_backend: str = "cpu"):
    from gsph6.engine import GSPH6Config, extract_gsph6_atoms
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_units = [u for i, u in enumerate(units) if i % nshards == shard]
    audit_path = out_dir / f"pd_ftdmp_s{shard:03d}.audit.jsonl"
    feat_path = out_dir / f"pd_ftdmp_s{shard:03d}.parquet"
    done = set()
    if audit_path.exists():
        for line in audit_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("status") not in {"error", "skipped_memory_limit"}:
                    done.add(rec["structure_id"])
            except Exception:
                pass
    rows = []
    if feat_path.exists():
        import pandas as pd
        rows = pd.read_parquet(feat_path).to_dict("records")
    mem_limit = float(os.environ.get("GSPH_MEM_LIMIT_MB", "0") or 0)
    try:
        import psutil
        _proc = psutil.Process()
    except ImportError:
        _proc = None
    cfg = GSPH6Config(interaction_backend=interaction_backend)
    with audit_path.open("a") as audit:
        for unit in shard_units:
            if unit["structure_id"] in done:
                continue
            if _proc is not None and mem_limit > 0:
                rss_mb = _proc.memory_info().rss / 1048576
                if rss_mb > mem_limit:
                    audit.write(json.dumps({
                        "structure_id": unit["structure_id"],
                        "group": unit["group"],
                        "status": "skipped_memory_limit",
                        "rss_mb": round(rss_mb, 1)}) + "\n")
                    break
            t0 = time.time()
            row = {k: v for k, v in unit.items() if k != "path"}
            try:
                atoms = _parse_atoms(unit["path"])
                res = extract_gsph6_atoms(list(atoms), scenario="protein_dna",
                                          config=cfg)
                meta = res.get("metadata") or {}
                row.update({k: float(v) for k, v in
                            (res.get("features") or {}).items()})
                rec = {"structure_id": unit["structure_id"],
                       "group": unit["group"], "kind": unit["kind"],
                       "bound_protocol": unit["bound_protocol"],
                       "elapsed_s": round(time.time() - t0, 2),
                       "status": meta.get("status"),
                       "topology_valid": meta.get("topology_valid"),
                       "support_incomplete": meta.get("support_incomplete"),
                       "hodge_incomplete": meta.get("hodge_incomplete"),
                       "persistence_valid": meta.get("persistence_valid"),
                       "timing": meta.get("timing") or {}}
            except Exception as exc:
                rec = {"structure_id": unit["structure_id"],
                       "group": unit["group"], "status": "error",
                       "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-2000:],
                       "elapsed_s": round(time.time() - t0, 2)}
            audit.write(json.dumps(rec) + "\n")
            audit.flush()
            rows.append(row)
            if len(rows) % 32 == 0:
                _write(feat_path, rows)
    _write(feat_path, rows)
    return len(shard_units)


def _write(path: Path, rows):
    if not rows:
        return
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(path, index=False)
    except ImportError:
        import csv
        keys = sorted({k for r in rows for k in r})
        with path.with_suffix(".csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Protein-DNA extraction root (contains *_docking dirs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    units = enumerate_units(Path(args.root))
    if args.limit:
        units = units[:args.limit]
    n = run_shard(units, Path(args.out), args.shard, args.nshards, args.backend)
    print(json.dumps({"shard": args.shard, "units": n}))


if __name__ == "__main__":
    main()
