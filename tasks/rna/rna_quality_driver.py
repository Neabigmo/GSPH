"""RNA-quality extraction driver (rsRNASP PM set).

Enumerates native + decoy structures, runs the frozen gsph6 intramolecular
pipeline per structure, joins official RMSD/DI labels, and writes
shard-level parquet + JSONL audit.  Incomplete results keep NaN features --
they are never zero-filled.

Usage:
    python rna_quality_driver.py --pm-root external/rsRNASP/PM_decoy_set \
        --out features/rna --shard 0 --nshards 1 [--max-events -1] [--limit N]
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

import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import intramolecular as I  # noqa: E402
import sparse as SP        # noqa: E402


def _labels(label_root: Path, target: str) -> dict:
    """Official PM labels: RMSD and DI per decoy filename."""
    out = {}
    for name, key in (("RMSD", "label_rmsd_official"), ("DI", "label_di")):
        f = None
        for cand in (label_root / name / f"{target}.{name.lower()}",
                     label_root / name / f"{target}.{name}"):
            if cand.is_file():
                f = cand
                break
        if f is None:
            continue
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out.setdefault(parts[0], {})[key] = float(parts[1])
                except ValueError:
                    pass
    return out


def enumerate_units(pm_root: Path) -> list:
    units = []
    for target_dir in sorted(pm_root.iterdir()):
        if not target_dir.is_dir() or target_dir.name in {"DI", "RMSD"}:
            continue
        target = target_dir.name
        labels = _labels(pm_root, target)
        for pdb in sorted(target_dir.glob("*.pdb")):
            kind = "native" if pdb.stem.upper() == target.upper() else "decoy"
            units.append({
                "structure_id": pdb.stem,
                "path": str(pdb),
                "group": target,
                "kind": kind,
                "label_rmsd_official": labels.get(pdb.name, {}).get("label_rmsd_official"),
                "label_di": labels.get(pdb.name, {}).get("label_di"),
            })
    return units


def run_shard(units, out_dir: Path, shard: int, nshards: int,
              intra_sep: int, config_kwargs: dict,
              sparse_fraction: float = 1.0, per_nt_quota=None,
              max_path_order: int = 3, compute_eta: bool = True,
              hodge_dense_max: int = 384):
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_units = [u for i, u in enumerate(units) if i % nshards == shard]
    audit_path = out_dir / f"rna_quality_s{shard:03d}.audit.jsonl"
    feat_path = out_dir / f"rna_quality_s{shard:03d}.parquet"
    done = set()
    if audit_path.exists():
        for line in audit_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                # Only successful extractions are "done"; errors and memory
                # skips are retried on the next run.  GSPH_RETRY_INCOMPLETE=1
                # also retries hodge/truncation incompletes (e.g. after
                # raising hodge_dense_max_dimension).
                retry_inc = os.environ.get("GSPH_RETRY_INCOMPLETE", "") == "1"
                st = rec.get("status")
                if st not in {"error", "skipped_memory_limit"} and not (
                        retry_inc and st == "incomplete"):
                    done.add(rec["structure_id"])
            except Exception:
                pass
    rows = []
    if feat_path.exists():
        import pandas as pd
        rows = pd.read_parquet(feat_path).to_dict("records")
    native_atoms = {}
    mem_limit = float(os.environ.get("GSPH_MEM_LIMIT_MB", "0") or 0)
    try:
        import psutil
        _proc = psutil.Process()
    except ImportError:
        _proc = None
    with audit_path.open("a") as audit:
        for unit in shard_units:
            if unit["structure_id"] in done:
                continue
            if _proc is not None and mem_limit > 0:
                rss_mb = _proc.memory_info().rss / 1048576
                if rss_mb > mem_limit:
                    audit.write(json.dumps({
                        "structure_id": unit["structure_id"],
                        "group": unit["group"], "status": "skipped_memory_limit",
                        "rss_mb": round(rss_mb, 1),
                        "limit_mb": mem_limit}) + "\n")
                    break
            t0 = time.time()
            row = {k: unit[k] for k in ("structure_id", "group", "kind",
                                        "label_rmsd_official", "label_di")}
            try:
                from gsph6.engine import GSPH6Config
                sparse_mode = (sparse_fraction < 1.0 or per_nt_quota is not None
                               or int(max_path_order) < 3 or not compute_eta)
                scenario = "rna_quality_sparse" if sparse_mode else "rna_quality"
                cfg = GSPH6Config(
                    interaction_backend="auto",
                    intra_min_bond_sep=intra_sep,
                    max_path_order=int(max_path_order),
                    compute_eta=bool(compute_eta),
                    hodge_dense_max_dimension=int(hodge_dense_max))
                atoms = I.parse_structure(unit["path"])
                event_filter, filter_meta = (None, {})
                if sparse_fraction < 1.0 or per_nt_quota is not None:
                    event_filter, filter_meta = SP.rna_event_filter(
                        atoms, fraction=sparse_fraction,
                        per_nt_quota=per_nt_quota)
                feats, meta = I.featurize_structure(
                    unit["path"], scenario, intra_sep=intra_sep,
                    config=cfg, atoms=atoms, event_filter=event_filter)
                row["gsph_mode"] = scenario
                meta.update({f"sparse_{k}": v for k, v in filter_meta.items()})
                row.update(feats)
                if unit["kind"] == "decoy":
                    target_native = native_atoms.get(unit["group"])
                    if target_native is None:
                        npath = Path(unit["path"]).parent / f"{unit['group']}.pdb"
                        if npath.is_file():
                            target_native = I.parse_structure(str(npath))
                            native_atoms[unit["group"]] = target_native
                    if target_native is not None:
                        row["label_c4_rmsd"] = I.c4_rmsd(target_native, atoms)
                        row["label_tm"] = I.tm_score(target_native, atoms)
                        row["label_lddt_c4"] = I.lddt_c4(target_native, atoms)
                audit_rec = {"structure_id": unit["structure_id"], "group": unit["group"],
                             "elapsed_s": round(time.time() - t0, 2), **meta}
            except Exception as exc:  # failure stays visible
                audit_rec = {"structure_id": unit["structure_id"],
                             "group": unit["group"], "status": "error",
                             "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()[-2000:],
                             "elapsed_s": round(time.time() - t0, 2)}
            audit.write(json.dumps(audit_rec) + "\n")
            audit.flush()
            # drop any stale row for this structure (e.g. retried incomplete)
            rows[:] = [r for r in rows
                       if r.get("structure_id") != unit["structure_id"]]
            rows.append(row)
            # checkpoint every 32 structures
            if len(rows) % 32 == 0:
                _write(feat_path, rows)
    _write(feat_path, rows)
    return len(shard_units), len(done)


def _write(path: Path, rows):
    if not rows:
        return
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(path, index=False)
    except ImportError:
        import csv
        csv_path = path.with_suffix(".csv")
        keys = sorted({k for r in rows for k in r})
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--intra-sep", type=int, default=I.DEFAULT_INTRA_SEP)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sparse-fraction", type=float, default=1.0,
                    help="GSPH-S: keep top fraction of stratified events "
                         "(1.0 = full-atom reference)")
    ap.add_argument("--per-nt-quota", type=int, default=0,
                    help="GSPH-S: max kept events touching one nucleotide "
                         "(0 = off)")
    ap.add_argument("--max-path-order", type=int, default=3,
                    help="GSPH-S: highest GLMY path order (<=3)")
    ap.add_argument("--no-eta", action="store_true",
                    help="GSPH-S: skip eta subset statistics (sparse schema)")
    ap.add_argument("--hodge-dense-max", type=int, default=384,
                    help="dense Hodge eigensolver dimension cap")
    args = ap.parse_args()
    units = enumerate_units(Path(args.pm_root))
    if args.limit:
        units = units[:args.limit]
    n, done = run_shard(units, Path(args.out), args.shard, args.nshards,
                        args.intra_sep, {},
                        sparse_fraction=args.sparse_fraction,
                        per_nt_quota=(args.per_nt_quota or None),
                        max_path_order=args.max_path_order,
                        compute_eta=not args.no_eta,
                        hodge_dense_max=args.hodge_dense_max)
    print(json.dumps({"shard": args.shard, "units": n, "already_done": done}))


if __name__ == "__main__":
    main()
