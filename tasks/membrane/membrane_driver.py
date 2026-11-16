"""M223 membrane-protein stability driver.

For each experimental mutation row: rebuild the mutant side chain
(deterministic, audited), run the frozen gsph6 ``protein_stability``
intramolecular pipeline on BOTH WT and mutant, and emit

    X_wt, X_mut, Delta X = X_mut - X_wt

as the feature row.  The label is ``Experimental_ddG`` following the
mCSM-membrane convention: negative = destabilizing.

WT feature extraction is cached per (pdb, chain set) -- the WT structure
never changes between mutations of the same protein.

Mutation convention: the M223 table lists ``original/resid/mutation``
without a chain; the protocol mutates chain ``A`` of the listed PDB entry
(first biological chain).  Every audit row records this choice.
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

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import intramolecular as I          # noqa: E402
import mutation_builder as MB        # noqa: E402
import sparse as SP                  # noqa: E402

_AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def load_m223(table_path: Path) -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_excel(table_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _resolve_chain(atoms, resid: int):
    """First chain (sorted) containing ``resid`` with a C-alpha atom."""
    for chain in sorted({str(getattr(a, "chain", "")) for a in atoms}):
        if SP.focus_ca_xyz(atoms, chain, int(resid)) is not None:
            return chain
    return None


def wt_features(pdb_path: Path, intra_sep: int, cache: dict, key: str,
                event_filter=None, atoms=None, config=None,
                scenario="protein_stability"):
    if key in cache:
        return cache[key]
    if atoms is None:
        atoms = I.parse_structure(str(pdb_path))
    res = I.extract_intra(atoms, scenario, intra_sep=intra_sep,
                          event_filter=event_filter, config=config)
    feats = {k: float(v) for k, v in (res.get("features") or {}).items()}
    meta = res.get("metadata") or {}
    cache[key] = (atoms, feats, meta)
    return cache[key]


def run_m223(table_path: Path, pdb_dir: Path, out_dir: Path,
             shard: int = 0, nshards: int = 1,
             intra_sep: int = I.DEFAULT_INTRA_SEP, limit: int = 0,
             local_radius: float = 0.0, max_path_order: int = 3,
             compute_eta: bool = True):
    from gsph6.engine import GSPH6Config
    sparse_mode = (local_radius > 0 or int(max_path_order) < 3
                   or not compute_eta)
    scenario = "protein_stability_sparse" if sparse_mode else "protein_stability"
    cfg = GSPH6Config(interaction_backend="auto",
                      intra_min_bond_sep=intra_sep,
                      max_path_order=int(max_path_order),
                      compute_eta=bool(compute_eta))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_m223(table_path)
    if limit:
        df = df.iloc[:limit]
    rows, audits = [], []
    cache: dict = {}
    atoms_cache: dict = {}
    mem_limit = float(os.environ.get("GSPH_MEM_LIMIT_MB", "0") or 0)
    try:
        import psutil
        _proc = psutil.Process()
    except ImportError:
        _proc = None
    audit_path = out_dir / f"membrane_s{shard:03d}.audit.jsonl"
    feat_path = out_dir / f"membrane_s{shard:03d}.parquet"
    with audit_path.open("a") as audit_f:
        for i, r in df.iterrows():
            if i % nshards != shard:
                continue
            if _proc is not None and mem_limit > 0:
                rss_mb = _proc.memory_info().rss / 1048576
                if rss_mb > mem_limit:
                    audit_f.write(json.dumps({
                        "row": int(i), "pdbid": str(r["pdbid"]),
                        "status": "skipped_memory_limit",
                        "rss_mb": round(rss_mb, 1),
                        "limit_mb": mem_limit}) + "\n")
                    break
            pdbid = str(r["pdbid"]).upper()
            t0 = time.time()
            rec = {"row": int(i), "pdbid": pdbid, "resid": int(r["resid"]),
                   "wt": str(r["original"]), "mut": str(r["mutation"]),
                   "status": "pending"}
            row = {"row": int(i), "pdbid": pdbid, "resid": int(r["resid"]),
                   "wt_aa": str(r["original"]), "mut_aa": str(r["mutation"]),
                   "protein": str(r.get("protein", "")),
                   "ddG": float(r["Experimental_ddG"])}
            try:
                mut3 = _AA1_TO_3[str(r["mutation"]).upper()]
                pdb_path = pdb_dir / f"{pdbid}.cif"
                if not pdb_path.is_file():
                    pdb_path = pdb_dir / f"{pdbid}.pdb"
                resid_i = int(r["resid"])
                if local_radius > 0:
                    # GSPH-S local mode: WT atoms parsed once per protein;
                    # WT and mutant share the SAME spatial focus (the WT
                    # residue C-alpha -- the backbone is unchanged by the
                    # side-chain rebuild) so Delta X stays consistent.
                    if pdbid not in atoms_cache:
                        atoms_cache[pdbid] = I.parse_structure(str(pdb_path))
                    wt_atoms = atoms_cache[pdbid]
                    chain_i = _resolve_chain(wt_atoms, resid_i)
                    if chain_i is None:
                        raise ValueError(
                            f"no residue with CA for resid {resid_i} in {pdbid}")
                    focus = SP.focus_ca_xyz(wt_atoms, chain_i, resid_i)
                    wt_key = f"{pdbid}|{intra_sep}|{scenario}|loc{local_radius}|r{resid_i}"
                    wt_filter, _ = SP.local_event_filter(
                        wt_atoms, focus, local_radius)
                    wt_atoms, wf, wm = wt_features(
                        pdb_path, intra_sep, cache, wt_key,
                        event_filter=wt_filter, atoms=wt_atoms,
                        config=cfg, scenario=scenario)
                    mut_atoms, mb_audit = MB.build_mutant_atoms(
                        wt_atoms, chain_i, resid_i, "", mut3)
                    mut_filter, mfilter_stats = SP.local_event_filter(
                        mut_atoms, focus, local_radius)
                    mres = I.extract_intra(
                        mut_atoms, scenario, intra_sep=intra_sep,
                        config=cfg, event_filter=mut_filter)
                else:
                    wt_atoms, wf, wm = wt_features(
                        pdb_path, intra_sep, cache,
                        f"{pdbid}|{intra_sep}|{scenario}",
                        config=cfg, scenario=scenario)
                    chain_i = _resolve_chain(wt_atoms, resid_i)
                    if chain_i is None:
                        raise ValueError(
                            f"no residue with CA for resid {resid_i} in {pdbid}")
                    mut_atoms, mb_audit = MB.build_mutant_atoms(
                        wt_atoms, chain_i, resid_i, "", mut3)
                    mres = I.extract_intra(mut_atoms, scenario,
                                           intra_sep=intra_sep, config=cfg)
                mf = {k: float(v) for k, v in (mres.get("features") or {}).items()}
                mmeta = mres.get("metadata") or {}
                for k in sorted(set(wf) | set(mf)):
                    row[f"d_{k}"] = mf.get(k, np.nan) - wf.get(k, np.nan)
                row["wt_status"] = str(wm.get("status"))
                row["mut_status"] = str(mmeta.get("status"))
                row["wt_topology_valid"] = bool(wm.get("topology_valid"))
                row["mut_topology_valid"] = bool(mmeta.get("topology_valid"))
                rec.update({"status": "ok", "wt_status": wm.get("status"),
                            "mut_status": mmeta.get("status"),
                            "local_radius": float(local_radius),
                            "wt_event_filter": wm.get("event_filter"),
                            "mut_event_filter": mmeta.get("event_filter"),
                            "mutation_audit": mb_audit,
                            "wt_timing": wm.get("timing", {}).get("total"),
                            "mut_timing": mmeta.get("timing", {}).get("total")})
            except Exception as exc:
                rec.update({"status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-2000:]})
            rec["elapsed_s"] = round(time.time() - t0, 2)
            audit_f.write(json.dumps(rec) + "\n")
            audit_f.flush()
            rows.append(row)
            if len(rows) % 8 == 0:
                _write(feat_path, rows)
    _write(feat_path, rows)
    return len(rows)


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
    ap.add_argument("--table", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--intra-sep", type=int, default=I.DEFAULT_INTRA_SEP)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--local-radius", type=float, default=0.0,
                    help="GSPH-S: keep only events within this many Angstrom "
                         "of the mutated residue C-alpha (0 = full protein)")
    ap.add_argument("--max-path-order", type=int, default=3,
                    help="GSPH-S: highest GLMY path order (<=3)")
    ap.add_argument("--no-eta", action="store_true",
                    help="GSPH-S: skip eta subset statistics (sparse schema)")
    a = ap.parse_args()
    n = run_m223(Path(a.table), Path(a.pdb_dir), Path(a.out),
                 a.shard, a.nshards, a.intra_sep, a.limit,
                 local_radius=a.local_radius,
                 max_path_order=a.max_path_order,
                 compute_eta=not a.no_eta)
    print(json.dumps({"rows": n}))


if __name__ == "__main__":
    main()
