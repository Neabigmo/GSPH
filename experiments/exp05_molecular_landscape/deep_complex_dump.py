# -*- coding: utf-8 -*-
"""Deep complex dump: per-structure path complex + all solved supports.

Extends deep_subset.py output with what filtration-level audits need:

  <sid>.complex_paths.parquet
      graph        : 'dir' (chemical directed edges) | 'und' (symmetrized)
      order        : GLMY order 0..3 (path length = order+1 patches)
      path         : ordered patch-id tuple as string "a,b,c"
      canon        : sorted unique patch-id set as string
      tau          : common-support scale (NaN if uncertified)
      valid        : certificate passed
  <sid>.complex_edges.parquet
      graph, u, v, w  (w = min_bond_distance edge weight)
  <sid>.complex_manifest.json

Tau for 'und' paths whose patch set was never solved under 'dir' is computed
on the fly with the SAME compute_patch_support solver (atoms/patches come
from the same extraction), so both graphs share one certified support table.
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
for _e in sorted((_R / "experiments").glob("exp*")):
    if _e.is_dir() and str(_e) not in _sys.path:
        _sys.path.insert(0, str(_e))
import argparse, json, sys, time, traceback
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_H3 = _HERE.parent
sys.path.insert(0, str(_H3 / "gsph" / "task_adapters"))
sys.path.insert(0, str(_H3.parent))
sys.path.insert(0, str(_HERE))
import deep_subset as DS          # noqa: E402  (pick_rna/pick_pd/run_*)
import deep_dump                  # noqa: E402


def _canon_tau(patch_map, atom_map, canon, cache, cfg_max_comb):
    key = tuple(sorted(set(canon)))
    if key in cache:
        return cache[key]
    from gsph6.geometry import compute_patch_support
    patches = [patch_map[i] for i in key]
    res = compute_patch_support(patches, atom_map, cfg_max_comb)
    tau = float(res.tau) if (res.certificate_passed and np.isfinite(res.tau)) else np.nan
    cache[key] = tau
    return tau


def dump_complex(res, atoms, out_dir: Path, sid: str):
    from gsph6.graph import enumerate_patch_paths
    meta = res.get("metadata") or {}
    cfg_scales = res.get("filtration_scales") or [0.0, 0.5, 2.0]
    edges = res.get("edges") or []
    patch_map = {int(p.patch_id): p for p in (res.get("patches") or [])}
    atom_map = {int(a.index): a for a in atoms}
    tau_cache = {}
    for k, v in (res.get("support_by_set") or {}).items():
        tau_cache[tuple(int(x) for x in k)] = (
            float(v) if v is not None and np.isfinite(float(v)) else np.nan)

    path_rows, edge_rows = [], []
    patch_ids = sorted(patch_map)

    # Directed paths: res["path_support"] keys are the enumerated tuples
    ps = res.get("path_support") or {}
    for path, info in ps.items():
        t = tuple(int(x) for x in path)
        canon = tuple(sorted(set(t)))
        order = len(t) - 1
        tau = tau_cache.get(canon, np.nan)
        if not np.isfinite(tau):
            tau = _canon_tau(patch_map, atom_map, canon, tau_cache, None)
        path_rows.append(dict(graph="dir", order=order,
                              path=",".join(map(str, t)),
                              canon=",".join(map(str, canon)),
                              tau=tau, valid=bool(info.get("valid"))))
    for u, v, w in edges:
        edge_rows.append(dict(graph="dir", u=int(u), v=int(v), w=float(w)))

    # Symmetrized graph: same nodes, undirected edges
    und = sorted({(int(u), int(v), float(w)) for u, v, w in edges} |
                 {(int(v), int(u), float(w)) for u, v, w in edges})
    try:
        paths_u, counts_u, backend_u = enumerate_patch_paths(
            patch_ids, und, max_order=3, use_cpp=False)
    except Exception as exc:
        paths_u, backend_u = {}, f"error:{type(exc).__name__}:{exc}"
    n_und = 0
    for order, rows in (paths_u or {}).items():
        for row in rows:
            t = tuple(int(x) for x in row)
            canon = tuple(sorted(set(t)))
            tau = _canon_tau(patch_map, atom_map, canon, tau_cache, None)
            path_rows.append(dict(graph="und", order=int(order),
                                  path=",".join(map(str, t)),
                                  canon=",".join(map(str, canon)),
                                  tau=tau, valid=bool(np.isfinite(tau))))
            n_und += 1
    for u, v, w in und:
        edge_rows.append(dict(graph="und", u=u, v=v, w=w))

    # Ensure every subset of size <=3 of every canon set is solved, so
    # alternative filtrations (pairwise-max, facet-max) are well-defined
    # offline.  Most are already cached; only genuinely new sets cost solves.
    all_canons = {tuple(int(x) for x in r["canon"].split(","))
                  for r in path_rows if r["canon"]}
    n_subset_solves = 0
    for canon in all_canons:
        for size in (1, 2, 3):
            if size >= len(canon):
                continue
            for sub in combinations(canon, size):
                key = tuple(sorted(set(sub)))
                if key not in tau_cache or not np.isfinite(tau_cache.get(key, np.nan)):
                    before = len(tau_cache)
                    _canon_tau(patch_map, atom_map, key, tau_cache, None)
                    n_subset_solves += len(tau_cache) - before

    pd.DataFrame(path_rows).to_parquet(out_dir / f"{sid}.complex_paths.parquet", index=False)
    pd.DataFrame(edge_rows).to_parquet(out_dir / f"{sid}.complex_edges.parquet", index=False)

    # Full certified support table: every canon set solved (path canons,
    # und-path canons, and all subsets of size <=3).  Graph-independent --
    # tau depends only on the patch SET, not on dir/und or path order.
    pd.DataFrame([dict(canon=",".join(map(str, k)), tau=v)
                  for k, v in sorted(tau_cache.items())]
                 ).to_parquet(out_dir / f"{sid}.tau_table.parquet", index=False)

    # P0-5 patch/chemistry audit: structural invariants of the patch layer.
    pa = patch_audit(res, atom_map)
    pd.DataFrame([pa]).to_parquet(out_dir / f"{sid}.patch_audit.parquet", index=False)

    man = dict(structure_id=sid, scales=cfg_scales,
               n_dir_paths=int(len(ps)), n_und_paths=int(n_und),
               und_backend=str(backend_u),
               n_dir_edges=int(len(edges)), n_und_edges=int(len(und)),
               n_canon_solved=int(len(tau_cache)),
               n_subset_solves=int(n_subset_solves))
    (out_dir / f"{sid}.complex_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def patch_audit(res, atom_map):
    """P0-5: verify patch-layer chemical/structural invariants."""
    from gsph6.geometry import _contact_support
    patches = list(res.get("patches") or [])
    n_contacts = 0; bad_contact = 0; tau_mismatch = 0; checked_tau = 0
    donor_overlap = 0; empty_patches = 0
    dists = []
    rng = np.random.default_rng(0)
    for p in patches:
        if not p.contacts:
            empty_patches += 1
        dm = set(int(x) for x in p.donor_members)
        am = set(int(x) for x in p.acceptor_members)
        if dm & am:
            donor_overlap += 1
        for c in p.contacts:
            n_contacts += 1
            try:
                d = float(np.linalg.norm(
                    np.asarray(atom_map[int(c.donor)].xyz) -
                    np.asarray(atom_map[int(c.acceptor)].xyz)))
                dists.append(d)
                if int(c.donor) == int(c.acceptor):
                    bad_contact += 1
            except Exception:
                bad_contact += 1
            # spot-check tau = 0.5*(d - r_d - r_a) on a sample
            if rng.random() < 0.02 and int(c.donor) in atom_map and int(c.acceptor) in atom_map:
                try:
                    expect = _contact_support(c, atom_map)
                    if np.isfinite(c.tau) and abs(float(c.tau) - expect) > 1e-6:
                        tau_mismatch += 1
                    checked_tau += 1
                except Exception:
                    pass
    dists = np.asarray(dists) if dists else np.array([np.nan])
    return dict(n_patches=len(patches), n_contacts=int(n_contacts),
                empty_patches=int(empty_patches),
                donor_acceptor_overlap=int(donor_overlap),
                bad_contacts=int(bad_contact),
                tau_checked=int(checked_tau), tau_mismatch=int(tau_mismatch),
                dist_min=float(np.nanmin(dists)), dist_max=float(np.nanmax(dists)),
                dist_mean=float(np.nanmean(dists)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rna", "pd"], required=True)
    ap.add_argument("--pm-root", default=str(DS._H3 / "external/rsRNASP/PM_decoy_set"))
    ap.add_argument("--ftdmp-root", default="")
    ap.add_argument("--n-targets", type=int, default=4)
    ap.add_argument("--n-per", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.task == "rna":
        units = DS.pick_rna(Path(a.pm_root), a.n_targets, a.n_per)
        runner = DS.run_rna
    else:
        units = DS.pick_pd(Path(a.ftdmp_root), a.n_targets, a.n_per)
        runner = DS.run_pd
    units = units[a.shard::a.n_shards]
    print(f"[complex] {len(units)} structures shard {a.shard}/{a.n_shards} -> {out}", flush=True)
    from gsph6.engine import GSPH6Config
    cfg = GSPH6Config(interaction_backend="auto", max_path_order=3,
                      compute_eta=True, hodge_dense_max_dimension=384)
    for u in units:
        sid = u["structure_id"]; t0 = time.time()
        try:
            if a.task == "rna":
                import intramolecular as I
                atoms = I.parse_structure(u["path"])
                import sparse as SP
                event_filter, _ = SP.rna_event_filter(atoms, fraction=0.25, per_nt_quota=8)
                res = I.extract_intra(atoms, "rna_quality_sparse",
                                      intra_sep=I.DEFAULT_INTRA_SEP,
                                      config=cfg, event_filter=event_filter)
            else:
                import pd_ftdmp_driver as D
                atoms = D._parse_atoms(u["path"])
                from gsph6.engine import extract_gsph6_atoms
                res = extract_gsph6_atoms(list(atoms), scenario="protein_dna", config=cfg)
            m = dump_complex(res, list(atoms), out, sid)
            print(f"[complex] {sid} ok {time.time()-t0:.0f}s "
                  f"dir={m['n_dir_paths']} und={m['n_und_paths']} "
                  f"sets={m['n_canon_solved']}", flush=True)
        except Exception as e:
            print(f"[complex] {sid} ERROR {type(e).__name__}: {e}", flush=True)
            (out / f"{sid}.complex_error.txt").write_text(traceback.format_exc())
    print("[complex] done", flush=True)


if __name__ == "__main__":
    main()
