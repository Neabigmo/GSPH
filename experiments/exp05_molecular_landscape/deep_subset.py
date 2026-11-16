# -*- coding: utf-8 -*-
"""Small-subset deep extraction: barcode + per-path tau/eta -> parquet.

Runs the UNMODIFIED math pipeline (engine patched only to serialize
``birth``/``eta``/``persistence`` that were already computed internally).

  RNA :  --task rna --pm-root external/rsRNASP/PM_decoy_set
         picks N_TARGETS x N_PER targets, RMSD-stratified decoys.
  PD  :  --task pd --ftdmp-root <dir with *_docking/*/relaxed_top_complexes>
         picks N_TARGETS x N_PER, CAD-stratified decoys (bound protocol).

Usage (from GSPH_H3 dir):
  python ablation/deep_subset.py --task rna --n-targets 4 --n-per 3
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
import argparse, os, sys, time, traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_H3 = _HERE.parent
sys.path.insert(0, str(_H3 / "gsph" / "task_adapters"))
sys.path.insert(0, str(_H3.parent))          # repo root for gsph6/runners
sys.path.insert(0, str(_HERE))
import deep_dump  # noqa: E402


def _n_atoms(path: str) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for ln in fh if ln[:6] in (b"ATOM  ", b"HETATM"))
    except OSError:
        return -1


def _stratified_targets(by_t: dict, label_key: str, n_targets: int,
                        size_key: str = "size"):
    """Pick targets spanning difficulty AND structure-size strata.

    Difficulty = median official label (RNA: RMSD, PD: CADscore; note the
    directions differ, so we stratify on RANK not raw value).  Size = atom
    count of the target's first decoy (proxy for extraction cost).
    Returns list of (target, stratum_tag).
    """
    stats = []
    for t, v in by_t.items():
        if len(v) < 3:
            continue
        labs = sorted(x[label_key] for x in v)
        med = labs[len(labs) // 2]
        iqr = labs[int(0.75 * (len(labs) - 1))] - labs[int(0.25 * (len(labs) - 1))]
        size = _n_atoms(v[0]["path"])
        stats.append(dict(target=t, med=med, iqr=iqr, size=size, n=len(v)))
    if not stats:
        return []
    # difficulty tertiles on median label rank
    order = sorted(stats, key=lambda s: s["med"])
    n = len(order)
    for i, s in enumerate(order):
        s["diff_stratum"] = min(2, i * 3 // max(n, 1))
    # size tertiles
    order_s = sorted(stats, key=lambda s: s["size"])
    for i, s in enumerate(order_s):
        s["size_stratum"] = min(2, i * 3 // max(n, 1))
    # greedy: cover (diff,size) cells, prefer larger candidate pools
    chosen, used = [], set()
    cells = sorted({(s["diff_stratum"], s["size_stratum"]) for s in stats})
    for cell in cells:
        cand = [s for s in stats if (s["diff_stratum"], s["size_stratum"]) == cell
                and s["target"] not in used]
        if not cand:
            continue
        best = max(cand, key=lambda s: (s["iqr"], s["n"]))
        chosen.append(best); used.add(best["target"])
        if len(chosen) >= n_targets:
            break
    if len(chosen) < n_targets:  # fill remaining by difficulty coverage
        for s in order:
            if s["target"] not in used:
                chosen.append(s); used.add(s["target"])
            if len(chosen) >= n_targets:
                break
    return [(s["target"], f"d{s['diff_stratum']}_s{s['size_stratum']}")
            for s in chosen[:n_targets]]


def pick_rna(pm_root: Path, n_targets: int, n_per: int):
    import rna_quality_driver as R
    units = [u for u in R.enumerate_units(pm_root)
             if u["kind"] == "decoy" and u["label_rmsd_official"] is not None]
    by_t = {}
    for u in units:
        by_t.setdefault(u["group"], []).append(u)
    picked = []
    for t, tag in _stratified_targets(by_t, "label_rmsd_official", n_targets):
        v = sorted(by_t[t], key=lambda x: x["label_rmsd_official"])
        idx = np_linspace(0, len(v) - 1, n_per)
        for i in idx:
            u = dict(v[int(i)]); u["stratum"] = tag
            picked.append(u)
    return picked


def np_linspace(a, b, n):
    if n == 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def pick_pd(root: Path, n_targets: int, n_per: int):
    import pd_ftdmp_driver as D
    units = [u for u in D.enumerate_units(root)
             if u.get("bound_protocol") and u.get("label_cadscore") is not None]
    by_t = {}
    for u in units:
        by_t.setdefault(u["group"], []).append(u)
    picked = []
    for t, tag in _stratified_targets(by_t, "label_cadscore", n_targets):
        v = sorted(by_t[t], key=lambda x: x["label_cadscore"])
        idx = np_linspace(0, len(v) - 1, n_per)
        for i in idx:
            u = dict(v[int(i)]); u["stratum"] = tag
            picked.append(u)
    return picked


def run_rna(unit, cfg):
    import intramolecular as I
    import sparse as SP
    atoms = I.parse_structure(unit["path"])
    # Match the production RNA extraction exactly: stratified sparse event
    # filter (fraction=0.25, per_nt_quota=8) + rna_quality_sparse scenario.
    # Full-atom RNA events are ~4x denser and cost many hours per structure.
    event_filter, _meta = SP.rna_event_filter(atoms, fraction=0.25,
                                              per_nt_quota=8)
    return I.extract_intra(atoms, "rna_quality_sparse",
                           intra_sep=I.DEFAULT_INTRA_SEP, config=cfg,
                           event_filter=event_filter)


def run_pd(unit, cfg):
    import pd_ftdmp_driver as D
    atoms = D._parse_atoms(unit["path"])
    from gsph6.engine import extract_gsph6_atoms
    return extract_gsph6_atoms(list(atoms), scenario="protein_dna", config=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rna", "pd"], required=True)
    ap.add_argument("--pm-root", default=str(_H3 / "external/rsRNASP/PM_decoy_set"))
    ap.add_argument("--ftdmp-root", default="")
    ap.add_argument("--n-targets", type=int, default=4)
    ap.add_argument("--n-per", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", default=str(_HERE.parent / "results" / "deep_subset"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.task == "rna":
        units = pick_rna(Path(a.pm_root), a.n_targets, a.n_per)
        runner = run_rna
    else:
        units = pick_pd(Path(a.ftdmp_root), a.n_targets, a.n_per)
        runner = run_pd
    if a.n_shards > 1:
        units = units[a.shard::a.n_shards]
    print(f"[deep] {len(units)} structures (shard {a.shard}/{a.n_shards}) -> {out}", flush=True)
    from gsph6.engine import GSPH6Config
    cfg = GSPH6Config(interaction_backend="auto", max_path_order=3,
                      compute_eta=True, hodge_dense_max_dimension=384)
    manifest = []
    for u in units:
        t0 = time.time()
        try:
            res = runner(u, cfg)
            m = deep_dump.dump_deep(res, out, u["structure_id"])
            m.update(group=u["group"], kind=u["kind"], stratum=u.get("stratum"),
                     label_rmsd=u.get("label_rmsd_official"),
                     label_cad=u.get("label_cadscore"),
                     elapsed_s=round(time.time() - t0, 1))
            manifest.append(m)
            print(f"[deep] {u['structure_id']} ok {m['elapsed_s']}s "
                  f"paths={m.get('support_n_paths')}", flush=True)
        except Exception as e:
            manifest.append(dict(structure_id=u["structure_id"], group=u["group"],
                                 status="error", error=f"{type(e).__name__}: {e}",
                                 tb=traceback.format_exc()[-500:]))
            print(f"[deep] {u['structure_id']} ERROR {e}", flush=True)
    import json as _j
    mname = ("deep_manifest.json" if a.n_shards == 1
             else f"deep_manifest_shard{a.shard}.json")
    (out / mname).write_text(_j.dumps(manifest, indent=1, default=str))
    print("[deep] done")


if __name__ == "__main__":
    main()
