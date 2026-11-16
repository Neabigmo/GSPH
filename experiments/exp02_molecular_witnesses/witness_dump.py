# -*- coding: utf-8 -*-
"""P1: common-witness geometry dump for real molecular paths.

Re-runs the UNMODIFIED extraction for one structure, selects directed
paths whose common-support tau exceeds the best pairwise-only bound
(tau(canon) - max_{pairs} tau(pair) > 0, |canon| >= 3), recomputes the
certified witness point for each canonical patch set, and writes a JSON
consumed by the local renderer::

  <out>/<sid>.witness.json
    examples[]: path, canon, order, tau, pairwise_max, gap,
                witness{center,tau,certificate_passed,route},
                patches[{patch_id,donor_site,acceptor_site,
                          contacts[{donor,acceptor,distance,tau}]}],
                atoms{idx:{xyz,element,name,resname,chain,resseq}}
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

_HERE = Path(__file__).resolve().parent
_H3 = _HERE.parent
sys.path.insert(0, str(_H3 / "gsph" / "task_adapters"))
sys.path.insert(0, str(_H3.parent))
sys.path.insert(0, str(_HERE))
import deep_subset as DS  # noqa: E402


def find_unit(task: str, root: Path, sid: str):
    if task == "rna":
        import rna_quality_driver as R
        units = [u for u in R.enumerate_units(root)
                 if u["kind"] == "decoy"]
    else:
        import pd_ftdmp_driver as D
        units = list(D.enumerate_units(root))
    for u in units:
        if sid == str(u.get("structure_id", "")):
            return u
    for u in units:
        if sid in str(u.get("structure_id", "")) or \
           sid in Path(str(u.get("path", ""))).name:
            return u
    raise KeyError(f"no unit matching {sid!r}")


def extract(task, unit, cfg):
    """Return (res, atoms) identical to deep_complex_dump."""
    if task == "rna":
        import intramolecular as I
        import sparse as SP
        atoms = I.parse_structure(unit["path"])
        event_filter, _ = SP.rna_event_filter(atoms, fraction=0.25,
                                              per_nt_quota=8)
        res = I.extract_intra(atoms, "rna_quality_sparse",
                              intra_sep=I.DEFAULT_INTRA_SEP,
                              config=cfg, event_filter=event_filter)
    else:
        import pd_ftdmp_driver as D
        atoms = D._parse_atoms(unit["path"])
        from gsph6.engine import extract_gsph6_atoms
        res = extract_gsph6_atoms(list(atoms), scenario="protein_dna",
                                  config=cfg)
    return res, list(atoms)


def _atom_info(a):
    return dict(
        xyz=[float(x) for x in np.asarray(a.xyz, dtype=float)],
        element=str(getattr(a, "element", "")),
        name=str(getattr(a, "name", "")),
        resname=str(getattr(a, "resname", "")),
        chain=str(getattr(a, "chain", getattr(a, "chain_id", ""))),
        resseq=int(getattr(a, "resseq", getattr(a, "res_id", -1)) or -1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rna", "pd"], required=True)
    ap.add_argument("--structure-id", required=True)
    ap.add_argument("--pm-root", default="")
    ap.add_argument("--ftdmp-root", default="")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.pm_root) if a.task == "rna" else Path(a.ftdmp_root)
    unit = find_unit(a.task, root, a.structure_id)
    sid = str(unit["structure_id"])
    from gsph6.engine import GSPH6Config
    cfg = GSPH6Config(interaction_backend="auto", max_path_order=3,
                      compute_eta=True, hodge_dense_max_dimension=384)
    t0 = time.time()
    res, atoms = extract(a.task, unit, cfg)
    patch_map = {int(p.patch_id): p for p in (res.get("patches") or [])}
    atom_map = {int(x.index): x for x in atoms}
    tau_of = {}
    for k, v in (res.get("support_by_set") or {}).items():
        try:
            kk = tuple(sorted(int(x) for x in k))
        except Exception:
            continue
        if v is not None and np.isfinite(float(v)):
            tau_of[kk] = float(v)
    from gsph6.geometry import compute_patch_support

    def canon_tau(canon):
        key = tuple(sorted(set(int(x) for x in canon)))
        if key in tau_of:
            return tau_of[key]
        patches = [patch_map[i] for i in key if i in patch_map]
        if len(patches) != len(key):
            return np.nan
        r = compute_patch_support(patches, atom_map, None)
        val = float(r.tau) if (r.certificate_passed
                               and np.isfinite(r.tau)) else np.nan
        tau_of[key] = val
        return val

    # Rank directed paths by higher-order gap.
    cands = {}
    ps = res.get("path_support") or {}
    for path in ps:
        t = tuple(int(x) for x in path)
        canon = tuple(sorted(set(t)))
        if len(canon) < 3 or canon in cands:
            continue
        full = canon_tau(canon)
        if not np.isfinite(full):
            continue
        pw = max((canon_tau(s) for s in combinations(canon, 2)),
                 default=np.nan)
        if np.isfinite(pw):
            cands[canon] = (full - pw, full, pw, t)
    ranked = sorted(cands.items(), key=lambda kv: -kv[1][0])
    examples = []
    for canon, (gap, full, pw, path) in ranked[:a.n_examples]:
        key = tuple(sorted(set(int(x) for x in canon)))
        patches = [patch_map[i] for i in key if i in patch_map]
        w = compute_patch_support(patches, atom_map, None)
        needed = set()
        for p in patches:
            for c in p.contacts:
                needed.add(int(c.donor)); needed.add(int(c.acceptor))
            needed.update(int(i) for i in p.atom_ids)
        atoms_out = {str(i): _atom_info(atom_map[i])
                     for i in sorted(needed) if i in atom_map}
        examples.append(dict(
            path=",".join(map(str, path)),
            canon=",".join(map(str, key)),
            order=len(path) - 1,
            tau=full, pairwise_max=pw, gap=float(gap),
            witness=dict(center=[float(x) for x in w.center],
                         tau=float(w.tau),
                         certificate_passed=bool(w.certificate_passed),
                         route=str(w.route)),
            patches=[dict(
                patch_id=int(p.patch_id),
                donor_site=[str(x) for x in p.donor_site],
                acceptor_site=[str(x) for x in p.acceptor_site],
                contacts=[dict(donor=int(c.donor), acceptor=int(c.acceptor),
                               distance=float(c.distance), tau=float(c.tau))
                          for c in p.contacts],
            ) for p in patches],
            atoms=atoms_out,
        ))
    doc = dict(structure_id=sid, task=a.task, elapsed_s=round(time.time() - t0, 1),
               n_dir_paths=len(ps), n_candidates=len(cands),
               examples=examples)
    fp = out / f"{sid}.witness.json"
    fp.write_text(json.dumps(doc, indent=1, default=str))
    print(f"[witness] {sid} ok {doc['elapsed_s']}s "
          f"paths={len(ps)} cands={len(cands)} -> {fp}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
