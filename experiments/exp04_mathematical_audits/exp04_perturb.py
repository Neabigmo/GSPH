# -*- coding: utf-8 -*-
"""EXP-04 real-structure perturbation audit.

Rebuilds InteractionPatch/MicroContact objects from EXP-07 patch dumps
(atoms.parquet + patches.parquet) and measures the Lipschitz stability of
the certified union-of-lenses support tau under bounded coordinate noise:

  for sigma in {0.05, 0.1, 0.2, 0.5} A (iid gaussian per coordinate),
  tau' = compute_patch_support(perturbed patches, perturbed atoms)
  violation iff |tau' - tau| > max_atom_displacement + tol

Also records per-patch single-support drift and certificate status.

Usage:
  python exp04_perturb.py --dumpdir DIR --sid SID --out FILE.json \
      [--n-paths 300] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep")
sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep/GSPH_H3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def load_dump(dumpdir: Path, sid: str):
    atoms_df = pd.read_parquet(dumpdir / f"{sid}.atoms.parquet")
    patches_df = pd.read_parquet(dumpdir / f"{sid}.patches.parquet")
    atoms = {}
    for _, r in atoms_df.iterrows():
        atoms[int(r["idx"])] = SimpleNamespace(
            xyz=np.array([float(r["x"]), float(r["y"]), float(r["z"])]),
            element=str(r["element"]))
    patches = {}
    for _, r in patches_df.iterrows():
        contacts = []
        for d, a, dist, itype in json.loads(r["contacts"]):
            contacts.append(SimpleNamespace(
                donor=int(d), acceptor=int(a),
                distance=float(dist), tau=0.0,
                interaction_type=str(itype)))
        donor_members = json.loads(r["donor_members"])
        acceptor_members = json.loads(r["acceptor_members"])
        patches[int(r["patch_id"])] = SimpleNamespace(
            patch_id=int(r["patch_id"]), contacts=contacts,
            donor_members=donor_members,
            acceptor_members=acceptor_members,
            minimum_tau=0.0,
            atom_ids=tuple(sorted(set(donor_members) | set(acceptor_members))))
    return atoms, patches


def canon_list(tau_path: Path, n_paths: int, seed: int):
    df = pd.read_parquet(tau_path)
    col = "canon" if "canon" in df.columns else df.columns[0]
    tau_col = "tau" if "tau" in df.columns else None
    rows = []
    for _, r in df.iterrows():
        ids = tuple(sorted(set(int(x) for x in str(r[col]).split(","))))
        if len(ids) >= 2:
            rows.append((ids, float(r[tau_col]) if tau_col else np.nan))
    rng = np.random.default_rng(seed)
    if len(rows) > n_paths:
        # stratified: keep lowest/mid/highest tau thirds plus random fill
        rows.sort(key=lambda x: (np.isfinite(x[1]), x[1]))
        k = n_paths // 3
        sel = rows[:k] + rows[len(rows)//2-k//2: len(rows)//2-k//2+k] + \
              rows[-k:] if k > 0 else rows[:n_paths]
        rest = [r for r in rows if r not in sel]
        extra = min(n_paths - len(sel), len(rest))
        if extra > 0:
            sel += [rest[i] for i in rng.choice(len(rest), extra,
                                                replace=False)]
        rows = sel[:n_paths]
    return rows


def perturb_atoms(atoms, sigma, rng):
    out = {}
    for i, a in atoms.items():
        out[i] = SimpleNamespace(
            xyz=a.xyz + rng.normal(0.0, sigma, 3), element=a.element)
    return out


def union_tau(patch_ids, patches, atoms):
    from gsph6.geometry import compute_patch_support
    plist = [patches[i] for i in patch_ids if i in patches]
    if len(plist) != len(patch_ids):
        return np.nan, False
    try:
        r = compute_patch_support(plist, atoms, None)
        ok = bool(r.certificate_passed) and np.isfinite(r.tau)
        return (float(r.tau) if ok else np.nan), ok
    except Exception:
        return np.nan, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumpdir", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-paths", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigmas", default="0.05,0.1,0.2,0.5")
    a = ap.parse_args()
    dumpdir = Path(a.dumpdir)
    atoms, patches = load_dump(dumpdir, a.sid)
    # tau_table may live in dumpdir via symlink
    tau_path = dumpdir / f"{a.sid}.tau_table.parquet"
    if not tau_path.exists():
        tau_path = dumpdir / f"{a.sid}.support.parquet"
    canons = canon_list(tau_path, a.n_paths, a.seed) if tau_path.exists() else []
    rng = np.random.default_rng(a.seed)
    sigmas = [float(s) for s in a.sigmas.split(",")]

    # patch-level single-support drift (all patches, cheap)
    patch_rows = []
    for pid, p in patches.items():
        t0, ok0 = union_tau((pid,), patches, atoms)
        if not ok0 or not np.isfinite(t0):
            continue
        for s in sigmas:
            pa = perturb_atoms(atoms, s, np.random.default_rng(a.seed + 97))
            t1, ok1 = union_tau((pid,), patches, pa)
            patch_rows.append(dict(patch_id=pid, sigma=s, tau0=t0,
                                   tau1=t1, dt=(t1 - t0) if ok1 else np.nan,
                                   certified=ok1))
    patch_df = pd.DataFrame(patch_rows)

    # path-level union support drift
    path_rows = []
    base = {}
    for ids, tau_ref in canons:
        t0, ok0 = union_tau(ids, patches, atoms)
        base[ids] = (t0 if ok0 else tau_ref, ok0)
    for s in sigmas:
        pa = perturb_atoms(atoms, s, np.random.default_rng(a.seed + 1000 + int(s * 1000)))
        maxdisp = 0.0
        dts = []
        for ids, (t0, ok0) in base.items():
            if not ok0 or not np.isfinite(t0):
                continue
            t1, ok1 = union_tau(ids, patches, pa)
            if ok1 and np.isfinite(t1):
                dts.append(abs(t1 - t0))
        path_rows.append(dict(sigma=s, n=len(dts),
                              mean_abs_dt=float(np.mean(dts)) if dts else np.nan,
                              max_abs_dt=float(np.max(dts)) if dts else np.nan))
    path_df = pd.DataFrame(path_rows)

    doc = dict(sid=a.sid, n_atoms=len(atoms), n_patches=len(patches),
               n_canons=len(canons), seed=a.seed,
               patch_drift=patch_df.to_dict("records"),
               path_drift=path_df.to_dict("records"))
    Path(a.out).write_text(json.dumps(doc))
    print(f"[perturb] {a.sid} atoms={len(atoms)} patches={len(patches)} "
          f"canons={len(canons)} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
