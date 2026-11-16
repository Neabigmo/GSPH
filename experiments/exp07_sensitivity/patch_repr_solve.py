# -*- coding: utf-8 -*-
"""EXP-07 patch-representation ablation: alternative support definitions.

For every canonical patch set in a structure's tau_table, computes four
path-birth values:

  union      - production tau (min_x max_p min_c support) [from tau_table]
  centroid   - 1-center of patch centroids: min_x max_p ||x - c_p||
               where c_p = mean of microcontact midpoints (no radii)
  enclosing  - max_p MEB_radius(patch atoms)  (Badoui-Clarkson approx,
               deterministic 200 iterations; per-patch, no coupling)
  single     - max_p min_c contact_tau(c)     (patch-intrinsic scalar,
               no spatial coupling)

Then maps canon -> path births via complex_paths.parquet and reports
per-structure birth deltas vs union. Barcode-level comparison reuses
exp03_fine.py with a --birth-column override (see README).

Usage: python patch_repr_solve.py --dumpdir <dir-with-.atoms/.patches/
        .tau_table/.complex_paths> --out repr.parquet
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
try:
    from cspph.geometry.patch_support import VDW_RADII
except Exception:
    sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep")
    from runners.gsph1 import VDW_RADII  # noqa: E402


def meb_radius(points, iters=200):
    """Deterministic Badoiu-Clarkson (1+eps)-MEB radius approximation."""
    P = np.asarray(points, float)
    if len(P) == 0:
        return 0.0
    c = P[0].copy()
    for t in range(1, iters + 1):
        d = np.linalg.norm(P - c, axis=1)
        i = int(np.argmax(d))
        c = c + (P[i] - c) / (t + 1)
    return float(np.max(np.linalg.norm(P - c, axis=1)))


def load_dump(dumpdir, sid):
    atoms = pd.read_parquet(Path(dumpdir) / f"{sid}.atoms.parquet")
    patches = pd.read_parquet(Path(dumpdir) / f"{sid}.patches.parquet")
    tau = pd.read_parquet(Path(dumpdir) / f"{sid}.tau_table.parquet")
    paths = pd.read_parquet(Path(dumpdir) / f"{sid}.complex_paths.parquet")
    return atoms, patches, tau, paths


def patch_geom(patch_row, atoms):
    """Return (centroid, meb_points, contact_taus) for one patch."""
    contacts = json.loads(patch_row["contacts"])
    donor_members = set(json.loads(patch_row["donor_members"]))
    acceptor_members = set(json.loads(patch_row["acceptor_members"]))
    all_ids = sorted(donor_members | acceptor_members)
    pts = atoms.set_index("idx").loc[all_ids, ["x", "y", "z"]].to_numpy(float) \
        if all_ids else np.zeros((0, 3))
    mids = []
    ct = []
    amap = atoms.set_index("idx")
    for d, a, dist, itype in contacts:
        try:
            da = amap.loc[d]; aa = amap.loc[a]
        except KeyError:
            continue
        mids.append([(da.x + aa.x) / 2, (da.y + aa.y) / 2, (da.z + aa.z) / 2])
        rd = VDW_RADII.get(str(da.element), VDW_RADII["X"])
        ra = VDW_RADII.get(str(aa.element), VDW_RADII["X"])
        dd = float(np.linalg.norm(
            np.array([da.x, da.y, da.z]) - np.array([aa.x, aa.y, aa.z])))
        ct.append(max(0.0, 0.5 * (dd - rd - ra)))
    centroid = np.mean(mids, axis=0) if mids else np.zeros(3)
    return centroid, pts, (min(ct) if ct else np.nan)


def one_center_radius(centroids):
    """min_x max_p ||x - c_p|| via Weiszfeld-like iteration (small k)."""
    C = np.asarray(centroids, float)
    if len(C) == 0:
        return 0.0
    x = C.mean(0)
    for _ in range(100):
        d = np.linalg.norm(C - x, axis=1)
        w = 1.0 / np.maximum(d, 1e-9)
        x_new = (C * w[:, None]).sum(0) / w.sum()
        if np.linalg.norm(x_new - x) < 1e-9:
            break
        x = x_new
    return float(np.max(np.linalg.norm(C - x, axis=1)))


def alt_births(canon_ids, geom):
    cents = [geom[i][0] for i in canon_ids]
    mebs = [geom[i][1] for i in canon_ids]
    singles = [geom[i][2] for i in canon_ids]
    tau_cent = one_center_radius(cents)
    tau_meb = max((meb_radius(p) for p in mebs if len(p)), default=0.0)
    finite_s = [s for s in singles if np.isfinite(s)]
    tau_single = max(finite_s) if finite_s else np.nan
    return tau_cent, tau_meb, tau_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumpdir", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dumpdir = Path(a.dumpdir)
    atoms, patches, tau, paths = load_dump(dumpdir, a.sid)
    geom = {}
    for _, pr in patches.iterrows():
        geom[int(pr["patch_id"])] = patch_geom(pr, atoms)
    # canon -> union tau
    union = {}
    for _, r in tau.iterrows():
        ids = tuple(int(x) for x in str(r["canon"]).split(","))
        union[ids] = float(r["tau"])
    _cache = {}

    def alt_cached(ids):
        if ids not in _cache:
            _cache[ids] = alt_births(ids, geom)
        return _cache[ids]

    rows = []
    for _, r in paths.iterrows():
        ids = tuple(int(x) for x in str(r["canon"]).split(","))
        if ids not in union:
            continue
        tc, tm, ts = alt_cached(ids)
        rows.append(dict(
            structure_id=a.sid, graph=r["graph"], order=int(r["order"]),
            path=r["path"], canon=r["canon"],
            tau_union=union[ids], tau_centroid=tc, tau_enclosing=tm,
            tau_single=ts))
    df = pd.DataFrame(rows)
    for col in ("tau_centroid", "tau_enclosing", "tau_single"):
        df[f"delta_{col[4:]}"] = df[col] - df["tau_union"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    summ = df[["tau_union", "tau_centroid", "tau_enclosing", "tau_single"]]\
        .corr().round(4)
    print(summ.to_string())
    print("frac |delta|>0.1A:",
          {c: float((df[f"delta_{c[4:]}"] .abs() > 0.1).mean())
           for c in ("tau_centroid", "tau_enclosing", "tau_single")})
    print("->", a.out, len(df), "rows")


if __name__ == "__main__":
    main()
