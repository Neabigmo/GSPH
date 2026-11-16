# -*- coding: utf-8 -*-
"""EXP-07 patch-level dump: capture InteractionPatch objects + atoms for
one structure, for alternative-representation support ablation.

Runs the SAME front pipeline as extract_gsph6_atoms (chain selection ->
pair batch -> events -> patches) but stops before graph/path enumeration
and serializes patches + atoms to parquet/pickle.

Usage:
  python exp07_patch_dump.py --pdb model.pdb --scenario protein_dna \
      --out patchdump.pkl [--clash 0.6] [--intra-sep 4]
"""
import argparse, os, pickle, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep")
sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep/GSPH_H3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _engine():
    try:
        from cspph.core import engine as E
        from cspph.core.gsph1 import GSPH1Config, _parse_atoms
        return E, GSPH1Config, _parse_atoms, "cspph"
    except Exception:
        from gsph6 import engine as E
        from runners.gsph1 import GSPH1Config, _parse_atoms
        return E, GSPH1Config, _parse_atoms, "gsph6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--scenario", default="protein_dna")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clash", type=float, default=0.6)
    ap.add_argument("--intra-sep", type=int, default=4)
    ap.add_argument("--sparse", type=float, default=0.0)
    ap.add_argument("--quota", type=int, default=0)
    a = ap.parse_args()

    E, GSPH1Config, _parse_atoms, flavor = _engine()
    atoms = list(_parse_atoms(a.pdb))
    cfg6 = E.GSPH6Config(
        interaction_backend="auto", max_path_order=3, compute_eta=True,
        intra_min_bond_sep=a.intra_sep,
        contact_min_vdw_ratio=a.clash)
    # replicate engine front pipeline
    sel_a, sel_b = E._chain_pair(atoms, a.scenario, None, None)
    left, right = E._side_atoms(atoms, sel_a), E._side_atoms(atoms, sel_b)
    # constants are re-exported through the engine namespace in both layouts
    pair_batch = E.build_interaction_pair_batch(
        atoms, left, right,
        max(float(cfg6.local_radius), float(cfg6.event_cutoff) + 2.0),
        E.ELEMENTS, E.RESIDUE_CLASS_INDEX, E.RDF_EDGES, E.LJ_SIGMA,
        E.LJ_EPSILON,
        backend=cfg6.interaction_backend,
        gpu_min_pairs=cfg6.gpu_min_pairs,
        gpu_batch_pairs=cfg6.gpu_batch_pairs,
        gpu_device=cfg6.gpu_device)
    legacy_cfg = GSPH1Config(
        event_cutoff=cfg6.event_cutoff, transfer_cutoff=cfg6.transfer_cutoff,
        local_radius=cfg6.local_radius, scales=cfg6.scales,
        max_events=cfg6.max_events,
        max_paths_per_order=cfg6.max_paths_per_order,
        max_local_pairs_per_event=cfg6.max_local_pairs_per_event,
        max_hodge_dimension=-1,
        max_bond_transfer_distance=cfg6.max_bond_transfer_distance,
        mode=cfg6.mode, interaction_backend=cfg6.interaction_backend,
        gpu_min_pairs=cfg6.gpu_min_pairs, gpu_batch_pairs=cfg6.gpu_batch_pairs,
        gpu_device=cfg6.gpu_device, interaction_batch_size=16,
        support_workers=1, support_gpu_min_paths=64,
        support_gpu_batch_paths=int(os.environ.get("GSPH_SUPPORT_CPP_BATCH", "4096")),
        path_backend="python_stream", surface_backend="none",
        hodge_backend="cpu", intra_min_bond_sep=a.intra_sep,
        contact_min_vdw_ratio=a.clash)
    bond_graph = None
    if a.intra_sep > 0:
        bond_graph = E._build_bond_graph(atoms)
    event_stats = {}
    events, _trunc = E._make_events(
        atoms, sel_a, sel_b, legacy_cfg, event_stats, pair_batch, None,
        bond_graph)
    # RNA sparse event filter parity (production uses sparse_fraction .25,
    # per_nt_quota 8); applied here only when requested.
    if a.sparse > 0:
        if flavor == "cspph":
            from tasks.rna.sparse import rna_event_filter
        else:
            from gsph.task_adapters.sparse import rna_event_filter
        flt, _fm = rna_event_filter(atoms, fraction=a.sparse,
                                    per_nt_quota=a.quota or None)
        events, fstats = flt(events, atoms)
        event_stats["sparse_filter"] = fstats
    patches = E.build_interaction_patches(events, atoms, None)
    # serialize: atoms + patches (contacts carry atom indices)
    atom_rows = []
    for i, at in enumerate(atoms):
        xyz = getattr(at, "xyz", None)
        if xyz is None:
            xyz = np.array([at.x, at.y, at.z])
        atom_rows.append(dict(idx=i, x=float(xyz[0]), y=float(xyz[1]),
                              z=float(xyz[2]),
                              element=str(getattr(at, "element", "X")),
                              name=str(getattr(at, "name", "")),
                              resname=str(getattr(at, "resname", "")),
                              chain=str(getattr(at, "chain", "")),
                              resid=int(getattr(at, "resid", -1))))
    import json as _json
    patch_rows = []
    for p in patches:
        patch_rows.append(dict(
            patch_id=int(p.patch_id),
            contacts=_json.dumps(
                [(int(c.donor), int(c.acceptor),
                  float(c.distance), str(c.interaction_type))
                 for c in p.contacts]),
            donor_members=_json.dumps([int(x) for x in p.donor_members]),
            acceptor_members=_json.dumps([int(x) for x in p.acceptor_members])))
    import pandas as pd
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(atom_rows).to_parquet(out.with_suffix(".atoms.parquet"))
    pd.DataFrame(patch_rows).to_parquet(out.with_suffix(".patches.parquet"))
    meta = dict(scenario=a.scenario, flavor=flavor,
                n_atoms=len(atoms), n_events=len(events),
                n_patches=len(patches),
                n_contacts=sum(len(p.contacts) for p in patches),
                event_stats=event_stats)
    Path(str(out) + ".meta.json").write_text(
        __import__("json").dumps(meta, indent=1, default=str))
    print("[patchdump]", meta)


if __name__ == "__main__":
    main()
