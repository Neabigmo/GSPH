# -*- coding: utf-8 -*-
"""EXP-02 post-processing: per-contact witness residuals.

For every certified witness example, evaluates the union-lens objective
decomposed per microcontact:

  residual(c) = min_{a in {donor,acceptor}} (|x - a| - r_a)
  patch_margin(p) = min_{c in p} residual(c)
  objective     = max_p patch_margin(p)   (== tau at optimum)

Outputs results/tables/witness_contact_residuals.csv and a per-example
summary JSON. Selection stratification (eta-quartile / median / relay)
requires re-running witness_dump with --select stratified on HPC; the
schema below already carries `selection_rule` for that extension.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cspph.geometry.patch_support import VDW_RADII  # noqa: E402

WITNESS_DIRS = [
    Path(r"H:\2026try\11.3GSPH\GSPH_H3\results\witness"),
    Path(r"H:\2026try\11.3GSPH\cspph-molecular\results\figures"),
]
OUT_CSV = Path(__file__).resolve().parents[2] / "results" / "tables" / \
    "witness_contact_residuals.csv"
OUT_JSON = OUT_CSV.with_suffix(".summary.json")


def atom_residual(a, x):
    r = VDW_RADII.get(str(a.get("element", "X")), VDW_RADII["X"])
    return float(np.linalg.norm(x - np.asarray(a["xyz"])) - r)


def contact_residual(contact, atoms, x):
    """Contact realization = BOTH atoms are constraints (intersection
    lens). Returns max(donor_resid, acceptor_resid)."""
    vals = []
    for aid in (contact["donor"], contact["acceptor"]):
        a = atoms.get(str(aid)) or atoms.get(aid)
        if a is None:
            continue
        vals.append(atom_residual(a, x))
    return max(vals) if vals else np.nan


def patch_distance(p, atoms, x):
    """min over contacts of the contact-level support at x."""
    vals = [contact_residual(c, atoms, x) for c in p.get("contacts", [])]
    vals = [v for v in vals if np.isfinite(v)]
    return min(vals) if vals else np.nan


def main():
    rows = []
    summaries = []
    seen = set()
    for d in WITNESS_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.witness.json")):
            if f.name in seen:
                continue
            seen.add(f.name)
            data = json.load(open(f))
            sid = data.get("structure_id", f.stem)
            for ex in data.get("examples", []):
                x = np.asarray(ex["witness"]["center"], float)
                atoms = ex.get("atoms", {})
                tau = float(ex["witness"]["tau"])
                patch_rows = []
                for p in ex.get("patches", []):
                    pdist = patch_distance(p, atoms, x)
                    patch_rows.append(dict(
                        structure_id=sid, path=ex.get("path"),
                        patch_id=p.get("patch_id"), n_contacts=len(p.get("contacts", [])),
                        patch_margin=pdist))
                # per-contact rows
                for p in ex.get("patches", []):
                    for c in p.get("contacts", []):
                        rows.append(dict(
                            structure_id=sid, path=ex.get("path"),
                            order=ex.get("order"), tau=tau,
                            gap=ex.get("gap"),
                            selection_rule="top_gap_v33",
                            patch_id=p.get("patch_id"),
                            donor=c.get("donor"), acceptor=c.get("acceptor"),
                            distance=c.get("distance"),
                            residual=contact_residual(c, atoms, x)))
                margins = [r["patch_margin"] for r in patch_rows
                           if np.isfinite(r["patch_margin"])]
                summaries.append(dict(
                    structure_id=sid, path=ex.get("path"),
                    order=ex.get("order"), tau=tau, gap=ex.get("gap"),
                    n_patches=len(patch_rows),
                    objective_recomputed=float(max(margins)) if margins else np.nan,
                    tau_match=bool(margins) and
                    abs(max(margins) - tau) < 5e-3,
                    n_active_patches=int(sum(
                        1 for m in margins if m <= tau + 1e-6))))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    sdf = pd.DataFrame(summaries)
    OUT_JSON.write_text(json.dumps({
        "n_examples": len(summaries),
        "n_contact_rows": len(rows),
        "tau_match_frac": float(sdf["tau_match"].mean()) if len(sdf) else None,
        "mean_active_patches": float(sdf["n_active_patches"].mean())
        if len(sdf) else None,
        "examples": summaries}, indent=1, default=str))
    print(sdf[["structure_id", "tau", "gap", "objective_recomputed",
               "tau_match", "n_active_patches"]].to_string())
    print("->", OUT_CSV)


if __name__ == "__main__":
    main()
