# -*- coding: utf-8 -*-
"""EXP-02 stratified common-witness selection.

Extends witness_dump.py: instead of only the top-n paths by higher-order
gap, selects certified witness examples across explicit strata so the
molecular witness gallery covers the candidate distribution, not just its
tail:

  gap_q1..q4 : median-gap candidate inside each gap quartile
  tau_med    : candidate whose common-support tau is closest to the median
  eta3_max   : directed path with the largest eta3 higher-order increment
  chem_hbond : candidate with the largest hydrogen-bond contact fraction
  chem_div   : candidate with the most diverse interaction-type set

Each example carries `selection_rule` for downstream audit. Runs the same
unmodified extraction as witness_dump (deep_subset.extract parity).

Usage:
  python witness_dump_stratified.py --task rna --structure-id SID \
      --pm-root ... --out DIR [--max-examples 8]
"""
import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
import witness_dump as WD  # noqa: E402


def _contact_types(patches):
    types = []
    for p in patches:
        for c in p.contacts:
            t = str(getattr(c, "interaction_type", "") or "")
            if t:
                types.append(t.lower())
    return types


def _build_example(canon, full, pw, path, patch_map, atom_map):
    from gsph6.geometry import compute_patch_support
    key = tuple(sorted(set(int(x) for x in canon)))
    patches = [patch_map[i] for i in key if i in patch_map]
    w = compute_patch_support(patches, atom_map, None)
    needed = set()
    for p in patches:
        for c in p.contacts:
            needed.add(int(c.donor)); needed.add(int(c.acceptor))
        needed.update(int(i) for i in getattr(p, "atom_ids", []) or [])
    atoms_out = {str(i): WD._atom_info(atom_map[i])
                 for i in sorted(needed) if i in atom_map}
    types = _contact_types(patches)
    return dict(
        path=",".join(map(str, path)),
        canon=",".join(map(str, key)),
        order=len(path) - 1,
        tau=float(full), pairwise_max=float(pw),
        gap=float(full - pw),
        witness=dict(center=[float(x) for x in w.center],
                     tau=float(w.tau),
                     certificate_passed=bool(w.certificate_passed),
                     route=str(w.route)),
        contact_types=types,
        patches=[dict(
            patch_id=int(p.patch_id),
            donor_site=[str(x) for x in p.donor_site],
            acceptor_site=[str(x) for x in p.acceptor_site],
            contacts=[dict(donor=int(c.donor), acceptor=int(c.acceptor),
                           distance=float(c.distance), tau=float(c.tau),
                           interaction_type=str(getattr(c, "interaction_type", "")))
                      for c in p.contacts],
        ) for p in patches],
        atoms=atoms_out,
    )


def stratified_select(cands, path_eta, max_examples=8):
    """cands: {canon: (gap, full, pw, path)}. Returns list of
    (rule, canon, payload)."""
    picks = []
    seen = set()

    def add(rule, canon):
        if canon in seen or canon not in cands:
            return False
        seen.add(canon)
        picks.append((rule, canon, cands[canon]))
        return True

    items = sorted(cands.items(), key=lambda kv: kv[1][0])
    n = len(items)
    if n == 0:
        return picks, seen, items, []
    # gap quartiles: median-gap candidate of each quartile
    for q in range(4):
        lo, hi = n * q // 4, max(n * (q + 1) // 4, n * q // 4 + 1)
        if hi <= lo:
            continue
        seg = items[lo:hi]
        mid = seg[len(seg) // 2]
        add(f"gap_q{q + 1}", mid[0])
    # tau median
    by_tau = sorted(cands.items(), key=lambda kv: kv[1][1])
    med = by_tau[len(by_tau) // 2][0]
    add("tau_med", med)
    # eta3 max (directed path lookup, tolerant of key form)
    eta3 = path_eta.get("3", {}) if isinstance(path_eta, dict) else {}
    best_eta, best_canon = -np.inf, None
    for canon, (_g, _f, _p, path) in cands.items():
        for key in (tuple(path), tuple(sorted(set(path)))):
            v = eta3.get(key)
            if v is None:
                v = eta3.get(str(key))
            if v is not None and float(v) > best_eta:
                best_eta, best_canon = float(v), canon
    if best_canon is not None:
        add("eta3_max", best_canon)
    # chemical strata (hbond fraction / type diversity) are resolved in
    # main() where patch contacts are available.
    return picks, seen, items, by_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rna", "pd"], required=True)
    ap.add_argument("--structure-id", required=True)
    ap.add_argument("--pm-root", default="")
    ap.add_argument("--ftdmp-root", default="")
    ap.add_argument("--max-examples", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.pm_root) if a.task == "rna" else Path(a.ftdmp_root)
    unit = WD.find_unit(a.task, root, a.structure_id)
    sid = str(unit["structure_id"])
    from gsph6.engine import GSPH6Config
    cfg = GSPH6Config(interaction_backend="auto", max_path_order=3,
                      compute_eta=True, hodge_dense_max_dimension=384)
    t0 = time.time()
    res, atoms = WD.extract(a.task, unit, cfg)
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
        pw = max((canon_tau(s) for s in itertools.combinations(canon, 2)),
                 default=np.nan)
        if np.isfinite(pw):
            cands[canon] = (full - pw, full, pw, t)
    path_eta = res.get("path_eta") or {}

    # strata selection (inline; keeps patch access for chemical rules)
    picks, seen, items, by_tau = stratified_select(cands, path_eta)
    # chemical strata need contact types -> resolve now
    def chem_stats(canon):
        patches = [patch_map[i] for i in canon if i in patch_map]
        types = _contact_types(patches)
        if not types:
            return 0.0, 0
        hb = sum(1 for t in types if "h" in t and ("bond" in t or t == "hb"))
        return hb / len(types), len(set(types))

    rest = [c for c in cands if c not in seen]
    if rest:
        best_hb = max(rest, key=lambda c: chem_stats(c)[0])
        if chem_stats(best_hb)[0] > 0:
            picks.append(("chem_hbond", best_hb, cands[best_hb])); seen.add(best_hb)
    rest = [c for c in cands if c not in seen]
    if rest:
        best_dv = max(rest, key=lambda c: chem_stats(c)[1])
        if chem_stats(best_dv)[1] > 1:
            picks.append(("chem_div", best_dv, cands[best_dv])); seen.add(best_dv)

    picks = picks[:a.max_examples]
    examples = []
    for rule, canon, (gap, full, pw, path) in picks:
        ex = _build_example(canon, full, pw, path, patch_map, atom_map)
        ex["selection_rule"] = rule
        ex["gap"] = float(gap)
        examples.append(ex)
    doc = dict(structure_id=sid, task=a.task,
               elapsed_s=round(time.time() - t0, 1),
               n_dir_paths=len(ps), n_candidates=len(cands),
               selection="stratified",
               examples=examples)
    fp = out / f"{sid}.witness_strat.json"
    fp.write_text(json.dumps(doc, indent=1, default=str))
    print(f"[witness-strat] {sid} ok {doc['elapsed_s']}s "
          f"cands={len(cands)} rules={[p[0] for p in picks]} -> {fp}",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
