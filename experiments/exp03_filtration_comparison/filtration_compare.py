# -*- coding: utf-8 -*-
"""P0-1: same-graph / same-path-complex filtration comparison.

Per dumped structure and per graph (dir = chemical directed edges,
und = symmetrized), rebuilds GLMY Omega chain complexes per filtration
scale under alternative birth rules and runs the SAME persistence
machinery as the production engine:

  tau        : GSPH  birth(pi) = tau(canon(pi))          [production rule]
  pairwise   : birth(pi) = max tau over all 2-subsets of canon(pi)
               (|canon| <= 2 -> tau(canon))
  facetmax   : recursive max-face birth
               birth_f(pi) = max( tau(canon), max_facets birth_f(facet) )
               with singleton birth = tau(singleton patch)
  shuffled   : tau values permuted across same-(graph,order) rows

Persistence = runners.glmy_sparse.sparse_chain +
runners.persistence.{build_omega_boundary_filtration,
build_omega_inclusion_chain_maps, persistent_intervals_from_filtration,
validate_chain_maps}  -- identical to gsph6.engine usage, so chain-map
commutation (P0-4) is audited on every complex as well.

Outputs:
  results/ablation/filtration_compare.json          (full intervals)
  results/ablation/filtration_compare_summary.csv   (compact table)
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
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

RES = _HERE.parent / "results"
OUT = RES / "ablation"
OUT.mkdir(parents=True, exist_ok=True)
SCALES = [0.0, 0.5, 2.0]
ORDERS = (0, 1, 2)


def canon_key(c):
    return ",".join(map(str, sorted(c)))


def load(d: Path, sid: str):
    paths = pd.read_parquet(d / f"{sid}.complex_paths.parquet")
    tau_df = pd.read_parquet(d / f"{sid}.tau_table.parquet")
    man = json.loads((d / f"{sid}.complex_manifest.json").read_text())
    tau_of = {r["canon"]: float(r["tau"]) for _, r in tau_df.iterrows()
              if np.isfinite(r["tau"])}
    return paths, tau_of, man


def pair_birth(canon, tau_of):
    canon = tuple(sorted(canon))
    if len(canon) <= 2:
        return tau_of.get(canon_key(canon), np.nan)
    vals = [tau_of[canon_key(sub)] for sub in combinations(canon, 2)
            if canon_key(sub) in tau_of]
    return max(vals) if vals else np.nan


def facet_births(canons, tau_of):
    memo = {}

    def fb(canon):
        canon = tuple(sorted(canon))
        if canon in memo:
            return memo[canon]
        key = canon_key(canon)
        tv = tau_of.get(key, np.nan)
        if len(canon) <= 1:
            v = tv
        else:
            fv = [fb(canon[:i] + canon[i + 1:]) for i in range(len(canon))]
            fv = [x for x in fv if np.isfinite(x)]
            v = max([x for x in fv if np.isfinite(x)] + ([tv] if np.isfinite(tv) else [])) \
                if (fv or np.isfinite(tv)) else np.nan
        memo[canon] = v
        return v

    return {canon: fb(canon) for canon in canons}


def active_sets(rows, birth_of, scale):
    active = {o: [] for o in range(4)}
    for r in rows:
        b = birth_of(r)
        if np.isfinite(b) and b <= scale:
            active[int(r["order"])].append(
                tuple(int(x) for x in r["path"].split(",")))
    return active


def run_persistence(paths_by_time):
    from runners.glmy_sparse import sparse_chain
    from runners.persistence import (build_omega_boundary_filtration,
                                     build_omega_inclusion_chain_maps,
                                     persistent_intervals_from_filtration,
                                     validate_chain_maps)
    chains = [sparse_chain(a) for a in paths_by_time]
    filtration = build_omega_boundary_filtration(chains)
    maps = build_omega_inclusion_chain_maps(chains, paths_by_time)
    check = validate_chain_maps(filtration, maps)
    res = {"chain_map_valid": bool(check.get("valid")),
           "chain_map_max_residual": float(check.get("max_residual", np.nan)),
           "kernel_backend": [getattr(c, "kernel_backend", "?") for c in chains],
           "betti_final": {str(k): int(v) for k, v in chains[-1].betti.items()},
           "boundary_sq_residual": {str(k): float(v) for k, v in
                                    chains[-1].boundary_square_residual.items()}}
    if check.get("valid"):
        for order in ORDERS:
            iv = persistent_intervals_from_filtration(filtration, maps, order)
            res[f"intervals_k{order}"] = [
                [int(i["birth"]), int(i["death"]),
                 bool(i.get("right_censored", False))]
                for i in iv]
    return res


def summarize(pres):
    out = {}
    for order in ORDERS:
        iv = pres.get(f"intervals_k{order}", [])
        out[f"n_k{order}"] = len(iv)
        out[f"tp_k{order}"] = float(sum(d - b for b, d, _ in iv))
    out["total_intervals"] = sum(out[f"n_k{o}"] for o in ORDERS)
    return out


def compare_structure(d: Path, sid: str, rng):
    paths, tau_of, man = load(d, sid)
    result = {"structure_id": sid, "source_dir": d.name,
              "scales": man.get("scales"), "n_dir_paths": man.get("n_dir_paths"),
              "n_und_paths": man.get("n_und_paths")}
    # edge weights per graph for the ordinary edge-weight filtration control
    ew = {}
    for graph in ("dir", "und"):
        ef = d / f"{sid}.complex_edges.parquet"
        if ef.exists():
            edf = pd.read_parquet(ef)
            edf = edf[edf["graph"] == graph]
            ew[graph] = {(int(r["u"]), int(r["v"])): float(r["w"])
                         for _, r in edf.iterrows()}
        else:
            ew[graph] = {}
    for graph in ("dir", "und"):
        gdf = paths[paths["graph"] == graph]
        if len(gdf) == 0:
            continue
        rows = list(gdf.to_dict("records"))
        wmap = ew.get(graph, {})
        canons = {tuple(int(x) for x in r["canon"].split(",")) for r in rows}
        facet_map = facet_births(canons, tau_of)

        def births_tau(r):
            return float(r["tau"]) if r["valid"] and np.isfinite(r["tau"]) else np.nan

        def births_pw(r):
            if not r["valid"]:
                return np.nan
            return pair_birth(tuple(int(x) for x in r["canon"].split(",")), tau_of)

        def births_facet(r):
            if not r["valid"]:
                return np.nan
            return facet_map[tuple(int(x) for x in r["canon"].split(","))]

        def births_ew(r):
            t = tuple(int(x) for x in r["path"].split(","))
            if len(t) <= 1:
                return 0.0
            ws = [wmap.get((t[i], t[i + 1]), np.nan) for i in range(len(t) - 1)]
            return max(ws) if all(np.isfinite(ws)) else np.nan

        # shuffled: permute tau within (graph, order)
        sh_map = {}
        for order in range(4):
            sub = [i for i, r in enumerate(rows) if int(r["order"]) == order]
            taus = np.array([rows[i]["tau"] if rows[i]["valid"] else np.nan
                             for i in sub], float)
            perm = rng.permutation(len(sub))
            for i, j in zip(sub, perm):
                sh_map[i] = float(taus[j]) if np.isfinite(taus[j]) else np.nan

        def births_shuf(r):
            return sh_map.get(id(r), np.nan)

        # rebuild row-indexed shuffled births properly
        sh_by_key = {}
        for order in range(4):
            idxs = [i for i, r in enumerate(rows) if int(r["order"]) == order]
            vals = [rows[i]["tau"] if rows[i]["valid"] else np.nan for i in idxs]
            perm = rng.permutation(len(idxs))
            for a, b in zip(idxs, perm):
                sh_by_key[(rows[a]["order"], rows[a]["path"])] = \
                    float(vals[b]) if np.isfinite(vals[b]) else np.nan

        def births_shuf2(r):
            return sh_by_key.get((int(r["order"]), r["path"]), np.nan)

        rules = {"tau": births_tau, "pairwise": births_pw,
                 "facetmax": births_facet, "edgew": births_ew,
                 "shuffled": births_shuf2}
        # Birth-value comparison: does higher-order tau exceed what
        # pairwise/facet summaries already determine?  Computed on the
        # BIRTH VALUES themselves, independent of persistence.
        deltas = {"tau_minus_pairwise": [], "tau_minus_facetmax": []}
        for r in rows:
            canon = tuple(int(x) for x in r["canon"].split(","))
            bt_ = births_tau(r); bp = births_pw(r); bf = births_facet(r)
            if np.isfinite(bt_) and np.isfinite(bp):
                deltas["tau_minus_pairwise"].append(float(bt_) - float(bp))
            if np.isfinite(bt_) and np.isfinite(bf):
                deltas["tau_minus_facetmax"].append(float(bt_) - float(bf))
        bd = {}
        for name, vals in deltas.items():
            a = np.asarray(vals, float)
            bd[name] = dict(n=int(len(a)),
                            frac_positive=float(np.mean(a > 1e-9)) if len(a) else None,
                            mean=float(np.mean(a)) if len(a) else None,
                            p90=float(np.quantile(a, 0.9)) if len(a) else None,
                            max=float(np.max(a)) if len(a) else None)
        result[f"{graph}_birth_deltas"] = bd
        for rule, bf in rules.items():
            paths_by_time = [active_sets(rows, bf, s) for s in SCALES]
            key = f"{graph}_{rule}"
            try:
                pres = run_persistence(paths_by_time)
                result[key] = dict(
                    chain_map_valid=pres["chain_map_valid"],
                    chain_map_max_residual=pres["chain_map_max_residual"],
                    kernel_backend=pres.get("kernel_backend"),
                    betti_final=pres.get("betti_final"),
                    boundary_sq_residual=pres.get("boundary_sq_residual"),
                    **summarize(pres),
                    intervals={k: v for k, v in pres.items()
                               if k.startswith("intervals_")})
            except Exception as exc:
                result[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def main():
    dirs = [RES / "deep_subset_pd_complex", RES / "deep_subset_rna_complex"]
    dirs = [d for d in dirs if d.exists()]
    rng = np.random.default_rng(7)
    all_rows = []
    for d in dirs:
        sids = sorted({f.name.split(".")[0]
                       for f in d.glob("*.complex_paths.parquet")})
        for sid in sids:
            try:
                r = compare_structure(d, sid, rng)
                all_rows.append(r)
                print(f"[cmp] {sid} ok", flush=True)
            except Exception as exc:
                print(f"[cmp] {sid} ERROR {type(exc).__name__}: {exc}", flush=True)
                all_rows.append(dict(structure_id=sid, source_dir=d.name,
                                     error=f"{type(exc).__name__}: {exc}"))
    (OUT / "filtration_compare.json").write_text(
        json.dumps(all_rows, indent=1, default=str))
    rows = []
    for r in all_rows:
        for key, val in r.items():
            if "_" in key and isinstance(val, dict) and "n_k0" in val:
                rows.append(dict(structure_id=r["structure_id"],
                                 source=r.get("source_dir"), arm=key,
                                 **{k: v for k, v in val.items()
                                    if not isinstance(v, (dict, list))}))
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "filtration_compare_summary.csv",
                                  index=False)
    print("[done] ->", OUT / "filtration_compare.json")


if __name__ == "__main__":
    main()
