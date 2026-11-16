# -*- coding: utf-8 -*-
"""EXP-03 extended filtration comparison.

Modes:
  grid        - fixed scale grid (fine grid supported via --scales)
  eventvalue  - exact event-value filtration: each rule uses its own
                sorted distinct finite birth values as the scale set
                (tau and pairwise rules only; capped by --max-scales
                with quantile subsampling and a logged warning)

Usage: python exp03_fine.py --dir D --sid S --out J --mode grid \
         --scales 0,0.25,0.5,1.0,1.5,2.0,3.0
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep/GSPH_H3/ablation")
sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep/GSPH_H3")
sys.path.insert(0, "/work/home/acn8s56xc0/gsph_deep")

import filtration_compare as FC  # noqa: E402


def _birth_fn(rule, rows, tau_of, facet_map, wmap, sh_by_key, rng):
    def canon(r):
        return tuple(int(x) for x in r["canon"].split(","))

    if rule == "tau":
        return lambda r: float(r["tau"]) if r["valid"] and np.isfinite(r["tau"]) else np.nan
    if rule == "pairwise":
        return lambda r: (np.nan if not r["valid"] else
                          FC.pair_birth(canon(r), tau_of))
    if rule == "facetmax":
        return lambda r: (np.nan if not r["valid"] else
                          facet_map[canon(r)])
    if rule == "edgew":
        def f(r):
            t = tuple(int(x) for x in r["path"].split(","))
            if len(t) <= 1:
                return 0.0
            ws = [wmap.get((t[i], t[i + 1]), np.nan) for i in range(len(t) - 1)]
            return max(ws) if all(np.isfinite(ws)) else np.nan
        return f
    if rule == "shuffled":
        return lambda r: sh_by_key.get((int(r["order"]), r["path"]), np.nan)
    raise ValueError(rule)


def _event_scales(births, max_scales):
    vals = np.sort(np.unique(np.asarray(
        [b for b in births if np.isfinite(b)], float)))
    if len(vals) > max_scales:
        qs = np.linspace(0, 1, max_scales)
        vals = np.unique(np.quantile(vals, qs))
        return vals, True
    return vals, False


def run_rule(rows, bf, scales):
    paths_by_time = [FC.active_sets(rows, bf, s) for s in scales]
    pres = FC.run_persistence(paths_by_time)
    return dict(
        n_scales=int(len(scales)),
        scale_min=float(scales[0]), scale_max=float(scales[-1]),
        chain_map_valid=pres["chain_map_valid"],
        chain_map_max_residual=pres["chain_map_max_residual"],
        kernel_backend=pres.get("kernel_backend"),
        betti_final=pres.get("betti_final"),
        boundary_sq_residual=pres.get("boundary_sq_residual"),
        **FC.summarize(pres),
        intervals={k: v for k, v in pres.items() if k.startswith("intervals_")})


def compare(d: Path, sid: str, mode: str, scales, max_scales: int, rng,
            graphs=("dir", "und")):
    paths, tau_of, man = FC.load(d, sid)
    result = {"structure_id": sid, "source_dir": d.name, "mode": mode,
              "n_dir_paths": man.get("n_dir_paths"),
              "n_und_paths": man.get("n_und_paths")}
    ew = {}
    for graph in graphs:
        ef = d / f"{sid}.complex_edges.parquet"
        if ef.exists():
            import pandas as pd
            edf = pd.read_parquet(ef)
            edf = edf[edf["graph"] == graph]
            ew[graph] = {(int(r["u"]), int(r["v"])): float(r["w"])
                         for _, r in edf.iterrows()}
        else:
            ew[graph] = {}
    for graph in graphs:
        gdf = paths[paths["graph"] == graph]
        if len(gdf) == 0:
            continue
        rows = list(gdf.to_dict("records"))
        wmap = ew.get(graph, {})
        canons = {tuple(int(x) for x in r["canon"].split(",")) for r in rows}
        facet_map = FC.facet_births(canons, tau_of)
        sh_by_key = {}
        for order in range(4):
            idxs = [i for i, r in enumerate(rows) if int(r["order"]) == order]
            vals = [rows[i]["tau"] if rows[i]["valid"] else np.nan for i in idxs]
            perm = rng.permutation(len(idxs))
            for a, b in zip(idxs, perm):
                sh_by_key[(rows[a]["order"], rows[a]["path"])] = \
                    float(vals[b]) if np.isfinite(vals[b]) else np.nan

        rules = ["tau", "pairwise", "facetmax", "edgew", "shuffled"] \
            if mode == "grid" else ["tau", "pairwise"]
        for rule in rules:
            bf = _birth_fn(rule, rows, tau_of, facet_map, wmap, sh_by_key, rng)
            if mode == "eventvalue":
                births = [bf(r) for r in rows]
                sc, subsampled = _event_scales(births, max_scales)
                if subsampled:
                    result[f"{graph}_{rule}_subsampled"] = True
            else:
                sc = np.asarray(scales, float)
            key = f"{graph}_{rule}"
            try:
                result[key] = run_rule(rows, bf, sc)
            except Exception as exc:
                result[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["grid", "eventvalue"], default="grid")
    ap.add_argument("--scales", default="0,0.25,0.5,1.0,1.5,2.0,3.0")
    ap.add_argument("--max-scales", type=int, default=4000)
    ap.add_argument("--graphs", default="dir,und")
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    scales = [float(x) for x in a.scales.split(",")]
    graphs = tuple(g for g in a.graphs.split(",") if g)
    r = compare(Path(a.dir), a.sid, a.mode, scales, a.max_scales, rng,
                graphs=graphs)
    r["source_dir"] = Path(a.dir).name
    Path(a.out).write_text(json.dumps(r, indent=1, default=str))
    print(f"[exp03] {a.sid} {a.mode} ok -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
