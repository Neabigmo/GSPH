"""Point-cloud controls and reuse of the frozen patch-filtration archives."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import gudhi
import miniball
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from cspph.core.engine import GSPH6Config
from cspph.core.gsph1 import _active_set_support, _support_certificate
from cspph.geometry.patch_support import compute_patch_support
from cspph.graph.patch_digraph import enumerate_patch_paths
from cspph.topology.event_persistence import compare_event_barcodes
from tasks.rna.intramolecular import extract_intra, parse_structure
from tasks.rna.sparse import rna_event_filter

ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-8
MAX_RADIUS = 2.0
FIELD = 2
PAIRS = (("und_tau", "und_pairwise"), ("dir_tau", "dir_pairwise"),
         ("dir_tau", "dir_edgew"))


def normalize_bars(bars):
    result = []
    for birth, death in bars:
        birth = float(birth)
        death = math.inf if death is None else float(death)
        if not math.isfinite(birth) or math.isnan(death) or death < birth:
            raise ValueError("invalid interval")
        if death - birth > TOL:
            result.append((birth, death))
    return sorted(result)


def endpoint_error(left, right):
    a, b = normalize_bars(left), normalize_bars(right)
    if len(a) != len(b):
        return math.inf
    if not a:
        return 0.
    a, b = np.asarray(a), np.asarray(b)
    costs = np.abs(a[:, 0, None] - b[None, :, 0])
    ad, bd = a[:, 1, None], b[None, :, 1]
    death_cost = np.abs(np.where(np.isfinite(ad), ad, 0) - np.where(np.isfinite(bd), bd, 0))
    costs = np.maximum(costs, death_cost)
    costs[np.isinf(ad) != np.isinf(bd)] = math.inf
    thresholds = np.unique(costs[np.isfinite(costs)])

    def matched(threshold):
        return np.all(maximum_bipartite_matching(csr_matrix(costs <= threshold), perm_type="column") >= 0)

    if not len(thresholds) or not matched(thresholds[-1]):
        return math.inf
    low, high = 0, len(thresholds) - 1
    while low < high:
        middle = (low + high) // 2
        if matched(thresholds[middle]):
            high = middle
        else:
            low = middle + 1
    return float(thresholds[low])


def distance(left, right):
    a, b = normalize_bars(left), normalize_bars(right)
    value = float(gudhi.bottleneck_distance(
        np.asarray(a, dtype=float).reshape(-1, 2),
        np.asarray(b, dtype=float).reshape(-1, 2), e=0.0))
    equal = endpoint_error(a, b) <= TOL
    return len(a), len(b), value, equal


def compare(structure, cohort, layer, left_name, right_name, left, right, **meta):
    rows = []
    for order in range(3):
        n_left, n_right, bottleneck, equal = distance(left[order], right[order])
        rows.append(dict(structure_id=structure, cohort=cohort, layer=layer,
                         left=left_name, right=right_name, order=order,
                         n_left=n_left, n_right=n_right, bottleneck=bottleneck,
                         max_matched_endpoint_difference=endpoint_error(left[order], right[order]),
                         identical_1e8=equal, status="ok", **meta))
    return rows


def simplex_radius(points):
    """Welzl miniball, with exhaustive active spheres for singular inputs."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise ValueError("miniball needs a nonempty simplex")
    points = points - points[0]
    try:
        center, squared = miniball.get_bounding_ball(points, rng=np.random.default_rng(7))
        if np.max(np.linalg.norm(points - center, axis=1)) <= math.sqrt(max(0., squared)) + 1e-10:
            return math.sqrt(max(0., squared))
    except np.linalg.LinAlgError:
        pass
    best = math.inf
    for size in range(1, min(4, len(points)) + 1):
        for indices in combinations(range(len(points)), size):
            active = points[list(indices)]
            delta = active[1:] - active[0]
            offset = np.linalg.lstsq(2 * delta, np.sum(delta * delta, axis=1), rcond=None)[0]
            radius = float(np.linalg.norm(offset))
            if np.max(np.linalg.norm(points - active[0] - offset, axis=1)) <= radius + 1e-10:
                best = min(best, radius)
    if not math.isfinite(best):
        raise ValueError("no enclosing sphere found")
    return best


def check_simplex(job):
    indices, points = job
    points = np.asarray(points, dtype=float)
    points = points - points[0]
    radius = simplex_radius(points)
    answer = _active_set_support(points, np.zeros(len(points)))
    if answer is None:
        raise ValueError(f"uncertified zero-radius support: {indices}")
    center, tau = answer
    certified, _, _ = _support_certificate(center, tau, points, np.zeros(len(points)))
    error = abs(float(tau) - radius)
    if not certified or error > TOL:
        raise ValueError(f"zero-radius support disagrees with miniball: {indices}, {error}")
    return indices, radius, float(tau), error


def diagrams(tree):
    tree.make_filtration_non_decreasing()
    tree.compute_persistence(homology_coeff_field=FIELD, persistence_dim_max=True)
    return {order: normalize_bars(tree.persistence_intervals_in_dimension(order))
            for order in range(3)}


def alpha_bars(points):
    if not len(points):
        return {order: [] for order in range(3)}
    tree = gudhi.AlphaComplex(points=points, precision="exact").create_simplex_tree(
        max_alpha_square=MAX_RADIUS ** 2)
    for simplex, value in list(tree.get_filtration()):
        tree.assign_filtration(simplex, math.sqrt(max(0., value)))
    return diagrams(tree)


def rips_bars(points):
    tree = gudhi.RipsComplex(points=points, max_edge_length=2 * MAX_RADIUS).create_simplex_tree(
        max_dimension=3)
    for simplex, value in list(tree.get_filtration()):
        tree.assign_filtration(simplex, value / 2)
    return diagrams(tree)


def point_controls(structure, coordinates, atom_ids, cap, workers):
    points = np.asarray(coordinates, dtype=float).reshape(-1, 3)
    if len(points) != len(atom_ids) or cap < 4:
        raise ValueError("invalid atom list or point cap")
    started = perf_counter()
    rows = compare(structure, "RNA_control", "points_full", "alpha", "rips",
                   alpha_bars(points), rips_bars(points), n_points=len(points),
                   scale_units="radius_A", max_scale=MAX_RADIUS, source="RCSB_control")
    small = points[:cap]
    jobs = [(indices, small[list(indices)]) for size in range(2, 5)
            for indices in combinations(range(len(small)), size)]
    if workers == 1:
        checked = list(map(check_simplex, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            checked = list(pool.map(check_simplex, jobs, chunksize=32))
    trees = [gudhi.SimplexTree(), gudhi.SimplexTree()]
    for tree in trees:
        for i in range(len(small)):
            tree.insert([i], filtration=0.)
    for indices, radius, tau, _ in checked:
        for tree, birth in zip(trees, (radius, tau)):
            if birth <= MAX_RADIUS:
                tree.insert(indices, filtration=birth)
    cech, zero_tau = map(diagrams, trees)
    alpha, rips = alpha_bars(small), rips_bars(small)
    for left, right, a, b in (("cech", "alpha", cech, alpha),
                              ("cech", "zero_tau", cech, zero_tau),
                              ("cech", "rips", cech, rips)):
        rows.extend(compare(structure, "RNA_control", "points_cech_subset", left, right,
                            a, b, n_points=len(small), scale_units="radius_A",
                            max_scale=MAX_RADIUS, source="RCSB_control"))
    checks = {str(size - 1): dict(n_simplices=sum(len(x[0]) == size for x in checked),
                                 max_abs_error_A=max((x[3] for x in checked if len(x[0]) == size), default=0.))
              for size in range(2, 5)}
    return rows, dict(n_full_points=len(points), n_cech_points=len(small),
                      cech_atom_ids=atom_ids[:cap], simplex_checks=checks,
                      seconds=perf_counter() - started)


def extract_control(structure, directory):
    cached = directory / f"{structure}.patch_control.json"
    if cached.exists():
        return json.loads(cached.read_text())
    started = perf_counter()
    pdb = directory / f"{structure}.pdb"
    if not pdb.exists():
        with urlopen(f"https://files.rcsb.org/download/{structure}.pdb", timeout=90) as response:
            pdb.write_bytes(response.read())
    atoms = parse_structure(pdb)
    settings = yaml.safe_load((ROOT / "configs/experiments/exp03_filtration_comparison.yaml").read_text())["rna_molecular_controls"]
    select, _ = rna_event_filter(atoms, **settings["event_filter"])
    result = extract_intra(atoms, "rna_quality_sparse", event_filter=select,
                           config=GSPH6Config(interaction_backend="cpu", use_cpp_path=False,
                                             **settings["geometry"]))
    meta = result["metadata"]
    if any(meta[k] for k in ("events_truncated", "paths_truncated", "support_incomplete")):
        raise ValueError("incomplete control extraction")
    patches = {p.patch_id: p for p in result["patches"]}
    atom_map = {a.index: a for a in atoms}
    ids = sorted({i for p in patches.values() for i in p.atom_ids
                  if str(atom_map[i].element).upper() not in ("H", "D")})
    supports = dict(result["support_by_set"])
    support_cache, pair_cache = {}, {}

    def support(path):
        canon = tuple(sorted(set(path)))
        if canon not in supports or supports[canon] is None:
            solved = compute_patch_support([patches[i] for i in canon], atom_map,
                                           support_cache=support_cache, pair_support_cache=pair_cache)
            if not solved.complete or not solved.certificate_passed:
                raise ValueError(f"uncertified patch support: {canon}")
            supports[canon] = float(solved.tau)
        value = float(supports[canon])
        if not math.isfinite(value):
            raise ValueError("nonfinite patch support")
        return value

    edges = [(int(u), int(v), float(w)) for u, v, w in result["edges"]]
    und_edges = sorted(set(edges) | {(v, u, w) for u, v, w in edges})
    und_paths, _, _ = enumerate_patch_paths(sorted(patches), und_edges, max_order=3, use_cpp=False)
    paths_dir = list(result["path_support"])
    if not all(info["valid"] for info in result["path_support"].values()):
        raise ValueError("invalid directed path")
    paths_und = [tuple(map(int, p)) for paths in und_paths.values() for p in paths]
    births = {}
    for graph, paths in (("dir", paths_dir), ("und", paths_und)):
        common, pairwise, edgew = [], [], []
        weights = {(u, v): float(w) for u, v, w in edges}
        for path in paths:
            canon = tuple(sorted(set(path)))
            tau = support(canon)
            pair = tau if len(canon) <= 2 else max(support(pair) for pair in combinations(canon, 2))
            common.append([list(path), tau])
            pairwise.append([list(path), pair])
            if graph == "dir":
                edgew.append([list(path), max((weights[path[i], path[i+1]]
                                              for i in range(len(path)-1)), default=0.)])
        births[f"{graph}_tau"], births[f"{graph}_pairwise"] = common, pairwise
        if graph == "dir":
            births["dir_edgew"] = edgew
    record = dict(structure_id=structure, pdb_sha256=hashlib.sha256(pdb.read_bytes()).hexdigest(),
                  atom_ids=ids, coordinates=[list(map(float, atom_map[i].xyz)) for i in ids],
                  n_patches=len(patches), n_dir_paths=len(paths_dir), n_und_paths=len(paths_und),
                  births=births, extraction_seconds=perf_counter() - started)
    cached.write_text(json.dumps(record, allow_nan=False) + "\n")
    return record


def patch_controls(record):
    started = perf_counter()
    births = {name: {tuple(p): float(t) for p, t in pairs if t <= MAX_RADIUS}
              for name, pairs in record["births"].items()}
    bars = {}
    for name, values in births.items():
        _, bars[name], _ = compare_event_barcodes(values, values)
    rows = []
    for left, right in PAIRS:
        units = "tau_A_vs_bond_steps" if right.endswith("edgew") else "inflation_A"
        rows.extend(compare(record["structure_id"], "RNA_control", "patch_events", left, right,
                            bars[left], bars[right], source="RCSB_control", scale_units=units,
                            max_scale=MAX_RADIUS))
    return rows, perf_counter() - started


def archived_bars(arm, scales, order):
    if not arm.get("chain_map_valid"):
        raise ValueError("unverified archived chain map")
    bars = []
    for birth, death, censored in arm["intervals"][f"intervals_k{order}"]:
        if censored != (death == len(scales)):
            raise ValueError("inconsistent censoring flag")
        bars.append((scales[birth], None if censored else scales[death]))
    return bars


def archive_rows():
    tables = ROOT / "results/tables"
    cohort = pd.read_csv(tables / "barcode_comparison_16structure.csv").structure_id.unique()
    details = {}
    for filename in ("filtration_compare.json", "filtration_compare_rna_clash06.json"):
        for record in json.loads((tables / filename).read_text()):
            details[record["structure_id"]] = (record, filename)
    counts = pd.read_csv(tables / "filtration_compare_summary.csv")
    rows = []
    for sid in cohort:
        task = "protein_DNA_archive" if "_ftd" in sid else "RNA_clash06_archive"
        raw, source = details.get(sid, ({}, "filtration_compare_summary.csv"))
        for left, right in PAIRS:
            if left in raw and right in raw and "intervals" in raw[left] and "intervals" in raw[right]:
                a = {k: archived_bars(raw[left], raw["scales"], k) for k in range(3)}
                b = {k: archived_bars(raw[right], raw["scales"], k) for k in range(3)}
                rows.extend(compare(sid, task, "patch_legacy_grid", left, right, a, b,
                                    source=source, max_scale=MAX_RADIUS,
                                    scale_units="tau_A_vs_bond_steps" if right.endswith("edgew") else "inflation_A"))
            else:
                available = counts[counts.structure_id == sid].set_index("arm")
                for order in range(3):
                    rows.append(dict(structure_id=sid, cohort=task, layer="patch_legacy_grid",
                                     left=left, right=right, order=order,
                                     n_left=available.loc[left, f"n_k{order}"] if left in available.index else None,
                                     n_right=available.loc[right, f"n_k{order}"] if right in available.index else None,
                                     bottleneck=None, identical_1e8=None, status="missing_intervals",
                                     source=source, max_scale=MAX_RADIUS,
                                     scale_units="tau_A_vs_bond_steps" if right.endswith("edgew") else "inflation_A"))
    return rows


def write_report(frame, audit, destination):
    lines = ["# Čech, α and Rips controls", "",
             "普通 Čech 是零半径点支撑下共同支撑的特例。在 patch 层，τ 是加厚支撑集族的 Čech 型神经出生值，pairwise 是它的 Rips 型对应。", "",
             "**控制队列，不是原 16 个结构的复现。** 1A9L、1A60 为 RCSB 第一模型；原队列仅复用归档，不以 native 替代 decoy。", "",
             "## Summary", "",
             "Counts pool H0–H2 per structure. Distances are GUDHI bottleneck distances on the stated parameter; infinity is retained.", "",
             "| Cohort | Layer | Comparison | Identical / available | Missing | Median | Maximum |",
             "|---|---|---|---:|---:|---:|---:|"]
    for keys, group in frame.groupby(["cohort", "layer", "left", "right"], sort=True):
        valid = group[group.status == "ok"]
        ds = valid.bottleneck.astype(float).to_numpy()
        median = float(np.median(ds)) if len(ds) else math.nan
        maximum = float(np.max(ds)) if len(ds) else math.nan
        lines.append(f"| {keys[0]} | {keys[1]} | {keys[2]} vs {keys[3]} | "
                     f"{int(valid.identical_1e8.sum())}/{len(valid)} | {len(group)-len(valid)} | {median:.9g} | {maximum:.9g} |")
    lines += ["", "## Scope and interpretation", "",
              "- Point-cloud scales are radii in Å, capped at 2 Å; H0–H2 uses simplices through dimension 3. Full α and Rips use all unique heavy-atom centers appearing in retained patches.",
              "- Exact Čech and zero-radius checks use the first 24 atoms in ascending parser atom-ID order, or the entire cloud if smaller. This additional computational truncation is not an EXP-03 truncation. All edges, triangles and tetrahedra of this subset are checked, including births above 2 Å; only barcode construction is capped at 2 Å. Subset α/Rips use exactly these same atoms.",
              "- α is GUDHI exact-precision Delaunay α with sqrt of squared filtration. Čech uses independent Welzl miniball (float64; exhaustive active-sphere fallback for singular simplices), not an approximate enclosing ball. Zero-radius τ calls the existing weighted active-set support solver with radii zero and checks its certificate; neither vdW radii nor patch definitions are edited.",
              "- Point simplicial homology uses F_2; the existing exact GLMY reducer uses F_1000003. No cross-layer equality is inferred across these coefficient fields. Numerical intervals of lifetime ≤1e-8 are discarded consistently. Identical means a multiplicity-preserving perfect interval matching with endpoint error ≤1e-8, without diagonal matches. This is checked independently of the GUDHI bottleneck (which permits the diagonal).",
              "- At the 2-unit cutoff, surviving classes have death +∞ in the restricted filtration. These are right-censored, not evidence of essential classes in the unbounded complex. Unequal surviving multiplicities can therefore give infinite bottleneck distance.",
              "- The frozen grid is [0, 0.5, 2]. Archived indices are mapped back to this shared grid; death index 3 is right-censored, never interpreted as a measured death at 3 Å. No full-event distances are inferred from the grid.",
              "- The summary CSV contains pre-clash RNA and omits arms; use its counts only when detailed intervals are absent. Matching detailed RNA clash06 JSON overrides pre-clash records. Missing detailed intervals stay missing; counts alone do not establish equal barcodes.",
              "- `und_*` archives and controls use the symmetrized graph's regular GLMY path complex, not simplicial homology of the support-family nerve. Their birth rules are Čech-/Rips-type, but these persistence objects must not be identified with ordinary point-cloud Čech/Rips.",
              "- `edgew` is max adjacent edge weight (Chowdhury–Mémoli-type path filtration). Here weights are minimum covalent bond-transfer distances in bond steps, not Å. τ/edgew distances and equality flags are numerical parameter controls, not geometrically calibrated Å distances.",
              "- Point-cloud and patch vertices differ. No direct bottleneck between their diagrams is claimed to isolate direction alone. These controls establish specialization and filtration differences, not predictive benefit or prevalence in the original 16 decoys.",
              "", "## Control audit", "",
              "| Structure | Full atoms | Čech atoms | Patches | Directed paths | Symmetrized paths | Extraction s | Points s | Patch persistence s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sid, rec in audit["controls"].items():
        lines.append(f"| {sid} | {rec['n_full_points']} | {rec['n_cech_points']} | {rec['n_patches']} | {rec['n_dir_paths']} | {rec['n_und_paths']} | {rec['extraction_seconds']:.3f} | {rec['seconds']:.3f} | {rec['patch_seconds']:.3f} |")
    for sid, rec in audit["controls"].items():
        lines += ["", f"- {sid} PDB SHA256: `{rec['pdb_sha256']}`.",
                  f"- {sid} retained Čech atom IDs: `{rec['cech_atom_ids']}`.",
                  f"- {sid} zero-radius checks by simplex dimension: `{json.dumps(rec['simplex_checks'], sort_keys=True)}`.", ""]
    lines += ["", f"Total command wall time: {audit['wall_seconds']:.3f} s (cached extraction times are listed separately).",
              "", f"Software: `{json.dumps(audit['versions'], sort_keys=True)}`.", "",
              "Sources: `experiments/exp03_filtration_comparison/EXPERIMENT_CARD.md`; `configs/experiments/exp03_filtration_comparison.yaml`; `results/tables/{filtration_compare_summary.csv,filtration_compare.json,filtration_compare_rna_clash06.json,barcode_comparison_16structure.csv}`.",
              "", "Run instructions and mathematical qualifications: [experiment card](../../experiments/external_topology_cech_rips/EXPERIMENT_CARD.md).", ""]
    destination.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Existing raw-data/cache directory outside the checkout")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    directory = args.data_dir.resolve()
    if not directory.is_dir() or directory == ROOT or ROOT in directory.parents:
        parser.error("--data-dir must be an existing directory outside the checkout")
    if args.workers < 1:
        parser.error("--workers must be positive")
    started = perf_counter()
    rows = archive_rows()
    audit = dict(controls={}, versions={p: importlib.metadata.version(p)
                                      for p in ("gudhi", "miniball", "numpy", "scipy", "pandas", "biopython")})
    for sid in ("1A9L", "1A60"):
        print(f"{sid}: extract patch control", flush=True)
        record = extract_control(sid, directory)
        print(f"{sid}: point controls", flush=True)
        point_rows, checked = point_controls(sid, record["coordinates"], record["atom_ids"], 24, args.workers)
        print(f"{sid}: GLMY patch controls", flush=True)
        patch_rows, seconds = patch_controls(record)
        rows.extend(point_rows + patch_rows)
        audit["controls"][sid] = {**checked, **{k: record[k] for k in (
            "n_patches", "n_dir_paths", "n_und_paths", "extraction_seconds", "pdb_sha256")},
            "patch_seconds": seconds}
    audit["wall_seconds"] = perf_counter() - started
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/tables/cech_rips_compare.csv", index=False)
    write_report(frame, audit, ROOT / "results/reports/cech_rips_compare.md")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
