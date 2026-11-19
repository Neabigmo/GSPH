"""Bounded, resumable ion/water geometry pilot; never trains a model."""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import gzip
import hashlib
import json
import random
import resource
import signal
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
import requests
import yaml
from scipy.spatial import cKDTree

from geometry import UnionSolver, cliques, crosscheck, pair_support

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/exp09_metal_site_pilot.yaml"
CLASSIC = {"HIS", "CYS", "ASP", "GLU"}
ZN_GROUPS = {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"], "HIS": ["ND1", "NE2"],
             "CYS": ["SG"], "MET": ["SD"], "SER": ["OG"], "THR": ["OG1"],
             "TYR": ["OH"], "ASN": ["OD1"], "GLN": ["OE1"]}


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(identifier):
    return int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "little")


def config_hash(cfg, region_radius=0.):
    payload = {"config": cfg, "region_radius": region_radius}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def reusable_result(path, fingerprint):
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    if row.get("config_hash") == fingerprint and row.get("status") not in {"download_missing", "worker_error", "error"}:
        return row
    return None


def load_config(path=DEFAULT_CONFIG):
    return yaml.safe_load(path.read_text())


def release_is_test(value, cfg=None):
    cfg = load_config() if cfg is None else cfg
    return date.fromisoformat(value) >= date.fromisoformat(cfg["time_split_date"])


def annotate_dates(data, cfg):
    cache = data / "release_dates.json"
    dates = json.loads(cache.read_text()) if cache.exists() else {}
    for cohort in current_samples():
        path = HERE / f"{cohort}_ids.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            identifier = row["pdb_id"]
            if identifier not in dates:
                block = gemmi.cif.read(str(data / f"{identifier}.cif.gz")).sole_block()
                values = list(block.find_values("_pdbx_audit_revision_history.revision_date"))
                if not values:
                    raise RuntimeError(f"Missing initial release date for {identifier}")
                dates[identifier] = values[0]
            row["initial_release_date"] = dates[identifier]
            row["time_split_test"] = release_is_test(dates[identifier], cfg)
        write_csv(path, rows)
    cache.write_text(json.dumps(dates, indent=2))
    return dates


def search_entries(session, query, data, name):
    target = data / f"search_{name}.json"
    if target.exists():
        response = json.loads(target.read_text())
    else:
        result = session.post("https://search.rcsb.org/rcsbsearch/v2/query", json={
            "query": query, "return_type": "entry", "request_options": {"return_all_hits": True}}, timeout=90)
        result.raise_for_status()
        response = result.json()
        target.write_text(json.dumps(response))
    return sorted({row["identifier"].lower() for row in response.get("result_set", [])})


def terminal(attribute, operator, value):
    return {"type": "terminal", "service": "text", "parameters": {
        "attribute": attribute, "operator": operator, "value": value}}


def sample_ids(biolip, data, cfg):
    rng = random.Random(cfg["seed"])
    zn = set()
    with gzip.open(biolip, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8 or fields[4].upper() != "ZN":
                continue
            try:
                good = 0 < float(fields[2]) <= cfg["resolution_max"]["zn"] and len(set(fields[7].split())) >= cfg["zn_min_binding_residues"]
            except ValueError:
                good = False
            if good:
                zn.add(fields[0].lower())
    session = requests.Session()
    water_query = {"type": "group", "logical_operator": "and", "nodes": [
        terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
        terminal("rcsb_entry_info.resolution_combined", "less_or_equal", cfg["resolution_max"]["water"]),
        terminal("rcsb_entry_info.polymer_entity_count_protein", "greater", 0)]}
    mg_nodes = [terminal("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
                terminal("rcsb_entry_info.resolution_combined", "less_or_equal", cfg["resolution_max"]["mg"]),
                terminal("rcsb_entry_info.polymer_entity_count_RNA", "greater", 0),
                terminal("rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id", "exact_match", "MG")]
    water = search_entries(session, water_query, data, "water")
    mg = search_entries(session, {"type": "group", "logical_operator": "and", "nodes": mg_nodes}, data, "mg")
    large = search_entries(session, {"type": "group", "logical_operator": "and", "nodes": mg_nodes + [
        terminal("rcsb_entry_info.deposited_atom_count", "greater_or_equal", cfg["large_rna_min_atoms"])]}, data, "large_rna")
    if not large:
        raise RuntimeError("No large RNA/Mg entry found")
    samples = {"zn": rng.sample(sorted(zn), min(len(zn), cfg["zn_count"])),
               "water": rng.sample(water, min(len(water), cfg["water_count"])),
               "mg": rng.sample(mg, min(len(mg), cfg["mg_count"]))}
    ribosome = rng.choice(large)
    if ribosome not in samples["mg"]:
        samples["mg"][-1] = ribosome
    for cohort, ids in samples.items():
        write_csv(HERE / f"{cohort}_ids.csv", [{"pdb_id": identifier, "cohort": cohort,
            "seed": cfg["seed"], "large_rna_required": int(cohort == "mg" and identifier == ribosome)} for identifier in ids])
    (data / "selection.json").write_text(json.dumps({"eligible_zn": len(zn), "eligible_water": len(water),
        "eligible_mg": len(mg), "large_rna_id": ribosome, "samples": samples}, indent=2))


def download(data, cfg):
    selection = json.loads((data / "selection.json").read_text())
    ledger_path = data / "downloads.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    used = sum(row["bytes"] for row in ledger)
    complete = {row["pdb_id"] for row in ledger if row["status"] == "complete"
                and (data / f"{row['pdb_id']}.cif.gz").exists()}
    session = requests.Session()
    for identifier in sorted({pdb for ids in selection["samples"].values() for pdb in ids}):
        if identifier in complete:
            continue
        received = 0
        status = "error"
        try:
            if used + cfg["download_reserve_bytes"] >= cfg["download_limit_bytes"]:
                raise RuntimeError("download_budget_stop")
            with session.get(f"https://files.rcsb.org/download/{identifier.upper()}.cif.gz", stream=True, timeout=60) as response:
                length = int(response.headers.get("Content-Length", 0))
                if length and used + length > cfg["download_limit_bytes"]:
                    raise RuntimeError("download_budget_stop")
                with (data / f"{identifier}.partial").open("wb") as handle:
                    while True:
                        remaining = cfg["download_limit_bytes"] - used
                        if remaining <= 0:
                            raise RuntimeError("download_budget_stop")
                        block = response.raw.read(min(65536, remaining), decode_content=True)
                        if not block:
                            break
                        received += len(block)
                        used += len(block)
                        handle.write(block)
                response.raise_for_status()
                (data / f"{identifier}.partial").replace(data / f"{identifier}.cif.gz")
                status = "complete"
        except Exception as exc:
            status = "budget_stop" if "download_budget_stop" in str(exc) else f"error_{type(exc).__name__}"
        ledger.append({"pdb_id": identifier, "bytes": received, "status": status})
        ledger_path.write_text(json.dumps(ledger, indent=2))
        print(f"download {identifier} {status} bytes={received} cumulative={used}", flush=True)
        if status == "budget_stop":
            raise RuntimeError("Download would exceed budget; stopped")


def read_structure(path, cohort, cfg=None):
    cfg = load_config() if cfg is None else cfg
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    atoms, groups, sites = [], [], []
    residues = []
    raw_sites = []
    for chain in structure[0]:
        for residue in chain:
            chosen = {}
            for atom in residue:
                if atom.element.is_hydrogen:
                    continue
                old = chosen.get(atom.name)
                if old is None or (atom.occ, atom.altloc) > (old.occ, old.altloc):
                    chosen[atom.name] = atom
            if residue.name in {"ZN", "MG", "HOH", "WAT", "DOD"}:
                expected = {"zn": "ZN", "mg": "MG", "water": "O"}[cohort]
                for atom in chosen.values():
                    if atom.element.name.upper() == expected and (atom.occ >= cfg["ion_min_occupancy"] if cohort != "water" else abs(atom.occ - cfg["water_occupancy"]) < cfg["occupancy_tolerance"]):
                        raw_sites.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
                continue
            info = gemmi.find_tabulated_residue(residue.name)
            polymer = info.kind == gemmi.ResidueKind.RNA if cohort == "mg" else info.is_amino_acid()
            if len(chosen) == 1 and residue.entity_type != gemmi.EntityType.Polymer:
                continue
            rid = len(residues)
            residues.append((chain.name, str(residue.seqid), residue.name))
            named = {}
            for name, atom in chosen.items():
                named[name] = len(atoms)
                atoms.append({"xyz": [atom.pos.x, atom.pos.y, atom.pos.z], "element": atom.element.name.upper(),
                              "name": name, "residue": rid})
            if not polymer or residue.entity_type != gemmi.EntityType.Polymer:
                continue
            definitions = []
            if cohort == "zn":
                definitions = [("side", ZN_GROUPS.get(residue.name, [])), ("backbone_O", ["O"])]
            elif cohort == "water":
                side = [name for name, idx in named.items() if atoms[idx]["element"] in {"N", "O"} and name not in {"N", "O", "OXT"}]
                definitions = [("side", side), ("backbone_N", ["N"]), ("backbone_O", ["O", "OXT"])]
            else:
                definitions = [("phosphate", ["OP1", "OP2", "O1P", "O2P"])]
                definitions += [(name, [name]) for name, idx in named.items() if atoms[idx]["element"] in {"N", "O"}
                                and name not in {"OP1", "OP2", "O1P", "O2P"}]
            for role, names in definitions:
                ids = tuple(named[name] for name in names if name in named and atoms[named[name]]["element"] in {"N", "O", "S"})
                if ids:
                    groups.append({"atoms": ids, "residue": rid,
                                   "classic": cohort == "zn" and role == "side" and residue.name in CLASSIC})
    xyz = np.asarray([atom["xyz"] for atom in atoms], dtype=float).reshape((-1, 3))
    coordination = cfg["coordination_radii"][cohort]
    radii = np.array([coordination.get(a["element"], coordination["other"]) for a in atoms])
    ligand_atoms = sorted({i for group in groups for i in group["atoms"]})
    if ligand_atoms:
        tree = cKDTree(xyz[ligand_atoms])
        for center in raw_sites:
            nearby = [ligand_atoms[i] for i in tree.query_ball_point(center, cfg["site_neighbor_radius"])]
            low, high = cfg["water_partner_distance"]
            partners = [i for i in nearby if (low <= np.linalg.norm(xyz[i] - center) <= high if cohort == "water"
                        else np.linalg.norm(xyz[i] - center) <= radii[i] + cfg["ion_partner_margin"])]
            if len(partners) >= cfg["site_min_partners"][cohort] and len({atoms[i]["residue"] for i in partners}) >= cfg["min_distinct_residues"]:
                patch_ids = [g for g, group in enumerate(groups) if set(partners) & set(group["atoms"])]
                sites.append({"center": center, "groups": patch_ids})
    return atoms, xyz, radii, groups, sites


def pair_graph(groups, xyz, radii, delta):
    atom_to_group = {atom: i for i, group in enumerate(groups) for atom in group["atoms"]}
    ids = sorted(atom_to_group)
    adjacency = [set() for _ in groups]
    pair_tau = {}
    if not ids:
        return adjacency, pair_tau
    tree = cKDTree(xyz[ids])
    for a, b in tree.query_pairs(2 * (float(radii[ids].max()) + delta)):
        left, right = sorted((atom_to_group[ids[a]], atom_to_group[ids[b]]))
        if left == right or (left, right) in pair_tau:
            continue
        tau = pair_support(groups[left]["atoms"], groups[right]["atoms"], xyz, radii)
        pair_tau[left, right] = tau
        if tau <= delta:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency, pair_tau


def clustered_count(points, cfg=None):
    cfg = load_config() if cfg is None else cfg
    if not points:
        return 0
    parent = list(range(len(points)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in cKDTree(points).query_pairs(cfg["clustering_threshold"]):
        parent[find(a)] = find(b)
    return len({find(i) for i in range(len(points))})


def crop_region(atoms, xyz, radii, groups, radius, seed, cfg=None):
    cfg = load_config() if cfg is None else cfg
    if not groups:
        raise ValueError("regional calibration requires ligand patches")
    anchor = random.Random(seed).choice(groups)["atoms"][0]
    center = xyz[anchor]
    distances = np.linalg.norm(xyz - center, axis=1)
    selected = [g for g in groups if all(distances[i] <= radius for i in g["atoms"])]
    keep = np.flatnonzero(distances <= radius + cfg["region_clash_shell"])
    remap = {old: new for new, old in enumerate(keep)}
    selected = [{**g, "atoms": tuple(remap[a] for a in g["atoms"])} for g in selected]
    return [atoms[i] for i in keep], xyz[keep], radii[keep], selected, center


def compute_structure(identifier, cohort, data, cfg, result, region_radius=0.):
    phase = time.perf_counter()
    atoms, xyz, radii, groups, sites = read_structure(data / f"{identifier}.cif.gz", cohort, cfg)
    if region_radius:
        if cohort != "mg":
            raise ValueError("regional calibration is separate from the main protein cohorts")
        result.update(full_atoms=len(atoms), full_patches=len(groups), region_radius_A=region_radius)
        atoms, xyz, radii, groups, center = crop_region(atoms, xyz, radii, groups, region_radius,
                                                       stable_seed(f"{cfg['seed']}:{identifier}:region"), cfg)
        result["region_center"] = center.tolist()
        sites = []
    result.update(atoms=len(atoms), patches=len(groups), true_sites=len(sites), parse_seconds=time.perf_counter() - phase)
    if len(atoms) > cfg["max_atoms"] or len(groups) > cfg["max_patches"]:
        result["status"] = "size_censored"
        return
    phase = time.perf_counter()
    adjacency, pair_tau = pair_graph(groups, xyz, radii, cfg["delta"])
    result.update(pair_edges=sum(map(len, adjacency)) // 2, pair_graph_seconds=time.perf_counter() - phase)
    phase = time.perf_counter()
    reservoirs, counts = defaultdict(list), defaultdict(int)
    rng = random.Random(stable_seed(f"{cfg['seed']}:{cohort}:{identifier}"))
    for order in cfg["candidate_orders"]:
        for candidate in cliques(adjacency, order):
            if len({groups[i]["residue"] for i in candidate}) < cfg["min_distinct_residues"]:
                continue
            classic = all(groups[i]["classic"] for i in candidate)
            key = (order, classic)
            counts[key] += 1
            if sum(counts.values()) > cfg["max_enumerated_candidates"]:
                result.update(status="enumeration_censored", enumerated_candidates=sum(counts.values()))
                return
            bucket = reservoirs[key]
            if len(bucket) < cfg["sample_per_order_stratum"]:
                bucket.append(candidate)
            else:
                index = rng.randrange(counts[key])
                if index < len(bucket):
                    bucket[index] = candidate
    result["enumeration_seconds"] = time.perf_counter() - phase
    result["pairwise_passing"] = sum(counts.values())
    result["candidate_sampled"] = any(counts[key] > len(reservoirs[key]) for key in counts)
    solver = UnionSolver(xyz, radii)
    heavy_tree = cKDTree(xyz) if len(xyz) else None
    metrics, predictions, site_rows, candidate_rows = [], [], [], []
    for order in cfg["candidate_orders"]:
        for classic in (False, True):
            if cohort != "zn" and classic:
                continue
            key = (order, classic)
            row = {"k": order, "stratum": "classic" if classic else "other", "pairwise_passing": counts[key],
                   "sampled": len(reservoirs[key]), "hard": 0, "clash_pass": 0, "hard_clash_pass": 0,
                   "positive": 0, "positive_clash_pass": 0, "candidate_seconds": 0.}
            for candidate in reservoirs[key]:
                started = time.perf_counter()
                point, tau = solver.solve([groups[i]["atoms"] for i in candidate])
                elapsed = time.perf_counter() - started
                row["candidate_seconds"] += elapsed
                hard = tau > cfg["delta"]
                ligand_ids = {a for i in candidate for a in groups[i]["atoms"]}
                clash_pass = not any(a not in ligand_ids for a in heavy_tree.query_ball_point(point, cfg["clash_distance"]))
                row["hard"] += int(hard)
                row["clash_pass"] += int(clash_pass)
                row["hard_clash_pass"] += int(hard and clash_pass)
                row["positive"] += int(not hard)
                row["positive_clash_pass"] += int(not hard and clash_pass)
                pw = max(pair_tau[tuple(sorted(pair))] for pair in combinations(candidate, 2))
                candidate_rows.append({"k": order, "classic": classic, "patches": candidate, "tau": tau,
                                       "tau_pw": pw, "eta": tau - pw, "clash_pass": clash_pass,
                                       "seconds": elapsed, "point": point.tolist()})
                if not hard:
                    predictions.append((point, clash_pass))
            metrics.append(row)
    points = [p for p, _ in predictions]
    unclashed = [p for p, ok in predictions if ok]
    for index, site in enumerate(sites):
        ids = site["groups"]
        eta = None
        if cfg["eta_min_patches"] <= len(ids) <= cfg["eta_max_patches"] and np.prod([len(groups[i]["atoms"]) for i in ids]) <= cfg["eta_max_alternatives"]:
            _, tau = solver.solve([groups[i]["atoms"] for i in ids])
            pw = max(pair_support(groups[a]["atoms"], groups[b]["atoms"], xyz, radii) for a, b in combinations(ids, 2))
            eta = tau - pw
        best = min((float(np.linalg.norm(p - site["center"])) for p in points), default=None)
        best_clash = min((float(np.linalg.norm(p - site["center"])) for p in unclashed), default=None)
        site_rows.append({"site_index": index, "ligand_patches": len(ids), "eta": eta,
                          "error": best, "error_clash": best_clash})
    result.update(status="complete", metrics=metrics, sites=site_rows, candidates=candidate_rows,
                  candidate_calls=solver.calls, weighted_solves=solver.solves, certificate_failures=solver.failures,
                  weighted_solver_seconds=solver.seconds, clusters=clustered_count(points, cfg),
                  clusters_clash=clustered_count(unclashed, cfg))


def worker(args, cfg):
    started = time.perf_counter()
    result = {"pdb_id": args.pdb_id, "cohort": args.cohort, "status": "error",
              "config_hash": config_hash(cfg, args.region_radius)}
    def timeout(signum, frame):
        raise TimeoutError("structure wall budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(cfg["timeout_seconds"])
    try:
        compute_structure(args.pdb_id, args.cohort, args.data, cfg, result, args.region_radius)
    except TimeoutError:
        result["status"] = "timeout_censored"
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(wall_seconds=time.perf_counter() - started, cpu_seconds=usage.ru_utime + usage.ru_stime,
                      peak_rss_kib=usage.ru_maxrss)
        args.output.write_text(json.dumps(result, allow_nan=False))


def execute(args, cfg):
    download(args.data, cfg)
    selection = json.loads((args.data / "selection.json").read_text())
    jobs = [(cohort, pdb) for cohort, ids in selection["samples"].items() for pdb in ids]
    if args.limit:
        jobs = [(c, p) for c in selection["samples"] for p in selection["samples"][c][:args.limit]]
    outdir = args.data / "structures"
    outdir.mkdir(exist_ok=True)
    def launch(job):
        cohort, pdb = job
        out = outdir / f"{cohort}_{pdb}.json"
        fingerprint = config_hash(cfg)
        cached = reusable_result(out, fingerprint)
        if cached is not None:
            return cached
        if not (args.data / f"{pdb}.cif.gz").exists():
            row = {"cohort": cohort, "pdb_id": pdb, "status": "download_missing", "config_hash": fingerprint}
            out.write_text(json.dumps(row))
            return row
        start = time.perf_counter()
        command = [sys.executable, str(Path(__file__).resolve()), "--stage", "worker", "--data", str(args.data),
                   "--config", str(args.config), "--pdb-id", pdb, "--cohort", cohort, "--output", str(out)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=cfg["timeout_seconds"] + cfg["worker_grace_seconds"])
            if completed.returncode or not out.exists():
                raise RuntimeError(f"worker exit {completed.returncode}")
            row = json.loads(out.read_text())
            row["process_wall_seconds"] = time.perf_counter() - start
            out.write_text(json.dumps(row))
        except Exception as exc:
            row = {"cohort": cohort, "pdb_id": pdb, "status": "worker_error", "error": type(exc).__name__,
                   "wall_seconds": time.perf_counter() - start, "config_hash": fingerprint}
            out.write_text(json.dumps(row))
        print(json.dumps({key: row.get(key) for key in ["cohort", "pdb_id", "status", "atoms", "patches", "pairwise_passing", "wall_seconds", "peak_rss_kib"]}), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=min(cfg["workers"], os.cpu_count() or 1)) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            future.result()


def current_samples():
    samples = {}
    for cohort in ("zn", "water", "mg"):
        with (HERE / f"{cohort}_ids.csv").open() as handle:
            samples[cohort] = {row["pdb_id"] for row in csv.DictReader(handle)}
    return samples


def selected_records(directory, samples):
    records = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
    return [r for r in records if r["pdb_id"] in samples.get(r["cohort"], set())]


def summarize(args, cfg):
    match_near, match_far = cfg["site_match_distances"]
    dates = annotate_dates(args.data, cfg)
    samples = current_samples()
    records = selected_records(args.data / "structures", samples)
    for record in records:
        record["initial_release_date"] = dates[record["pdb_id"]]
        record["time_split_test"] = release_is_test(dates[record["pdb_id"]], cfg)
    per_structure = [{k: v for k, v in row.items() if k not in {"metrics", "sites", "candidates", "error"}} for row in records]
    write_csv(ROOT / "results/tables/exp09_metal_site_pilot_per_structure.csv", per_structure)
    site_rows = [{"cohort": r["cohort"], "pdb_id": r["pdb_id"], **site} for r in records for site in r.get("sites", [])]
    write_csv(ROOT / "results/tables/exp09_metal_site_pilot_sites.csv", site_rows)
    detail = [{"cohort": r["cohort"], "pdb_id": r["pdb_id"], **m} for r in records for m in r.get("metrics", [])]
    write_csv(ROOT / "results/tables/exp09_metal_site_pilot_candidate_strata.csv", detail)
    summary = []
    for cohort in ("zn", "water", "mg"):
        for k in cfg["candidate_orders"]:
            for stratum in (["all", "classic"] if cohort == "zn" else ["all"]):
                for split in (["all", False, True] if cohort != "mg" else ["all"]):
                    complete = [r for r in records if r["cohort"] == cohort and r["status"] == "complete"
                                and (split == "all" or r["time_split_test"] == split)]
                    rows = [m for r in complete for m in r["metrics"] if m["k"] == k and (stratum == "all" or m["stratum"] == "classic")]
                    denom = sum(m["pairwise_passing"] for m in rows)
                    weighted_hard = sum(m["hard"] * m["pairwise_passing"] / m["sampled"] for m in rows if m["sampled"])
                    clash_denom = sum(m["clash_pass"] * m["pairwise_passing"] / m["sampled"] for m in rows if m["sampled"])
                    clash_hard = sum(m["hard_clash_pass"] * m["pairwise_passing"] / m["sampled"] for m in rows if m["sampled"])
                    summary.append({"cohort": cohort, "k": k, "stratum": stratum, "time_split_test": split,
                        "complete_structures": len(complete), "pairwise_passing": denom,
                        "structures_with_candidates": sum(any(m["k"] == k and m["pairwise_passing"] and
                            (stratum == "all" or m["stratum"] == "classic") for m in r["metrics"]) for r in complete),
                        "sampled_candidates": sum(m["sampled"] for m in rows),
                        "hard_sample_count": sum(m["hard"] for m in rows), "estimated_hard_fraction": weighted_hard / denom if denom else None,
                        "estimated_clash_pass_count": clash_denom,
                        "estimated_hard_fraction_after_clash": clash_hard / clash_denom if clash_denom else None,
                        "candidate_seconds": sum(m["candidate_seconds"] for m in rows)})
    write_csv(ROOT / "results/tables/exp09_metal_site_pilot_summary.csv", summary)
    measurements = {}
    for cohort in ("zn", "water", "mg"):
        rows = [r for r in records if r["cohort"] == cohort]
        complete = [r for r in rows if r["status"] == "complete"]
        truth = [s for r in complete for s in r["sites"]]
        times = [r["wall_seconds"] for r in rows if "wall_seconds" in r]
        errors = [s["error"] for s in truth if s["error"] is not None]
        eta = [s["eta"] for s in truth if s["eta"] is not None]
        sample_count = sum(m["sampled"] for r in complete for m in r["metrics"])
        candidate_seconds = sum(m["candidate_seconds"] for r in complete for m in r["metrics"])
        candidate_times = [c["seconds"] for r in complete for c in r["candidates"]]
        projected = sum(m["candidate_seconds"] * m["pairwise_passing"] / m["sampled"]
                        for r in complete for m in r["metrics"] if m["sampled"])
        measurements[cohort] = {"selected": len(samples[cohort]),
            "attempted": len(rows), "complete": len(complete), "statuses": dict((status, sum(r["status"] == status for r in rows)) for status in sorted({r["status"] for r in rows})),
            "sampled_structures": sum(bool(r.get("candidate_sampled")) for r in complete),
            "wall_median_p90_max_seconds": np.quantile(times, [.5, .9, 1.]).tolist() if times else None,
            "complete_wall_median_p90_max_seconds": np.quantile([r["wall_seconds"] for r in complete], [.5,.9,1.]).tolist() if complete else None,
            "max_peak_rss_kib": max((r.get("peak_rss_kib", 0) for r in rows), default=0),
            "total_cpu_seconds": sum(r.get("cpu_seconds", 0.) for r in rows),
            "candidate_mean_seconds": candidate_seconds / sample_count if sample_count else None,
            "candidate_median_p90_max_seconds": np.quantile(candidate_times, [.5,.9,1.]).tolist() if candidate_times else None,
            "projected_candidate_core_hours_complete_structures": projected / 3600,
            "true_sites": len(truth), "eta_evaluable": len(eta), "eta_positive_fraction": sum(e > cfg["eta_positive_tolerance"] for e in eta) / len(eta) if eta else None,
            "recall_1A": sum(s["error"] is not None and s["error"] <= match_near for s in truth) / len(truth) if truth else None,
            "recall_2A": sum(s["error"] is not None and s["error"] <= match_far for s in truth) / len(truth) if truth else None,
            "recall_clash_1A": sum(s["error_clash"] is not None and s["error_clash"] <= match_near for s in truth) / len(truth) if truth else None,
            "recall_clash_2A": sum(s["error_clash"] is not None and s["error_clash"] <= match_far for s in truth) / len(truth) if truth else None,
            "median_nearest_localization_error_A": float(np.median(errors)) if errors else None,
            "median_matched_1A_error_A": float(np.median([e for e in errors if e <= match_near])) if any(e <= match_near for e in errors) else None,
            "median_matched_2A_error_A": float(np.median([e for e in errors if e <= match_far])) if any(e <= match_far for e in errors) else None,
            "clusters_per_true_site": sum(r["clusters"] for r in complete) / len(truth) if truth else None,
            "clusters_clash_per_true_site": sum(r["clusters_clash"] for r in complete) / len(truth) if truth else None,
            "weighted_solves": sum(r["weighted_solves"] for r in complete),
            "certificate_failures": sum(r["certificate_failures"] for r in complete),
            "weighted_solver_seconds": sum(r["weighted_solver_seconds"] for r in complete)}
    ledger = json.loads((args.data / "downloads.json").read_text())
    measurements["downloads"] = {"bytes": sum(r["bytes"] for r in ledger), "complete_files": sum(r["status"] == "complete" for r in ledger),
                                 "search_json_disk_bytes": sum(p.stat().st_size for p in args.data.glob("search_*.json"))}
    measurements["solver_crosscheck"] = json.loads((args.data / "crosscheck.json").read_text())
    regional = selected_records(args.data / "regional", samples)
    regional_rows = []
    for record in regional:
        for metric in record.get("metrics", []):
            regional_rows.append({"pdb_id": record["pdb_id"], "cohort": record["cohort"],
                "scope": "seeded_region", "region_radius_A": record["region_radius_A"],
                "full_atoms": record["full_atoms"], "full_patches": record["full_patches"],
                "atoms": record["atoms"], "patches": record["patches"], "wall_seconds": record["wall_seconds"],
                "parse_seconds": record["parse_seconds"], "peak_rss_kib": record["peak_rss_kib"], **metric,
                "estimated_hard_fraction": metric["hard"] / metric["sampled"] if metric["sampled"] else None,
                "estimated_hard_fraction_after_clash": metric["hard_clash_pass"] / metric["clash_pass"] if metric["clash_pass"] else None})
    if regional_rows:
        write_csv(ROOT / "results/tables/exp09_metal_site_pilot_regional.csv", regional_rows)
    measurements["regional_calibration"] = regional_rows
    (ROOT / "results/reports/exp09_metal_site_pilot_measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    print(json.dumps(measurements, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["select", "download", "crosscheck", "run", "summarize", "worker"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--biolip", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pdb-id")
    parser.add_argument("--cohort", choices=["zn", "water", "mg"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--region-radius", type=float, default=0., help="separate RNA/Mg regional calibration radius")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.data.is_dir():
        parser.error("--data must be an existing directory outside the repository")
    if ROOT == args.data.resolve() or ROOT in args.data.resolve().parents:
        parser.error("raw data must stay outside the repository")
    if args.stage == "select":
        sample_ids(args.biolip, args.data, cfg)
    elif args.stage == "download":
        download(args.data, cfg)
    elif args.stage == "crosscheck":
        result = crosscheck(cfg["seed"], cfg["crosscheck_cases"])
        if result["max_abs_delta_tau"] > cfg["crosscheck_tolerance"]:
            raise RuntimeError(f"crosscheck mismatch: {result}")
        (args.data / "crosscheck.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    elif args.stage == "worker":
        worker(args, cfg)
    elif args.stage == "run":
        execute(args, cfg)
    else:
        summarize(args, cfg)


if __name__ == "__main__":
    main()
