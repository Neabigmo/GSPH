"""GSPH6 end-to-end extractor on the dual-resolution patch representation."""
from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from cspph.core.gsph1 import (
    Atom, ELEMENTS, GSPH1Config, LJ_EPSILON, LJ_SIGMA, RDF_EDGES, RESIDUE_CLASS_INDEX,
    _build_bond_graph, _chain_pair, _make_events, _side_atoms,
)
from cspph.chemistry.interaction_backend import build_interaction_pair_batch
from cspph.graph.patch_digraph import build_patch_transfer_graph, enumerate_patch_paths, summarize_path_tokens
from cspph.spectral.hodge import shared_hodge_readout
from cspph.chemistry.patch_builder import build_interaction_patches
from cspph.features.physical import aggregate_all_patches
from cspph.geometry.patch_support import compute_patch_support


@dataclass(frozen=True)
class GSPH6Config:
    event_cutoff: float = 5.0
    transfer_cutoff: float = 6.0
    local_radius: float = 4.0
    scales: Tuple[float, ...] = (0.0, 0.5, 2.0)
    max_events: int = -1
    max_paths_per_order: int = -1
    max_local_pairs_per_event: int = -1
    hodge_dense_max_dimension: int = 384
    hodge_max_eigenvalues: int = 32
    hodge_tol: float = 1e-8
    max_bond_transfer_distance: int = 3
    # Highest GLMY path order enumerated (0..3).  Lower values define a
    # sparser task-adapted complex; the math on that complex is unchanged.
    max_path_order: int = 3
    # eta_k statistics need one extra support solve per path subset and can
    # dominate runtime on dense complexes.  ``compute_eta=False`` is only
    # valid for ``*_sparse`` scenarios whose schema does not require the
    # eta keys; on standard scenarios the row correctly stays incomplete.
    compute_eta: bool = True
    support_max_combinations: Optional[int] = None
    mode: str = "benchmark"
    interaction_backend: str = "auto"
    gpu_min_pairs: int = 250000
    gpu_batch_pairs: int = 500000
    gpu_device: str = "cuda:0"
    use_cpp_path: bool = True
    # H3 intramolecular tasks (rna_quality / protein_stability): pairs closer
    # than this many covalent bonds are excluded from microcontact events.
    # 0 = legacy intermolecular behaviour (unchanged).
    intra_min_bond_sep: int = 0
    # Steric-clash floor forwarded to GSPH1Config (fraction of summed vdW
    # radii).  0 = disabled; intramolecular adapter enables it.
    contact_min_vdw_ratio: float = 0.0


def _path_tuple(row) -> Tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(row).tolist())


def _path_channel_weights(paths, patches, channel):
    patch_lookup = patches if isinstance(patches, Mapping) else {int(p.patch_id): p for p in patches}
    result = []
    for path in paths:
        vals = []
        for patch_id in path:
            patch = patch_lookup[int(patch_id)]
            key = {"rdf": "patch_rdf_mass_mean", "contact": "patch_contact_mass_mean", "lj": "patch_lj_abs_mean", "shape": "patch_shape_facing_mean"}[channel]
            vals.append(float(patch.physical.get(key, 0.0)))
        result.append(float(np.mean(vals)) if vals else 1.0)
    return result


def _glmy_features(chain) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for order in range(4):
        dim_a = float(len(chain.paths.get(order, []))); dim_o = float(chain.omega_dims[order])
        rank = float(chain.boundary_ranks[order]); next_rank = float(chain.boundary_ranks[order + 1] if order < 3 else 0)
        kernel = max(0.0, dim_o - rank)
        out[f"glmy_dim_A_{order}"] = dim_a; out[f"glmy_dim_Omega_{order}"] = dim_o
        out[f"glmy_dim_gap_{order}"] = max(0.0, dim_a - dim_o); out[f"glmy_rank_boundary_{order}"] = rank
        out[f"glmy_kernel_{order}"] = kernel; out[f"glmy_admissibility_{order}"] = dim_o / max(dim_a, 1.0)
        out[f"glmy_filling_{order}"] = next_rank / max(kernel, 1e-8); out[f"glmy_omega_truncated_{order}"] = 0.0
        if order <= 2: out[f"glmy_betti_{order}"] = float(chain.betti[order])
    for order, residual in chain.boundary_square_residual.items(): out[f"glmy_boundary_square_residual_{order}"] = float(residual)
    out["glmy_omega_truncated"] = 0.0; out["glmy_engine_sparse_exact"] = 1.0
    return out


def extract_gsph6_atoms(atoms: Sequence[Atom], scenario: str = "unknown", chain_a: Optional[str] = None, chain_b: Optional[str] = None, config: Optional[GSPH6Config] = None, static_bond_graph: Optional[Mapping[int, set]] = None, target_cache: Optional[object] = None, event_filter: Optional[object] = None) -> Dict[str, object]:
    cfg = config or GSPH6Config(); atoms = list(atoms); timing: Dict[str, float] = {}; started = time.perf_counter()
    cache_identity_valid = bool(target_cache is not None and (not hasattr(target_cache, "identity_matches") or target_cache.identity_matches(atoms)))
    # ``*_sparse`` is a schema/audit contract suffix, not a new entity type:
    # chain selection rules are identical to the base scenario.
    chain_scenario = scenario[:-7] if str(scenario).endswith("_sparse") else scenario
    selected_a, selected_b = _chain_pair(atoms, chain_scenario, chain_a, chain_b)
    pair_start = time.perf_counter()
    left, right = _side_atoms(atoms, selected_a), _side_atoms(atoms, selected_b)
    pair_batch = build_interaction_pair_batch(atoms, left, right, max(float(getattr(cfg, "local_radius", 10.0)), float(cfg.event_cutoff) + 2.0), ELEMENTS, RESIDUE_CLASS_INDEX, RDF_EDGES, LJ_SIGMA, LJ_EPSILON, backend=cfg.interaction_backend, gpu_min_pairs=cfg.gpu_min_pairs, gpu_batch_pairs=cfg.gpu_batch_pairs, gpu_device=cfg.gpu_device)
    event_stats: Dict[str, object] = {}
    legacy_cfg = GSPH1Config(
        event_cutoff=cfg.event_cutoff, transfer_cutoff=cfg.transfer_cutoff,
        local_radius=cfg.local_radius, scales=cfg.scales, max_events=cfg.max_events,
        max_paths_per_order=cfg.max_paths_per_order,
        max_local_pairs_per_event=cfg.max_local_pairs_per_event,
        max_hodge_dimension=-1, max_bond_transfer_distance=cfg.max_bond_transfer_distance,
        mode=cfg.mode, interaction_backend=cfg.interaction_backend,
        gpu_min_pairs=cfg.gpu_min_pairs, gpu_batch_pairs=cfg.gpu_batch_pairs,
        gpu_device=cfg.gpu_device, interaction_batch_size=16, support_workers=1,
        support_gpu_min_paths=64,
        support_gpu_batch_paths=int(os.environ.get("GSPH_SUPPORT_CPP_BATCH", "4096")),
        path_backend="python_stream", surface_backend="none", hodge_backend="cpu",
        intra_min_bond_sep=int(getattr(cfg, "intra_min_bond_sep", 0) or 0),
        contact_min_vdw_ratio=float(getattr(cfg, "contact_min_vdw_ratio", 0.0) or 0.0),
    )
    if target_cache is not None and hasattr(target_cache, "local_index_for_candidate"):
        cached_local_index = target_cache.local_index_for_candidate(atoms)
    else:
        cached_local_index = getattr(target_cache, "local_index", None) if target_cache is not None and bool(getattr(target_cache, "local_index_is_candidate", False)) else None
    # Intramolecular tasks need the covalent graph before event generation so
    # covalently linked pairs can be excluded from the microcontact set.  The
    # graph is reused afterwards for the relay graph -- it is built once.
    if int(getattr(cfg, "intra_min_bond_sep", 0) or 0) > 0 and static_bond_graph is None:
        if cache_identity_valid and target_cache is not None and hasattr(target_cache, "bond_graph"):
            static_bond_graph = dict(target_cache.bond_graph)
        else:
            static_bond_graph = _build_bond_graph(atoms)
    events, events_truncated = _make_events(atoms, selected_a, selected_b, legacy_cfg, event_stats, pair_batch, cached_local_index, static_bond_graph)
    timing["interaction_and_event_construction"] = time.perf_counter() - pair_start
    # GSPH-S: optional task-adaptive event selection.  The filter runs AFTER
    # the exact event builder and BEFORE patch construction, so the frozen
    # patch->graph->GLMY->persistence math is untouched; only the input event
    # set changes.  ``event_filter`` is a callable
    # ``(events, atoms) -> (kept_events, stats_dict)``.  Sparse results are
    # task-defined sparse GSPH and must never be reported as full-atom GSPH.
    event_filter_stats: Dict[str, object] = {}
    event_filter_name = ""
    if event_filter is not None:
        _t0 = time.perf_counter()
        n_events_before = len(events)
        events, event_filter_stats = event_filter(events, atoms)
        event_filter_stats = dict(event_filter_stats or {})
        event_filter_stats["events_before_filter"] = n_events_before
        event_filter_stats["events_after_filter"] = int(len(events))
        event_filter_name = str(event_filter_stats.get("filter") or getattr(event_filter, "__name__", "custom"))
        event_filter_stats["event_filter_seconds"] = time.perf_counter() - _t0
        timing["event_filter"] = event_filter_stats["event_filter_seconds"]
    patch_catalog = getattr(target_cache, "patch_catalog", None) if cache_identity_valid else None
    patch_start = time.perf_counter(); patches = build_interaction_patches(events, atoms, patch_catalog); physical_global = aggregate_all_patches(patches); timing["patch_construction_and_physical_aggregation"] = time.perf_counter() - patch_start
    if cache_identity_valid and target_cache is not None and hasattr(target_cache, "note_active_patches"):
        target_cache.note_active_patches([patch.patch_id for patch in patches])
    if static_bond_graph is not None:
        bond_graph = dict(static_bond_graph)
    elif cache_identity_valid and target_cache is not None and hasattr(target_cache, "bond_graph"):
        # Rigid-body decoys reuse the immutable target graph.  Coordinates and
        # all cross-chain physics still come from the candidate atoms above.
        bond_graph = dict(target_cache.bond_graph)
    else:
        bond_graph = _build_bond_graph(atoms)
    graph_start = time.perf_counter(); site_transfer_catalog = None
    if cache_identity_valid and target_cache is not None and hasattr(target_cache, "build_site_transfer_catalog"):
        site_transfer_catalog = target_cache.build_site_transfer_catalog(cfg.max_bond_transfer_distance)
    edges, witnesses = build_patch_transfer_graph(patches, bond_graph, cfg.max_bond_transfer_distance, site_transfer_catalog=site_transfer_catalog); timing["transfer_graph"] = time.perf_counter() - graph_start
    path_start = time.perf_counter(); cached_paths = None
    patch_ids = [p.patch_id for p in patches]
    max_order = int(getattr(cfg, "max_path_order", 3) or 3)
    if cache_identity_valid and target_cache is not None and hasattr(target_cache, "cached_paths"):
        cached_paths = target_cache.cached_paths(patch_ids, edges, max_order, cfg.use_cpp_path)
    if cached_paths is None:
        incremental_paths = target_cache.incremental_paths(patch_ids, edges, max_order) if cache_identity_valid and target_cache is not None and hasattr(target_cache, "incremental_paths") else None
        if incremental_paths is not None:
            path_arrays, path_counts, path_engine = incremental_paths
        else:
            path_arrays, path_counts, path_engine = enumerate_patch_paths(patch_ids, edges, max_order=max_order, use_cpp=cfg.use_cpp_path)
        if cache_identity_valid and target_cache is not None and hasattr(target_cache, "store_paths"):
            target_cache.store_paths(patch_ids, edges, max_order, cfg.use_cpp_path, (path_arrays, path_counts, path_engine))
    else:
        path_arrays, path_counts, path_engine = cached_paths
        path_arrays = {int(order): np.array(rows, copy=True) for order, rows in path_arrays.items()}
        path_counts = dict(path_counts); path_engine = f"{path_engine}:target_cache"
    if cache_identity_valid and target_cache is not None and hasattr(target_cache, "remember_paths"):
        target_cache.remember_paths(patch_ids, edges, (path_arrays, path_counts, path_engine))
    timing["path_enumeration"] = time.perf_counter() - path_start
    if cfg.max_paths_per_order >= 0:
        path_truncated = any(int(path_counts.get(o, 0)) > cfg.max_paths_per_order for o in range(4))
        path_arrays = {o: rows[:cfg.max_paths_per_order] for o, rows in path_arrays.items()}
    else: path_truncated = False
    # Canonical path order: C++ enumeration, incremental updates and cached
    # templates may return different row orders.  GLMY chain construction,
    # Hodge channel weights and persistence maps must all see ONE order.
    def _canon_sort(rows):
        arr = np.asarray(rows)
        if arr.ndim != 2 or arr.shape[0] <= 1:
            return arr
        view = np.ascontiguousarray(arr).view(
            np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).ravel()
        return arr[np.argsort(view, kind="stable")]
    path_arrays = {o: _canon_sort(rows) for o, rows in path_arrays.items()}
    patch_map = {p.patch_id: p for p in patches}; atom_map = {a.index: a for a in atoms}; support_start = time.perf_counter(); birth: Dict[Tuple[int, ...], object] = {}; support_by_set: Dict[Tuple[int, ...], object] = {}; support_incomplete = False; support_cache = {}; pair_support_cache = {}
    support_stats: Dict[str, object] = {
        "support_cache_hits": 0, "support_cache_misses": 0,
        "support_cpp_attempted": 0, "support_cpp_successes": 0,
        "support_cpp_seconds": 0.0, "support_fallbacks": 0,
        "support_solver_failures": 0, "support_certificate_failures": 0,
        "support_branch_nodes": 0, "support_pruned_nodes": 0,
        "eta_subset_cache_hits": 0, "eta_subset_cache_misses": 0,
        "support_solver_routes": Counter(),
    }

    def _valid_support(res) -> bool:
        return bool(res.complete and res.certificate_passed and np.isfinite(res.tau))

    # Support is computed once per UNIQUE patch set: Q_{P1} n Q_{P2} n Q_{P1}
    # equals Q_{P1} n Q_{P2}, so revisiting a patch cannot change tau.  This
    # is exact and also gives rigid decoys far more cache hits.
    canon_cache: Dict[Tuple[int, ...], object] = {}
    for order, rows in path_arrays.items():
        for row in rows:
            path = _path_tuple(row); canon = tuple(sorted(set(path)))
            warm = target_cache.warm_support(canon) if cache_identity_valid and target_cache is not None and hasattr(target_cache, "warm_support") else None
            if canon in canon_cache:
                result = canon_cache[canon]
            else:
                result = compute_patch_support([patch_map[i] for i in canon], atom_map, cfg.support_max_combinations, support_cache, support_stats, pair_support_cache, warm_start=warm)
                canon_cache[canon] = result
            birth[path] = result
            if cache_identity_valid and target_cache is not None and hasattr(target_cache, "remember_support"):
                target_cache.remember_support(canon, result)
            support_by_set.setdefault(canon, result); support_stats["support_branch_nodes"] = int(support_stats.get("support_branch_nodes", 0)) + int(result.explored); support_stats["support_pruned_nodes"] = int(support_stats.get("support_pruned_nodes", 0)) + int(result.pruned); support_incomplete = support_incomplete or not _valid_support(result)
            support_stats["support_solver_routes"][result.route] += 1
    timing["patch_support_search"] = time.perf_counter() - support_start
    # The cache is shared across all path orders.  The exact solver route is
    # retained in every result; timing is therefore auditable even when the
    # solver falls back from C++ to the independently certified reference.
    support_stats["support_pipeline_seconds"] = timing["patch_support_search"]
    feature_lifting_start = time.perf_counter()
    features: Dict[str, float] = dict(physical_global); features.update({"patch_count": float(len(patches)), "patch_directed_edge_count": float(len(edges))})
    # Path chemistry and geometry are scale-independent tokens.  They are
    # generated once here and reused by every filtration scale below.
    features.update(summarize_path_tokens(path_arrays, patch_map))
    # Explicit patch-support summaries.  Order k corresponds to k+1 patches:
    # tau_2/tau_3/tau_4 are never inferred from an anchor event.  Only VALID
    # support (complete AND certified) may enter a formal statistic -- an
    # uncertified solve is audit metadata, never a zero and never a tau.
    for order, label in ((1, "tau_2"), (2, "tau_3"), (3, "tau_4")):
        rows = path_arrays.get(order, [])
        values = [float(birth[_path_tuple(row)].tau) for row in rows if _valid_support(birth[_path_tuple(row)])]
        # Paths that exist but ALL failed certification yield NaN, never a
        # silent zero that would look like a real "no support" measurement.
        # NB: ``rows`` is a NumPy array -- ``not rows``/``bool(rows)`` raises
        # "truth value of an array is ambiguous"; always use len().
        n_rows = int(len(rows)) if rows is not None else 0
        features[f"{label}_mean"] = float(np.mean(values)) if values else (0.0 if n_rows == 0 else float("nan"))
        features[f"{label}_min"] = float(np.min(values)) if values else (0.0 if n_rows == 0 else float("nan"))
        features[f"{label}_max"] = float(np.max(values)) if values else (0.0 if n_rows == 0 else float("nan"))
    for order in range(4):
        rows = path_arrays.get(order, np.empty((0, order + 1), dtype=np.int32))
        taus = [float(birth[_path_tuple(row)].tau) for row in rows if _valid_support(birth[_path_tuple(row)])]
        n_rows = int(len(rows)) if rows is not None else 0
        features[f"path_count_k{order}"] = float(n_rows)
        features[f"path_support_mean_k{order}"] = float(np.mean(taus)) if taus else (0.0 if n_rows == 0 else float("nan"))
        features[f"path_support_max_k{order}"] = float(np.max(taus)) if taus else (0.0 if n_rows == 0 else float("nan"))
    # Fixed-length patch token: aggregate every patch without embedding patch
    # IDs into column names.  All microcontact information is already retained
    # inside each patch's physical summaries.
    patch_keys = sorted({key for patch in patches for key in patch.physical})
    for key in patch_keys:
        values = np.asarray([float(patch.physical.get(key, 0.0)) for patch in patches], dtype=float)
        safe = str(key).replace("patch_", "", 1)
        features[f"patch_token_{safe}_mean"] = float(values.mean()) if len(values) else 0.0
        features[f"patch_token_{safe}_std"] = float(values.std()) if len(values) else 0.0
        features[f"patch_token_{safe}_max"] = float(values.max()) if len(values) else 0.0
    subset_birth_cache: Dict[Tuple[int, ...], object] = dict(support_by_set)
    support_audit: Dict[str, int] = {"subset_solves": 0, "subset_invalid": 0,
                                     "eta_paths_skipped": 0}
    def _higher_increment(row, order):
        nonlocal support_incomplete
        path = _path_tuple(row)
        if not _valid_support(birth[path]):
            support_audit["eta_paths_skipped"] += 1
            return float("nan")
        high = float(birth[path].tau); low = []
        for size in (order,):
            for subset in combinations(path, size):
                subset_key = tuple(sorted(set(subset)))
                if subset_key in birth:
                    res = birth[subset_key]
                    support_stats["eta_subset_cache_hits"] = int(support_stats.get("eta_subset_cache_hits", 0)) + 1
                elif subset_key in subset_birth_cache:
                    res = subset_birth_cache[subset_key]
                    support_stats["eta_subset_cache_hits"] = int(support_stats.get("eta_subset_cache_hits", 0)) + 1
                else:
                    support_stats["eta_subset_cache_misses"] = int(support_stats.get("eta_subset_cache_misses", 0)) + 1
                    res = compute_patch_support([patch_map[i] for i in subset_key], atom_map, cfg.support_max_combinations, support_cache, support_stats, pair_support_cache)
                    subset_birth_cache[subset_key] = res
                    support_audit["subset_solves"] += 1
                if not _valid_support(res):
                    support_audit["subset_invalid"] += 1
                    support_incomplete = True
                    return float("nan")
                low.append(float(res.tau))
        return max(0.0, high - max(low, default=0.0))
    eta3_map, eta4_map = {}, {}
    if not bool(getattr(cfg, "compute_eta", True)):
        # Sparse-scenario mode: eta statistics are omitted entirely (never
        # zero-filled).  ``*_sparse`` manifests do not require these keys.
        support_audit["eta_skipped"] = True
    elif cfg.support_max_combinations == 0:
        features["eta_3_mean"] = 0.0; features["eta_4_mean"] = 0.0
    else:
        eta3 = []
        for _row in path_arrays.get(2, []):  # noqa: F811
            _v = _higher_increment(_row, 2)
            if np.isfinite(_v):
                eta3.append(_v); eta3_map[_path_tuple(_row)] = float(_v)
        eta4 = []
        for _row in path_arrays.get(3, []):
            _v = _higher_increment(_row, 3)
            if np.isfinite(_v):
                eta4.append(_v); eta4_map[_path_tuple(_row)] = float(_v)
        features["eta_3_mean"] = float(np.mean(eta3)) if eta3 else (0.0 if len(path_arrays.get(2, [])) == 0 else float("nan"))
        features["eta_4_mean"] = float(np.mean(eta4)) if eta4 else (0.0 if len(path_arrays.get(3, [])) == 0 else float("nan"))
    # Delta-tau is the path support minus the strongest one-patch support.
    singleton = [float(birth[_path_tuple(row)].tau) for row in path_arrays.get(0, []) if _valid_support(birth[_path_tuple(row)])]
    singleton_max = max(singleton, default=0.0)
    for order, label in ((1, "delta_tau_2"), (2, "delta_tau_3"), (3, "delta_tau_4")):
        values = [max(0.0, float(birth[_path_tuple(row)].tau) - singleton_max) for row in path_arrays.get(order, []) if _valid_support(birth[_path_tuple(row)])]
        features[f"{label}_mean"] = float(np.mean(values)) if values else (0.0 if len(path_arrays.get(order, [])) == 0 else float("nan"))
    support_audit["paths_total"] = int(sum(len(rows) for rows in path_arrays.values()))
    support_audit["paths_invalid"] = int(sum(1 for rows in path_arrays.values() for row in rows if not _valid_support(birth[_path_tuple(row)])))
    timing["feature_lifting"] = time.perf_counter() - feature_lifting_start
    chains = []; hodge_start = time.perf_counter(); hodge_incomplete = False
    from cspph.topology.glmy_sparse import sparse_chain
    def _active_paths(scale):
        return {o: [_path_tuple(row) for row in path_arrays.get(o, np.empty((0, o + 1), dtype=np.int32))
                    if float(birth[_path_tuple(row)].tau) <= float(scale) and _valid_support(birth[_path_tuple(row)])]
                for o in range(4)}
    for scale in cfg.scales:
        active = _active_paths(scale)
        chain = target_cache.cached_chain(active) if cache_identity_valid and target_cache is not None and hasattr(target_cache, "cached_chain") else None
        if chain is None:
            chain = sparse_chain(active)
            if cache_identity_valid and target_cache is not None and hasattr(target_cache, "remember_chain"):
                target_cache.remember_chain(active, chain)
        chains.append(chain); prefix = f"scale_{float(scale):g}_"; features.update({prefix + k: v for k, v in _glmy_features(chain).items()})
        # Shared base spectrum is candidate-invariant for a fixed chain and
        # is cached on the target; only the four channel projections are
        # recomputed per candidate.
        ckey = tuple(tuple(sorted(tuple(int(x) for x in p) for p in active.get(o, ()))) for o in range(4))
        channel_weights = {channel: {o: _path_channel_weights(active[o], patches, channel) for o in range(4)} for channel in ("rdf", "contact", "lj", "shape")}
        spectra = {}
        spectra_valid = True
        if cache_identity_valid and target_cache is not None and hasattr(target_cache, "cached_hodge"):
            spectra = target_cache.cached_hodge(ckey) or {}
        for order in range(3):
            spec = spectra.get(order)
            if spec is None:
                from cspph.spectral.hodge import hodge_base_spectrum
                spec = hodge_base_spectrum(chain, order, dense_limit=cfg.hodge_dense_max_dimension, max_eigenvalues=cfg.hodge_max_eigenvalues, tol=cfg.hodge_tol)
                spectra[order] = spec
            spectra_valid = spectra_valid and bool(spec.get("converged", False)) or int(chain.omega_dims.get(order, 0)) == 0
        if cache_identity_valid and target_cache is not None and hasattr(target_cache, "remember_hodge") and spectra:
            target_cache.remember_hodge(ckey, spectra)
        from cspph.spectral.hodge import hodge_channel_readout
        readout = hodge_channel_readout(chain, channel_weights, spectra=spectra, dense_limit=cfg.hodge_dense_max_dimension)
        features.update({prefix + k: float(v) for k, v in readout.items()}); hodge_incomplete = hodge_incomplete or any(str(k).endswith("converged") and float(v) == 0.0 for k, v in readout.items() if "shared_hodge_" in k) or not spectra_valid
    timing["shared_hodge_and_glmy"] = time.perf_counter() - hodge_start
    persistence_start = time.perf_counter(); persistence_valid = False; persistence_error = ""; persistence_details: Dict[str, object] = {}
    try:
        from cspph.topology.persistence import build_omega_boundary_filtration, build_omega_inclusion_chain_maps, persistent_intervals_from_filtration, validate_chain_maps
        # Same validity rule as the GLMY activation above: a path enters the
        # filtration only with complete AND certified support.
        paths_by_time = [_active_paths(scale) for scale in cfg.scales]
        # Reuse validated chain maps when the identical active-path complex
        # was already persisted for this target (rigid decoys often share it).
        pkey = tuple(tuple(tuple(sorted(tuple(int(x) for x in p) for p in active.get(o, ())))
                          for o in range(4)) for active in paths_by_time)
        cached_pm = target_cache.cached_persistence(pkey) if cache_identity_valid and target_cache is not None and hasattr(target_cache, "cached_persistence") else None
        if cached_pm is not None:
            filtration, maps, check = cached_pm
        else:
            filtration = build_omega_boundary_filtration(chains)
            maps = build_omega_inclusion_chain_maps(chains, paths_by_time)
            check = validate_chain_maps(filtration, maps)
            if cache_identity_valid and target_cache is not None and hasattr(target_cache, "remember_persistence"):
                target_cache.remember_persistence(pkey, (filtration, maps, check))
        persistence_details = {"scales": [float(s) for s in cfg.scales], "orders": {}}
        features["persistence_max_chain_residual"] = float(check.get("max_residual", 0.0))
        # persistence_valid is set ONLY after chain-map validation AND all
        # interval computations succeed; any exception leaves it False.
        if bool(check.get("valid", False)):
            ok_all = True
            for order in (0, 1, 2):
                intervals = persistent_intervals_from_filtration(filtration, maps, order)
                features[f"persistence_interval_count_k{order}"] = float(len(intervals))
                norm = []
                for iv in intervals:
                    if isinstance(iv, Mapping):
                        norm.append([int(iv.get("birth", 0)),
                                     int(iv.get("death", 0)),
                                     bool(iv.get("right_censored", False))])
                    else:
                        norm.append([int(iv[0]), int(iv[1]),
                                     bool(iv[2]) if len(iv) > 2 else False])
                persistence_details["orders"][order] = {"intervals": norm}
            # H3 is intentionally not computed: there is no partial_4, so a
            # rank-invariant H3 would be a pseudo-result.  Record the omission
            # explicitly instead of emitting zero intervals.
            persistence_details["orders"][3] = {"intervals": [],
                                                "skipped": "no_boundary_4_by_design"}
            persistence_valid = ok_all
    except Exception as exc:
        import traceback as _tb
        persistence_error = f"{type(exc).__name__}: {exc} | {_tb.format_exc().splitlines()[-3:]}"
    timing["persistence"] = time.perf_counter() - persistence_start; timing["serialization"] = 0.0
    # Validate only after every structural block (Hodge and persistence
    # included) has been materialized.  Validating before those blocks would
    # make a complete candidate look schema-incomplete.
    from cspph.io.schema import load_feature_manifest, validate_feature_record
    try:
        schema = load_feature_manifest(scenario)
        # A missing manifest key is not always a failed computation: an
        # interface with no microcontacts has an EMPTY patch/path complex,
        # whose aggregate features are defined structural zeros.  Fill only
        # keys the manifest explicitly marks structural-zero-safe; failed
        # support/Hodge/persistence flags still make the row incomplete.
        structural_zero_filled = sorted(
            name for name, spec in schema.features.items()
            if name not in features and spec.structural_zero_allowed)
        for name in structural_zero_filled:
            features[name] = 0.0
        schema_audit = validate_feature_record(features, scenario, schema=schema)
        schema_audit["structural_zero_filled_count"] = len(structural_zero_filled)
    except Exception as exc:
        schema_audit = validate_feature_record(features, scenario)
        schema_audit.update({"valid": False, "manifest_error": f"{type(exc).__name__}: {exc}"})
    # GSPH6_SCHEMA_REGEN=1 rewrites the checked-in manifest from the observed
    # feature set.  This is the ONLY sanctioned way to add/remove feature
    # keys; normal runs must match the frozen manifest exactly.
    import os as _os
    if _os.environ.get("GSPH6_SCHEMA_REGEN") == "1" and not schema_audit.get("valid"):
        try:
            from cspph.io.schema import schema_from_features
            from pathlib import Path as _Path
            manifest = schema_from_features(scenario, features)
            import json as _json
            out_path = _Path(__file__).resolve().parents[1] / "schemas" / f"{scenario}_v6_features.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(_json.dumps({"scenario": scenario, "version": "gsph6.v1",
                "features": {n: {"name": n, "dtype": "float64", "unit": "dimensionless",
                                 "block": n.split("_", 1)[0], "definition": "fixed GSPH6 structural feature",
                                 "structural_zero_allowed": True, "missing_allowed": False}
                             for n in sorted(manifest.features)}}, indent=1) + "\n", encoding="utf-8")
            schema_audit = validate_feature_record(features, scenario, schema=manifest)
            schema_audit["schema_regenerated"] = str(out_path)
        except Exception as exc2:
            schema_audit.update({"valid": False, "regen_error": f"{type(exc2).__name__}: {exc2}"})
    timing["support_solver"] = float(support_stats.get("support_solver_seconds", 0.0))
    timing["cache_lookup"] = 0.0
    incomplete = bool(events_truncated or path_truncated or support_incomplete or hodge_incomplete or not persistence_valid or not schema_audit["valid"])
    metadata = {"schema_version": "gsph6.v1", "feature_schema_id": schema_audit["schema_id"], "schema_audit": schema_audit, "scenario": scenario, "chain_a": selected_a, "chain_b": selected_b, "status": "incomplete" if incomplete else ("empty_events" if not events else "complete"), "topology_valid": not incomplete, "events_truncated": events_truncated, "paths_truncated": path_truncated, "support_incomplete": support_incomplete, "hodge_incomplete": hodge_incomplete, "persistence_valid": persistence_valid, "persistence_error": persistence_error, "path_engine": path_engine, "glmy_kernel_backend": getattr(chain, "kernel_backend", "structured_sparse"), "directed_edge_count": len(edges), "patch_count": len(patches), "target_cache_identity_valid": cache_identity_valid, "stable_patch_catalog_size": int(len(getattr(target_cache, "patch_catalog", ()))) if getattr(target_cache, "patch_catalog", None) is not None and cache_identity_valid else 0, "support_solver_routes": dict(support_stats["support_solver_routes"]), "support_fallback_used": bool(support_stats["support_fallbacks"]), "support_certificate_passed": not bool(support_stats["support_solver_failures"]), "support_stats": {k: v for k, v in support_stats.items() if k != "support_solver_routes"}, "target_cache_reuse": dict(getattr(target_cache, "reuse_stats", {}) or {}), "timing": {**timing, "total": time.perf_counter() - started}, "hodge_dense_max_dimension": cfg.hodge_dense_max_dimension, "hodge_method": "shared_glmy_hodge_shared_basis", "feature_count": len(features),
            "event_filter": event_filter_name or None,
            "event_filter_stats": event_filter_stats or None,
            "event_stats": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in event_stats.items()},
            "sparse_gsph": bool(event_filter is not None),
            "max_path_order": int(max_order)}
    return {"metadata": metadata, "features": features, "patches": patches, "transfer_witnesses": witnesses,
            "persistence": persistence_details, "filtration_scales": [float(s) for s in cfg.scales],
            "support_audit": support_audit,
            "path_support": {p: {"tau": float(s.tau), "valid": bool(_valid_support(s))}
                             for p, s in birth.items()},
            "path_eta": {"3": eta3_map, "4": eta4_map},
            # Serialization-only additions for downstream filtration audits:
            # directed transfer edges (u, v, min_bond_distance) and every
            # patch-SET support actually solved (None if not certified).
            "edges": [[int(u), int(v), float(w)] for u, v, w in edges],
            "support_by_set": {tuple(int(x) for x in k):
                               (float(v.tau) if _valid_support(v) else None)
                               for k, v in {**subset_birth_cache,
                                            **support_by_set}.items()}}


__all__ = ["GSPH6Config", "extract_gsph6_atoms"]
