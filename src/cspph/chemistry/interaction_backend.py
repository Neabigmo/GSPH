"""Global interface-pair preparation and fused interaction statistics.

The extractor used to rebuild a local dense distance matrix for every typed
event.  This module makes the interface pair table a first-class object: a
single KD-tree pass produces candidate pairs, then one vectorized CPU pass (or
an optional CUDA/PyTorch pass) computes distances and fixed element-class
LJ-like values.  GPU distances are used for ordinary rows while float64 CPU
values remain the reference around thresholds and histogram edges.

The CUDA path is intentionally optional.  It is selected only when a CUDA
enabled PyTorch is already present and ``backend='auto'``/``'cuda'`` permits
it.  No CUDA toolkit or custom extension is required for the portable path;
custom kernels can be added behind this interface later without changing the
GSPH feature contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class InteractionPairBatch:
    """Compact interface pair table shared by event and local-feature code."""

    left_indices: np.ndarray
    right_indices: np.ndarray
    distances: np.ndarray
    left_element_codes: np.ndarray
    right_element_codes: np.ndarray
    left_residue_codes: np.ndarray
    right_residue_codes: np.ndarray
    rdf_bins: np.ndarray
    lj_attraction: np.ndarray
    lj_repulsion: np.ndarray
    pair_cutoff: float
    backend: str
    gpu_available: bool
    cpu_build_seconds: float
    gpu_compute_seconds: float = 0.0
    transfer_seconds: float = 0.0
    gpu_max_distance_error: float = 0.0
    threshold_rechecks: int = 0
    backend_reason: str = ""
    same_side: bool = False
    _rows_by_left: Dict[int, Tuple[int, ...]] = field(default_factory=dict, repr=False)
    _rows_by_right: Dict[int, Tuple[int, ...]] = field(default_factory=dict, repr=False)
    _row_left_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)
    _row_right_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)

    @property
    def size(self) -> int:
        return int(self.distances.size)

    def rows_for_event(self, donor_index: int, acceptor_index: int,
                       donor_side_ids: Sequence[int], acceptor_side_ids: Sequence[int],
                       local_radius: float, atoms: Mapping[int, object]) -> Tuple[np.ndarray, bool]:
        """Return rows near one directed event.

        The pair table is stored in side-A -> side-B orientation.  The boolean
        says whether the event direction matches that orientation.  Selecting
        through per-atom row lists avoids a second dense distance matrix.
        """
        donor_ids = {int(value) for value in donor_side_ids}
        acceptor_ids = {int(value) for value in acceptor_side_ids}
        donor = atoms[int(donor_index)]
        acceptor = atoms[int(acceptor_index)]

        # A->B orientation: donor is table-left, acceptor table-right.
        # The table orientation is encoded by the row indexes, not by the
        # caller's local donor/acceptor sets.  This matters for B->A events:
        # their donor is a table-right atom and their acceptor a table-left
        # atom.
        forward = (int(donor_index) in self._rows_by_left and
                   int(acceptor_index) in self._rows_by_right)
        if forward:
            cand_parts = [np.asarray(self._rows_by_left[atom_id], dtype=np.int64)
                          for atom_id in donor_ids if atom_id in self._rows_by_left]
            if not cand_parts:
                return np.zeros(0, dtype=np.int64), True
            cand = np.concatenate(cand_parts)
            acceptor_arr = np.fromiter(acceptor_ids, dtype=np.int64)
            keep = np.isin(self.right_indices[cand], acceptor_arr)
            keep &= np.linalg.norm(self._row_left_xyz[cand] - donor.xyz, axis=1) <= local_radius + 1e-12
            keep &= np.linalg.norm(self._row_right_xyz[cand] - acceptor.xyz, axis=1) <= local_radius + 1e-12
            return np.unique(cand[keep]), True

        # B->A orientation: table-left is the event acceptor and table-right
        # is the event donor; callers swap the type/residue channels.
        cand_parts = [np.asarray(self._rows_by_left[atom_id], dtype=np.int64)
                      for atom_id in acceptor_ids if atom_id in self._rows_by_left]
        if not cand_parts:
            return np.zeros(0, dtype=np.int64), False
        cand = np.concatenate(cand_parts)
        donor_arr = np.fromiter(donor_ids, dtype=np.int64)
        keep = np.isin(self.right_indices[cand], donor_arr)
        keep &= np.linalg.norm(self._row_left_xyz[cand] - acceptor.xyz, axis=1) <= local_radius + 1e-12
        keep &= np.linalg.norm(self._row_right_xyz[cand] - donor.xyz, axis=1) <= local_radius + 1e-12
        return np.unique(cand[keep]), False


def _element_code(element: str, elements: Sequence[str]) -> int:
    try:
        return int(elements.index(element))
    except ValueError:
        return int(len(elements) - 1)


def _pair_candidates(left_atoms: Sequence[object], right_atoms: Sequence[object], cutoff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build all cross-side pairs within ``cutoff`` without an NxM matrix."""
    started = perf_counter()
    if not left_atoms or not right_atoms:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    left_xyz = np.asarray([atom.xyz for atom in left_atoms], dtype=np.float64)
    right_xyz = np.asarray([atom.xyz for atom in right_atoms], dtype=np.float64)
    part_i: List[np.ndarray] = []
    part_j: List[np.ndarray] = []
    part_d: List[np.ndarray] = []
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(right_xyz)
        for i, neighbours in enumerate(tree.query_ball_point(left_xyz, float(cutoff))):
            if not neighbours:
                continue
            nb = np.asarray(neighbours, dtype=np.int64)
            delta = right_xyz[nb] - left_xyz[i]
            distances = np.linalg.norm(delta, axis=1)
            keep = distances <= cutoff + 1e-12
            nb = nb[keep]; distances = distances[keep]
            part_i.append(np.full(nb.size, i, dtype=np.int64))
            part_j.append(nb); part_d.append(distances)
    except Exception:
        # Deterministic chunked fallback; it never materializes the full
        # interface matrix, even when SciPy is unavailable.
        block = 4096
        for start in range(0, len(left_xyz), block):
            chunk = left_xyz[start:start + block]
            distances = np.linalg.norm(chunk[:, None, :] - right_xyz[None, :, :], axis=2)
            li, ri = np.where(distances <= cutoff + 1e-12)
            part_i.append(start + li.astype(np.int64))
            part_j.append(ri.astype(np.int64))
            part_d.append(distances[li, ri].astype(np.float64))
    if not part_i:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    li_all = np.concatenate(part_i); ri_all = np.concatenate(part_j); d_all = np.concatenate(part_d)
    order = np.lexsort((ri_all, li_all, d_all))
    return (
        li_all[order].astype(np.int32),
        ri_all[order].astype(np.int32),
        d_all[order],
    )


def _torch_cuda_values(left_xyz: np.ndarray, right_xyz: np.ndarray,
                       sigma: np.ndarray, epsilon: np.ndarray,
                       batch_size: int, device_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool, str]:
    """Compute distances/LJ values with optional CUDA PyTorch.

    The return values include GPU compute time, transfer time, maximum
    distance discrepancy against CPU float64 and availability.  Importing
    PyTorch is lazy so CPU-only installations remain lightweight.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.device(device_name)
    except Exception:
        return (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32), 0.0, 0.0, False, "cuda_unavailable")

    extension = None
    backend_name = "torch_cuda"
    try:
        try:
            from cspph.chemistry.cuda_interactions import load_fused_interactions
        except ImportError:  # pragma: no cover - direct script execution
            from cspph.chemistry.cuda_interactions import load_fused_interactions
        extension = load_fused_interactions()
        backend_name = "cuda_extension"
    except Exception:
        # A CUDA-enabled PyTorch remains a valid fallback when nvcc or an
        # extension compiler is unavailable.  The selected backend is still
        # reported so performance audits cannot mistake it for the extension.
        extension = None
        backend_name = "torch_cuda"

    distances = np.empty(len(left_xyz), dtype=np.float32)
    attractions = np.empty(len(left_xyz), dtype=np.float32)
    repulsions = np.empty(len(left_xyz), dtype=np.float32)
    transfer_seconds = 0.0
    started = perf_counter()
    with torch.inference_mode():
        for start in range(0, len(left_xyz), max(1, int(batch_size))):
            stop = min(len(left_xyz), start + max(1, int(batch_size)))
            t0 = perf_counter()
            x = torch.as_tensor(left_xyz[start:stop], dtype=torch.float32, device=device)
            y = torch.as_tensor(right_xyz[start:stop], dtype=torch.float32, device=device)
            s = torch.as_tensor(sigma[start:stop], dtype=torch.float32, device=device)
            e = torch.as_tensor(epsilon[start:stop], dtype=torch.float32, device=device)
            transfer_seconds += perf_counter() - t0
            if extension is not None:
                d, attraction_values, repulsion_values = extension.fused_interactions(x, y, s, e)
            else:
                d = torch.linalg.vector_norm(x - y, dim=1).clamp_min(1e-6)
                ratio = s / d.clamp_min(0.8)
                energy = 4.0 * e * (ratio.pow(12) - ratio.pow(6))
                attraction_values = torch.minimum(energy, torch.zeros_like(energy))
                repulsion_values = torch.maximum(energy, torch.zeros_like(energy))
            t1 = perf_counter()
            distances[start:stop] = d.detach().cpu().numpy()
            attractions[start:stop] = attraction_values.detach().cpu().numpy()
            repulsions[start:stop] = repulsion_values.detach().cpu().numpy()
            transfer_seconds += perf_counter() - t1
    return distances, attractions, repulsions, perf_counter() - started, transfer_seconds, True, backend_name


def build_interaction_pair_batch(atoms: Sequence[object], left_atoms: Sequence[object],
                                 right_atoms: Sequence[object], pair_cutoff: float,
                                 elements: Sequence[str], residue_class_index: Mapping[str, int],
                                 rdf_edges: np.ndarray, lj_sigma: Mapping[str, float],
                                 lj_epsilon: Mapping[str, float], backend: str = "auto",
                                 gpu_min_pairs: int = 250_000, gpu_batch_pairs: int = 500_000,
                                 gpu_device: str = "cuda:0") -> InteractionPairBatch:
    """Build one globally shared, fused interaction table."""
    started = perf_counter()
    li_local, ri_local, cpu_distances = _pair_candidates(left_atoms, right_atoms, pair_cutoff)
    li = li_local.astype(np.int64, copy=False)
    ri = ri_local.astype(np.int64, copy=False)
    # Per-atom vectors are computed once, then gathered by pair row.  Every
    # value is identical to the previous per-pair Python lookups.
    left_index_arr = np.asarray([atom.index for atom in left_atoms], dtype=np.int32)
    right_index_arr = np.asarray([atom.index for atom in right_atoms], dtype=np.int32)
    # Intramolecular tasks pass the SAME molecule on both sides.  The query
    # then yields each unordered pair twice ((i,j) and (j,i)) plus self-pairs
    # (i,i).  Keep only the canonical li<ri rows so the fused table still
    # represents each physical pair exactly once; event direction still comes
    # from donor/acceptor roles, not row orientation.
    same_side = bool(left_index_arr.size == right_index_arr.size
                     and np.array_equal(left_index_arr, right_index_arr))
    if same_side and cpu_distances.size:
        keep = li < ri
        li, ri, cpu_distances = li[keep], ri[keep], cpu_distances[keep]
    n = int(cpu_distances.size)
    left_elem_arr = np.asarray([_element_code(atom.element, elements) for atom in left_atoms], dtype=np.int16)
    right_elem_arr = np.asarray([_element_code(atom.element, elements) for atom in right_atoms], dtype=np.int16)
    left_res_arr = np.asarray([residue_class_index.get(str(atom.resname), 0) for atom in left_atoms], dtype=np.int16)
    right_res_arr = np.asarray([residue_class_index.get(str(atom.resname), 0) for atom in right_atoms], dtype=np.int16)
    left_indices = left_index_arr[li]
    right_indices = right_index_arr[ri]
    left_elements = left_elem_arr[li]
    right_elements = right_elem_arr[ri]
    left_residue = left_res_arr[li]
    right_residue = right_res_arr[ri]

    # CPU float64 values are retained for threshold/bin-edge audits.  When the
    # GPU path is selected, its distances become the effective values for
    # ordinary rows; only pairs near a scientific cutoff/bin edge keep the CPU
    # reference so roundoff cannot change membership.
    xyz_left_arr = np.asarray([atom.xyz for atom in left_atoms], dtype=np.float64)
    xyz_right_arr = np.asarray([atom.xyz for atom in right_atoms], dtype=np.float64)
    left_xyz = xyz_left_arr[li] if n else np.zeros((0, 3), dtype=np.float64)
    right_xyz = xyz_right_arr[ri] if n else np.zeros((0, 3), dtype=np.float64)
    sig_left = np.asarray([lj_sigma[atom.element] for atom in left_atoms], dtype=np.float64)
    sig_right = np.asarray([lj_sigma[atom.element] for atom in right_atoms], dtype=np.float64)
    eps_left = np.asarray([lj_epsilon[atom.element] for atom in left_atoms], dtype=np.float64)
    eps_right = np.asarray([lj_epsilon[atom.element] for atom in right_atoms], dtype=np.float64)
    sigma = np.sqrt(sig_left[li] * sig_right[ri])
    epsilon = np.sqrt(eps_left[li] * eps_right[ri])

    selected_backend = "cpu_numpy"
    backend_reason = "forced_cpu" if backend == "cpu" else "cuda_unavailable_or_below_threshold"
    gpu_available = False
    gpu_seconds = 0.0
    transfer_seconds = 0.0
    gpu_max_error = 0.0
    effective_distances = np.asarray(cpu_distances, dtype=np.float64).copy()
    if n and backend in {"auto", "cuda"} and (backend == "cuda" or n >= int(gpu_min_pairs)):
        gpu_dist, gpu_att, gpu_rep, gpu_seconds, transfer_seconds, gpu_available, gpu_backend_name = _torch_cuda_values(
            left_xyz, right_xyz, sigma, epsilon, int(gpu_batch_pairs), gpu_device
        )
        if gpu_available:
            selected_backend = gpu_backend_name
            backend_reason = gpu_backend_name
            gpu_max_error = float(np.max(np.abs(gpu_dist.astype(np.float64) - cpu_distances))) if n else 0.0
            gpu_distances = gpu_dist.astype(np.float64, copy=False)
            edge_distances = np.min(np.abs(cpu_distances[:, None] - np.asarray(rdf_edges, dtype=np.float64)[None, :]), axis=1) if n else np.zeros(0)
            safe = (np.abs(gpu_distances - cpu_distances) <= 1e-5)
            safe &= np.abs(cpu_distances - float(pair_cutoff)) > 1e-5
            safe &= edge_distances > 1e-6
            effective_distances[safe] = gpu_distances[safe]
            lj_att = gpu_att.astype(np.float64)
            lj_rep = gpu_rep.astype(np.float64)
        else:
            backend_reason = "cuda_unavailable"
            lj_att = np.minimum(4.0 * epsilon * (sigma / np.maximum(cpu_distances, 0.8)) ** 12 -
                                4.0 * epsilon * (sigma / np.maximum(cpu_distances, 0.8)) ** 6, 0.0)
            energy = 4.0 * epsilon * ((sigma / np.maximum(cpu_distances, 0.8)) ** 12 -
                                      (sigma / np.maximum(cpu_distances, 0.8)) ** 6)
            lj_rep = np.maximum(energy, 0.0)
    else:
        if backend == "auto" and n >= int(gpu_min_pairs):
            backend_reason = "cuda_unavailable"
        elif backend == "auto":
            backend_reason = "below_gpu_min_pairs"
        ratio = sigma / np.maximum(cpu_distances, 0.8)
        energy = 4.0 * epsilon * (ratio ** 12 - ratio ** 6)
        lj_att = np.minimum(energy, 0.0)
        lj_rep = np.maximum(energy, 0.0)

    # Histogram binning is done from CPU float64 distances so a GPU roundoff
    # cannot move a pair across a scientific threshold.  Near an edge we mark
    # a CPU recheck explicitly for audit.
    rdf_bins = np.searchsorted(np.asarray(rdf_edges, dtype=np.float64), cpu_distances, side="right") - 1
    near_edges = np.min(np.abs(cpu_distances[:, None] - np.asarray(rdf_edges)[None, :]), axis=1) <= 1e-6 if n else np.zeros(0, dtype=bool)
    threshold_rechecks = int(np.count_nonzero(near_edges))

    rows_by_left: Dict[int, List[int]] = {}
    rows_by_right: Dict[int, List[int]] = {}
    for row, (left_index, right_index) in enumerate(zip(left_indices, right_indices)):
        rows_by_left.setdefault(int(left_index), []).append(row)
        rows_by_right.setdefault(int(right_index), []).append(row)
    elapsed = perf_counter() - started
    return InteractionPairBatch(
        left_indices=left_indices,
        right_indices=right_indices,
        distances=effective_distances,
        left_element_codes=left_elements,
        right_element_codes=right_elements,
        left_residue_codes=left_residue,
        right_residue_codes=right_residue,
        rdf_bins=rdf_bins.astype(np.int16),
        lj_attraction=np.asarray(lj_att, dtype=np.float64),
        lj_repulsion=np.asarray(lj_rep, dtype=np.float64),
        pair_cutoff=float(pair_cutoff),
        backend=selected_backend,
        gpu_available=bool(gpu_available),
        cpu_build_seconds=float(elapsed - gpu_seconds),
        gpu_compute_seconds=float(gpu_seconds),
        transfer_seconds=float(transfer_seconds),
        gpu_max_distance_error=float(gpu_max_error),
        threshold_rechecks=threshold_rechecks,
        backend_reason=backend_reason,
        same_side=bool(same_side),
        _rows_by_left={key: tuple(value) for key, value in rows_by_left.items()},
        _rows_by_right={key: tuple(value) for key, value in rows_by_right.items()},
        _row_left_xyz=left_xyz,
        _row_right_xyz=right_xyz,
    )


def batch_gpu_interactions(batch_specs: Sequence[Tuple[InteractionPairBatch, Sequence[object]]],
                           elements: Sequence[str], lj_sigma: Mapping[str, float],
                           lj_epsilon: Mapping[str, float], rdf_edges: np.ndarray,
                           pair_cutoff: float, gpu_batch_pairs: int = 500000,
                           gpu_device: str = "cuda:0") -> Dict[str, object]:
    """Apply one fused GPU distance/LJ pass to several same-target candidates.

    Each pair table remains isolated by an offset in the concatenated arrays;
    outputs are split back into their original ``InteractionPairBatch``
    objects, so no event or feature can cross candidate boundaries.  CPU-built
    pair tables remain the deterministic fallback when CUDA is unavailable.
    """
    if not batch_specs:
        return {"available": False, "backend": "none", "pairs": 0, "seconds": 0.0,
                "transfer_seconds": 0.0, "max_distance_error": 0.0}
    left_rows = []; right_rows = []; sigmas = []; epsilons = []; offsets = [0]
    for pair_batch, atoms in batch_specs:
        by_index = {int(atom.index): atom for atom in atoms}
        for left_index, right_index in zip(pair_batch.left_indices, pair_batch.right_indices):
            left = by_index[int(left_index)]; right = by_index[int(right_index)]
            left_rows.append(np.asarray(left.xyz, dtype=np.float64))
            right_rows.append(np.asarray(right.xyz, dtype=np.float64))
            sigmas.append(np.sqrt(lj_sigma[left.element] * lj_sigma[right.element]))
            epsilons.append(np.sqrt(lj_epsilon[left.element] * lj_epsilon[right.element]))
        offsets.append(offsets[-1] + pair_batch.size)
    total = offsets[-1]
    if total == 0:
        return {"available": False, "backend": "empty", "pairs": 0, "seconds": 0.0,
                "transfer_seconds": 0.0, "max_distance_error": 0.0}
    gpu_dist, gpu_att, gpu_rep, gpu_seconds, transfer_seconds, available, backend_name = _torch_cuda_values(
        np.asarray(left_rows, dtype=np.float64), np.asarray(right_rows, dtype=np.float64),
        np.asarray(sigmas, dtype=np.float64), np.asarray(epsilons, dtype=np.float64),
        int(gpu_batch_pairs), gpu_device,
    )
    if not available:
        return {"available": False, "backend": backend_name, "pairs": int(total),
                "seconds": float(gpu_seconds), "transfer_seconds": float(transfer_seconds),
                "max_distance_error": 0.0}
    edges = np.asarray(rdf_edges, dtype=np.float64)
    max_error = 0.0
    for index, (pair_batch, atoms) in enumerate(batch_specs):
        start, stop = offsets[index], offsets[index + 1]
        cpu_distances = np.asarray(pair_batch.distances, dtype=np.float64).copy()
        distances = cpu_distances.copy()
        gpu_distances = gpu_dist[start:stop].astype(np.float64, copy=False)
        edge_distances = np.min(np.abs(cpu_distances[:, None] - edges[None, :]), axis=1) if pair_batch.size else np.zeros(0)
        safe = np.abs(gpu_distances - cpu_distances) <= 1e-5
        safe &= np.abs(cpu_distances - float(pair_cutoff)) > 1e-5
        safe &= edge_distances > 1e-6
        distances[safe] = gpu_distances[safe]
        pair_batch.distances = distances
        pair_batch.lj_attraction = gpu_att[start:stop].astype(np.float64, copy=False)
        pair_batch.lj_repulsion = gpu_rep[start:stop].astype(np.float64, copy=False)
        pair_batch.backend = backend_name
        pair_batch.gpu_available = True
        pair_batch.gpu_compute_seconds = float(gpu_seconds)
        pair_batch.transfer_seconds = float(transfer_seconds)
        pair_batch.gpu_max_distance_error = float(np.max(np.abs(gpu_distances - cpu_distances))) if pair_batch.size else 0.0
        pair_batch.backend_reason = "target_batched_" + backend_name
        max_error = max(max_error, pair_batch.gpu_max_distance_error)
    return {"available": True, "backend": backend_name, "pairs": int(total),
            "seconds": float(gpu_seconds), "transfer_seconds": float(transfer_seconds),
            "max_distance_error": float(max_error)}


__all__ = ["InteractionPairBatch", "build_interaction_pair_batch", "batch_gpu_interactions"]
