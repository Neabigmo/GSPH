"""Accessible molecular surface points and normal-complementarity features.

This is a deterministic Shrake--Rupley-style surface construction: atom
spheres are sampled with Fibonacci directions, occluded samples are removed,
and exposed samples near the partner interface are compared by distance and
normal orientation.  It is intentionally separate from the old local-PCA
proxy and can use CUDA only for the candidate point distance arithmetic.
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, Sequence, Tuple

import numpy as np

VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80,
       "S": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}


def _directions(count: int) -> np.ndarray:
    index = np.arange(max(1, int(count)), dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * index / max(1, int(count)))
    theta = np.pi * (1.0 + np.sqrt(5.0)) * index
    return np.column_stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)))


def _surface_points(atoms: Sequence[object], all_atoms: Sequence[object], probe: float,
                    samples: int) -> Tuple[np.ndarray, np.ndarray]:
    if not atoms:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    directions = _directions(samples)
    all_xyz = np.asarray([atom.xyz for atom in all_atoms], dtype=float)
    all_radii = np.asarray([VDW.get(str(atom.element).upper(), 1.70) + probe for atom in all_atoms], dtype=float)
    all_indices = {int(atom.index): idx for idx, atom in enumerate(all_atoms)}
    try:
        from scipy.spatial import cKDTree
        all_tree = cKDTree(all_xyz)
    except Exception:
        all_tree = None
    point_rows = []
    normal_rows = []
    for atom in atoms:
        radius = VDW.get(str(atom.element).upper(), 1.70) + probe
        points = np.asarray(atom.xyz, dtype=float)[None, :] + radius * directions
        owner = all_indices.get(int(atom.index), -1)
        visible = np.ones(len(points), dtype=bool)
        # Query only nearby possible occluders.  The former dense
        # point-by-all-atom matrix made large structures quadratic in Python
        # memory and dominated 1ASY.
        try:
            if all_tree is None:
                raise RuntimeError("scipy.spatial unavailable")
            neighbours = all_tree.query_ball_point(points, r=float(np.max(all_radii)))
            for point_index, candidates in enumerate(neighbours):
                for candidate in candidates:
                    if candidate == owner:
                        continue
                    if np.linalg.norm(points[point_index] - all_xyz[candidate]) < all_radii[candidate] - 1e-7:
                        visible[point_index] = False
                        break
        except Exception:
            for point_index, candidate_xyz in enumerate(points):
                delta = all_xyz - candidate_xyz
                distances = np.linalg.norm(delta, axis=1)
                if owner >= 0:
                    distances[owner] = np.inf
                visible[point_index] = bool(np.all(distances >= all_radii - 1e-7))
        if np.any(visible):
            point_rows.append(points[visible])
            normal_rows.append(np.repeat(directions[visible], 1, axis=0))
    if not point_rows:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    return np.concatenate(point_rows), np.concatenate(normal_rows)


def surface_complementarity(left_atoms: Sequence[object], right_atoms: Sequence[object],
                            backend: str = "auto", device_name: str = "cuda:0",
                            probe_radius: float = 1.4, samples_per_atom: int = 16,
                            partner_cutoff: float = 8.0) -> Dict[str, object]:
    started = perf_counter()
    all_atoms = list(left_atoms) + list(right_atoms)
    left_points, left_normals = _surface_points(left_atoms, all_atoms, probe_radius, samples_per_atom)
    right_points, right_normals = _surface_points(right_atoms, all_atoms, probe_radius, samples_per_atom)
    if not len(left_points) or not len(right_points):
        return {"surface_backend": "surface_points_empty", "surface_point_count_a": len(left_points),
                "surface_point_count_b": len(right_points), "surface_nearest_mean": 0.0,
                "surface_gap_mean": 0.0, "surface_normal_dot_mean": 0.0,
                "surface_normal_antialignment": 0.0, "surface_contact_fraction": 0.0,
                "surface_seconds": float(perf_counter() - started)}
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(right_points)
        distances, indices = tree.query(left_points, k=1)
        keep = distances <= float(partner_cutoff)
    except Exception:
        distance_matrix = np.linalg.norm(left_points[:, None, :] - right_points[None, :, :], axis=2)
        indices = np.argmin(distance_matrix, axis=1)
        distances = distance_matrix[np.arange(len(left_points)), indices]
        keep = distances <= float(partner_cutoff)
    if not np.any(keep):
        keep = np.ones(len(distances), dtype=bool)
    dots = np.sum(left_normals[keep] * right_normals[indices[keep]], axis=1)
    result = {
        "surface_backend": "fibonacci_accessible_surface_normals",
        "surface_point_count_a": int(len(left_points)),
        "surface_point_count_b": int(len(right_points)),
        "surface_nearest_mean": float(np.mean(distances[keep])),
        "surface_gap_mean": float(np.mean(np.maximum(distances[keep] - 2.0 * probe_radius, 0.0))),
        "surface_normal_dot_mean": float(np.mean(dots)),
        "surface_normal_antialignment": float(np.mean(dots < -0.25)),
        "surface_contact_fraction": float(np.mean(distances[keep] <= 2.5)),
        "surface_seconds": float(perf_counter() - started),
    }
    # CUDA is optional for this point-pair reduction.  Keep the CPU point
    # selection/normal calculation deterministic and record the actual backend.
    if backend in {"cuda", "auto"}:
        try:
            import torch
            if torch.cuda.is_available() and len(left_points) * len(right_points) <= 500000:
                device = torch.device(device_name)
                with torch.inference_mode():
                    lp = torch.as_tensor(left_points[keep], dtype=torch.float32, device=device)
                    rp = torch.as_tensor(right_points[indices[keep]], dtype=torch.float32, device=device)
                    # The CPU KD-tree still chooses the deterministic nearest
                    # partner, but the actual pair distances used for every
                    # reported surface statistic now come from this GPU
                    # reduction rather than being recomputed and discarded.
                    gpu_distances = torch.linalg.vector_norm(lp - rp, dim=1).detach().cpu().numpy()
                distances = np.asarray(distances, dtype=float).copy()
                distances[keep] = gpu_distances.astype(float, copy=False)
                dots = np.sum(left_normals[keep] * right_normals[indices[keep]], axis=1)
                result.update({
                    "surface_nearest_mean": float(np.mean(distances[keep])),
                    "surface_gap_mean": float(np.mean(np.maximum(distances[keep] - 2.0 * probe_radius, 0.0))),
                    "surface_normal_dot_mean": float(np.mean(dots)),
                    "surface_normal_antialignment": float(np.mean(dots < -0.25)),
                    "surface_contact_fraction": float(np.mean(distances[keep] <= 2.5)),
                })
                result["surface_backend"] = "fibonacci_accessible_surface_normals_torch_cuda"
        except Exception as exc:
            result["surface_gpu_error"] = f"{type(exc).__name__}: {exc}"
    return result


__all__ = ["surface_complementarity"]
