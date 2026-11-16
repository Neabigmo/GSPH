"""Compact CSR directed-path enumeration.

The engine keeps paths as fixed-width NumPy integer arrays instead of Python
tuples while expanding orders 0--3.  It is exact for unbounded runs and
reports every generated/retained count; bounded smoke runs continue to use the
audited streaming generator so truncation semantics stay unchanged.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

LAST_BACKEND = "uninitialized"
LAST_ERROR = ""


def enumerate_csr_paths(event_ids: Sequence[int], edges: Iterable[Tuple[int, int, float]],
                        max_order: int = 3) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    global LAST_BACKEND, LAST_ERROR
    # Backend status is per invocation.  Without resetting it, a preceding
    # CUDA/C++ call could make a forced Python call report ``cpp17_csr`` even
    # though no C++ path engine was used.
    LAST_BACKEND = "uninitialized"
    LAST_ERROR = ""
    ids = np.asarray(sorted({int(value) for value in event_ids}), dtype=np.int32)
    position = {int(value): index for index, value in enumerate(ids)}
    adjacency = [[] for _ in range(len(ids))]
    for source, target, _ in edges:
        if int(source) in position and int(target) in position and int(source) != int(target):
            # The CSR engine addresses rows/columns by compact positions, not
            # by the (possibly sparse) stable patch IDs.
            adjacency[position[int(source)]].append(position[int(target)])
    adjacency = [np.asarray(sorted(set(values)), dtype=np.int32) for values in adjacency]

    if os.environ.get("GSPH_ENABLE_CPP_PATH", "0") == "1":
        try:
            import torch
            from cspph.graph.path_engine_backend import load_cpp_path_engine
            indptr = np.zeros(len(ids) + 1, dtype=np.int32)
            flat = []
            for row, values in enumerate(adjacency):
                flat.extend(int(value) for value in values)
                indptr[row + 1] = len(flat)
            outputs = load_cpp_path_engine().enumerate_csr(
                torch.as_tensor(indptr, dtype=torch.int32),
                torch.as_tensor(np.asarray(flat, dtype=np.int32), dtype=torch.int32),
                int(max_order),
            )
            compact = {}
            for order, tensor in enumerate(outputs):
                positions = tensor.detach().cpu().numpy().astype(np.int32, copy=False)
                compact[order] = ids[positions]
            LAST_BACKEND = "cpp17_csr"
            LAST_ERROR = ""
            return compact, {order: int(value.shape[0]) for order, value in compact.items()}
        except Exception as exc:
            # The Python CSR implementation below is exact and remains the
            # portable fallback when a compiler/toolchain is unavailable.
            LAST_BACKEND = "exact_numpy_csr"
            LAST_ERROR = f"{type(exc).__name__}: {exc}"

    output: Dict[int, np.ndarray] = {0: np.arange(len(ids), dtype=np.int32).reshape((-1, 1))}
    generated: Dict[int, int] = {0: int(len(ids))}
    frontier = output[0]
    for order in range(1, int(max_order) + 1):
        if frontier.size == 0:
            output[order] = np.empty((0, order + 1), dtype=np.int32)
            generated[order] = 0
            frontier = output[order]
            continue
        rows = []
        for previous in frontier:
            for target in adjacency[int(previous[-1])]:
                # Regular paths forbid only adjacent repeats; non-adjacent
                # repeated event IDs remain valid.
                if int(target) == int(previous[-1]):
                    continue
                rows.append(np.concatenate((previous, np.asarray([target], dtype=np.int32))))
        output[order] = (np.asarray(rows, dtype=np.int32).reshape((-1, order + 1))
                         if rows else np.empty((0, order + 1), dtype=np.int32))
        generated[order] = int(output[order].shape[0])
        frontier = output[order]
    output = {order: ids[rows] for order, rows in output.items()}
    if LAST_BACKEND == "uninitialized":
        LAST_BACKEND = "exact_numpy_csr"
    return output, generated


def path_engine_status() -> Tuple[str, str]:
    return LAST_BACKEND, LAST_ERROR


__all__ = ["enumerate_csr_paths", "path_engine_status"]
