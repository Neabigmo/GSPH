"""Lazy CPU C++ fixed-dimensional weighted-support backend.

The C++ solver enumerates the 2/3/4-ball Helly active sets and returns only a
candidate that is independently checked by the Python float64 certificate.
Any rejected or unavailable candidate is counted and routed through the
reference solver; it is never silently converted to a benchmark value.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


_ENGINE = None
_ERROR = None
_C_LIB = None
_C_LIB_FAILED = False


def _load_c_support_lib():
    """Torch-free C solver (libgsph_support_c.so) via ctypes.

    The cluster's torch 2.6 + pybind11 build rejects at::Tensor arguments
    with ``RuntimeError: Unknown layout`` for ANY extension, so the pure-C
    port in cpp_support_solver_c.c is the production accelerator.  Candidates
    are still certified by the Python float64 KKT/Helly routine.
    """
    global _C_LIB, _C_LIB_FAILED
    if _C_LIB is not None or _C_LIB_FAILED:
        return _C_LIB
    try:
        import ctypes
        here = Path(__file__).resolve().parent
        candidates = []
        build = os.environ.get("GSPH_C_BUILD_DIR") or os.environ.get("GSPH_CPP_BUILD_DIR")
        if build:
            candidates.append(Path(build) / "libgsph_support_c.so")
        candidates.append(here / "libgsph_support_c.so")
        for path in candidates:
            if path.is_file():
                lib = ctypes.CDLL(str(path))
                fn = lib.gsph_support_solve_batch
                fn.restype = ctypes.c_int
                fn.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 6
                _C_LIB = lib
                return lib
        _C_LIB_FAILED = True
    except Exception:
        _C_LIB_FAILED = True
    return None


def load_support_active_engine():
    global _ENGINE, _ERROR
    if _ENGINE is not None:
        return _ENGINE
    if _ERROR is not None:
        raise RuntimeError("C++ support active-set engine unavailable") from _ERROR
    try:
        from torch.utils.cpp_extension import load
        # Remote jobs often invoke an absolute conda Python without activating
        # the shell environment.  PyTorch's extension loader discovers Ninja
        # through PATH, so make the interpreter's bin directory explicit.
        python_bin = str(Path(sys.executable).resolve().parent)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if python_bin not in path_parts:
            os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
        source = Path(__file__).resolve().parent / "cpp_support_solver.cpp"
        build_dir = Path(os.environ.get("GSPH_CPP_BUILD_DIR", str(Path(tempfile.gettempdir()) / "gsph1_cpp_build")))
        build_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MAX_JOBS", "4")
        _ENGINE = load(name="gsph1_cpp_support_active", sources=[str(source)],
                       build_directory=str(build_dir), verbose=False, with_cuda=False,
                       extra_cflags=["-O3", "-std=c++17"])
        return _ENGINE
    except BaseException as exc:
        _ERROR = exc
        raise RuntimeError(f"C++ support active-set build/load failed: {exc}") from exc


def build_support_candidates(atom_keys: Sequence[Tuple[int, ...]], atoms: Mapping[int, object]):
    stats: Dict[str, object] = {
        "support_cpp_attempted": False, "support_cpp_available": False,
        "support_cpp_batch": 0, "support_cpp_seconds": 0.0,
        "support_cpp_error": "",
    }
    if not atom_keys:
        return {}, stats

    lib = _load_c_support_lib()
    if lib is not None:
        try:
            import ctypes
            from cspph.core.gsph1 import _vdw_radius
            n_keys = len(atom_keys)
            centers = np.zeros((n_keys, 8, 3), dtype=np.float64)
            radii = np.zeros((n_keys, 8), dtype=np.float64)
            counts = np.zeros(n_keys, dtype=np.int64)
            for row, key in enumerate(atom_keys):
                key = tuple(int(x) for x in key)
                counts[row] = min(8, len(key))
                for col, atom_id in enumerate(key[:8]):
                    atom = atoms[int(atom_id)]
                    centers[row, col] = np.asarray(atom.xyz, dtype=np.float64)
                    radii[row, col] = _vdw_radius(atom.element)
            stats["support_cpp_attempted"] = True
            started = perf_counter()
            out_centers = np.zeros((n_keys, 3), dtype=np.float64)
            out_tau = np.zeros(n_keys, dtype=np.float64)
            out_ok = np.zeros(n_keys, dtype=np.int32)
            rc = lib.gsph_support_solve_batch(
                n_keys,
                centers.ctypes.data_as(ctypes.c_void_p),
                radii.ctypes.data_as(ctypes.c_void_p),
                counts.ctypes.data_as(ctypes.c_void_p),
                out_centers.ctypes.data_as(ctypes.c_void_p),
                out_tau.ctypes.data_as(ctypes.c_void_p),
                out_ok.ctypes.data_as(ctypes.c_void_p))
            elapsed = perf_counter() - started
            if rc != 0:
                raise RuntimeError(f"gsph_support_solve_batch rc={rc}")
            stats["support_cpp_available"] = True
            stats["support_cpp_backend"] = "c_ctypes"
            stats["support_cpp_batch"] = n_keys
            stats["support_cpp_seconds"] = float(elapsed)
            return {
                key: (np.asarray(out_centers[row], dtype=np.float64),
                      float(max(0.0, out_tau[row])))
                for row, key in enumerate(atom_keys) if int(out_ok[row])
            }, stats
        except Exception as exc:
            stats["support_cpp_error"] = f"c_ctypes:{type(exc).__name__}: {exc}"

    try:
        import torch
        from cspph.core.gsph1 import _vdw_radius
        centers = np.zeros((len(atom_keys), 8, 3), dtype=np.float64)
        radii = np.zeros((len(atom_keys), 8), dtype=np.float64)
        counts = np.zeros(len(atom_keys), dtype=np.int64)
        for row, key in enumerate(atom_keys):
            counts[row] = min(8, len(key))
            for col, atom_id in enumerate(key[:8]):
                atom = atoms[int(atom_id)]
                centers[row, col] = np.asarray(atom.xyz, dtype=np.float64)
                radii[row, col] = _vdw_radius(atom.element)
        stats["support_cpp_attempted"] = True
        started = perf_counter()
        out_centers, out_tau, out_ok = load_support_active_engine().weighted_support_active_set(
            torch.as_tensor(centers), torch.as_tensor(radii), torch.as_tensor(counts)
        )
        out_centers = out_centers.numpy(); out_tau = out_tau.numpy(); out_ok = out_ok.numpy()
        stats["support_cpp_available"] = True
        stats["support_cpp_backend"] = "torch"
        stats["support_cpp_batch"] = len(atom_keys)
        stats["support_cpp_seconds"] = float(perf_counter() - started)
        return {
            key: (np.asarray(out_centers[row], dtype=np.float64), float(max(0.0, out_tau[row])))
            for row, key in enumerate(atom_keys) if int(out_ok[row])
        }, stats
    except Exception as exc:
        stats["support_cpp_error"] = f"{type(exc).__name__}: {exc}"
        return {}, stats


def solve_support_batch(atom_keys: Sequence[Tuple[int, ...]], atoms: Mapping[int, object]):
    """Compatibility API returning explicit candidate records."""
    raw, stats = build_support_candidates(atom_keys, atoms)
    records = {key: {"center": tuple(map(float, center)), "tau": float(tau), "solver_ok": True}
               for key, (center, tau) in raw.items()}
    return records, stats


__all__ = ["load_support_active_engine", "build_support_candidates", "solve_support_batch"]
