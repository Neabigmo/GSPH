"""Lazy loader for the optional C++17 compact CSR path engine."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

_ENGINE: Optional[Any] = None
_ERROR: Optional[BaseException] = None


def load_cpp_path_engine() -> Any:
    global _ENGINE, _ERROR
    if _ENGINE is not None:
        return _ENGINE
    if _ERROR is not None:
        raise RuntimeError("C++ path engine unavailable") from _ERROR
    try:
        import torch
        from torch.utils.cpp_extension import load
        python_bin = str(Path(sys.executable).resolve().parent)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if python_bin not in path_parts:
            os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
        source = Path(__file__).resolve().parent / "cpp_path_engine.cpp"
        build_dir = Path(os.environ.get(
            "GSPH_CPP_BUILD_DIR", str(Path(tempfile.gettempdir()) / "gsph1_cpp_build")
        ))
        build_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MAX_JOBS", "4")
        _ENGINE = load(
            name="gsph1_cpp_path_engine",
            sources=[str(source)], build_directory=str(build_dir),
            verbose=False, with_cuda=False, extra_cflags=["-O3", "-std=c++17"],
        )
        return _ENGINE
    except BaseException as exc:
        _ERROR = exc
        raise RuntimeError(f"C++ path engine build/load failed: {exc}") from exc


def path_engine_error() -> Optional[str]:
    return None if _ERROR is None else f"{type(_ERROR).__name__}: {_ERROR}"


__all__ = ["load_cpp_path_engine", "path_engine_error"]
