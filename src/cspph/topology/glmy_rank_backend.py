"""Lazy C++17 sparse modular-rank backend for GLMY incidence operators."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

_ENGINE: Optional[Any] = None
_ERROR: Optional[BaseException] = None


def load_glmy_rank_engine() -> Any:
    global _ENGINE, _ERROR
    if _ENGINE is not None:
        return _ENGINE
    if _ERROR is not None:
        raise RuntimeError("C++ GLMY rank engine unavailable") from _ERROR
    try:
        from torch.utils.cpp_extension import load
        source = Path(__file__).resolve().parent / "cpp_glmy_rank.cpp"
        build_dir = Path(os.environ.get("GSPH_CPP_BUILD_DIR", str(Path(tempfile.gettempdir()) / "gsph1_cpp_build")))
        build_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MAX_JOBS", "4")
        _ENGINE = load(name="gsph1_cpp_glmy_rank", sources=[str(source)], build_directory=str(build_dir),
                       verbose=False, with_cuda=False, extra_cflags=["-O3", "-std=c++17"])
        return _ENGINE
    except BaseException as exc:
        _ERROR = exc
        raise RuntimeError(f"C++ GLMY rank build/load failed: {exc}") from exc


def glmy_rank_engine_error() -> Optional[str]:
    return None if _ERROR is None else f"{type(_ERROR).__name__}: {_ERROR}"


__all__ = ["load_glmy_rank_engine", "glmy_rank_engine_error"]
