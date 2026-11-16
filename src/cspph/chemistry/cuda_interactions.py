"""Lazy loader for the optional SM75 fused-interaction CUDA extension.

Geometric support is deliberately not a CUDA kernel: its fixed-dimensional
weighted 1-center problem is solved by the CPU C++ candidate backend followed
by the float64 Helly/KKT certificate in :mod:`runners.gsph1`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional


_EXTENSION: Optional[Any] = None
_ERROR: Optional[BaseException] = None


def load_fused_interactions() -> Any:
    """Build/load the extension once, without changing the Python/CUDA stack."""
    global _EXTENSION, _ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _ERROR is not None:
        raise RuntimeError("GSPH1 CUDA extension is unavailable") from _ERROR
    try:
        import torch
        from torch.utils.cpp_extension import load
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the active PyTorch environment")
        source = Path(__file__).resolve().parent / "cuda" / "fused_interactions.cu"
        build_dir = Path(os.environ.get(
            "GSPH_CUDA_BUILD_DIR",
            str(Path(tempfile.gettempdir()) / "gsph1_cuda_build"),
        ))
        build_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
        os.environ.setdefault("MAX_JOBS", "4")
        _EXTENSION = load(
            name="gsph1_fused_interactions_sm75",
            sources=[str(source)],
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
            extra_cuda_cflags=["-O3"],
        )
        return _EXTENSION
    except BaseException as exc:  # keep a transparent fallback for CPU-only hosts
        _ERROR = exc
        raise RuntimeError(f"GSPH1 CUDA extension build/load failed: {exc}") from exc


def extension_error() -> Optional[str]:
    return None if _ERROR is None else f"{type(_ERROR).__name__}: {_ERROR}"


__all__ = [
]
