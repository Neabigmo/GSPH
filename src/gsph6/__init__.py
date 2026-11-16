"""Backward-compatibility package for `gsph6.*` imports.

Canonical implementation lives in `cspph.*`; each submodule is a sys.modules alias shim. New code should use `cspph` directly.
"""

from cspph.chemistry.patch_builder import (  # noqa: F401
    InteractionPatch, MicroContact, build_interaction_patches)
from cspph.geometry.patch_support import (  # noqa: F401
    PatchSupportResult, compute_patch_support, higher_order_increments)
from cspph.core.engine import GSPH6Config, extract_gsph6_atoms  # noqa: F401
from cspph.core.cache import TargetGeometryCache  # noqa: F401
from cspph.core.incremental import TargetIncrementalEngine  # noqa: F401

__all__ = [
    "InteractionPatch", "MicroContact", "build_interaction_patches",
    "PatchSupportResult", "compute_patch_support", "higher_order_increments",
    "GSPH6Config", "extract_gsph6_atoms",
    "TargetGeometryCache", "TargetIncrementalEngine",
]
