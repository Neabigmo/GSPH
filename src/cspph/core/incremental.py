"""Target-level candidate engine for safe cross-decoy reuse.

The extractor remains the scientific authority.  This wrapper keeps one
``TargetGeometryCache`` alive for a target, so static site-transfer adjacency,
stable patch IDs, rigid KD-trees, path templates and validated support
warm-starts are shared across a candidate stream.  It never changes candidate
order or silently substitutes an incomplete result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from cspph.core.cache import TargetGeometryCache
from cspph.core.engine import GSPH6Config, extract_gsph6_atoms


@dataclass
class TargetIncrementalEngine:
    native_atoms: Sequence[object]
    scenario: str
    config: GSPH6Config
    chain_a: Optional[str] = None
    chain_b: Optional[str] = None

    def __post_init__(self):
        self.cache = TargetGeometryCache.from_atoms(self.native_atoms)
        self.candidate_count = 0

    def extract(self, atoms: Sequence[object], candidate_id: Optional[str] = None):
        """Extract one candidate using the shared target state."""
        result = extract_gsph6_atoms(
            atoms,
            scenario=self.scenario,
            chain_a=self.chain_a,
            chain_b=self.chain_b,
            config=self.config,
            target_cache=self.cache,
        )
        self.candidate_count += 1
        return result

    def extract_many(self, candidates: Iterable[Tuple[str, Sequence[object]]]):
        """Yield ``(candidate_id, result)`` in input order.

        The iterator is intentionally streaming: callers can checkpoint each
        result and bound memory to one target/candidate at a time.
        """
        for candidate_id, atoms in candidates:
            yield candidate_id, self.extract(atoms, candidate_id)

    def reuse_report(self) -> Mapping[str, object]:
        stats = dict(self.cache.reuse_stats)
        observations = int(stats.get("patch_overlap_observations", 0))
        stats["patch_reuse_ratio"] = float(stats.get("patch_overlap_sum", 0.0) / observations) if observations else 0.0
        stats["static_transfer_reuse_ratio"] = 1.0 if self.candidate_count > 1 and stats.get("site_transfer_builds", 0) == 1 else 0.0
        path_hits = int(stats.get("path_template_hits", 0)); path_misses = int(stats.get("path_template_misses", 0))
        stats["path_reuse_ratio"] = float(path_hits / (path_hits + path_misses)) if path_hits + path_misses else 0.0
        stats["glmy_structure_reuse_ratio"] = stats["path_reuse_ratio"]
        stats["candidate_count"] = int(self.candidate_count)
        return stats


__all__ = ["TargetIncrementalEngine"]
