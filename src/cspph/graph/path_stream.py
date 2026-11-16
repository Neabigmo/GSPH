"""Deterministic directed regular-path streaming for GSPH extraction.

The generator is intentionally independent of geometry and scoring.  It emits
paths breadth-first by order, keeps source ranges disjoint through explicit
per-order counters, and treats a negative per-order limit as *unbounded*.
When a finite limit is requested it stops only after setting ``truncated``;
callers must propagate that flag instead of treating the retained prefix as a
complete path complex.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass
class PathStreamStats:
    generated: Dict[int, int] = field(default_factory=dict)
    retained: Dict[int, int] = field(default_factory=dict)
    truncated: bool = False
    stopped_order: int = -1


def stream_directed_paths(events: Sequence[object], edges: Iterable[Tuple[int, int, float]],
                          max_order: int = 3, max_paths_per_order: int = -1,
                          stats: Optional[PathStreamStats] = None) -> Iterator[Tuple[int, Tuple[int, ...]]]:
    """Yield regular directed paths without silently applying a cap.

    ``events`` only needs stable ``index`` attributes.  Edge weights are
    ignored by the combinatorial stream but accepted so the same edge table
    can be shared with support-birth code.
    """
    stats = stats if stats is not None else PathStreamStats()
    adjacency: Dict[int, List[int]] = {}
    for source, target, _ in edges:
        adjacency.setdefault(int(source), []).append(int(target))
    for source in adjacency:
        adjacency[source] = sorted(set(adjacency[source]))

    frontier = [(int(getattr(event, "index")),) for event in events]
    for path in frontier:
        stats.generated[0] = stats.generated.get(0, 0) + 1
        stats.retained[0] = stats.retained.get(0, 0) + 1
        yield 0, path

    for order in range(1, max_order + 1):
        next_frontier: List[Tuple[int, ...]] = []
        for previous in frontier:
            for target in adjacency.get(previous[-1], []):
                if target == previous[-1]:
                    continue
                path = previous + (target,)
                stats.generated[order] = stats.generated.get(order, 0) + 1
                if max_paths_per_order >= 0 and len(next_frontier) >= max_paths_per_order:
                    stats.truncated = True
                    stats.stopped_order = order
                    break
                next_frontier.append(path)
                stats.retained[order] = stats.retained.get(order, 0) + 1
                yield order, path
            if stats.truncated and stats.stopped_order == order:
                break
        frontier = next_frontier
        if stats.truncated and stats.stopped_order == order:
            # Higher-order paths depend on the complete preceding frontier;
            # once it is capped, do not fabricate partial higher-order chains.
            break


__all__ = ["PathStreamStats", "stream_directed_paths"]
