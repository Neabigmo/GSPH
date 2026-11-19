"""GLMY persistence at all birth events, using exact finite-field algebra."""
from __future__ import annotations

from itertools import groupby
from math import isfinite
from typing import Mapping

from .glmy_sparse import RANK_PRIMES, PathTuple, _is_regular

_PRIME = RANK_PRIMES[0]
_Vector = dict[int, int]
_Births = Mapping[PathTuple, float]
_Interval = tuple[float, float | None]


def _add_scaled(dst: _Vector, src: _Vector, scale: int) -> None:
    for row, value in src.items():
        updated = (dst.get(row, 0) + scale * value) % _PRIME
        if updated:
            dst[row] = updated
        else:
            dst.pop(row, None)


def _pivot_inverse(column: _Vector) -> int:
    return pow(column[max(column)], -1, _PRIME)


def _faces(path: PathTuple):
    if len(path) > 1:
        for i in range(len(path)):
            face = path[:i] + path[i + 1:]
            if _is_regular(face):
                yield face, (-1 if i % 2 else 1)


def _omega_basis(paths, lower_paths):
    """Return (birth, vector) columns triangular in their ambient paths.

    The time-filtered kernel of [boundary | -I] projects isomorphically
    to Omega: slack columns become available when their faces are allowed.
    A dependent column yields one new kernel vector at its birth.
    """
    face_rows = {}
    raw_boundaries = []
    for _, path in paths:
        boundary: _Vector = {}
        for face, sign in _faces(path):
            row = face_rows.setdefault(face, len(face_rows))
            _add_scaled(boundary, {row: 1}, sign)
        raw_boundaries.append(boundary)

    events = [(birth, 0, i) for i, (birth, _) in enumerate(paths)]
    events.extend((birth, 1, face_rows[path]) for birth, path in lower_paths if path in face_rows)
    boundary_pivots = {}
    omega_pivots = {}
    basis = []
    for birth, is_slack, i in sorted(events):
        boundary = {i: _PRIME - 1} if is_slack else raw_boundaries[i].copy()
        vector = {} if is_slack else {i: 1}
        while boundary:
            row = max(boundary)
            if row not in boundary_pivots:
                boundary_pivots[row] = (boundary, vector, _pivot_inverse(boundary))
                break
            prev_boundary, prev_vector, inverse = boundary_pivots[row]
            scale = -boundary[row] * inverse
            _add_scaled(boundary, prev_boundary, scale)
            _add_scaled(vector, prev_vector, scale)
        else:
            while vector:
                row = max(vector)
                if row not in omega_pivots:
                    omega_pivots[row] = len(basis)
                    inverse = _pivot_inverse(vector)
                    basis.append((birth, {j: v * inverse % _PRIME for j, v in vector.items()}))
                    break
                _add_scaled(vector, basis[omega_pivots[row]][1], -vector[row])
            else:
                raise ValueError("dependent projected Omega generator")
    return basis, omega_pivots


def _boundary_coordinates(vector, paths, lower_index, lower_basis, lower_pivots, birth):
    boundary: _Vector = {}
    for i, value in vector.items():
        for face, sign in _faces(paths[i][1]):
            row = lower_index.get(face)
            if row is not None:
                _add_scaled(boundary, {row: 1}, value * sign)
            # A regular face unavailable even at the last scale cancels in Omega.
    coordinates = {}
    while boundary:
        row = max(boundary)
        j = lower_pivots[row]
        lower_birth, lower_vector = lower_basis[j]
        if lower_birth > birth:
            raise ValueError("Omega boundary is not filtration-preserving")
        coordinates[j] = value = boundary[row]
        _add_scaled(boundary, lower_vector, -value)
    return coordinates


def _reduce_persistence(births: _Births, scale_index: Mapping[float, int]):
    if any(not 1 <= len(path) <= 4 or not _is_regular(path) for path in births):
        raise ValueError("expected regular paths of orders 0 through 3")
    events = sorted((len(path), scale_index[birth], path) for path, birth in births.items())
    paths_by_order = {
        size - 1: [(birth, path) for _, birth, path in level]
        for size, level in groupby(events, key=lambda item: item[0])
    }
    lower_paths, lower_basis, lower_pivots = [], [], {}
    lower_creators: set[int] = set()
    diagrams = {}
    for order in range(4):
        paths = paths_by_order.get(order, [])
        basis, pivots = _omega_basis(paths, lower_paths)
        lower_index = {path: i for i, (_, path) in enumerate(lower_paths)}
        reduced = {}
        deaths = {}
        creators = set()
        for j, (birth, vector) in enumerate(basis):
            boundary = _boundary_coordinates(
                vector, paths, lower_index, lower_basis, lower_pivots, birth
            )
            while boundary:
                row = max(boundary)
                if row not in reduced:
                    if row not in lower_creators:
                        raise ValueError("Omega boundary does not square to zero")
                    reduced[row] = (boundary, _pivot_inverse(boundary))
                    deaths[row] = birth
                    break
                prev, inverse = reduced[row]
                _add_scaled(boundary, prev, -boundary[row] * inverse)
            else:
                creators.add(j)
        if order:
            diagrams[order - 1] = [
                (lower_basis[i][0], deaths.get(i))
                for i in sorted(lower_creators)
                if lower_basis[i][0] != deaths.get(i)
            ]
        lower_paths, lower_basis, lower_pivots = paths, basis, pivots
        lower_creators = creators
    return diagrams


def compare_event_barcodes(
    common: _Births, pairwise: _Births
) -> tuple[list[float], dict[int, list[_Interval]], dict[int, list[_Interval]]]:
    """Compare H0–H2 in Å for the same paths on the full union of birth values.

    The only rounding is in the supplied geometric births, not in the
    chain algebra. Duplicate intervals are retained; None denotes infinity.
    """
    if common.keys() != pairwise.keys():
        raise ValueError("the two birth rules must use exactly the same paths")
    scales = sorted(set(common.values()) | set(pairwise.values()))
    if any(not isfinite(value) for value in scales):
        raise ValueError("all paths must have finite births under both rules")
    index = {value: i for i, value in enumerate(scales)}

    def diagram(births):
        return {
            order: sorted(
                ((scales[b], None if d is None else scales[d]) for b, d in intervals),
                key=lambda bar: (bar[0], float("inf") if bar[1] is None else bar[1]),
            )
            for order, intervals in _reduce_persistence(births, index).items()
        }

    return scales, diagram(common), diagram(pairwise)
