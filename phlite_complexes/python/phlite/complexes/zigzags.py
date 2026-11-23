"""Zigzag persistence: of a sequence of simplicial complexes, each obtained from the last by inserting or deleting
simplices, by FastZigzag (Dey and Hou, *Fast computation of zigzag persistence*, ESA 2022) on phlite's reductions.

Build a zigzag by chaining insertions and deletions, or from a sequence of complexes, and take its barcode:

```python
from phlite.complexes import Zigzag

barcode = (
    Zigzag()
    .insert((0,), (1,), (2,), (0, 1), (1, 2), (0, 2))   # a triangle's boundary: a loop is born
    .delete((0, 2))                                     # ... and dies
    .barcode()
)
barcode.in_dimension(1)       # [ZigzagBar(dimension=1, birth=6, death=6, birth_closed=True, death_closed=True)]

# A zigzag of complexes K_0 -> K_0 ∪ K_1 <- K_1 -> ...: intervals in levels (K_i at 2i, the unions in between)
Zigzag.from_complexes([window_0, window_1, window_2]).barcode(levels=True)
```

Positions: a cell-wise zigzag `K_0 <-> K_1 <-> ... <-> K_m` starts from the empty complex `K_0`, and `K_{i+1}` is the
result of operation `i`. An interval `[birth, death]` is homology alive in `K_birth, ..., K_death`; its birth is
*closed* if made by an insertion (open if by a deletion), its death *closed* if followed by a deletion or the end (open
if by an insertion).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple, Optional

import numpy as np

from phlite.core import ZigzagBar, ZigzagBarcode

from ._complexes import _timed_zigzag, _zigzag

__all__ = ["TimedZigzagBar", "TimedZigzagBarcode", "Zigzag", "ZigzagBar", "ZigzagBarcode", "timed_zigzag", "zigzag"]


def _canonical(simplex) -> tuple:
    simplex = tuple(sorted(simplex))
    if len(set(simplex)) != len(simplex):
        raise ValueError(f"{simplex} has a repeated vertex")
    return simplex


def zigzag(
    operations: Iterable[tuple[Sequence[int], bool]],
    *,
    algorithm=None,
    method: str = "cohomology",
    union_find: bool = True,
) -> ZigzagBarcode:
    """The zigzag barcode of a sequence of operations `(simplex, insert)` from the empty complex (see `Zigzag`, which
    builds them): `True` inserts the simplex (its faces must be there), `False` deletes it (its cofaces must be gone).

    The zigzag is turned into an ordinary filtration (FastZigzag), reduced by `method` (`'cohomology'`, the default,
    or `'homology'`) with `algorithm`, and with `union_find` (by cohomology) dimension 0 and the top dimension by
    union-find; the barcode is the same either way.

    Raises:
        ValueError: If an operation is impossible.
    """
    operations = list(operations)
    bars = _zigzag(operations, algorithm=algorithm, method=method, union_find=union_find)
    return ZigzagBarcode((ZigzagBar(*bar) for bar in bars), len(operations) + 1)


class Zigzag:
    """A zigzag of simplicial complexes, built by chaining insertions and deletions (each returns the zigzag):

    ```python
    Zigzag().insert((0,), (1,), (0, 1)).delete((0, 1)).barcode()
    ```

    `insert` adds missing faces first if `with_faces=True`, and `delete` removes cofaces first if `with_cofaces=True`.
    `Zigzag.from_complexes` builds the zigzag `K_0 -> K_0 ∪ K_1 <- K_1 -> ...` of a sequence of complexes.
    """

    def __init__(self, operations: Optional[Iterable[tuple[Sequence[int], bool]]] = None):
        self._operations: list[tuple[tuple, bool]] = []
        self._current: set[tuple] = set()
        # Positions of the complexes given to from_complexes and their unions, as (level, position)
        self._levels: list[tuple[int, int]] = []
        for simplex, insert in operations or []:
            (self.insert if insert else self.delete)(simplex)

    @property
    def operations(self) -> list[tuple[tuple, bool]]:
        """The operations so far, as `(simplex, insert)` pairs."""
        return list(self._operations)

    @property
    def current(self) -> frozenset:
        """The simplices of the last complex."""
        return frozenset(self._current)

    @property
    def position(self) -> int:
        """The position of the last complex (the number of operations)."""
        return len(self._operations)

    def insert(self, *simplices, with_faces: bool = False) -> "Zigzag":
        """Insert simplices, in order (each one's faces must be there already, or, with `with_faces`, are inserted
        first)."""
        for simplex in simplices:
            simplex = _canonical(simplex)
            if with_faces:
                for k in range(1, len(simplex)):
                    for face in _faces_of_size(simplex, k):
                        if face not in self._current:
                            self._push(face, True)
            if simplex in self._current:
                raise ValueError(f"{simplex} is already there")
            self._push(simplex, True)
        return self

    def delete(self, *simplices, with_cofaces: bool = False) -> "Zigzag":
        """Delete simplices, in order (each one's cofaces must be gone already, or, with `with_cofaces`, are deleted
        first, largest first)."""
        for simplex in simplices:
            simplex = _canonical(simplex)
            if simplex not in self._current:
                raise ValueError(f"{simplex} isn't there")
            if with_cofaces:
                cofaces = [s for s in self._current if len(s) > len(simplex) and set(simplex) <= set(s)]
                for coface in sorted(cofaces, key=lambda s: (-len(s), s)):
                    self._push(coface, False)
            self._push(simplex, False)
        return self

    def _push(self, simplex: tuple, insert: bool) -> None:
        self._operations.append((simplex, insert))
        if insert:
            self._current.add(simplex)
        else:
            self._current.discard(simplex)

    def move_to(self, simplices: Iterable[Sequence[int]], *, join: str = "union") -> "Zigzag":
        """Go to the complex `simplices` (closed under faces): through the union with the current one (`join='union'`:
        inserting what is new, faces first, then deleting what is gone, cofaces first), or through the intersection
        (`join='intersection'`: deleting first, then inserting)."""
        if join not in ("union", "intersection"):
            raise ValueError(f"join must be 'union' or 'intersection', got {join!r}")
        target = {_canonical(s) for s in simplices}
        missing = [s for s in target for f in _all_faces(s) if f not in target]
        if missing:
            raise ValueError(f"not a complex: {missing[0]} is missing a face")
        new = sorted(target - self._current, key=lambda s: (len(s), s))
        gone = sorted(self._current - target, key=lambda s: (-len(s), s))
        if join == "union":
            self.insert(*new)
            middle = self.position
            self.delete(*gone)
        else:
            self.delete(*gone)
            middle = self.position
            self.insert(*new)
        if self._levels:
            self._levels.append((self._levels[-1][0] + 1, middle))
        self._levels.append((len(self._levels), self.position))
        return self

    @classmethod
    def from_complexes(cls, complexes: Iterable[Iterable[Sequence[int]]], *, join: str = "union") -> "Zigzag":
        """The zigzag `K_0 -> K_0 ∪ K_1 <- K_1 -> K_1 ∪ K_2 <- ...` of simplicial complexes (each a collection of
        simplices, closed under faces), cell by cell; with `join='intersection'`, `K_0 <- K_0 ∩ K_1 -> K_1 <- ...`.
        `barcode(levels=True)` gives intervals in levels: `K_i` at `2 i`, the join of `K_i` and `K_{i+1}` at
        `2 i + 1`. (For point clouds and networks, `phlite.clouds.cloud_zigzag` and `phlite.networks.network_zigzag`
        build the complexes in Rust.)"""
        zz = cls()
        for complex_ in complexes:
            zz.move_to(complex_, join=join)
        return zz

    @classmethod
    def rips_windows(
        cls,
        points,
        windows,
        threshold: float,
        dimension: int = 2,
        *,
        metric: str = "euclidean",
    ) -> "Zigzag":
        """The zigzag of the Rips complexes, at scale `threshold`, of windows of a point cloud (for instance sliding
        along a time series): `K_0 -> K_0 ∪ K_1 <- K_1 -> ...` with `K_i` the Rips complex of the points
        `windows[i]` (simplices of dimension up to `dimension`, on the points' indices in `points`).

        Args:
            points: The points, an array of shape `(n, d)`.
            windows: Each window, a sequence of point indices; or an int `w`, for the windows `[0, w)`, `[1, w + 1)`,
                ... sliding by one point.
            threshold: The scale: the largest distance within a simplex.
            dimension: The largest dimension of a simplex.
            metric: As for `phlite.clouds.rips`.

        `barcode(levels=True)` gives the intervals over the windows (window `i` at `2 i`).
        """
        import numpy as np
        from phlite.clouds import rips_complex

        points = np.asarray(points, dtype=float)
        if isinstance(windows, int):
            windows = [range(i, i + windows) for i in range(len(points) - windows + 1)]
        complexes = []
        for window in windows:
            index = np.asarray(sorted(window), dtype=int)
            simplices = rips_complex(points=points[index], dimension=dimension, threshold=threshold, metric=metric)
            complexes.append([tuple(int(index[v]) for v in s) for s, _ in simplices])
        return cls.from_complexes(complexes)

    def barcode(self, *, levels: bool = False, algorithm=None, method: str = "cohomology", union_find: bool = True) -> ZigzagBarcode:
        """The zigzag barcode. With `levels` (for a zigzag from `from_complexes`), intervals are in levels: `K_i` at
        `2 i`, `K_i ∪ K_{i+1}` at `2 i + 1`, each interval the levels it contains (both ends inclusive, so reported as
        closed), and intervals living only between levels (in partly built complexes) left out."""
        cellwise = zigzag(self._operations, algorithm=algorithm, method=method, union_find=union_find)
        if not levels:
            return cellwise
        if not self._levels:
            raise ValueError("levels=True is for zigzags built by from_complexes or move_to")
        out = []
        for bar in cellwise:
            inside = [level for level, position in self._levels if bar.birth <= position <= bar.death]
            if inside:
                out.append(ZigzagBar(bar.dimension, min(inside), max(inside), True, True))
        return ZigzagBarcode(out, self._levels[-1][0] + 1)

    def __len__(self) -> int:
        return len(self._operations)

    def __repr__(self) -> str:
        return f"Zigzag({len(self._operations)} operations, {len(self._current)} simplices now)"


def _faces_of_size(simplex: tuple, k: int):
    from itertools import combinations

    return combinations(simplex, k)


def _all_faces(simplex: tuple):
    for k in range(1, len(simplex)):
        yield from _faces_of_size(simplex, k)


# ---- Cells with lifetimes


class TimedZigzagBar(NamedTuple):
    """An interval of a timed zigzag: homology of `dimension` alive from `birth` to `death` (`inf` if it lasts),
    each end included if closed. `levels` is the interval in levels (`K(τ_i)` at `2 i`, the join at `τ_{i+1}` at
    `2 i + 1`)."""

    dimension: int
    birth: float
    death: float
    birth_closed: bool
    death_closed: bool

    def contains(self, t: float) -> bool:
        """Whether the class is alive at time `t`."""
        after = self.birth < t or (self.birth_closed and self.birth == t)
        before = t < self.death or (self.death_closed and self.death == t)
        return after and before

    @property
    def length(self) -> float:
        return self.death - self.birth


class TimedZigzagBarcode(Sequence):
    """The barcode of a timed zigzag (`timed_zigzag`): `TimedZigzagBar`s sorted by dimension, birth and death, with
    the event times `times` and the same barcode over levels as `levels` (a `ZigzagBarcode`)."""

    def __init__(self, bars: Iterable[TimedZigzagBar], times: Sequence[float], levels: ZigzagBarcode):
        self._bars = tuple(sorted(bars))
        self.times = np.asarray(times, dtype=float)
        self.levels = levels

    def __len__(self) -> int:
        return len(self._bars)

    def __getitem__(self, index):
        return self._bars[index]

    def __eq__(self, other) -> bool:
        if isinstance(other, TimedZigzagBarcode):
            return self._bars == other._bars and np.array_equal(self.times, other.times)
        return list(self._bars) == list(other)

    __hash__ = None

    def in_dimension(self, dim: int) -> list[TimedZigzagBar]:
        return [b for b in self._bars if b.dimension == dim]

    def betti(self, t: float, dim: int) -> int:
        """The rank of homology in dimension `dim` at time `t`: of `K(t)`, the cells alive at `t` (after any change
        at `t`, so not counting classes that live only for the instant of a join)."""
        return sum(1 for b in self._bars if b.dimension == dim and b.birth <= t < b.death)

    def diagram(self, dim: int) -> np.ndarray:
        """`(birth, death)` of each interval in dimension `dim`, as an `(n, 2)` array."""
        return np.array([(b.birth, b.death) for b in self.in_dimension(dim)], dtype=float).reshape(-1, 2)

    def __repr__(self) -> str:
        dims = sorted({b.dimension for b in self._bars})
        counts = ", ".join(f"H{d}: {len(self.in_dimension(d))}" for d in dims)
        return f"TimedZigzagBarcode({len(self)} intervals over {len(self.times)} event times; {counts})"


def _lifetimes(value) -> list[tuple[float, float]]:
    """`(start, end)`, or a list of them, as a list of float pairs (`None` for `end` is `inf`)."""
    if len(value) == 2 and not isinstance(value[0], (tuple, list)):
        value = [value]
    return [(float(a), float("inf") if b is None else float(b)) for a, b in value]


def timed_zigzag(
    cells,
    *,
    join: str = "union",
    algorithm=None,
    method: str = "cohomology",
    union_find: bool = True,
) -> TimedZigzagBarcode:
    """The zigzag barcode of cells that each live over some intervals of time.

    A cell is there at time `t` when `start <= t < end` for one of its lifetimes, and its faces must be there whenever
    it is. Between the *event times* `τ_0 < τ_1 < ...` (where some cell starts or ends) the complex `K(t)` is
    constant; at each event the zigzag passes through the union (`join='union'`) or the intersection
    (`join='intersection'`) of the complexes before and after, so a class may live for an instant `[τ, τ]`. Computed
    by FastZigzag (Dey and Hou).

    Args:
        cells: A dict `{cell: lifetimes}` or an iterable of `(cell, lifetimes)` pairs, each cell a simplex (a tuple of
            integer vertices: every face must be a cell too); or `(cell, lifetimes, faces)` triples for cells of any
            hashable kind (cubes, cells of a CW complex) with their faces. Lifetimes are `(start, end)` or a list of
            them (`end` may be `None` or `inf`).
        join: `'union'` or `'intersection'`.
        algorithm, method, union_find: As for `zigzag`.

    Returns:
        A `TimedZigzagBarcode`.

    ```python
    timed_zigzag({(0,): (0, None), (1,): (0, None), (2,): (0, None),
                  (0, 1): (0, None), (1, 2): (1, None), (0, 2): [(2, 3), (4, 6)]})
    # H1: [2, 3) and [4, 6), the loop while (0, 2) is there
    ```
    """
    items = list(cells.items()) if isinstance(cells, dict) else list(cells)
    converted = []
    for item in items:
        if len(item) == 3:
            cell, lifetimes, faces = item
            converted.append((cell, _lifetimes(lifetimes), list(faces)))
        else:
            cell, lifetimes = item
            converted.append((tuple(cell) if not isinstance(cell, tuple) else cell, _lifetimes(lifetimes)))
    if any(len(c) == 3 for c in converted):
        converted = [c if len(c) == 3 else (c[0], c[1], []) for c in converted]
    bars, times = _timed_zigzag(converted, join=join, algorithm=algorithm, method=method, union_find=union_find)
    levels = ZigzagBarcode((ZigzagBar(d, lb, ld, True, True) for d, *_, lb, ld in bars), max(1, 2 * len(times) - 1))
    return TimedZigzagBarcode((TimedZigzagBar(d, b, e, bc, dc) for d, b, e, bc, dc, _, _ in bars), times, levels)
