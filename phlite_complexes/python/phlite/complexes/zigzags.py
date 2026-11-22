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

from ._complexes import _zigzag

__all__ = ["Zigzag", "ZigzagBar", "ZigzagBarcode", "zigzag"]


def _canonical(simplex) -> tuple:
    simplex = tuple(sorted(simplex))
    if len(set(simplex)) != len(simplex):
        raise ValueError(f"{simplex} has a repeated vertex")
    return simplex


def zigzag(operations: Iterable[tuple[Sequence[int], bool]], *, algorithm=None) -> ZigzagBarcode:
    """The zigzag barcode of a sequence of operations `(simplex, insert)` from the empty complex (see `Zigzag`, which
    builds them): `True` inserts the simplex (its faces must be there), `False` deletes it (its cofaces must be gone).

    Raises:
        ValueError: If an operation is impossible.
    """
    operations = list(operations)
    bars = _zigzag(operations, algorithm=algorithm)
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

    def move_to(self, simplices: Iterable[Sequence[int]]) -> "Zigzag":
        """Go to the complex `simplices` (closed under faces): through the union with the current one, inserting what
        is new (faces first), then deleting what is gone (cofaces first)."""
        target = {_canonical(s) for s in simplices}
        missing = [s for s in target for f in _all_faces(s) if f not in target]
        if missing:
            raise ValueError(f"not a complex: {missing[0]} is missing a face")
        self.insert(*sorted(target - self._current, key=lambda s: (len(s), s)))
        union = self.position
        self.delete(*sorted(self._current - target, key=lambda s: (-len(s), s)))
        if self._levels:
            self._levels.append((self._levels[-1][0] + 1, union))
        self._levels.append((len(self._levels), self.position))
        return self

    @classmethod
    def from_complexes(cls, complexes: Iterable[Iterable[Sequence[int]]]) -> "Zigzag":
        """The zigzag `K_0 -> K_0 ∪ K_1 <- K_1 -> K_1 ∪ K_2 <- ...` of simplicial complexes (each a collection of
        simplices, closed under faces), cell by cell. `barcode(levels=True)` gives intervals in levels: `K_i` at
        `2 i`, `K_i ∪ K_{i+1}` at `2 i + 1`."""
        zz = cls()
        for complex_ in complexes:
            zz.move_to(complex_)
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

    def barcode(self, *, levels: bool = False, algorithm=None) -> ZigzagBarcode:
        """The zigzag barcode. With `levels` (for a zigzag from `from_complexes`), intervals are in levels: `K_i` at
        `2 i`, `K_i ∪ K_{i+1}` at `2 i + 1`, each interval the levels it contains (both ends inclusive, so reported as
        closed), and intervals living only between levels (in partly built complexes) left out."""
        cellwise = zigzag(self._operations, algorithm=algorithm)
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
