"""Zigzag barcodes, shared by phlite's packages: `phlite.complexes.zigzag` and `Zigzag` (zigzags of complexes) and
`phlite.images.video_zigzag` (zigzags of the frames of a video).

A zigzag `K_0 <-> K_1 <-> ...` has a barcode of intervals over its positions; each `ZigzagBar` is homology of
`dimension` alive in the complexes at positions `birth` to `death` (inclusive), its ends closed or open by the kind of
arrow (see `phlite.complexes.zigzag`; for zigzags given by levels, such as videos, both ends are closed).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import numpy as np

__all__ = ["ZigzagBar", "ZigzagBarcode"]


class ZigzagBar(NamedTuple):
    """An interval of a zigzag barcode: homology of `dimension` alive in the complexes at positions `birth` to `death`
    (inclusive)."""

    dimension: int
    birth: int
    death: int
    birth_closed: bool
    death_closed: bool

    def alive_at(self, position: int) -> bool:
        """Whether the interval contains `position`."""
        return self.birth <= position <= self.death

    @property
    def length(self) -> int:
        """The number of complexes it spans, minus one."""
        return self.death - self.birth

    @property
    def kind(self) -> str:
        """`'closed-closed'`, `'closed-open'`, `'open-closed'` or `'open-open'`."""
        return f"{'closed' if self.birth_closed else 'open'}-{'closed' if self.death_closed else 'open'}"

    def __str__(self) -> str:
        return f"H{self.dimension} {'[' if self.birth_closed else '('}{self.birth}, {self.death}{']' if self.death_closed else ')'}"


class ZigzagBarcode(Sequence):
    """The barcode of a zigzag: a sequence of `ZigzagBar`s sorted by (dimension, birth, death), over the positions
    `0, ..., n_positions - 1`."""

    def __init__(self, bars: Iterable[ZigzagBar], n_positions: int):
        self._bars = tuple(sorted(bars))
        self.n_positions = n_positions

    def __len__(self) -> int:
        return len(self._bars)

    def __getitem__(self, index):
        return self._bars[index]

    def __iter__(self):
        return iter(self._bars)

    def __eq__(self, other) -> bool:
        if isinstance(other, ZigzagBarcode):
            return self._bars == other._bars
        return list(self._bars) == list(other)

    @property
    def max_dim(self) -> int:
        """The largest dimension with an interval (-1 if there are none)."""
        return max((b.dimension for b in self._bars), default=-1)

    def in_dimension(self, dim: int) -> list[ZigzagBar]:
        """The intervals of dimension `dim`."""
        return [b for b in self._bars if b.dimension == dim]

    def diagram(self, dim: int) -> np.ndarray:
        """The intervals of dimension `dim` as an array of shape `(k, 2)` of `(birth, death)` positions (inclusive)."""
        return np.array([(b.birth, b.death) for b in self.in_dimension(dim)], dtype=float).reshape(-1, 2)

    def rank(self, position: int, dim: int) -> int:
        """The Betti number of dimension `dim` at `position`: the number of intervals containing it."""
        return sum(1 for b in self._bars if b.dimension == dim and b.alive_at(position))

    def betti_curve(self, dim: int) -> np.ndarray:
        """The Betti numbers of dimension `dim` at every position."""
        curve = np.zeros(self.n_positions, dtype=int)
        for b in self.in_dimension(dim):
            curve[b.birth : b.death + 1] += 1
        return curve

    def to_dionysus(self) -> list[list[tuple[int, float]]]:
        """Dionysus 2's convention (`zigzag_homology_persistence` with operation `i` at time `i + 1`): by dimension,
        `(birth, death + 1)` pairs, with `inf` for intervals alive at the end."""
        out: list[list[tuple[int, float]]] = [[] for _ in range(self.max_dim + 1)]
        last = self.n_positions - 1
        for b in self._bars:
            out[b.dimension].append((b.birth, float("inf") if b.death == last else b.death + 1))
        return out

    def plot(self, ax: Any = None) -> Any:
        """Draw the intervals with matplotlib over the positions, grouped and coloured by dimension, with closed ends
        as filled dots and open ends as hollow ones. Returns the axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 3))
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        y = 0
        for dim in range(self.max_dim + 1):
            colour = colours[dim % len(colours)]
            for i, b in enumerate(self.in_dimension(dim)):
                ax.plot([b.birth, b.death], [y, y], color=colour, lw=2, label=f"H{dim}" if i == 0 else None)
                for x, closed in ((b.birth, b.birth_closed), (b.death, b.death_closed)):
                    ax.plot([x], [y], "o", ms=4, color=colour, mfc=colour if closed else "white")
                y += 1
            y += 1
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.n_positions - 0.5)
        ax.set_xlabel("position")
        if self._bars:
            ax.legend(loc="lower right", fontsize="small", frameon=False)
        return ax

    def __repr__(self) -> str:
        counts = ", ".join(f"H{d}: {len(self.in_dimension(d))}" for d in range(self.max_dim + 1))
        return f"ZigzagBarcode({len(self)} intervals over {self.n_positions} positions{'; ' + counts if counts else ''})"
