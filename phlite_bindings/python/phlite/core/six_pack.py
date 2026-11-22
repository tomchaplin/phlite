"""Six-packs of persistence barcodes, for a filtration-preserving map `f: L -> K` (an inclusion of a subcomplex, a
simplicial map, a map of point clouds, ...): the barcodes of

* `domain`: `L`;
* `codomain`: `K`;
* `image`, `kernel`, `cokernel`: of `f_*: H(L_t) -> H(K_t)`;
* `relative`: of the map (its mapping cone; for an inclusion, the pair `(K, L)`).

Each bar has a representative (if asked for): a cycle of `L` for the domain, image and kernel (an image bar's cycle
maps to the image class; a kernel bar's maps to a boundary); a cycle of `K` for the codomain and cokernel; a relative
cycle for the relative barcode, on cells of `K` and cones `Cone(σ)` on cells `σ` of `L`. A representative is a list of
`(cell, coefficient)` pairs.

```python
pack = filtered.six_pack(subcomplex)
pack.kernel.diagram(0)            # an array of (birth, death)
for bar in pack.image.in_dimension(1):
    print(bar.birth, bar.death, bar.representative)
pack.ranks(0.5, dim=1)            # how many bars of each barcode are alive at 0.5
```
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NamedTuple, Optional

import numpy as np

__all__ = ["SixPack", "PackBarcode", "PackBar", "Cone"]

NAMES = ("domain", "codomain", "image", "kernel", "cokernel", "relative")


class Cone(NamedTuple):
    """The cone on a cell of the domain, a cell of the mapping cone one dimension up (for a map of complexes, the
    cylinder `σ × [0, 1]`), as appears in representatives of the relative barcode."""

    cell: Any

    def __repr__(self) -> str:
        return f"Cone({self.cell!r})"


class PackBar:
    """A bar of one of the six barcodes: homology of `dimension` born at `birth` and dying at `death` (`inf` if it never
    does), with a representative (`(cell, coefficient)` pairs), or `None` if representatives weren't asked for."""

    __slots__ = ("dimension", "birth", "death", "representative")

    def __init__(self, dimension: int, birth: float, death: float, representative: Optional[list] = None):
        self.dimension = dimension
        self.birth = birth
        self.death = death
        self.representative = representative

    def __eq__(self, other) -> bool:
        if not isinstance(other, PackBar):
            return NotImplemented
        return (self.dimension, self.birth, self.death, self.representative) == (
            other.dimension, other.birth, other.death, other.representative)

    __hash__ = None

    @property
    def essential(self) -> bool:
        """Whether the bar never dies."""
        return math.isinf(self.death)

    @property
    def persistence(self) -> float:
        """Its length, `death - birth`."""
        return self.death - self.birth

    def alive_at(self, t: float) -> bool:
        """Whether the bar is alive at `t`: `birth <= t < death`."""
        return self.birth <= t < self.death

    @property
    def cells(self) -> Optional[list]:
        """The cells of the representative, without coefficients."""
        return None if self.representative is None else [cell for cell, _ in self.representative]

    def __repr__(self) -> str:
        return f"PackBar(dimension={self.dimension}, birth={self.birth!r}, death={self.death!r})"


class PackBarcode(Sequence):
    """One barcode of a six-pack: a sequence of `PackBar`s, sorted by (dimension, birth, death)."""

    def __init__(self, name: str, bars: Sequence[PackBar]):
        self.name = name
        self._bars = tuple(sorted(bars, key=lambda b: (b.dimension, b.birth, b.death)))

    def __len__(self) -> int:
        return len(self._bars)

    def __getitem__(self, index):
        return self._bars[index]

    def __iter__(self) -> Iterator[PackBar]:
        return iter(self._bars)

    @property
    def bars(self) -> tuple[PackBar, ...]:
        return self._bars

    @property
    def max_dim(self) -> int:
        """The largest dimension with a bar (-1 if there are none)."""
        return max((b.dimension for b in self._bars), default=-1)

    def in_dimension(self, dim: int) -> list[PackBar]:
        """The bars of dimension `dim`."""
        return [b for b in self._bars if b.dimension == dim]

    def diagram(self, dim: int) -> np.ndarray:
        """The persistence diagram in dimension `dim`: an array of shape `(k, 2)` of `(birth, death)`, sorted, with
        death `inf` for essential bars."""
        return np.array([(b.birth, b.death) for b in self.in_dimension(dim)], dtype=float).reshape(-1, 2)

    def diagrams(self, max_dim: Optional[int] = None) -> list[np.ndarray]:
        """The diagrams in dimensions `0, ..., max_dim` (default: the largest with a bar)."""
        top = self.max_dim if max_dim is None else max_dim
        return [self.diagram(d) for d in range(top + 1)]

    def essential(self, dim: Optional[int] = None) -> list[PackBar]:
        """The bars that never die (in dimension `dim`, or all)."""
        return [b for b in self._bars if b.essential and (dim is None or b.dimension == dim)]

    def finite(self, dim: Optional[int] = None) -> list[PackBar]:
        """The bars that die (in dimension `dim`, or all)."""
        return [b for b in self._bars if not b.essential and (dim is None or b.dimension == dim)]

    def representatives(self, dim: int) -> list:
        """The representatives of the bars of dimension `dim`, in the order of `in_dimension(dim)`."""
        return [b.representative for b in self.in_dimension(dim)]

    def rank(self, t: float, dim: int) -> int:
        """The number of bars of dimension `dim` alive at `t`: the rank of this barcode's homology there."""
        return sum(1 for b in self._bars if b.dimension == dim and b.alive_at(t))

    def betti(self, dim: Optional[int] = None) -> list[int] | int:
        """The number of essential bars in each dimension (or in `dim`)."""
        if dim is not None:
            return len(self.essential(dim))
        return [len(self.essential(d)) for d in range(self.max_dim + 1)]

    def to_gudhi(self) -> list[tuple[int, tuple[float, float]]]:
        """The bars as GUDHI's `persistence()` gives them: `(dimension, (birth, death))`."""
        return [(b.dimension, (b.birth, b.death)) for b in self._bars]

    def __repr__(self) -> str:
        counts = ", ".join(f"H{d}: {len(self.in_dimension(d))}" for d in range(self.max_dim + 1))
        return f"PackBarcode({self.name}, {len(self)} bars{'; ' + counts if counts else ''})"


class SixPack(Mapping):
    """The six barcodes of a map `f: L -> K`: `domain`, `codomain`, `image`, `kernel`, `cokernel` and `relative`, each a
    `PackBarcode`. A mapping from those names to the barcodes, in that order."""

    def __init__(self, barcodes: Mapping[str, PackBarcode], representatives: bool):
        self._barcodes = {name: barcodes[name] for name in NAMES}
        self.has_representatives = representatives

    @classmethod
    def _from_parts(cls, parts: Mapping[str, Sequence[tuple]], representatives: bool) -> "SixPack":
        """From each barcode's bars as `(dimension, birth, death, representative)`, with representatives lists of
        `(cell, coefficient)`, or `(cell, coefficient, True)` for the cone on a cell of the domain."""

        def cone(chain):
            return None if chain is None else [(Cone(c[0]), c[1]) if len(c) == 3 else c for c in chain]

        barcodes = {
            name: PackBarcode(
                name, [PackBar(d, b, e, cone(r) if name == "relative" else r) for d, b, e, r in parts[name]]
            )
            for name in NAMES
        }
        return cls(barcodes, representatives)

    @property
    def domain(self) -> PackBarcode:
        """The barcode of `L`."""
        return self._barcodes["domain"]

    @property
    def codomain(self) -> PackBarcode:
        """The barcode of `K`."""
        return self._barcodes["codomain"]

    @property
    def image(self) -> PackBarcode:
        """The barcode of the image of `H(L) -> H(K)`, represented by cycles of `L`."""
        return self._barcodes["image"]

    @property
    def kernel(self) -> PackBarcode:
        """The barcode of the kernel of `H(L) -> H(K)`, represented by cycles of `L` that map to boundaries."""
        return self._barcodes["kernel"]

    @property
    def cokernel(self) -> PackBarcode:
        """The barcode of the cokernel of `H(L) -> H(K)`, represented by cycles of `K`."""
        return self._barcodes["cokernel"]

    @property
    def relative(self) -> PackBarcode:
        """The barcode of the map's relative homology (the mapping cone; `(K, L)` for an inclusion)."""
        return self._barcodes["relative"]

    def __getitem__(self, name: str) -> PackBarcode:
        return self._barcodes[name]

    def __iter__(self) -> Iterator[str]:
        return iter(NAMES)

    def __len__(self) -> int:
        return len(NAMES)

    def ranks(self, t: float, dim: int) -> dict[str, int]:
        """The number of bars of dimension `dim` of each barcode alive at `t`: the ranks of `H(L_t)`, `H(K_t)`, the image,
        kernel and cokernel of `H(L_t) -> H(K_t)`, and the relative homology. (They satisfy `domain = image + kernel`
        and `codomain = image + cokernel`.)"""
        return {name: barcode.rank(t, dim) for name, barcode in self._barcodes.items()}

    def __repr__(self) -> str:
        top = max(b.max_dim for b in self._barcodes.values())
        header = "SixPack" + ("" if self.has_representatives else " (no representatives)")
        if top < 0:
            return f"{header}: no bars"
        width = max(map(len, NAMES))
        lines = [header + ", bars by dimension:"]
        lines.append(" " * (width + 2) + " ".join(f"{'H' + str(d):>6}" for d in range(top + 1)))
        for name, barcode in self._barcodes.items():
            lines.append(f"  {name:<{width}}" + " ".join(f"{len(barcode.in_dimension(d)):>6}" for d in range(top + 1)))
        return "\n".join(lines)
