"""Magnitude homology results: `MagnitudeHomology`, what `phlite.networks.magnitude` returns.

```python
from phlite.networks import magnitude

mh = magnitude(G, l_max=4)
print(mh)                      # the ranks as a table, k across and l down (as gramag prints them)
mh.rank(1, 2)                  # the rank of MH_{1,2}
mh[2][1]                       # the same: indexed [l][k], as a list of lists
mh.magnitude_coefficients()    # sum_k (-1)^k rank MH_{k,l}: the coefficients of the magnitude in powers of q

mh = magnitude(G, l_max=4, integral=True)
mh.group(2, 3)                 # e.g. "Z^4 ⊕ Z/2"
```
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np

__all__ = ["MagnitudeHomology"]


class MagnitudeHomology(Sequence):
    """The ranks of magnitude homology `MH_{k,l}` for lengths `l = 0, ..., l_max` and degrees `k <= min(l, k_max)`,
    over a field (`modulus`) or the integers (`modulus == 0`, with `torsion`).

    A sequence over `l` of lists over `k`, so `mh[l][k]` is the rank of `MH_{k,l}` (`None` where over the integers the
    Smith form overflowed). Compares equal to such a list of lists.
    """

    def __init__(self, ranks: Sequence[Sequence[Optional[int]]], modulus: int = 2,
                 torsion: Optional[Mapping[tuple[int, int], Mapping[int, int]]] = None):
        self._ranks = [list(row) for row in ranks]
        self.modulus = modulus
        self._torsion = {tuple(k): dict(v) for k, v in (torsion or {}).items()}

    # ---- As a list of lists
    def __getitem__(self, index):
        if isinstance(index, str):
            # (The dict this used to be over the integers)
            if index == "free":
                return self._ranks
            if index == "torsion":
                return self._torsion
            raise KeyError(index)
        return self._ranks[index]

    def __len__(self) -> int:
        return len(self._ranks)

    def __eq__(self, other) -> bool:
        if isinstance(other, MagnitudeHomology):
            return (self._ranks, self.modulus, self._torsion) == (other._ranks, other.modulus, other._torsion)
        if isinstance(other, Mapping):
            # (The dict this used to be over the integers)
            return self.integral and set(other) == {"free", "torsion"} and self._ranks == other["free"] and (
                self._torsion == {tuple(k): dict(v) for k, v in other["torsion"].items()}
            )
        return self._ranks == other

    __hash__ = None

    # ---- Entries
    @property
    def l_max(self) -> int:
        return len(self._ranks) - 1

    @property
    def integral(self) -> bool:
        """Whether over the integers."""
        return self.modulus == 0

    def rank(self, k: int, l: int) -> Optional[int]:
        """The rank (dimension over the field; free rank over the integers) of `MH_{k,l}`; 0 outside the table."""
        if 0 <= l < len(self._ranks) and 0 <= k < len(self._ranks[l]):
            return self._ranks[l][k]
        return 0

    def torsion(self, k: int, l: int) -> dict[int, int]:
        """Over the integers, the torsion of `MH_{k,l}`: `{order: multiplicity}` of its invariant factors."""
        return dict(self._torsion.get((k, l), {}))

    def group(self, k: int, l: int) -> str:
        """`MH_{k,l}` written out, e.g. `"Z^4 ⊕ Z/2"`, or `"F_3^2"` over a field."""
        r = self.rank(k, l)
        if r is None:
            return "?"
        base = "Z" if self.integral else f"F_{self.modulus}"
        parts = [] if r == 0 else [base if r == 1 else f"{base}^{r}"]
        for order, mult in sorted(self.torsion(k, l).items()):
            parts.append(f"Z/{order}" if mult == 1 else f"(Z/{order})^{mult}")
        return " ⊕ ".join(parts) or "0"

    def magnitude_coefficients(self) -> list[Optional[int]]:
        """`sum_k (-1)^k rank MH_{k,l}` for each `l`: the coefficient of `q^l` in the magnitude of the graph (by
        Hepworth and Willerton's categorification), when `k_max >= l`; `None` where a rank is unknown."""
        out = []
        for row in self._ranks:
            out.append(None if any(r is None for r in row) else sum((-1) ** k * r for k, r in enumerate(row)))
        return out

    def to_array(self) -> np.ndarray:
        """The ranks as a float array indexed `[l, k]`, with `nan` where unknown or out of range (`k > l`)."""
        width = max((len(row) for row in self._ranks), default=0)
        out = np.full((len(self._ranks), width), np.nan)
        for l, row in enumerate(self._ranks):
            for k, r in enumerate(row):
                if r is not None:
                    out[l, k] = r
        return out

    def table(self) -> str:
        """The ranks as a table: `k` across, `l` down; over the integers with torsion, e.g. `3+Z/2`."""
        width = max((len(row) for row in self._ranks), default=0)

        def entry(k, l):
            r = self._ranks[l][k]
            text = "?" if r is None else str(r)
            for order, mult in sorted(self.torsion(k, l).items()):
                text += f"+Z/{order}" + (f"^{mult}" if mult > 1 else "")
            return text

        cells = [[entry(k, l) if k < len(row) else "" for k in range(width)] for l, row in enumerate(self._ranks)]
        w = max([len(c) for row in cells for c in row] + [len(str(width - 1)), 1])
        lines = ["l\\k " + " ".join(f"{k:>{w}}" for k in range(width))]
        for l, row in enumerate(cells):
            lines.append(f"{l:>3} " + " ".join(f"{c:>{w}}" for c in row))
        return "\n".join(lines)

    def __repr__(self) -> str:
        field = "Z" if self.integral else f"Z/{self.modulus}"
        return f"MagnitudeHomology over {field}, l = 0..{self.l_max}:\n{self.table()}"

    def _repr_html_(self) -> str:
        return "<pre>" + self.__repr__().replace("&", "&amp;").replace("<", "&lt;") + "</pre>"


def wrap(result, *, modulus, by_pair: bool, integral: bool):
    """`phlite.networks.magnitude`'s result as `MagnitudeHomology` objects."""
    if integral:
        if by_pair:
            return {pair: MagnitudeHomology(r["free"], 0, r["torsion"]) for pair, r in result.items()}
        return MagnitudeHomology(result["free"], 0, result["torsion"])
    if isinstance(modulus, (list, tuple)):
        return {p: wrap(r, modulus=p, by_pair=by_pair, integral=False) for p, r in result.items()}
    p = 2 if modulus is None else modulus
    if by_pair:
        return {pair: MagnitudeHomology(r, p) for pair, r in result.items()}
    return MagnitudeHomology(result, p)
