"""Spectral sequences, computed one page at a time.

A `SpectralSequence` is returned by the packages (`phlite.complexes.FilteredComplex.spectral_sequence`,
`phlite.networks.magnitude_path_spectral_sequence`); nothing is computed until a page is asked for:

```python
ss = filtered.spectral_sequence()
e1 = ss.next()          # E^1, then E^2 with the next call, ...
print(e1)               # a table of the page, with its differentials
e1[2, 1]                # the entry at step p = 2 in degree k = 1
g = e1.group(2, 1)      # ... as a group: g.rank, g.torsion, g.elementary_divisors, str(g) = "Z^3 ⊕ Z/2"
g.generators            # a generator of each cyclic summand: .order (None if free) and .chain ({cell: coefficient})
g.classify(chain)       # the class of a chain, as coordinates on the generators
g.d(chain)              # d^r of its class, as coordinates on the generators of g.d_target
ss.page(3)              # any page, directly
for page in ss: ...     # every page, until the sequence degenerates
ss.infinity             # E^infinity
e1.plot()               # a chart, the differentials as arrows (needs matplotlib)
```

Over a field each entry is a dimension; over the integers it is a group, given as `(rank, torsion)`, the free rank and
the invariant factors of the torsion. The page `E^r` has differentials `d^r` from step `p` in degree `k` to step
`p - r` in degree `k - 1`, and `E^{r+1}` is their homology.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence, Union

__all__ = ["SpectralSequence", "Page", "Group", "Generator", "Differential", "Entry"]

#: An entry: a dimension (over a field), `(rank, torsion)` (over the integers), or `None` if not determined.
Entry = Union[int, tuple[int, list[int]], None]


def _format(entry: Entry, integral: bool) -> str:
    """An entry as in the tables: a rank as a number, torsion as `Z/n`, zero as `.`, unknown as `?`."""
    if entry is None:
        return "?"
    if integral:
        rank, torsion = entry
        parts = ([str(rank)] if rank else []) + [f"Z/{t}" for t in torsion]
        return "⊕".join(parts) if parts else "."
    return str(entry) if entry else "."


def _is_zero(entry: Entry) -> bool:
    if entry is None:
        return False
    if isinstance(entry, tuple):
        return entry[0] == 0 and not entry[1]
    return entry == 0


def _prime_powers(n: int) -> list[int]:
    """The prime-power factors of `n` (its elementary divisors as a cyclic group)."""
    out, p = [], 2
    while p * p <= n:
        if n % p == 0:
            q = 1
            while n % p == 0:
                n //= p
                q *= p
            out.append(q)
        p += 1
    if n > 1:
        out.append(n)
    return out


@dataclass(frozen=True)
class Generator:
    """A generator of one cyclic summand of a `Group`: its `order` (`None` for a free summand; over a field every
    summand is free) and a `chain` representing it, a dict from cells (the complex's own: input cells, or tuples of
    nodes for the MPSS) to coefficients. The chain `x` has degree `k`, lies in `F_p` and has `dx` in `F_{p-r}`; it is
    determined only modulo `Z^{r-1}_{p-1} + d Z^{r-1}_{p+r-1}` (so not canonically, over the integers)."""

    order: Optional[int]
    chain: dict

    def __repr__(self) -> str:
        order = "∞" if self.order is None else str(self.order)
        terms = ""
        for i, (cell, c) in enumerate(list(self.chain.items())[:6]):
            sign = ("-" if c < 0 else "") if i == 0 else (" - " if c < 0 else " + ")
            terms += f"{sign}{abs(c)}·{cell}"
        more = f" + ... ({len(self.chain)} terms)" if len(self.chain) > 6 else ""
        return f"<Generator of order {order}: {terms}{more}>"


class Group:
    """An entry `E^r_p` (degree `k`) of a spectral sequence as a group: its isomorphism class (`rank`, `torsion`, the
    invariant factors with each dividing the next, and `elementary_divisors`), generators of its cyclic summands, and
    the class of any chain. Over a field it is a vector space: `rank` is its dimension and there is no torsion.

    Groups compare equal when isomorphic (to another `Group`, a `(rank, torsion)` pair, or over a field an `int`)."""

    def __init__(self, sequence: "SpectralSequence", r: int, p: int, k: int, entry: Entry):
        if entry is None:
            raise ValueError(f"E^{r} at ({p}, {k}) is not determined (the filtration is truncated)")
        self.sequence, self.r, self.p, self.k = sequence, r, p, k
        rank, torsion = entry if isinstance(entry, tuple) else (entry, [])
        #: The free rank (over a field, the dimension).
        self.rank: int = rank
        #: The invariant factors of the torsion, each dividing the next.
        self.torsion: list[int] = list(torsion)
        self._generators: Optional[list[Generator]] = None

    @property
    def elementary_divisors(self) -> list[int]:
        """The torsion as prime powers, sorted: `Z/6 ⊕ Z/12` gives `[2, 3, 3, 4]`."""
        return sorted(q for t in self.torsion for q in _prime_powers(t))

    @property
    def is_zero(self) -> bool:
        return self.rank == 0 and not self.torsion

    @property
    def order(self) -> Optional[int]:
        """The number of elements, `None` if infinite."""
        if self.rank:
            return None if self.sequence.integral or self.sequence.modulus is None else self.sequence.modulus ** self.rank
        out = 1
        for t in self.torsion:
            out *= t
        return out

    def _invariants(self) -> tuple[int, tuple[int, ...]]:
        return self.rank, tuple(self.elementary_divisors)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Group):
            return self._invariants() == other._invariants()
        if isinstance(other, tuple) and len(other) == 2:
            rank, torsion = other
            return self._invariants() == (rank, tuple(sorted(q for t in torsion for q in _prime_powers(t))))
        if isinstance(other, int) and not isinstance(other, bool):
            return not self.torsion and self.rank == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._invariants())

    def __str__(self) -> str:
        if self.sequence.integral:
            parts = ([{1: "Z"}.get(self.rank, f"Z^{self.rank}")] if self.rank else []) + [f"Z/{t}" for t in self.torsion]
            return " ⊕ ".join(parts) if parts else "0"
        if not self.rank:
            return "0"
        field = f"F_{self.sequence.modulus}" if self.sequence.modulus else "F"
        return field if self.rank == 1 else f"{field}^{self.rank}"

    def __repr__(self) -> str:
        s, d = self.sequence.step_label, self.sequence.degree_label
        return f"<Group E^{self.r} at ({s}={self.p}, {d}={self.k}): {self}>"

    @property
    def generators(self) -> list[Generator]:
        """A generator of each cyclic summand, torsion first (in the order of `torsion`), then free. Computed (exactly,
        from the cells in the entry's window of steps) when first asked for."""
        if self._generators is None:
            backend = self.sequence._backend
            if not backend.has_generators:
                raise ValueError("this spectral sequence has no generators")
            result = backend.generators(self.r, self.p, self.k)
            if result is None:
                raise ValueError("not determined (the filtration is truncated)")
            _, generators = result
            self._generators = [Generator(order, chain) for order, chain in generators]
        return self._generators

    def classify(self, chain: Any) -> list[int]:
        """The class of `chain` (a dict from cells to coefficients, or `(cell, coefficient)` pairs) in this group, as
        its coordinates on `generators` (each reduced modulo the generator's order).

        Raises:
            ValueError: If the chain doesn't represent an element: it isn't of degree `k`, in `F_p`, with boundary in
                `F_{p-r}` (or a cell isn't one of the complex's).
        """
        out = self.sequence._backend.classify(self.r, self.p, self.k, chain)
        if out is None:
            raise ValueError(f"the chain doesn't represent an element of E^{self.r} at ({self.p}, {self.k})")
        return out

    def contains(self, chain: Any) -> bool:
        """Whether `chain` represents an element (see `classify`)."""
        return self.sequence._backend.classify(self.r, self.p, self.k, chain) is not None

    @property
    def d_target(self) -> Optional["Group"]:
        """The group `d^r` maps this one to, `E^r` at `(p - r, k - 1)` (`None` if there is none: `p < r` or `k = 0`)."""
        if self.p < self.r or self.k == 0:
            return None
        return self.sequence.page(self.r).group(self.p - self.r, self.k - 1)

    def d(self, chain: Any) -> list[int]:
        """The differential `d^r` on the class of `chain`: the class of its boundary in `d_target`, as coordinates on
        that group's generators (empty if there is no target).

        Raises:
            ValueError: If the chain doesn't represent an element of this group.
        """
        self.classify(chain)
        target = self.d_target
        if target is None:
            return []
        return target.classify(self.sequence._backend.boundary(chain))


@dataclass(frozen=True)
class Differential:
    """A non-zero differential `d^r` of a page, from `source` to `target` (each `(p, k)`), with its image."""

    r: int
    source: tuple[int, int]
    target: tuple[int, int]
    image: Entry


class Page:
    """The page `E^r` of a `SpectralSequence`: its entries, indexed `page[p, k]` (step, degree), and its non-zero
    differentials. Printing it shows a table like gramag's (steps down, degrees across); in a notebook it shows as an
    HTML table, and `plot()` draws it with matplotlib."""

    def __init__(self, sequence: "SpectralSequence", r: int, rows: list[list[Entry]], differentials: list[Differential]):
        self.sequence = sequence
        #: The page number.
        self.r = r
        #: The entries: `rows[p][k]` (rows may be shorter than the number of degrees).
        self.rows = rows
        #: The non-zero differentials `d^r` (those not determined are left out).
        self.differentials = differentials

    def __getitem__(self, key: tuple[int, int]) -> Entry:
        p, k = key
        if not 0 <= p < len(self.rows) or not 0 <= k < len(self.rows[p]):
            raise KeyError(f"({p}, {k}) is not an entry of the page")
        return self.rows[p][k]

    def group(self, p: int, k: int) -> Group:
        """The entry at `(p, k)` as a `Group`: its isomorphism class, generators and the class of any chain."""
        return Group(self.sequence, self.r, p, k, self[p, k])

    def rank(self, p: int, k: int) -> Optional[int]:
        """The dimension (over a field) or free rank (over the integers) at `(p, k)`, or `None` if not determined."""
        entry = self[p, k]
        return entry[0] if isinstance(entry, tuple) else entry

    def torsion(self, p: int, k: int) -> list[int]:
        """The invariant factors of the torsion at `(p, k)` (always empty over a field)."""
        entry = self[p, k]
        return list(entry[1]) if isinstance(entry, tuple) else []

    def entries(self) -> Iterator[tuple[tuple[int, int], Entry]]:
        """Every non-zero entry, as `((p, k), entry)`."""
        for p, row in enumerate(self.rows):
            for k, entry in enumerate(row):
                if not _is_zero(entry):
                    yield (p, k), entry

    @property
    def is_degenerate(self) -> bool:
        """Whether every differential of this page is (known to be) zero, so that the next page is the same."""
        return not self.differentials

    def _row_label(self, p: int) -> str:
        sequence = self.sequence
        label = f"{sequence.step_label}={p}"
        steps = sequence.steps
        if steps is not None and steps[p] != p:
            label += f" ({steps[p]:g})" if isinstance(steps[p], float) else f" ({steps[p]})"
        return label

    def table(self) -> str:
        """The page as a text table: steps down, degrees across, zero as `.` and unknown as `?`."""
        integral = self.sequence.integral
        n_degrees = max((len(row) for row in self.rows), default=0)
        cells = [[_format(entry, integral) for entry in row] for row in self.rows]
        widths = [max([len(str(k))] + [len(row[k]) for row in cells if k < len(row)]) for k in range(n_degrees)]
        labels = [self._row_label(p) for p in range(len(self.rows))]
        corner = f"{self.sequence.degree_label}="
        label_width = max([len(corner)] + [len(label) for label in labels])

        def line(label: str, values: Sequence[str]) -> str:
            body = "  ".join(v.ljust(w) for v, w in zip(values, widths))
            body = body.ljust(sum(widths) + 2 * max(n_degrees - 1, 0))
            return f"│ {label.ljust(label_width)} │ {body} │"

        inner = sum(widths) + 2 * max(n_degrees - 1, 0)
        out = [
            f"╭─{'─' * label_width}─┬─{'─' * inner}─╮",
            line(corner, [str(k) for k in range(n_degrees)]),
            f"├─{'─' * label_width}─┼─{'─' * inner}─┤",
        ]
        out += [line(label, row) for label, row in zip(labels, cells)]
        out.append(f"╰─{'─' * label_width}─┴─{'─' * inner}─╯")
        return "\n".join(out)

    def _differential_lines(self, limit: int = 20) -> list[str]:
        s, d = self.sequence.step_label, self.sequence.degree_label
        lines = [
            f"d^{self.r}: ({s}={a.source[0]}, {d}={a.source[1]}) → ({s}={a.target[0]}, {d}={a.target[1]}), image "
            f"{_format(a.image, self.sequence.integral)}"
            for a in self.differentials[:limit]
        ]
        if len(self.differentials) > limit:
            lines.append(f"... and {len(self.differentials) - limit} more")
        return lines

    def _name(self, sup: str = "^{}", infinity: str = "∞") -> str:
        name = "E" + sup.format(self.r)
        last = self.sequence.degenerates_at
        if last is not None and self.r >= last:
            name += " = E" + sup.format(infinity)
        return name

    def __str__(self) -> str:
        title = self._name() + (f" of {self.sequence.title}" if self.sequence.title else "")
        lines = [title, self.table()]
        if self.differentials:
            lines += self._differential_lines()
        else:
            lines.append(f"(every d^{self.r} is zero)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Page E^{self.r}: {sum(1 for _ in self.entries())} non-zero entries, {len(self.differentials)} non-zero differentials>"

    def _repr_html_(self) -> str:
        integral = self.sequence.integral
        n_degrees = max((len(row) for row in self.rows), default=0)
        sources = {a.source for a in self.differentials}
        targets = {a.target for a in self.differentials}
        style = (
            "border-collapse:collapse;font-family:monospace;text-align:center;"
            "--ss-line:color-mix(in srgb,currentColor 25%,transparent)"
        )
        cell = "border:1px solid var(--ss-line);padding:2px 8px"
        head = "".join(f'<th style="{cell}">{k}</th>' for k in range(n_degrees))
        body = []
        for p, row in enumerate(self.rows):
            tds = []
            for k in range(n_degrees):
                if k >= len(row):
                    tds.append(f'<td style="{cell}"></td>')
                    continue
                entry = row[k]
                shade = ""
                if (p, k) in sources and (p, k) in targets:
                    shade = "background:rgba(160,90,220,.22);"
                elif (p, k) in sources:
                    shade = "background:rgba(220,90,60,.20);"
                elif (p, k) in targets:
                    shade = "background:rgba(60,120,220,.20);"
                elif not _is_zero(entry):
                    shade = "background:rgba(128,128,128,.12);"
                weight = "" if _is_zero(entry) else "font-weight:600;"
                tds.append(f'<td style="{cell};{shade}{weight}">{html.escape(_format(entry, integral))}</td>')
            body.append(f'<tr><th style="{cell};text-align:left">{html.escape(self._row_label(p))}</th>{"".join(tds)}</tr>')
        title = self._name("<sup>{}</sup>", "&infin;") + (
            f" of {html.escape(self.sequence.title)}" if self.sequence.title else ""
        )
        corner = html.escape(f"{self.sequence.degree_label}=")
        table = (
            f'<table style="{style}"><caption style="caption-side:top;text-align:left;font-weight:600">{title}</caption>'
            f'<tr><th style="{cell}">{corner}</th>{head}</tr>{"".join(body)}</table>'
        )
        if self.differentials:
            legend = (
                '<div style="font-size:smaller;opacity:.8">sources of d<sup>r</sup> shaded red, targets blue</div>'
            )
            items = "".join(f"<li>{html.escape(line)}</li>" for line in self._differential_lines())
            return f"{table}{legend}<ul style='font-family:monospace;font-size:smaller'>{items}</ul>"
        return f"{table}<div style='font-size:smaller;opacity:.8'>every d<sup>{self.r}</sup> is zero</div>"

    def plot(self, ax: Any = None, *, show_zero: bool = False) -> Any:
        """Draw the page with matplotlib: degrees across, steps down (as in the table), each non-zero entry labelled,
        and each non-zero differential `d^r` as an arrow from its source to its target, labelled with its image.
        Returns the axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            n_degrees = max((len(row) for row in self.rows), default=1)
            _, ax = plt.subplots(figsize=(1 + 0.9 * n_degrees, 1 + 0.6 * len(self.rows)))
        integral = self.sequence.integral
        for p, row in enumerate(self.rows):
            for k, entry in enumerate(row):
                if _is_zero(entry):
                    if show_zero:
                        ax.plot(k, p, ".", color="0.75")
                    continue
                ax.annotate(
                    _format(entry, integral),
                    (k, p),
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3", lw=0.8),
                    zorder=3,
                )
        for a in self.differentials:
            ax.annotate(
                "",
                xy=(a.target[1], a.target[0]),
                xytext=(a.source[1], a.source[0]),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2, shrinkA=10, shrinkB=10),
                zorder=2,
            )
            # The image, at the middle of the arrow
            ax.annotate(
                _format(a.image, integral),
                ((a.source[1] + a.target[1]) / 2, (a.source[0] + a.target[0]) / 2),
                xytext=(6, 0),
                textcoords="offset points",
                color="tab:red",
                fontsize="small",
                va="center",
            )
        n_degrees = max((len(row) for row in self.rows), default=1)
        ax.set_xlim(-0.6, n_degrees - 0.4)
        ax.set_ylim(len(self.rows) - 0.4, -0.6)
        ax.set_xticks(range(n_degrees))
        ax.set_yticks(range(len(self.rows)))
        ax.set_yticklabels([self._row_label(p) for p in range(len(self.rows))])
        ax.set_xlabel(self.sequence.degree_label)
        ax.set_ylabel(self.sequence.step_label)
        ax.grid(True, color="0.92", zorder=0)
        ax.set_axisbelow(True)
        title = self._name("$^{{{}}}$", r"\infty") + (f" of {self.sequence.title}" if self.sequence.title else "")
        ax.set_title(title)
        return ax


class SpectralSequence:
    """A spectral sequence, computed one page at a time (see the module docs).

    Pages are computed when first asked for, and kept. `next()` moves on to the next page; iterating gives every page
    until the sequence degenerates (when it is known to: a page from which every differential is zero, `E^infinity`).
    """

    def __init__(
        self,
        backend: Any,
        *,
        steps: Optional[Sequence[Any]] = None,
        step_label: str = "p",
        degree_label: str = "k",
        title: Optional[str] = None,
    ):
        self._backend = backend
        #: The filtration value of each step, if the steps stand for values (else `None`).
        self.steps = list(steps) if steps is not None else None
        #: The names of the axes, for display.
        self.step_label = step_label
        self.degree_label = degree_label
        self.title = title
        self._pages: dict[int, Page] = {}
        self._current = 0

    @property
    def integral(self) -> bool:
        """Whether entries are groups over the integers (`(rank, torsion)`) rather than dimensions over a field."""
        return self._backend.integral

    @property
    def n_steps(self) -> int:
        return self._backend.n_steps

    @property
    def n_degrees(self) -> int:
        return self._backend.n_degrees

    @property
    def modulus(self) -> Optional[int]:
        """The characteristic of the coefficient field, `None` over the integers."""
        return self._backend.modulus

    @property
    def degenerates_at(self) -> Optional[int]:
        """A page from which every differential is zero, so that it is `E^infinity` (if known)."""
        return self._backend.degenerates_at

    def page(self, r: int) -> Page:
        """The page `E^r` (`r >= 1`), computed if it hasn't been."""
        if r < 1:
            raise ValueError(f"pages start at E^1, got r = {r}")
        if r not in self._pages:
            rows = self._backend.page(r)
            differentials = [
                Differential(r, (p, k), (p - r, k - 1), image) for p, k, image in self._backend.differentials(r)
            ]
            self._pages[r] = Page(self, r, rows, differentials)
        return self._pages[r]

    def next(self) -> Page:
        """Compute the next page (`E^1` first) and make it the current one."""
        self._current += 1
        return self.page(self._current)

    @property
    def current(self) -> Optional[Page]:
        """The last page `next()` gave (`None` before the first call)."""
        return self._pages.get(self._current) if self._current else None

    def __iter__(self) -> Iterator[Page]:
        r = 1
        while True:
            page = self.page(r)
            yield page
            last = self.degenerates_at
            if (last is not None and r >= last) or (last is None and page.is_degenerate and r > self.n_steps):
                return
            r += 1

    @property
    def infinity(self) -> Page:
        """`E^infinity`, the page from which every differential is zero."""
        last = self.degenerates_at
        return self.page(last if last is not None else self.n_steps + 1)

    def __repr__(self) -> str:
        over = "the integers" if self.integral else "a field"
        computed = ", ".join(f"E^{r}" for r in sorted(self._pages)) or "none"
        name = f" ({self.title})" if self.title else ""
        return f"<SpectralSequence{name} over {over}: {self.n_steps} steps, {self.n_degrees} degrees; pages computed: {computed}>"

    def _repr_html_(self) -> str:
        page = self.current
        intro = html.escape(repr(self))
        return intro if page is None else f"<div>{intro}</div>{page._repr_html_()}"
