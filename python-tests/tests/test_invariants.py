"""Properties every barcode from phlite_py must have, checked on random filtered chain complexes over Z/p.

Unlike `test_sparse.py` (which compares with PHAT and LoPHAT over Z/2, on cells listed in filtration order), the
complexes here have coefficients in Z/p and cells listed in a shuffled order, so a cell's index is not its position in
the filtration. The checks need no other library: a reference reduction, and linear algebra over Z/p for the
invariants persistence must satisfy (Betti numbers, the Euler characteristic and the rank invariant).
"""

import itertools
import random

import numpy as np
import phat
import phlite
import pytest
from hypothesis import example, given, note
from hypothesis import strategies as st

from grid import algorithm_grid
from modp import Echelon, boundary_of, kernel

METHODS = ["cohomology", "homology"]
INF = float("inf")


# ====== Random filtered complexes over Z/p ===============================


class Complex:
    """A filtered chain complex over Z/p: `boundaries[i]` is {face: coefficient}, cells ordered by (value, index)."""

    def __init__(self, p, dims, boundaries, values):
        self.p, self.dims, self.boundaries, self.values = p, dims, boundaries, values

    def __len__(self):
        return len(self.dims)

    def __repr__(self):
        return f"Complex(p={self.p}, dims={self.dims}, boundaries={self.boundaries}, values={self.values})"

    def key(self, cell):
        return (self.values[cell], cell)

    @property
    def top(self):
        return max(self.dims)

    def pairs(self):
        """For `ChainComplex.from_pairs`: (dimension, faces) and the coefficients."""
        pairs = [(d, sorted(b)) for d, b in zip(self.dims, self.boundaries)]
        coefficients = [[b[f] for f in sorted(b)] for b in self.boundaries]
        return pairs, coefficients

    def filtered(self, values=None):
        pairs, coefficients = self.pairs()
        chain = phlite.ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=self.p)
        return chain.filtration(self.values if values is None else values)

    def relabelled(self, order):
        """The same complex with cell `order[i]` renamed `i`."""
        new = {old: i for i, old in enumerate(order)}
        return Complex(
            self.p,
            [self.dims[old] for old in order],
            [{new[f]: c for f, c in self.boundaries[old].items()} for old in order],
            [self.values[old] for old in order],
        )


def random_complex(rng, p, max_dim, n_vertices, cells_per_dim, max_value):
    """Cells of each dimension get as boundary a random combination of a few cycles of the dimension below (so there
    are multi-edges, loops, zero boundaries and coefficients other than ±1), and values no smaller than their faces'.
    The cells are then listed in a random order, constrained only so that a face comes before its cofaces of equal
    value (as the filtration breaks ties by index)."""
    dims, boundaries, values = [], [], []
    for _ in range(n_vertices):
        dims.append(0)
        boundaries.append({})
        values.append(rng.randint(0, max_value))
    for dim in range(1, max_dim + 1):
        faces = [i for i, d in enumerate(dims) if d == dim - 1]
        if not faces:
            break
        cycles = kernel(faces, boundaries, p) if dim > 1 else None
        for _ in range(cells_per_dim):
            boundary = {}
            if dim == 1:
                a, b = rng.choice(faces), rng.choice(faces)
                if a != b:
                    c = rng.randrange(1, p)
                    boundary = {a: c, b: -c % p}
            else:
                for cycle in rng.sample(cycles, min(len(cycles), rng.choice([0, 1, 1, 2, 3]))):
                    c = rng.randrange(1, p)
                    for f, d in cycle.items():
                        boundary[f] = (boundary.get(f, 0) + c * d) % p
                boundary = {f: c for f, c in boundary.items() if c}
            dims.append(dim)
            boundaries.append(boundary)
            values.append(max([rng.randint(0, max_value)] + [values[f] for f in boundary]))
    complex = Complex(p, dims, boundaries, values)
    return complex.relabelled(shuffled_order(rng, complex))


def shuffled_order(rng, complex):
    """A random order of the cells in which each face comes before its cofaces of the same value: each value's cells
    in a random order respecting faces, then those sequences shuffled together at random."""
    by_value = {}
    for value in sorted(set(complex.values)):
        pending = [i for i, v in enumerate(complex.values) if v == value]
        placed, sequence = set(), []
        while pending:
            ready = [i for i in pending if all(f in placed or complex.values[f] < value for f in complex.boundaries[i])]
            cell = rng.choice(ready)
            pending.remove(cell)
            placed.add(cell)
            sequence.append(cell)
        by_value[value] = sequence[::-1]  # popped from the end
    order = []
    while by_value:
        value = rng.choice(sorted(by_value))
        order.append(by_value[value].pop())
        if not by_value[value]:
            del by_value[value]
    return order


@st.composite
def complexes(draw, moduli=(2, 3, 5), max_cells_per_dim=10):
    p = draw(st.sampled_from(moduli))
    max_dim = draw(st.integers(1, 3))
    n_vertices = draw(st.integers(1, 8))
    cells_per_dim = draw(st.integers(0, max_cells_per_dim))
    # Few distinct values means many ties, so many zero-length (and apparent) pairs
    max_value = draw(st.sampled_from([0, 1, 2, 5, 100]))
    seed = draw(st.integers(0, 2**32 - 1))
    complex = random_complex(random.Random(seed), p, max_dim, n_vertices, cells_per_dim, max_value)
    note(f"p={p} dims={complex.dims} values={complex.values} boundaries={complex.boundaries}")
    return complex


# ====== The reference =====================================================


def reference(complex):
    """The standard algorithm over Z/p, in (value, index) order: (pairs, essential), including zero-length pairs."""
    p = complex.p
    order = sorted(range(len(complex)), key=complex.key)
    owner, reduced, pairs = {}, {}, []
    for j in order:
        column = dict(complex.boundaries[j])
        while column:
            pivot = max(column, key=complex.key)
            if pivot not in owner:
                break
            other = reduced[owner[pivot]]
            factor = column[pivot] * pow(other[pivot], p - 2, p) % p
            for r, c in other.items():
                column[r] = (column.get(r, 0) - factor * c) % p
                if column[r] == 0:
                    del column[r]
        reduced[j] = column
        if column:
            pivot = max(column, key=complex.key)
            owner[pivot] = j
            pairs.append((pivot, j))
    paired = {cell for pair in pairs for cell in pair}
    return sorted(pairs), [i for i in range(len(complex)) if i not in paired]


def expected(complex, max_dim=None):
    """The reference barcode as `(pairings, essential)`, without zero-length bars, up to dimension `max_dim`."""
    pairs, essential = reference(complex)
    pairs = [(b, d) for b, d in pairs if complex.values[b] != complex.values[d]]
    if max_dim is not None:
        pairs = [(b, d) for b, d in pairs if complex.dims[b] <= max_dim]
        essential = [b for b in essential if complex.dims[b] <= max_dim]
    return pairs, essential


def bars_of(barcode):
    return (barcode.pairings, barcode.essential)


# ====== Every algorithm agrees with the reference ==========================


@given(complex=complexes(), data=st.data())
def test_every_algorithm_mod_p(complex, data):
    """Over Z/p, and with cells out of filtration order, every algorithm and option gives the reference barcode."""
    filtered = complex.filtered()
    max_dim = data.draw(st.none() | st.integers(0, complex.top))
    want = expected(complex, max_dim)
    for method, algorithm in itertools.product(METHODS, algorithm_grid(phlite.Algorithm, complex.top)):
        note(f"method={method} algorithm={algorithm!r}")
        assert bars_of(filtered.persistence(max_dim, method=method, algorithm=algorithm)) == want


# Two vertices joined by two edges, all at time 0: every pair has length zero, so is skipped, yet the involution
# (computing cohomology representatives) needs one of them. (Once gave a representative that wasn't a cycle.)
DOUBLE_EDGE = Complex(2, [0, 0, 1, 1], [{}, {}, {0: 1, 1: 1}, {0: 1, 1: 1}], [0, 0, 0, 0])


@given(complex=complexes(), data=st.data())
@example(complex=DOUBLE_EDGE, data=None).via("regression")
def test_representatives_mod_p(complex, data):
    """Every representative is a cycle in the bar's dimension, led by the birth cell, and a boundary from the death
    on (never, for essential bars). Both methods every time: cohomology computes representatives by a separate
    reduction (the involution), whatever the algorithm."""
    p = complex.p
    grid = list(algorithm_grid(phlite.Algorithm, complex.top))
    algorithm = phlite.Algorithm.serial() if data is None else data.draw(st.sampled_from(grid))
    for method in METHODS:
        barcode = complex.filtered().persistence(method=method, algorithm=algorithm, representatives=True)
        check_representatives(complex, barcode)


def check_representatives(complex, barcode):
    p = complex.p
    assert bars_of(barcode) == expected(complex)

    def boundaries_until(death, inclusive):
        cofaces = [
            i
            for i in range(len(complex))
            if complex.key(i) < complex.key(death) or (inclusive and i == death)
        ]
        return Echelon(p, [complex.boundaries[i] for i in cofaces])

    for bar in barcode.bars:
        note(f"bar={bar.birth, bar.death} representative={bar.representative}")
        rep = dict(bar.representative)
        assert len(rep) == len(bar.representative), "repeated cell"
        assert all(0 < c < p for c in rep.values())
        cells = [cell for cell, _ in bar.representative]
        # Decreasing filtration order, so the birth comes first
        assert cells == sorted(cells, key=complex.key, reverse=True) and cells[0] == bar.birth
        assert all(complex.dims[cell] == bar.dimension for cell in cells)
        assert not boundary_of(rep, complex.boundaries, p), "not a cycle"
        if bar.death is None:
            assert rep not in Echelon(p, complex.boundaries)
        else:
            assert rep in boundaries_until(bar.death, inclusive=True)
            assert rep not in boundaries_until(bar.death, inclusive=False)


# ====== Invariance =========================================================


def diagrams(barcode, top):
    return [barcode.diagram(dim).tolist() for dim in range(top + 1)]


@given(complex=complexes(), seed=st.integers(0, 2**32 - 1))
def test_relabelling_cells(complex, seed):
    """Listing the cells in another (valid) order changes which cells pair up among ties, but not the diagrams; and
    each bar's cells are the relabelled cells when no two cells share a value."""
    order = shuffled_order(random.Random(seed), complex)
    other = complex.relabelled(order)
    barcode = complex.filtered().persistence()
    relabelled = other.filtered().persistence()
    assert diagrams(barcode, complex.top) == diagrams(relabelled, complex.top)
    if len(set(complex.values)) == len(complex):
        new = {old: i for i, old in enumerate(order)}
        pairs = sorted((new[b], new[d]) for b, d in barcode.pairings)
        assert (pairs, sorted(new[b] for b in barcode.essential)) == bars_of(relabelled)


@given(complex=complexes())
def test_monotone_change_of_values(complex):
    """A strictly increasing change of values keeps the order (ties included), so the same pairs, with new values."""
    barcode = complex.filtered().persistence()
    transforms = [lambda v: 3 * v + 7, lambda v: v**3 - 1000, lambda v: float(np.exp(v / 50))]
    for transform in transforms:
        values = [transform(v) for v in complex.values]
        moved = complex.filtered(values).persistence()
        assert bars_of(moved) == bars_of(barcode)
        for before, after in zip(barcode.bars, moved.bars):
            assert after.birth_value == values[before.birth]
            assert after.death_value == (INF if before.death is None else values[before.death])


@given(complex=complexes())
def test_one_chain_many_filtrations(complex):
    """Filtering one chain complex several times gives the same as building it afresh each time."""
    pairs, coefficients = complex.pairs()
    chain = phlite.ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=complex.p)
    for values in [complex.values, [3 * v for v in complex.values], complex.values]:
        again = complex.filtered(values).persistence()
        assert bars_of(chain.filtration(values).persistence()) == bars_of(again)


# ====== Invariants of any barcode ===========================================


@given(complex=complexes())
def test_structure_of_bars(complex):
    """Bars pair a cell with one of the dimension above that comes later, with a larger value; no cell is in two
    bars. With the index filtration (no zero-length bars) every cell is in exactly one bar."""
    barcode = complex.filtered().persistence()
    births = [bar.birth for bar in barcode.bars]
    deaths = [bar.death for bar in barcode.bars if bar.death is not None]
    assert len(set(births + deaths)) == len(births) + len(deaths)
    for bar in barcode.bars:
        assert bar.dimension == complex.dims[bar.birth]
        assert bar.birth_value == complex.values[bar.birth]
        if bar.death is not None:
            assert complex.dims[bar.death] == bar.dimension + 1
            assert complex.key(bar.birth) < complex.key(bar.death)
            assert bar.birth_value < bar.death_value == complex.values[bar.death]
        else:
            assert bar.death_value == INF

    in_order = sorted(range(len(complex)), key=complex.key)
    index_filtered = complex.relabelled(in_order).filtered([float(i) for i in range(len(complex))])
    barcode = index_filtered.persistence()
    cells = list(barcode.essential) + [cell for pair in barcode.pairings for cell in pair]
    assert sorted(cells) == list(range(len(complex)))


def betti_numbers(complex, cells):
    """The Betti numbers of the subcomplex on `cells`: cycles minus boundaries, in each dimension."""
    betti = []
    for dim in range(complex.top + 1):
        in_dim = [i for i in cells if complex.dims[i] == dim]
        boundaries_into = Echelon(complex.p, [complex.boundaries[i] for i in cells if complex.dims[i] == dim + 1])
        boundaries_out = Echelon(complex.p, [complex.boundaries[i] for i in in_dim])
        betti.append(len(in_dim) - len(boundaries_out) - len(boundaries_into))
    return betti


@given(complex=complexes())
def test_betti_numbers_and_euler_characteristic(complex):
    """The essential bars in each dimension count its homology, and their alternating sum is the Euler
    characteristic."""
    barcode = complex.filtered().persistence()
    essential = [0] * (complex.top + 1)
    for bar in barcode.bars:
        if bar.death is None:
            essential[bar.dimension] += 1
    assert essential == betti_numbers(complex, range(len(complex)))
    cells = [complex.dims.count(dim) for dim in range(complex.top + 1)]
    assert sum((-1) ** d * n for d, n in enumerate(cells)) == sum((-1) ** d * n for d, n in enumerate(essential))


@given(complex=complexes(), data=st.data())
def test_rank_invariant(complex, data):
    """For values s <= t, the bars of dimension d alive from s to t count the d-cycles at s that are not yet
    boundaries at t: dim (Z_d(s) + B_d(t)) - dim B_d(t). This determines the barcode completely."""
    p = complex.p
    barcode = complex.filtered().persistence()
    values = sorted(set(complex.values))
    if len(values) > 6:
        values = sorted(data.draw(st.lists(st.sampled_from(values), min_size=1, max_size=6, unique=True)))
    for s, t in itertools.combinations_with_replacement(values, 2):
        for dim in range(complex.top + 1):
            alive = [i for i in range(len(complex)) if complex.values[i] <= s and complex.dims[i] == dim]
            cycles = kernel(alive, complex.boundaries, p)
            boundaries = [
                complex.boundaries[i]
                for i in range(len(complex))
                if complex.values[i] <= t and complex.dims[i] == dim + 1
            ]
            span = Echelon(p, boundaries)
            rank = len(span)
            for cycle in cycles:
                span.add(cycle)
            persistent_betti = len(span) - rank
            bars = sum(
                1
                for bar in barcode.bars
                if bar.dimension == dim and bar.birth_value <= s and bar.death_value > t
            )
            assert bars == persistent_betti, f"dim {dim}, s={s}, t={t}"


# ====== Consistency of the API ==============================================


@given(complex=complexes(), data=st.data())
def test_views_of_a_barcode_agree(complex, data):
    """`bars`, `pairings`, `essential`, `diagram`, `bar` and `representative` describe the same bars; asking for
    representatives or fewer dimensions changes nothing else."""
    filtered = complex.filtered()
    method = data.draw(st.sampled_from(METHODS))
    barcode = filtered.persistence(method=method, representatives=True)
    plain = filtered.persistence(method=method)
    assert bars_of(plain) == bars_of(barcode)
    assert all(bar.representative is None for bar in plain.bars)

    bars = barcode.bars
    assert len(barcode) == len(bars)
    assert [bar.birth for bar in bars] == sorted(bar.birth for bar in bars)
    assert barcode.pairings == [(bar.birth, bar.death) for bar in bars if bar.death is not None]
    assert barcode.essential == [bar.birth for bar in bars if bar.death is None]
    for bar in bars:
        looked_up = barcode.bar(bar.birth)
        assert (looked_up.birth, looked_up.death, looked_up.dimension) == (bar.birth, bar.death, bar.dimension)
        assert barcode.representative(bar.birth) == bar.representative
    diagrams = barcode.diagrams()
    assert len(diagrams) == complex.top + 1
    for dim, diagram in enumerate(diagrams):
        points = sorted((bar.birth_value, bar.death_value) for bar in bars if bar.dimension == dim)
        assert diagram.shape == (len(points), 2)
        assert diagram.tolist() == [list(point) for point in points]

    max_dim = data.draw(st.integers(0, complex.top))
    truncated = filtered.persistence(max_dim, method=method)
    assert [(b.birth, b.death) for b in truncated.bars] == [
        (b.birth, b.death) for b in bars if b.dimension <= max_dim
    ]
    assert len(truncated.diagrams()) == max_dim + 1


# ====== Larger complexes ======================================================


def expected_by_phat(complex):
    """As `expected`, over Z/2, from PHAT (much faster than the reference reduction on large complexes)."""
    assert complex.p == 2
    order = sorted(range(len(complex)), key=complex.key)
    position = {cell: i for i, cell in enumerate(order)}
    columns = [(complex.dims[c], sorted(position[f] for f in complex.boundaries[c])) for c in order]
    matrix = phat.boundary_matrix(columns=columns)
    pairs = [(order[int(b)], order[int(d)]) for b, d in matrix.compute_persistence_pairs()]
    paired = {cell for pair in pairs for cell in pair}
    pairs = sorted((b, d) for b, d in pairs if complex.values[b] != complex.values[d])
    return pairs, [i for i in range(len(complex)) if i not in paired]


def cubical_grid(n, seed):
    """The cubical complex of an n x n image of random values (each cell the max of its vertices), with cells listed
    in a random order: large enough for several levels of the pivot columns' bit tree."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 20, size=(n, n))
    cells, index = [], {}

    def add(dim, key, faces, value):
        index[key] = len(cells)
        cells.append((dim, [index[f] for f in faces], value))

    for i, j in itertools.product(range(n), range(n)):
        add(0, (i, j, 0), [], int(pixels[i, j]))
    for i, j in itertools.product(range(n), range(n)):
        if i + 1 < n:
            add(1, (i, j, "v"), [(i, j, 0), (i + 1, j, 0)], int(max(pixels[i, j], pixels[i + 1, j])))
        if j + 1 < n:
            add(1, (i, j, "h"), [(i, j, 0), (i, j + 1, 0)], int(max(pixels[i, j], pixels[i, j + 1])))
    for i, j in itertools.product(range(n - 1), range(n - 1)):
        faces = [(i, j, "h"), (i + 1, j, "h"), (i, j, "v"), (i, j + 1, "v")]
        add(2, (i, j, 2), faces, int(pixels[i : i + 2, j : j + 2].max()))
    dims = [d for d, _, _ in cells]
    boundaries = [{f: 1 for f in faces} for _, faces, _ in cells]
    complex = Complex(2, dims, boundaries, [v for _, _, v in cells])
    return complex.relabelled(shuffled_order(random.Random(seed), complex))


@pytest.mark.parametrize("seed", range(3))
def test_larger_cubical_complexes(seed):
    """A cubical complex of ~20k cells (so ranks span three levels of the bit tree), every representation."""
    complex = cubical_grid(70, seed)
    want = expected_by_phat(complex)
    filtered = complex.filtered()
    Algorithm = phlite.Algorithm
    for method, columns in itertools.product(METHODS, ["merged", "heap", "sorted", "pivot"]):
        for algorithm in [
            Algorithm.serial().with_columns(columns),
            Algorithm.serial().with_columns(columns).with_store_r(False).with_cache(True),
            Algorithm.parallel(4).with_columns(columns),
        ]:
            assert bars_of(filtered.persistence(method=method, algorithm=algorithm)) == want, f"{method} {algorithm!r}"
