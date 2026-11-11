"""Fuzz phlite_py (the `phlite` package) against PHAT and LoPHAT on random filtered chain complexes over Z/2.

The complexes are cellular rather than simplicial: each k-cell's boundary is a random (k-1)-cycle of the cells before
it, so there are multi-edges, loops, cells with zero boundary, and k-cells with long boundaries. Cells are listed in a
valid filtration order (every face before its cofaces, values non-decreasing), so the index order PHAT and LoPHAT use
is also phlite's order, and the pairings agree exactly.
"""

import itertools
import random

import lophat
import numpy as np
import phat
import phlite
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from mod2 import Span

Algorithm = phlite.Algorithm
METHODS = ["cohomology", "homology"]


# ====== Random filtered chain complexes ==================================


def _cycle_basis(cells, boundaries):
    """A basis, over Z/2, of the cycles among `cells` (which all have the same dimension), as sets of cells."""
    # Gaussian elimination on the boundaries, tracking which cells each reduced column is a sum of
    pivots = {}
    basis = []
    for cell in cells:
        column, combination = set(boundaries[cell]), {cell}
        while column and max(column) in pivots:
            other_column, other_combination = pivots[max(column)]
            column ^= other_column
            combination ^= other_combination
        if column:
            pivots[max(column)] = (column, combination)
        else:
            basis.append(combination)
    return basis


def random_complex(rng, max_dim, n_vertices, cells_per_dim, max_value, density=None):
    """A random filtered chain complex: `(columns, values)` in filtration order, `columns[i] = (dim, boundary)`.

    Each cell above dimension 1 has as boundary the sum of a few basis cycles or, if `density` is given, of each basis
    cycle with that probability (which gives a dense matrix).
    """
    dims, boundaries, values = [], [], []

    def add(dim, boundary, value):
        dims.append(dim)
        boundaries.append(sorted(boundary))
        values.append(value)

    for _ in range(n_vertices):
        add(0, [], rng.randint(0, max_value))
    for dim in range(1, max_dim + 1):
        faces = [i for i, d in enumerate(dims) if d == dim - 1]
        if not faces:
            break
        cycles = _cycle_basis(faces, boundaries) if dim > 1 else None
        for _ in range(cells_per_dim):
            if dim == 1:
                # An edge between two vertices, or occasionally a loop
                a, b = rng.choice(faces), rng.choice(faces)
                boundary = set() if a == b else {a, b}
            else:
                # A random cycle (a set of faces whose boundaries cancel): the sum of a few basis cycles, so that
                # boundaries stay sparse, and occasionally zero
                if density is None:
                    chosen = rng.sample(cycles, min(len(cycles), rng.choice([0, 1, 1, 1, 2, 3])))
                else:
                    chosen = [cycle for cycle in cycles if rng.random() < density]
                boundary = set()
                for cycle in chosen:
                    boundary ^= cycle
            add(dim, boundary, max([rng.randint(0, max_value)] + [values[f] for f in boundary]))

    order = _filtration_order(rng, dims, boundaries, values)
    position = {old: new for new, old in enumerate(order)}
    columns = [(dims[i], sorted(position[f] for f in boundaries[i])) for i in order]
    return columns, [values[i] for i in order]


def _filtration_order(rng, dims, boundaries, values):
    """A random order of the cells with non-decreasing values in which every face comes before its cofaces."""
    placed, order = set(), []
    for value in sorted(set(values)):
        pending = [i for i, v in enumerate(values) if v == value]
        while pending:
            ready = [i for i in pending if all(f in placed for f in boundaries[i])]
            cell = rng.choice(ready)
            pending.remove(cell)
            placed.add(cell)
            order.append(cell)
    return order


@st.composite
def complexes(draw, max_cells_per_dim=12):
    """A random filtered complex, drawn via a seed so that Hypothesis can still shrink the sizes."""
    max_dim = draw(st.integers(1, 3))
    n_vertices = draw(st.integers(1, 10))
    cells_per_dim = draw(st.integers(0, max_cells_per_dim))
    # Few distinct values means many ties, so many zero-length (and apparent) pairs
    max_value = draw(st.sampled_from([0, 1, 2, 5, 100]))
    seed = draw(st.integers(0, 2**32 - 1))
    return random_complex(random.Random(seed), max_dim, n_vertices, cells_per_dim, max_value)


# ====== The references ===================================================


def phat_pairs(columns, reduction=phat.reductions.twist_reduction, representation=None):
    kwargs = {} if representation is None else {"representation": representation}
    matrix = phat.boundary_matrix(columns=columns, **kwargs)
    pairs = matrix.compute_persistence_pairs(reduction=reduction)
    return sorted((int(b), int(d)) for b, d in pairs)


def lophat_pairs(columns, anti_transpose=True, threads=0, clearing=True):
    options = lophat.LoPhatOptions(num_threads=threads, clearing=clearing)
    diagram = lophat.compute_pairings(columns, anti_transpose, options)
    pairs = sorted((b, d) for b, d in diagram.items() if d is not None)
    # Every cell is either paired or listed as essential
    assert len(diagram) + len(pairs) == len(columns)
    return pairs


def essential_of(columns, pairs):
    paired = {cell for pair in pairs for cell in pair}
    return [i for i in range(len(columns)) if i not in paired]


def nonzero(pairs, values):
    return [(b, d) for b, d in pairs if values[b] != values[d]]


# ====== Every algorithm ==================================================


def algorithms(top, threads=2):
    """Every algorithm and option, for a complex whose top dimension is `top`."""
    bases = [Algorithm.standard(), Algorithm.clearing(), Algorithm.parallel_clearing()]
    for base, columns in itertools.product(bases, ["merged", "heap"]):
        algorithm = base.with_columns(columns)
        if base.name == "standard":
            yield algorithm
            continue
        for skip in [True, False]:
            for order in [None, list(range(top + 1)), list(range(top, -1, -1))]:
                configured = algorithm.with_skip_zero_apparent_pairs(skip).with_dimension_order(order)
                yield configured
                if base.name == "parallel-clearing":
                    yield configured.with_threads(threads)
                    yield configured.with_storage("lock-free").with_threads(threads)


def top_dim(columns):
    return max(dim for dim, _ in columns)


# ====== Tests ============================================================


@given(complex=complexes())
def test_references_agree(complex):
    """Sanity check the references against each other, with several of their algorithms."""
    columns, _ = complex
    expected = phat_pairs(columns)
    reductions = phat.reductions
    representations = phat.representations
    for reduction, representation in [
        (reductions.standard_reduction, representations.vector_vector),
        (reductions.chunk_reduction, representations.bit_tree_pivot_column),
        (reductions.row_reduction, representations.vector_heap),
        (reductions.spectral_sequence_reduction, representations.full_pivot_column),
    ]:
        assert phat_pairs(columns, reduction, representation) == expected, (reduction, representation)
    for anti_transpose, clearing, threads in itertools.product([True, False], [True, False], [1, 2]):
        assert lophat_pairs(columns, anti_transpose, threads, clearing) == expected


@given(complex=complexes())
def test_index_filtration_matches_phat(complex):
    """With the index filtration no bar has length zero, so phlite gives exactly PHAT's pairs."""
    columns, _ = complex
    expected_pairs = phat_pairs(columns)
    expected = (expected_pairs, essential_of(columns, expected_pairs))
    filtered = phlite.ChainComplex.from_columns(columns).filtration()
    for method, algorithm in itertools.product(METHODS, algorithms(top_dim(columns))):
        note(f"method={method} algorithm={algorithm!r}")
        barcode = filtered.persistence(method=method, algorithm=algorithm)
        assert (barcode.pairings, barcode.essential) == expected


@given(complex=complexes(), data=st.data())
def test_values_match_phat_and_lophat(complex, data):
    """With filtration values, phlite gives PHAT's (and LoPHAT's) pairs of positive length."""
    columns, values = complex
    pairs = phat_pairs(columns)
    assert lophat_pairs(columns) == pairs
    expected = (nonzero(pairs, values), essential_of(columns, pairs))
    # Both ways of building the complex
    chain = data.draw(st.sampled_from(["columns", "csc"]))
    if chain == "columns":
        chain = phlite.ChainComplex.from_columns(columns)
    else:
        indptr = np.cumsum([0] + [len(c) for _, c in columns], dtype=np.int64)
        indices = np.array([f for _, c in columns for f in c], dtype=np.int64)
        dimensions = np.array([d for d, _ in columns], dtype=np.int64)
        chain = phlite.ChainComplex.from_csc(indptr, indices, dimensions)
    values = data.draw(st.sampled_from([values, np.array(values, dtype=np.float64)]))
    filtered = chain.filtration(values)
    max_dim = data.draw(st.none() | st.integers(0, top_dim(columns)))
    if max_dim is not None:
        dims = [d for d, _ in columns]
        expected = (
            [(b, d) for b, d in expected[0] if dims[b] <= max_dim],
            [b for b in expected[1] if dims[b] <= max_dim],
        )
    for method, algorithm in itertools.product(METHODS, algorithms(top_dim(columns))):
        note(f"method={method} algorithm={algorithm!r}")
        barcode = filtered.persistence(max_dim, method=method, algorithm=algorithm)
        assert (barcode.pairings, barcode.essential) == expected


@given(complex=complexes(), data=st.data())
def test_representatives(complex, data):
    """Every representative is a cycle whose last cell is the birth, and is a boundary exactly when it should be."""
    columns, values = complex
    pairs = nonzero(phat_pairs(columns), values)
    method = data.draw(st.sampled_from(METHODS))
    algorithm = data.draw(st.sampled_from(list(algorithms(top_dim(columns)))))
    barcode = phlite.ChainComplex.from_columns(columns).filtration(values).persistence(
        method=method, algorithm=algorithm, representatives=True
    )
    assert barcode.pairings == pairs
    boundaries = [set(c) for _, c in columns]

    def boundary_of(chain):
        out = set()
        for cell in chain:
            out ^= boundaries[cell]
        return out

    for bar in barcode.bars:
        note(f"bar={bar.birth, bar.death} representative={bar.representative}")
        assert all(coefficient == 1 for _, coefficient in bar.representative)
        rep = [cell for cell, _ in bar.representative]
        assert len(set(rep)) == len(rep)
        # Decreasing filtration order, so the birth comes first
        assert rep[0] == bar.birth and rep == sorted(rep, reverse=True)
        assert all(columns[cell][0] == bar.dimension for cell in rep)
        assert not boundary_of(rep)
        # A boundary by the death, and not before; never a boundary if essential
        if bar.death is None:
            assert set(rep) not in Span(boundaries)
        else:
            assert set(rep) in Span(boundaries[: bar.death + 1])
            assert set(rep) not in Span(boundaries[: bar.death])


@pytest.mark.parametrize("seed", range(6))
def test_larger_complexes(seed):
    """Fewer, larger complexes, so the parallel algorithms have enough columns to race on."""
    rng = random.Random(seed)
    columns, values = random_complex(rng, max_dim=3, n_vertices=100, cells_per_dim=500, max_value=rng.choice([3, 50]))
    pairs = phat_pairs(columns)
    assert lophat_pairs(columns, threads=4) == pairs
    expected = (nonzero(pairs, values), essential_of(columns, pairs))
    filtered = phlite.ChainComplex.from_columns(columns).filtration(values)
    for method in METHODS:
        for algorithm in [
            Algorithm.serial(),
            Algorithm.serial().with_columns("heap"),
            Algorithm.parallel(4),
            Algorithm.parallel(4).with_storage("lock-free"),
            Algorithm.parallel(4).with_columns("heap").with_skip_zero_apparent_pairs(False),
        ]:
            for _ in range(3):
                barcode = filtered.persistence(method=method, algorithm=algorithm)
                assert (barcode.pairings, barcode.essential) == expected, f"{method} {algorithm!r}"
