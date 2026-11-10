import itertools
import math
import random

import numpy as np
import pytest
from reference import boundary_of_chain, in_span, random_simplicial_complex, reduce, rp2

import phlite

METHODS = ["cohomology", "homology"]
ALGORITHMS = ["standard", "clearing", "parallel", "parallel-lock-free"]
MODULI = [2, 3, 5, 7, 11, 13]


def expected_bars(columns, coefficients, values, p, max_dim=None, dimensions=None):
    """Reference barcode: (pairings, essential), dropping zero-length bars."""
    pairs, essential = reduce(columns, coefficients, values, p)
    pairs = [(b, d) for b, d in pairs if values[b] != values[d]]
    if max_dim is not None:
        pairs = [(b, d) for b, d in pairs if dimensions[b] <= max_dim]
        essential = [b for b in essential if dimensions[b] <= max_dim]
    return sorted(pairs), sorted(essential)


def complex_for(seed, p=2, top_dim=3):
    rng = random.Random(seed)
    simplices, values, dimensions, columns, coefficients = random_simplicial_complex(
        rng, n_vertices=rng.randint(4, 9), n_top=rng.randint(1, 8), top_dim=top_dim
    )
    complex = phlite.FilteredComplex(
        columns,
        dimensions,
        values,
        coefficients=coefficients if p != 2 else None,
        modulus=p,
    )
    return complex, simplices, values, dimensions, columns, coefficients


# ====== Agreement with the reference reduction ============================


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_matches_reference_z2(seed, method, algorithm):
    complex, _, values, _, columns, coefficients = complex_for(seed)
    barcode = complex.persistence(method=method, algorithm=algorithm)
    assert (barcode.pairings, barcode.essential) == expected_bars(
        columns, coefficients, values, 2
    )


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("p", MODULI[1:])
@pytest.mark.parametrize("method", METHODS)
def test_matches_reference_zp(seed, p, method):
    complex, _, values, _, columns, coefficients = complex_for(seed, p)
    barcode = complex.persistence(method=method)
    assert (barcode.pairings, barcode.essential) == expected_bars(
        columns, coefficients, values, p
    )


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("max_dim", [0, 1, 2])
@pytest.mark.parametrize("method", METHODS)
def test_max_dim(seed, max_dim, method):
    complex, _, values, dimensions, columns, coefficients = complex_for(seed)
    barcode = complex.persistence(max_dim, method=method)
    expected = expected_bars(columns, coefficients, values, 2, max_dim, dimensions)
    assert (barcode.pairings, barcode.essential) == expected
    assert len(barcode.diagrams()) == max_dim + 1


@pytest.mark.parametrize("seed", range(30))
def test_matches_lophat(seed):
    lophat = pytest.importorskip("lophat")
    _, _, _, dimensions, columns, _ = complex_for(seed)
    # Index filtration, so that no bar has length zero and lophat's output is directly comparable
    pairs = list(zip(dimensions, columns))
    barcode = phlite.FilteredComplex(pairs).persistence()
    expected = lophat.compute_pairings(pairs)
    assert dict(barcode.pairings) == {
        b: d for b, d in expected.items() if d is not None
    }
    assert barcode.essential == sorted(b for b, d in expected.items() if d is None)


def test_rp2_torsion():
    """Over Z/2, RP^2 has H_1 = H_2 = Z/2; over odd primes, only H_0 survives."""
    _, dimensions, columns, coefficients = rp2()
    assert sum((-1) ** d for d in dimensions) == 1  # Euler characteristic
    betti = {}
    for p in MODULI:
        complex = phlite.FilteredComplex(
            columns, dimensions, coefficients=coefficients, modulus=p
        )
        for method in METHODS:
            barcode = complex.persistence(method=method)
            betti[p, method] = [
                sum(dimensions[b] == k for b in barcode.essential) for k in range(3)
            ]
    assert betti[2, "cohomology"] == betti[2, "homology"] == [1, 1, 1]
    for p in MODULI[1:]:
        assert betti[p, "cohomology"] == betti[p, "homology"] == [1, 0, 0]


# ====== Representatives ===================================================


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("p", [2, 3, 5])
@pytest.mark.parametrize("method", METHODS)
def test_representatives(seed, p, method):
    complex, _, values, dimensions, columns, coefficients = complex_for(
        seed, p, top_dim=2
    )
    if p == 2:
        coefficients = [[1] * len(col) for col in columns]
    barcode = complex.persistence(method=method, representatives=True)
    reps = barcode.representatives
    deaths = dict(barcode.pairings)
    assert set(reps) == set(deaths) | set(barcode.essential)
    key = lambda i: (values[i], i)
    for birth, rep in reps.items():
        chain = {cell: 1 for cell in rep} if p == 2 else dict(rep)
        assert all(0 < c < p for c in chain.values())
        # Cells listed in decreasing filtration order, the leading one is the birth cell, all the same dimension
        cells = [cell for cell in rep] if p == 2 else [cell for cell, _ in rep]
        assert cells == sorted(cells, key=key, reverse=True)
        assert cells[0] == birth
        assert all(dimensions[c] == dimensions[birth] for c in cells)
        # A cycle
        assert boundary_of_chain(chain, columns, coefficients, p) == {}
        # For a finite bar, a boundary by the death time
        if birth in deaths:
            death = deaths[birth]
            available = [
                boundary_of_chain({j: 1}, columns, coefficients, p)
                for j in range(len(columns))
                if key(j) <= key(death) and dimensions[j] == dimensions[birth] + 1
            ]
            assert in_span(chain, available, p)


def test_no_representatives_by_default():
    complex, *_ = complex_for(0)
    assert complex.persistence().representatives is None


# ====== Input formats =====================================================


def test_input_formats_agree():
    complex, _, values, dimensions, columns, _ = complex_for(3)
    expected = complex.persistence()

    def same(other):
        barcode = other.persistence()
        assert (barcode.pairings, barcode.essential) == (
            expected.pairings,
            expected.essential,
        )

    # (dimension, column) pairs, from a generator, numpy arrays everywhere
    same(
        phlite.FilteredComplex(
            ((d, c) for d, c in zip(dimensions, columns)), values=values
        )
    )
    same(
        phlite.FilteredComplex(
            [np.array(c, dtype=np.int32) for c in columns],
            np.array(dimensions),
            np.array(values, dtype=np.float32),
        )
    )
    # CSC arrays, and scipy
    indptr = np.cumsum([0] + [len(c) for c in columns])
    indices = np.array(list(itertools.chain.from_iterable(columns)))
    same(phlite.FilteredComplex.from_csc(indptr, indices, dimensions, values))
    scipy_sparse = pytest.importorskip("scipy.sparse")
    n = len(columns)
    matrix = scipy_sparse.csc_matrix(
        (np.ones(len(indices)), indices, indptr), shape=(n, n)
    )
    matrix = matrix.tocsc()
    same(
        phlite.FilteredComplex.from_csc(
            matrix.indptr, matrix.indices, dimensions, values, data=matrix.data
        )
    )


def test_coefficients_reduce_mod_p():
    # A single edge whose boundary is 3*v1 - 3*v0: zero mod 3 (so both vertices essential), not mod 2
    columns = [[], [], [0, 1]]
    dims = [0, 0, 1]
    coeffs = [[], [], [-3, 3]]
    z3 = phlite.FilteredComplex(columns, dims, coefficients=coeffs, modulus=3)
    assert z3.n_entries == 0
    assert z3.persistence().essential == [0, 1, 2]
    z2 = phlite.FilteredComplex(columns, dims, coefficients=coeffs)
    assert z2.persistence().pairings == [(1, 2)]


def test_diagram_values():
    # A hollow triangle at times 0, 1, 1, 2, filled at time 2 (zero-length) and a lone vertex at time inf
    columns = [[], [], [], [0, 1], [1, 2], [0, 2], [3, 4, 5], []]
    dims = [0, 0, 0, 1, 1, 1, 2, 0]
    values = [0.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, math.inf]
    barcode = phlite.FilteredComplex(columns, dims, values).persistence()
    # (1, 3) and (5, 6) have length zero, so are omitted
    assert barcode.pairings == [(2, 4)]
    assert barcode.essential == [0, 7]
    # Vertex 7 is born at infinity, so never appears in a diagram
    np.testing.assert_array_equal(barcode.diagram(0), [[0.0, math.inf], [1.0, 1.5]])
    assert barcode.diagram(1).shape == (0, 2)
    assert barcode.dimension_of(2) == 0
    with pytest.raises(ValueError, match="No bar"):
        barcode.dimension_of(3)


def test_empty_complex():
    complex = phlite.FilteredComplex([])
    assert len(complex) == 0 and complex.dimension is None
    barcode = complex.persistence()
    assert len(barcode) == 0 and barcode.diagrams() == []


# ====== Errors ============================================================


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(boundaries=[[], [0]]), "dimension of every cell"),
        (dict(boundaries=[[], [0]], dimensions=[0, 0]), "dimension one lower"),
        (dict(boundaries=[[], [5]], dimensions=[0, 1]), "only 2 cells"),
        (dict(boundaries=[[1], []], dimensions=[1, 0]), "enter the filtration first"),
        (
            dict(boundaries=[[], [0]], dimensions=[0, 1], values=[1.0, 0.0]),
            "enter the filtration first",
        ),
        (dict(boundaries=[[], [0]], dimensions=[0, 1], values=[0.0, math.nan]), "NaN"),
        (
            dict(boundaries=[[], [0]], dimensions=[0, 1], values=[0.0]),
            "values has length",
        ),
        (
            dict(boundaries=[[], [0]], dimensions=[0, 1], modulus=4),
            "Unsupported modulus",
        ),
        (
            dict(boundaries=[[], [0]], dimensions=[0, 1], coefficients=[[], [1, 1]]),
            "coefficients\\[1\\] has length",
        ),
        (dict(boundaries=[(0, []), [0]]), "either every column"),
        (dict(boundaries=[(0, [])], dimensions=[0]), "both"),
        (dict(boundaries=[[], [-1]], dimensions=[0, 1]), "out of range"),
    ],
)
def test_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        phlite.FilteredComplex(**kwargs)


def test_bad_arguments():
    complex, *_ = complex_for(0)
    with pytest.raises(ValueError, match="Unknown algorithm"):
        complex.persistence(algorithm="fast")
    with pytest.raises(ValueError, match="Unknown method"):
        complex.persistence(method="both")
    with pytest.raises(ValueError, match="threads can only be set"):
        complex.persistence(threads=2)
    with pytest.raises(TypeError, match="sequence of integers"):
        phlite.FilteredComplex([[], "ab"], [0, 1])
    with pytest.raises(ValueError, match="1-dimensional"):
        phlite.FilteredComplex.from_csc(np.zeros((2, 2), dtype=np.int64), [], [0])
