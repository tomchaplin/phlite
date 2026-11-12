"""Fuzz phlite_rips against ripser and giotto-ph, and check its representatives from first principles."""

import itertools
import math

import numpy as np
from gph import ripser_parallel
from hypothesis import given, note
from hypothesis import strategies as st
from ripser import ripser

from grid import algorithm_grid
from mod2 import Span, boundary, chain_sum
import phlite_rips
from phlite_rips import rips

METHODS = ["cohomology", "homology"]
ALGORITHMS = [
    phlite_rips.Algorithm.standard(),
    phlite_rips.Algorithm.serial(),
    phlite_rips.Algorithm.parallel(),
    phlite_rips.Algorithm.parallel().with_storage("lock-free"),
    phlite_rips.Algorithm.clearing().with_store_r(),
    # The other representation and caching from this package's default
    phlite_rips.Algorithm.serial().with_columns("merged").with_cache(True),
]
INF = math.inf


@st.composite
def distance_matrices(draw, max_points=9):
    """Symmetric distance matrices, with entries exactly representable as float32 (ripser and giotto-ph use float32).

    Mixes Euclidean point clouds with (possibly non-metric) small-integer matrices,
    which have lots of ties, optionally with infinite entries.
    """
    n = draw(st.integers(1, max_points))
    kind = draw(st.sampled_from(["points", "integers", "integers with infinity"]))
    if kind == "points":
        dim = draw(st.integers(1, 3))
        coords = st.floats(-10, 10, allow_nan=False, width=32)
        points = np.array(draw(st.lists(st.lists(coords, min_size=dim, max_size=dim), min_size=n, max_size=n)))
        distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    else:
        values = st.integers(0, 4).map(float)
        if kind == "integers with infinity":
            values = st.one_of(values, st.just(INF))
        upper = draw(st.lists(values, min_size=n * (n - 1) // 2, max_size=n * (n - 1) // 2))
        distances = np.zeros((n, n))
        distances[np.triu_indices(n, 1)] = upper
        distances = distances + distances.T
    distances = distances.astype(np.float32).astype(np.float64)
    np.fill_diagonal(distances, 0.0)
    return distances


def diagram(barcode, dim):
    """All bars in dimension `dim` as sorted (birth, death) pairs, with death inf for essential bars."""
    # Keep finite and essential bars distinguishable: a "finite" bar must die at a finite time
    assert all(death != INF for _, death in barcode.pairings[dim]), "finite bar dies at infinity"
    assert all(birth != INF for birth, _ in barcode.pairings[dim]) and INF not in barcode.essential[dim]
    return sorted(list(barcode.pairings[dim]) + [(birth, INF) for birth in barcode.essential[dim]])


def reference_diagram(dgms, dim):
    """Same as `diagram`, for the output of ripser or giotto-ph."""
    return sorted((float(b), float(d)) for b, d in dgms[dim] if b < d and b != INF)


def diameter(distances, simplex):
    return max((distances[a, b] for a, b in itertools.combinations(simplex, 2)), default=0.0)


def check_representatives(distances, barcode, max_dim):
    """Check every representative against the definition of persistent homology, using linear algebra over Z/2."""
    n = len(distances)
    for dim in range(max_dim + 1):
        cofaces = [
            (diameter(distances, s), s) for s in itertools.combinations(range(n), dim + 2)
        ]
        cofaces = [(f, s) for f, s in cofaces if f != INF]

        def boundaries_before(time, strict):
            return Span(boundary(s) for f, s in cofaces if (f < time if strict else f <= time))

        bars = [(bar, rep) for bar, rep in zip(barcode.pairings[dim], barcode.pairing_representatives[dim])]
        bars += [((b, INF), rep) for b, rep in zip(barcode.essential[dim], barcode.essential_representatives[dim])]
        for (birth, death), rep in bars:
            note(f"dim={dim} bar=({birth}, {death}) rep={rep}")
            assert len(rep) > 0
            assert len(set(rep)) == len(rep), "repeated cell"
            assert all(len(cell) == dim + 1 and list(cell) == sorted(cell) for cell in rep)
            chain = set(rep)
            if dim > 0:
                assert not chain_sum(boundary(cell) for cell in rep), "not a cycle"
            assert max(diameter(distances, cell) for cell in rep) == birth, "not born at birth"
            assert chain not in boundaries_before(death, strict=True), "already a boundary before death"
            if death != INF:
                assert chain in boundaries_before(death, strict=False), "not a boundary at death"


@given(distances=distance_matrices(), max_dim=st.integers(0, 2))
def test_rips_matches_references(distances, max_dim):
    expected_ripser = ripser(distances, distance_matrix=True, maxdim=max_dim)["dgms"]
    expected_gph = ripser_parallel(distances, maxdim=max_dim, metric="precomputed")["dgms"]
    for dim in range(max_dim + 1):
        # Sanity check the references against each other
        assert reference_diagram(expected_ripser, dim) == reference_diagram(expected_gph, dim)

    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        note(f"method={method} algorithm={algorithm}")
        barcode = rips(distances, max_dim, method=method, algorithm=algorithm, representatives=True)
        assert len(barcode.essential) == len(barcode.pairings) == max_dim + 1
        for dim in range(max_dim + 1):
            assert diagram(barcode, dim) == reference_diagram(expected_ripser, dim), f"dim {dim}"
        check_representatives(distances, barcode, max_dim)


@given(distances=distance_matrices(), max_dim=st.integers(0, 2), threads=st.integers(1, 4))
def test_representatives_flag_and_threads_do_not_change_barcode(distances, max_dim, threads):
    plain = rips(distances, max_dim)
    with_reps = rips(distances, max_dim, algorithm=phlite_rips.Algorithm.parallel(threads), representatives=True)
    assert plain.pairing_representatives is None and plain.essential_representatives is None
    assert (plain.essential, plain.pairings) == (with_reps.essential, with_reps.pairings)


@given(distances=distance_matrices(), max_dim=st.integers(0, 2), data=st.data())
def test_threshold_matches_references(distances, max_dim, data):
    # A threshold at one of the entries (exact in float32), or just below/above everything
    finite = sorted({float(d) for d in distances.flat if d != INF})
    threshold = data.draw(st.sampled_from(finite + [-1.0, INF]))
    expected = ripser(distances, distance_matrix=True, maxdim=max_dim, thresh=threshold)["dgms"]
    expected_gph = ripser_parallel(distances, maxdim=max_dim, metric="precomputed", thresh=threshold)["dgms"]
    for method in METHODS:
        barcode = rips(distances, max_dim, threshold=threshold, method=method, representatives=True)
        for dim in range(max_dim + 1):
            assert diagram(barcode, dim) == reference_diagram(expected, dim), f"dim {dim}"
            assert diagram(barcode, dim) == reference_diagram(expected_gph, dim), f"dim {dim} (giotto-ph)"


@given(distances=distance_matrices(), max_dim=st.integers(0, 2))
def test_input_types_agree(distances, max_dim):
    # float64 arrays are read directly; float32 arrays too (the entries are exactly representable); lists via Python
    expected = rips(distances, max_dim)
    for other in [distances.astype(np.float32), distances.tolist(), np.asfortranarray(distances), distances.T]:
        barcode = rips(other, max_dim)
        assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings)


def test_input_validation():
    import pytest

    for bad, message in [
        (np.zeros((2, 3)), "square"),
        (np.zeros((2, 2, 2)), "2-dimensional"),
        ([[0.0, 1.0]], "square"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), "symmetric"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float32), "non-negative"),
    ]:
        with pytest.raises(ValueError, match=message):
            rips(bad, 1)
    # Integer arrays go through the general path
    assert rips(np.array([[0, 1], [1, 0]]), 0).pairings == [[(0.0, 1.0)]]


def test_default_columns():
    """Each package's presets use its own column representation and caching."""
    for algorithm in [phlite_rips.Algorithm.serial(), phlite_rips.Algorithm.parallel(), phlite_rips.Algorithm.standard()]:
        assert not algorithm.store_r
        assert (algorithm.columns, algorithm.cache) == ('heap', False)


@given(distances=distance_matrices(), max_dim=st.integers(0, 2))
def test_every_algorithm_agrees(distances, max_dim):
    """Every algorithm and option of the full grid (as for phlite_py) gives the default's barcode."""
    expected = rips(distances, max_dim)
    for method, algorithm in itertools.product(METHODS, algorithm_grid(phlite_rips.Algorithm)):
        note(f"method={method} algorithm={algorithm!r}")
        barcode = rips(distances, max_dim, method=method, algorithm=algorithm)
        assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings)
