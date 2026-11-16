"""Fuzz phlite.clouds.rips against ripser and giotto-ph, and check its representatives from first principles."""

import itertools
import math

import numpy as np
import pytest
import scipy.sparse
from gph import ripser_parallel
from hypothesis import given, note
from hypothesis import strategies as st
from ripser import ripser

from grid import algorithm_grid
from mod2 import Span, boundary, chain_sum
from phlite.core import Algorithm
from phlite.clouds import rips

METHODS = ["cohomology", "homology"]
ALGORITHMS = [
    Algorithm.standard(),
    Algorithm.serial(),
    Algorithm.parallel(),
    Algorithm.parallel().with_storage("lock-free"),
    Algorithm.clearing().with_store_r(),
    # The other representation and caching from this pipeline's default
    Algorithm.serial().with_columns("merged").with_cache(True),
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
    with_reps = rips(distances, max_dim, algorithm=Algorithm.parallel(threads), representatives=True)
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
    """The pipeline's defaults fill in what the presets leave unset, and are what runs."""
    for algorithm in [Algorithm.serial(), Algorithm.parallel(), Algorithm.standard()]:
        resolved = algorithm.resolve("clouds.rips")
        assert not resolved.store_r
        assert (resolved.columns, resolved.cache) == ('heap', False)
        # These matrices give no row ranks, so pivot columns would merge: refused rather than silently
        with pytest.raises(ValueError, match="'pivot' columns"):
            algorithm.with_columns("pivot").resolve("clouds.rips")


@given(distances=distance_matrices(), max_dim=st.integers(0, 2))
def test_every_algorithm_agrees(distances, max_dim):
    """Every algorithm and option of the full grid (as for explicit complexes) gives the default's barcode."""
    expected = rips(distances, max_dim)
    for method, algorithm in itertools.product(METHODS, algorithm_grid(Algorithm, row_ranks=False, union_find=True)):
        note(f"method={method} algorithm={algorithm!r}")
        barcode = rips(distances, max_dim, method=method, algorithm=algorithm)
        assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings)


# ====== Sparse distance matrices and points ================================


def sparse_matrix(distances, threshold, data):
    """The edges of `distances` up to `threshold` as a scipy sparse matrix: in COO, CSR or CSC format, each edge given
    once (either way round) or both ways round, keeping zero-length edges as explicit entries."""
    n = len(distances)
    rows, cols, values = [], [], []
    for i, j in itertools.combinations(range(n), 2):
        d = distances[i, j]
        if d == INF or d > threshold:
            continue
        how = data.draw(st.sampled_from(["upper", "lower", "both"]))
        if how in ("upper", "both"):
            rows.append(i), cols.append(j), values.append(d)
        if how in ("lower", "both"):
            rows.append(j), cols.append(i), values.append(d)
    matrix = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(n, n))
    return {"coo": matrix, "csr": matrix.tocsr(), "csc": matrix.tocsc()}[data.draw(st.sampled_from(["coo", "csr", "csc"]))]


@given(distances=distance_matrices(), max_dim=st.integers(0, 2), data=st.data())
def test_sparse_matches_references(distances, max_dim, data):
    """A sparse matrix of the edges up to a threshold gives ripser's barcode for it (missing entries never appear), and
    phlite's own for the dense matrix with that threshold, with every algorithm, and valid representatives."""
    finite = sorted({float(d) for d in distances.flat if d != INF})
    threshold = data.draw(st.sampled_from(finite + [INF]))
    matrix = sparse_matrix(distances, threshold, data)
    note(f"threshold={threshold} format={matrix.format} nnz={matrix.nnz}")
    # ripser reads only the upper triangle of a sparse matrix (and its diagonal as the vertices' births: there is none
    # here, so every vertex enters at 0), so give it each edge once that way round
    coo = matrix.tocoo()
    upper = scipy.sparse.coo_matrix(
        (coo.data, (np.minimum(coo.row, coo.col), np.maximum(coo.row, coo.col))), shape=coo.shape
    )
    upper.sum_duplicates()
    upper.data = np.array([distances[i, j] for i, j in zip(upper.row, upper.col)])
    expected = ripser(upper, distance_matrix=True, maxdim=max_dim)["dgms"]
    truncated = np.where(distances <= threshold, distances, INF)
    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        note(f"method={method} algorithm={algorithm}")
        barcode = rips(matrix, max_dim, method=method, algorithm=algorithm, representatives=True)
        for dim in range(max_dim + 1):
            assert diagram(barcode, dim) == reference_diagram(expected, dim), f"dim {dim}"
        dense = rips(distances, max_dim, threshold=threshold, method=method, algorithm=algorithm)
        assert (barcode.essential, barcode.pairings) == (dense.essential, dense.pairings)
        check_representatives(truncated, barcode, max_dim)



@given(distances=distance_matrices(), max_dim=st.integers(0, 2), data=st.data())
def test_explicit_and_implicit_agree(distances, max_dim, data):
    """Listing the complex (`explicit=True`) and computing it implicitly (`explicit=False`) give the same barcode as
    the default's choice, for sparse and dense input, with thresholds, weights and every algorithm, and valid
    representatives."""
    finite = sorted({float(d) for d in distances.flat if d != INF})
    threshold = data.draw(st.sampled_from(finite + [INF]))
    sparse = data.draw(st.booleans())
    matrix = sparse_matrix(distances, threshold, data) if sparse else distances
    weights = data.draw(st.none() | st.lists(st.integers(0, 3).map(float), min_size=len(distances), max_size=len(distances)))
    note(f"sparse={sparse} threshold={threshold} weights={weights}")
    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        note(f"method={method} algorithm={algorithm}")
        kwargs = dict(threshold=threshold, weights=weights, method=method, algorithm=algorithm)
        default = rips(matrix, max_dim, **kwargs)
        for explicit in [True, False]:
            barcode = rips(matrix, max_dim, explicit=explicit, representatives=weights is None, **kwargs)
            assert (barcode.essential, barcode.pairings) == (default.essential, default.pairings), f"explicit={explicit}"
            if weights is None:
                check_representatives(np.where(distances <= threshold, distances, INF), barcode, max_dim)


METRICS = {
    "euclidean": lambda diffs: np.sqrt((diffs**2).sum(axis=-1)),
    "manhattan": lambda diffs: np.abs(diffs).sum(axis=-1),
    "chebyshev": lambda diffs: np.abs(diffs).max(axis=-1, initial=0.0),
}


@given(
    points=st.integers(1, 9).flatmap(
        lambda n: st.integers(1, 3).flatmap(
            lambda dim: st.lists(
                st.lists(st.floats(-10, 10, allow_nan=False, width=32), min_size=dim, max_size=dim),
                min_size=n,
                max_size=n,
            )
        )
    ),
    metric=st.sampled_from(sorted(METRICS)),
    max_dim=st.integers(0, 2),
    data=st.data(),
)
def test_points_match_their_distance_matrix(points, metric, max_dim, data):
    """Points give the barcode of their distance matrix under the metric (with a threshold, built sparse)."""
    points = np.array(points)
    distances = METRICS[metric](points[:, None] - points[None])
    threshold = data.draw(st.sampled_from(sorted({float(d) for d in distances.flat}) + [INF]))
    for method in METHODS:
        barcode = rips(points=points, metric=metric, threshold=threshold, max_dim=max_dim, method=method)
        expected = rips(distances, max_dim, threshold=threshold, method=method)
        assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings)
    # Lists of points are read too
    from_list = rips(points=points.tolist(), metric=metric, max_dim=max_dim)
    assert from_list.pairings == rips(distances, max_dim).pairings


def _rips_of_points(_):
    points = np.random.default_rng(1).uniform(size=(200, 2))
    return rips(points=points, threshold=0.2).pairings


def test_fork_after_use():
    """Every path (points with and without a threshold, the parallel algorithm) still works in a process forked after
    using it: no thread pool may be left behind half-copied."""
    import multiprocessing

    points = np.random.default_rng(1).uniform(size=(200, 2))
    expected = rips(points=points, threshold=0.2).pairings
    rips(points=points)
    rips(points=points, algorithm=Algorithm.parallel(2))
    with multiprocessing.get_context("fork").Pool(1) as pool:
        result = pool.map_async(_rips_of_points, [0])
        assert result.get(timeout=60) == [expected]


def test_sparse_and_points_validation():
    for bad, message in [
        (scipy.sparse.coo_matrix(np.zeros((2, 3))), "square"),
        (scipy.sparse.coo_matrix(([1.0, 2.0], ([0, 1], [1, 0])), shape=(2, 2)), "both 1 and 2"),
        (scipy.sparse.coo_matrix(([-1.0], ([0], [1])), shape=(2, 2)), "non-negative"),
    ]:
        with pytest.raises(ValueError, match=message):
            rips(bad, 1)
    for kwargs, message in [
        ({}, "distance matrix"),
        ({"distance_matrix": np.zeros((2, 2)), "points": np.zeros((2, 2))}, "not both"),
        ({"points": np.array([[0.0], [np.nan]])}, "finite"),
        ({"points": [[0.0, 1.0], [1.0]]}, "same length"),
        ({"points": np.zeros((2, 2)), "metric": "cosine"}, "Unknown metric"),
    ]:
        with pytest.raises(ValueError, match=message):
            rips(**kwargs)
    # The diagonal of a sparse matrix is ignored, and an edge may be given both ways round
    both_ways = scipy.sparse.coo_matrix(([5.0, 1.0, 1.0], ([0, 0, 1], [0, 1, 0])), shape=(2, 2))
    assert rips(both_ways, 0).pairings == [[(0.0, 1.0)]]


# ====== Invariants (no reference library needed) ===========================


def rips_simplices(distances, max_dim):
    """Every simplex of dimension at most `max_dim + 1` that ever appears, with its diameter."""
    n = len(distances)
    return {
        s: diameter(distances, s)
        for k in range(1, max_dim + 3)
        for s in itertools.combinations(range(n), k)
        if diameter(distances, s) != INF
    }


@given(distances=distance_matrices(), max_dim=st.integers(0, 2), data=st.data())
def test_rank_invariant(distances, max_dim, data):
    """For diameters s <= t, the bars of dimension d alive from s to t count the d-cycles of the Rips complex at s
    that are not yet boundaries at t (by linear algebra over Z/2). This determines the barcode completely."""
    from modp import Echelon, kernel

    simplices = rips_simplices(distances, max_dim)
    boundaries = {s: {f: 1 for f in boundary(s)} for s in simplices}
    barcode = rips(distances, max_dim)
    times = sorted(set(simplices.values()))
    if len(times) > 5:
        times = sorted(data.draw(st.lists(st.sampled_from(times), min_size=1, max_size=5, unique=True)))
    for s, t in itertools.combinations_with_replacement(times, 2):
        for dim in range(max_dim + 1):
            cycles = kernel([c for c, f in simplices.items() if f <= s and len(c) == dim + 1], boundaries, 2)
            span = Echelon(2, [boundaries[c] for c, f in simplices.items() if f <= t and len(c) == dim + 2])
            rank = len(span)
            for cycle in cycles:
                span.add(cycle)
            bars = sum(1 for birth, death in diagram(barcode, dim) if birth <= s and death > t)
            assert bars == len(span) - rank, f"dim {dim}, s={s}, t={t}"


@given(distances=distance_matrices(), max_dim=st.integers(0, 2), seed=st.integers(0, 2**32 - 1))
def test_relabelling_and_monotone_change(distances, max_dim, seed):
    """Permuting the points changes nothing, and a strictly increasing change of distances (keeping 0) changes each
    bar's values by the same function, since a simplex's diameter is the largest of its distances. The change is to
    each distance's rank among the distances, which is exact in float32 (as the distances are read)."""
    barcode = rips(distances, max_dim)
    order = np.random.default_rng(seed).permutation(len(distances))
    permuted = rips(distances[np.ix_(order, order)], max_dim)
    rank = {d: float(i) for i, d in enumerate(sorted(set(distances.flat) - {INF}))}
    rank[INF] = INF
    ranked = rips(np.vectorize(rank.__getitem__)(distances), max_dim)
    for dim in range(max_dim + 1):
        assert diagram(permuted, dim) == diagram(barcode, dim)
        assert diagram(ranked, dim) == [(rank[b], rank[d]) for b, d in diagram(barcode, dim)]
