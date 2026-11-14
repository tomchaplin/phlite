"""Fuzz weighted Rips (phlite.clouds.rips with `weights`) and DTM-Rips against GUDHI's WeightedRipsComplex and
DTMRipsComplex, and check representatives from first principles."""

import itertools
import math

import numpy as np
import pytest
import scipy.sparse
from gudhi.dtm_rips_complex import DTMRipsComplex
from gudhi.point_cloud.dtm import DistanceToMeasure
from gudhi.weighted_rips_complex import WeightedRipsComplex
from hypothesis import given, note
from hypothesis import strategies as st

from mod2 import Span, boundary, chain_sum
from phlite.core import Algorithm
from phlite.clouds import dtm, rips
from test_clouds_rips import ALGORITHMS, INF, METHODS, diagram, distance_matrices

# Weights exactly representable in float32 (like the distances), including ties and zeros
WEIGHTS = st.one_of(st.integers(0, 3).map(lambda w: w / 4), st.floats(0, 2, allow_nan=False, width=32))


def weighted_value(distances, weights, simplex):
    """GUDHI's weighted Rips value of a simplex: the largest of its vertices' 2 w and edges' values."""
    value = max(2 * weights[v] for v in simplex)
    for i, j in itertools.combinations(sorted(simplex), 2):
        # (The larger index's weight first, as GUDHI and phlite add them)
        value = max(value, distances[j, i] + weights[j] + weights[i])
    return value


def gudhi_diagram(distances, weights, max_dim, threshold=INF):
    tree = WeightedRipsComplex(distances, weights=list(weights), max_filtration=threshold).create_simplex_tree(
        max_dim + 1
    )
    tree.compute_persistence(min_persistence=-1, persistence_dim_max=True)
    return [
        sorted((float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(dim) if b < d)
        for dim in range(max_dim + 1)
    ]


def check_representatives(distances, weights, barcode, max_dim):
    n = len(distances)
    value = lambda s: weighted_value(distances, weights, s)  # noqa: E731
    for dim in range(max_dim + 1):
        cofaces = [(value(s), s) for s in itertools.combinations(range(n), dim + 2)]
        cofaces = [(f, s) for f, s in cofaces if f != INF]

        def boundaries_before(time, strict):
            return Span(boundary(s) for f, s in cofaces if (f < time if strict else f <= time))

        bars = list(zip(barcode.pairings[dim], barcode.pairing_representatives[dim]))
        bars += [((b, INF), rep) for b, rep in zip(barcode.essential[dim], barcode.essential_representatives[dim])]
        for (birth, death), rep in bars:
            note(f"dim={dim} bar=({birth}, {death}) rep={rep}")
            if dim > 0:
                assert not chain_sum(boundary(cell) for cell in rep), "not a cycle"
            assert max(value(cell) for cell in rep) == birth, "not born at birth"
            chain = set(rep)
            assert chain not in boundaries_before(death, strict=True), "already a boundary before death"
            if death != INF:
                assert chain in boundaries_before(death, strict=False), "not a boundary at death"


@given(distances=distance_matrices(max_points=8), max_dim=st.integers(0, 2), data=st.data())
def test_weighted_rips_matches_gudhi(distances, max_dim, data):
    n = len(distances)
    weights = np.array(data.draw(st.lists(WEIGHTS, min_size=n, max_size=n)))
    note(f"weights={weights.tolist()}")
    expected = gudhi_diagram(distances, weights, max_dim)
    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        note(f"method={method} algorithm={algorithm}")
        barcode = rips(distances, max_dim, weights=weights, method=method, algorithm=algorithm, representatives=True)
        for dim in range(max_dim + 1):
            assert diagram(barcode, dim) == expected[dim], f"dim {dim}"
        check_representatives(distances, weights, barcode, max_dim)


@given(distances=distance_matrices(max_points=8), max_dim=st.integers(0, 2), data=st.data())
def test_weighted_rips_threshold_and_sparse(distances, max_dim, data):
    """A threshold, and the same as a sparse matrix (the edges up to it): GUDHI's barcode with `max_filtration`."""
    n = len(distances)
    weights = np.array(data.draw(st.lists(WEIGHTS, min_size=n, max_size=n)))
    values = sorted({weighted_value(distances, weights, s) for s in itertools.combinations(range(n), 2)} - {INF})
    threshold = data.draw(st.sampled_from(values + [INF]))
    note(f"weights={weights.tolist()} threshold={threshold}")
    expected = gudhi_diagram(distances, weights, max_dim, threshold)
    for method in METHODS:
        barcode = rips(distances, max_dim, weights=weights, threshold=threshold, method=method)
        for dim in range(max_dim + 1):
            assert diagram(barcode, dim) == expected[dim], f"dim {dim}"
    # Every edge of finite length, as a sparse matrix, with the threshold
    rows, cols = np.nonzero(np.triu(np.isfinite(distances), 1))
    sparse = scipy.sparse.coo_matrix((distances[rows, cols], (rows, cols)), shape=(n, n))
    barcode = rips(sparse, max_dim, weights=weights, threshold=threshold)
    for dim in range(max_dim + 1):
        assert diagram(barcode, dim) == expected[dim], f"dim {dim} (sparse)"


@given(
    points=st.integers(1, 9).flatmap(
        lambda n: st.lists(st.lists(st.floats(-10, 10, allow_nan=False, width=32), min_size=2, max_size=2), min_size=n, max_size=n)
    ),
    k=st.integers(1, 9),
    q=st.sampled_from([1.0, 2.0, 3.5]),
)
def test_dtm_matches_gudhi(points, k, q):
    points = np.array(points)
    k = min(k, len(points))
    expected = DistanceToMeasure(k, q=q).fit_transform(points)
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    for ours in [dtm(points=points, k=k, q=q), dtm(distances, k=k, q=q)]:
        assert np.allclose(ours, expected, rtol=1e-12, atol=1e-12)
    # DTM-Rips: weighted Rips with these weights (GUDHI's own weights, for exact values)
    weights = DistanceToMeasure(k, q=2.0, metric="precomputed").fit_transform(distances)
    tree = DTMRipsComplex(distance_matrix=distances, k=k).create_simplex_tree(2)
    tree.compute_persistence(min_persistence=-1, persistence_dim_max=True)
    barcode = rips(distances, 1, weights=weights)
    for dim in range(2):
        expected = sorted((float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(dim) if b < d)
        assert diagram(barcode, dim) == expected


def test_points_and_weights():
    """Points with weights (and a threshold, so built sparse) give the dense matrix's weighted barcode."""
    rng = np.random.default_rng(0)
    points = rng.random((60, 2))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    weights = dtm(points=points, k=6)
    for threshold in [None, 0.5]:
        a = rips(points=points, weights=weights, threshold=threshold, max_dim=1)
        b = rips(distances, 1, weights=weights, threshold=threshold)
        assert (a.pairings, a.essential) == (b.pairings, b.essential)


def test_weights_validation():
    distances = np.zeros((3, 3))
    for weights, message in [([0.0, 1.0], "2 weights for 3 points"), ([0.0, math.inf, 1.0], "finite")]:
        with pytest.raises(ValueError, match=message):
            rips(distances, weights=weights)
    with pytest.raises(ValueError, match="k must be"):
        dtm(distances, k=4)
    with pytest.raises(ValueError, match="dense"):
        dtm(scipy.sparse.coo_matrix(distances), k=1)
    # A heavy point appears late, and its edges after it
    barcode = rips([[0.0, 1.0], [1.0, 0.0]], 0, weights=[0.0, 2.0])
    # Point 1 appears at 4, and its edge at max(0, 4, 1 + 0 + 2) = 4 too: a bar of length zero, left out
    assert (barcode.essential, barcode.pairings) == ([[0.0]], [[]])
