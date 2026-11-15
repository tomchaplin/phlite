"""Fuzz phlite.clouds.delaunay_cech against GUDHI's Delaunay–Čech complexes (exact), and check its representatives from
first principles. (The point clouds and checks are the alpha tests'.)"""

import itertools
import math

import gudhi
import numpy as np
from hypothesis import given, note

from grid import algorithm_grid
from phlite.core import Algorithm
from phlite.clouds import alpha_complex, delaunay_cech, delaunay_cech_complex
from test_clouds_alpha import ALGORITHMS, INF, METHODS, check_representatives, diagram, point_clouds, same, scale


def gudhi_diagram(points, dim, tolerance):
    tree = gudhi.DelaunayCechComplex(points=points, precision="exact").create_simplex_tree()
    if tree.num_vertices() == 1:
        # GUDHI gives no bars at all for a single point
        return [(0.0, INF)] if dim == 0 else []
    tree.compute_persistence(min_persistence=-1)
    return sorted((float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(dim) if d - b > tolerance)


@given(points=point_clouds())
def test_delaunay_cech_matches_gudhi(points):
    note(f"points={points.tolist()}")
    max_dim = points.shape[1] - 1
    tolerance = scale(points)
    expected = [gudhi_diagram(points, dim, tolerance) for dim in range(max_dim + 1)]
    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        note(f"method={method} algorithm={algorithm}")
        barcode = delaunay_cech(points, squared=True, method=method, algorithm=algorithm, representatives=True)
        for dim in range(max_dim + 1):
            ours = diagram(barcode, dim, tolerance)
            assert same(ours, expected[dim], tolerance), f"dim {dim}: {ours} != {expected[dim]}"
        check_representatives(points, barcode, max_dim, tolerance, complex=delaunay_cech_complex)


@given(points=point_clouds(max_points=8))
def test_every_algorithm_agrees(points):
    expected = delaunay_cech(points)
    for method, algorithm in itertools.product(METHODS, algorithm_grid(Algorithm)):
        barcode = delaunay_cech(points, method=method, algorithm=algorithm)
        assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings), f"{method} {algorithm!r}"


@given(points=point_clouds())
def test_values_are_minimal_enclosing_balls(points):
    """Every value is at least the alpha value (an enclosing ball, not necessarily an empty one), the edges' are half
    their lengths, and faces come no later than their cofaces."""
    cech = dict(delaunay_cech_complex(points))
    alpha_values = dict(alpha_complex(points))
    assert cech.keys() == alpha_values.keys()
    for simplex, value in cech.items():
        assert value <= alpha_values[simplex] * (1 + 1e-12) + 1e-300
        if len(simplex) == 2:
            half = float(np.linalg.norm(points[simplex[0]] - points[simplex[1]])) / 2
            assert math.isclose(value, half, rel_tol=1e-12, abs_tol=1e-300)
        for face in itertools.combinations(simplex, len(simplex) - 1):
            if face:
                assert cech[face] <= value


def test_larger_clouds_match_gudhi():
    rng = np.random.default_rng(1)
    grid = np.array(list(itertools.product(range(6), range(6), range(5))), dtype=float)
    for points in [rng.random((3000, 2)), rng.random((2000, 3)), grid]:
        tolerance = scale(points)
        barcode = delaunay_cech(points, squared=True)
        for dim in range(points.shape[1]):
            assert same(diagram(barcode, dim, tolerance), gudhi_diagram(points, dim, tolerance), tolerance), dim


def test_default_columns():
    resolved = Algorithm.serial().resolve("clouds.delaunay_cech")
    assert (resolved.columns, resolved.cache, resolved.store_r) == ("pivot", False, True)
