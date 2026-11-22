"""Fuzz phlite.clouds.delaunay_cech against GUDHI's Delaunay–Čech complexes (exact), and check its representatives from
first principles. (The point clouds and checks are the alpha tests'.)"""

import itertools
import math
from fractions import Fraction

import gudhi
import numpy as np
import pytest
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


def exact_meb_squared(points):
    """The squared radius of the minimal enclosing ball of a few points (rows of floats), exactly: the smallest
    circumball of a subset (in its affine hull) that encloses them all, over the rationals."""
    pts = [[Fraction(float(x)) for x in p] for p in points]

    def circumball(sub):
        p0 = sub[0]
        v = [[a - b for a, b in zip(p, p0)] for p in sub[1:]]
        k = len(v)
        m = [[2 * sum(x * y for x, y in zip(v[i], v[j])) for j in range(k)] + [sum(x * x for x in v[i])] for i in range(k)]
        for i in range(k):
            pivot = next((r for r in range(i, k) if m[r][i] != 0), None)
            if pivot is None:
                return None  # affinely dependent: no circumball of its own
            m[i], m[pivot] = m[pivot], m[i]
            for r in range(k):
                if r != i and m[r][i] != 0:
                    f = m[r][i] / m[i][i]
                    m[r] = [x - f * y for x, y in zip(m[r], m[i])]
        t = [m[i][k] / m[i][i] for i in range(k)]
        centre = [p0[d] + sum(t[i] * v[i][d] for i in range(k)) for d in range(len(p0))]
        return centre, sum((x - y) ** 2 for x, y in zip(centre, p0))

    best = None
    for r in range(1, len(pts) + 1):
        for sub in itertools.combinations(pts, r):
            ball = circumball(list(sub))
            if ball is None:
                continue
            centre, r2 = ball
            if all(sum((x - y) ** 2 for x, y in zip(centre, q)) <= r2 for q in pts):
                best = r2 if best is None else min(best, r2)
    return best


def exact_diagram(points, dim, tolerance):
    """The Delaunay–Čech barcode with every value an exact minimal enclosing ball (on phlite's simplices, through a
    GUDHI simplex tree): the reference when GUDHI's own Delaunay–Čech values are inexact, as on nearly degenerate
    clouds."""
    tree = gudhi.SimplexTree()
    for simplex, _ in delaunay_cech_complex(points, squared=True):
        tree.insert(list(simplex), float(exact_meb_squared(points[list(simplex)])))
    tree.make_filtration_non_decreasing()
    if tree.num_vertices() == 1:
        return [(0.0, INF)] if dim == 0 else []
    tree.compute_persistence(min_persistence=-1)
    return sorted((float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(dim) if d - b > tolerance)


@given(points=point_clouds())
def test_delaunay_cech_matches_gudhi(points):
    """The same barcode as GUDHI's Delaunay–Čech complex; where they differ (GUDHI's values can be off by 1e-8 on
    nearly degenerate clouds, e.g. one coordinate 6e-8), the same as with exact minimal enclosing balls, and every
    value of phlite's exact."""
    note(f"points={points.tolist()}")
    max_dim = points.shape[1] - 1
    tolerance = scale(points)
    expected = [gudhi_diagram(points, dim, tolerance) for dim in range(max_dim + 1)]
    first = delaunay_cech(points, squared=True)
    if not all(same(diagram(first, dim, tolerance), expected[dim], tolerance) for dim in range(max_dim + 1)):
        expected = [exact_diagram(points, dim, tolerance) for dim in range(max_dim + 1)]
        for simplex, value in delaunay_cech_complex(points, squared=True):
            exact = float(exact_meb_squared(points[list(simplex)]))
            assert math.isclose(value, exact, rel_tol=1e-12, abs_tol=1e-300), (simplex, value, exact)
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
            # Coordinates below 2^-90 of the largest are flushed to zero first (as for the triangulation), which can
            # move a length by about that much: e.g. points 3.5e-26 apart at coordinates near 1
            flushed = float(np.abs(points).max()) * 2.0**-88
            assert math.isclose(value, half, rel_tol=1e-12, abs_tol=max(flushed, 1e-300))
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


# ====== The full Čech complex ======


def _bars(barcode, top):
    return [
        sorted([tuple(round(x, 9) for x in p) for p in barcode.pairings[k]] + [(round(b, 9), math.inf) for b in barcode.essential[k]])
        for k in range(top + 1)
    ]


@pytest.mark.parametrize("dim", [2, 3])
def test_cech_has_the_delaunay_cech_barcode(dim):
    """Čech and Delaunay–Čech filtrations have the same barcode (Bauer and Edelsbrunner)."""
    from phlite.clouds import cech, delaunay_cech

    rng = np.random.default_rng(dim)
    for n in (5, 9, 13):
        points = rng.random((n, dim))
        top = dim - 1
        assert _bars(cech(points, top), top) == _bars(delaunay_cech(points, top), top)
        assert _bars(cech(points, top, squared=True), top) == _bars(delaunay_cech(points, top, squared=True), top)


@given(points=point_clouds(max_points=12))
def test_cech_random_has_the_delaunay_cech_barcode(points):
    """The same, on random clouds in the plane and in space (Čech complexes listed in full, so small ones)."""
    from phlite.clouds import cech, delaunay_cech

    points = np.asarray(points, dtype=float)
    top = points.shape[1] - 1

    # Compared up to a tolerance at the points' scale, not by rounding (which can land two equal values on either side
    # of a boundary), leaving out bars shorter than it: Delaunay–Čech treats coordinates negligible next to the largest
    # as zero (merging points 1e-31 apart, say), where Čech gives them a bar that short
    tol = 1e-9 * max(1.0, float(np.abs(points).max()))

    def bars(b):
        out = []
        for k in range(top + 1):
            dim = [(x, y) for x, y in b.pairings[k] if y - x > tol] + [(x, math.inf) for x in b.essential[k]]
            out.append(sorted(dim))
        return out

    ours, theirs = bars(cech(points, top)), bars(delaunay_cech(points, top))
    assert [len(d) for d in ours] == [len(d) for d in theirs], (ours, theirs)

    def close(p, q):
        return math.isclose(p[0], q[0], abs_tol=tol) and (p[1] == q[1] or math.isclose(p[1], q[1], abs_tol=tol))

    # Matched as multisets up to the tolerance (bars born within it of each other may sort either way)
    for a, b in zip(ours, theirs):
        left = list(b)
        for p in a:
            match = next((i for i, q in enumerate(left) if close(p, q)), None)
            assert match is not None, (ours, theirs)
            left.pop(match)


def test_cech_in_more_dimensions():
    from phlite.clouds import cech, cech_complex
    from phlite.core import Barcode

    # A circle in R^5: one H_1 bar, dying at the circumradius
    t = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    circle = np.zeros((10, 5))
    circle[:, 0], circle[:, 1] = np.cos(t), np.sin(t)
    barcode = cech(circle, 1)
    assert isinstance(barcode, Barcode) and len(barcode.pairings[1]) == 1
    assert barcode.pairings[1][0][1] == pytest.approx(1.0)
    # max_radius truncates, leaving essential what is still alive
    assert len(cech(circle, 1, max_radius=0.9).essential[1]) == 1
    # The complex: edges at half their length, faces never after cofaces
    simplices = dict(cech_complex(circle, 2))
    assert simplices[(0, 1)] == pytest.approx(np.linalg.norm(circle[0] - circle[1]) / 2)
    for s, v in simplices.items():
        for i in range(len(s)):
            if len(s) > 1:
                assert simplices[s[:i] + s[i + 1:]] <= v


def _enclosing_radius2(points):
    """The squared radius of the minimal enclosing ball, independently: the dual of min r^2 s.t. |c - p_i|^2 <= r^2,
    max_λ sum λ_i |p_i|^2 - |sum λ_i p_i|^2 over the simplex (a concave quadratic), by scipy."""
    from scipy.optimize import minimize

    p = np.asarray(points, dtype=float)
    k = len(p)
    if k == 1:
        return 0.0
    sq = (p * p).sum(1)

    def negative(lam):
        c = lam @ p
        return -(lam @ sq - c @ c)

    best = None
    for start in [np.full(k, 1.0 / k)] + [np.eye(k)[i] * 0.5 + 0.5 / k for i in range(k)]:
        res = minimize(negative, start / start.sum(), method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=[{"type": "eq", "fun": lambda lam: lam.sum() - 1}], options={"ftol": 1e-14, "maxiter": 500})
        if best is None or res.fun < best:
            best = res.fun
    return -best


@pytest.mark.parametrize("dim", [4, 5, 6])
def test_cech_radii_against_convex_programming(dim):
    """Every simplex of cech_complex in 4 to 6 dimensions (where there is no Delaunay–Čech to compare with) has the
    radius of its minimal enclosing ball, computed independently as a convex program; and cech's barcode is that of
    the complex it exports."""
    from phlite.clouds import cech, cech_complex
    from phlite.complexes import FilteredComplex

    rng = np.random.default_rng(dim)
    for _ in range(3):
        points = rng.random((8, dim))
        simplices = cech_complex(points, 3)
        for s, r in simplices:
            assert r * r == pytest.approx(_enclosing_radius2(points[list(s)]), rel=1e-6, abs=1e-9), s
        ours = _bars(cech(points, 2), 2)
        explicit = FilteredComplex.from_simplices(simplices).persistence(2).diagrams()
        expected = [sorted(tuple(round(x, 9) for x in row) for row in explicit[k].tolist()) for k in range(3)]
        assert ours == expected
