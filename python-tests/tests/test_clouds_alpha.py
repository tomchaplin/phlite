"""Fuzz phlite.clouds.alpha against GUDHI's alpha complexes (exact), and check its representatives from first
principles."""

import bisect
import itertools
import math

import gudhi
import numpy as np
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from grid import algorithm_grid
from mod2 import Span, boundary, chain_sum
from phlite.core import Algorithm
from phlite.clouds import alpha, alpha_complex

INF = math.inf
METHODS = ["cohomology", "homology"]
ALGORITHMS = [
    Algorithm.standard(),
    Algorithm.serial(),
    Algorithm.parallel(),
    Algorithm.serial().with_columns("merged").with_cache(True),
    Algorithm.clearing().with_store_r(False),
]


@st.composite
def point_clouds(draw, max_points=12):
    """Points in the plane or in space: random (float32, so exact in float64), or small integers (lots of ties:
    repeated, collinear, coplanar and cospherical points)."""
    dim = draw(st.sampled_from([2, 3]))
    n = draw(st.integers(1, max_points))
    if dim == 3 and draw(st.integers(0, 3)) == 0:
        # On a plane or a line in space (exactly: small integers), so triangulated in their own span
        small = st.integers(-2, 2)
        if draw(st.booleans()):
            a, b, c = draw(small), draw(small), draw(small)
            uv = draw(st.lists(st.tuples(small, small), min_size=n, max_size=n))
            points = [[u, v, a * u + b * v + c] for u, v in uv]
        else:
            direction = draw(st.tuples(small, small, small))
            ts = draw(st.lists(small, min_size=n, max_size=n))
            points = [[t * direction[0], t * direction[1] + 1, t * direction[2] - 1] for t in ts]
        return np.array(points, dtype=np.float64)
    if draw(st.booleans()):
        coord = st.floats(-10, 10, allow_nan=False, width=32)
    else:
        coord = st.integers(-2, 2).map(float)
    points = draw(st.lists(st.lists(coord, min_size=dim, max_size=dim), min_size=n, max_size=n))
    return np.array(points, dtype=np.float64)


def scale(points):
    """A tolerance for squared radii: they are computed in floating point (GUDHI's exactly)."""
    return 1e-9 * max(1.0, float(np.abs(points).max()) ** 2)


def diagram(barcode, dim, tolerance):
    bars = list(barcode.pairings[dim]) + [(b, INF) for b in barcode.essential[dim]]
    return sorted((b, d) for b, d in bars if d - b > tolerance)


def gudhi_complex(points):
    """GUDHI's alpha complex and its simplex tree."""
    complex = gudhi.AlphaComplex(points=points, precision="exact")
    return complex, complex.create_simplex_tree()


def gudhi_diagram(tree, dim, tolerance):
    if tree.num_vertices() == 1:
        # GUDHI gives no bars at all for a single point
        return [(0.0, INF)] if dim == 0 else []
    tree.compute_persistence(min_persistence=-1)
    return sorted((float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(dim) if d - b > tolerance)


def same(ours, theirs, tolerance):
    """Whether the two diagrams match bar for bar, up to `tolerance` (as multisets: values an ulp apart can sort
    differently, so each of ours is matched to an unmatched bar of theirs with nearby values)."""
    if len(ours) != len(theirs):
        return False
    theirs = sorted(theirs)
    births = [b for b, _ in theirs]
    used = [False] * len(theirs)
    for b, d in sorted(ours):
        start = bisect.bisect_left(births, b - tolerance)
        for i in range(start, len(theirs)):
            c, e = theirs[i]
            if c > b + tolerance:
                return False
            if not used[i] and (d == e or abs(d - e) <= tolerance):
                used[i] = True
                break
        else:
            return False
    return True


def check_representatives(points, barcode, max_dim, tolerance, complex=alpha_complex):
    """Each representative is a cycle of simplices of the alpha complex, born at its birth, and a boundary at its death
    but not before (over Z/2). Checked against phlite's own complex (`alpha_complex`), since with cospherical points
    GUDHI may triangulate differently (with the same barcode)."""
    value = dict(complex(points, squared=True))
    for dim in range(max_dim + 1):
        cofaces = [(f, s) for s, f in value.items() if len(s) == dim + 2]

        def boundaries_before(time, strict):
            return Span(boundary(s) for f, s in cofaces if (f < time if strict else f <= time))

        bars = list(zip(barcode.pairings[dim], barcode.pairing_representatives[dim]))
        bars += [((b, INF), rep) for b, rep in zip(barcode.essential[dim], barcode.essential_representatives[dim])]
        for (birth, death), rep in bars:
            note(f"dim={dim} bar=({birth}, {death}) rep={rep}")
            assert all(tuple(cell) in value and len(cell) == dim + 1 for cell in rep), "not a simplex of the complex"
            if dim > 0:
                assert not chain_sum(boundary(cell) for cell in rep), "not a cycle"
            assert max(value[tuple(cell)] for cell in rep) == birth, "not born at birth"
            chain = set(map(tuple, rep))
            assert chain not in boundaries_before(death, strict=True), "already a boundary before death"
            if death != INF:
                assert chain in boundaries_before(death, strict=False), "not a boundary at death"


def test_alpha_complex_is_a_filtration():
    """Every face of a simplex is listed, before it, with a value no larger; and the complex is the triangulation's."""
    rng = np.random.default_rng(0)
    for points in [rng.random((50, 2)), rng.random((40, 3)), np.array(list(itertools.product(range(3), repeat=3)), float)]:
        complex = alpha_complex(points)
        position = {s: i for i, (s, _) in enumerate(complex)}
        value = dict(complex)
        assert len(position) == len(complex)
        for s, v in complex:
            for face in itertools.combinations(s, len(s) - 1):
                if face:
                    assert position[face] < position[s] and value[face] <= v
        # The Euler characteristic of a triangulation of a convex set is 1
        assert sum((-1) ** (len(s) - 1) for s, _ in complex) == 1


@given(points=point_clouds())
def test_alpha_matches_gudhi(points):
    note(f"points={points.tolist()}")
    _, tree = gudhi_complex(points)
    max_dim = points.shape[1] - 1
    tolerance = scale(points)
    expected = [gudhi_diagram(tree, dim, tolerance) for dim in range(max_dim + 1)]
    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        note(f"method={method} algorithm={algorithm}")
        barcode = alpha(points, squared=True, method=method, algorithm=algorithm, representatives=True)
        assert len(barcode.pairings) == max_dim + 1
        for dim in range(max_dim + 1):
            ours = diagram(barcode, dim, tolerance)
            assert same(ours, expected[dim], tolerance), f"dim {dim}: {ours} != {expected[dim]}"
        check_representatives(points, barcode, max_dim, tolerance)


@given(points=point_clouds(max_points=8))
def test_every_algorithm_agrees(points):
    expected = alpha(points)
    for method, algorithm in itertools.product(METHODS, algorithm_grid(Algorithm)):
        barcode = alpha(points, method=method, algorithm=algorithm)
        assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings), f"{method} {algorithm!r}"


@given(points=point_clouds())
def test_radii_are_square_roots(points):
    """Radii are the square roots of the squared radii (a bar shorter than rounding may vanish when rooted)."""
    radii, squared = alpha(points), alpha(points, squared=True)

    def longer_than_rounding(bars):
        # (On the squared scale for both: a bar's relative length halves when rooted)
        return sorted((b, d) for b, d in bars if d - b > 1e-12 * abs(d))

    for dim in range(len(radii.pairings)):
        rooted = [(b * b, d * d) for b, d in radii.pairings[dim]]
        ours = [(math.sqrt(b), math.sqrt(d)) for b, d in longer_than_rounding(rooted)]
        theirs = longer_than_rounding(squared.pairings[dim])
        assert len(ours) == len(theirs)
        for (b, d), (b2, d2) in zip(ours, theirs):
            assert math.isclose(b * b, b2, rel_tol=1e-12, abs_tol=1e-300)
            assert math.isclose(d * d, d2, rel_tol=1e-12, abs_tol=1e-300)


@given(points=point_clouds(), seed=st.integers(0, 2**32 - 1))
def test_permuting_and_moving_points(points, seed):
    """Relabelling the points changes nothing, exactly (each circumsphere is computed from its vertices in a canonical
    order). Rotating (in the plane, by a quarter turn) and scaling by a power of two (both exact, unlike translating)
    change the values accordingly, up to rounding: values are accurate to a few units in the last place, so a bar
    shorter than that (exactly) may round to zero length or not."""
    note(f"points={points.tolist()}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    expected = alpha(points, squared=True)
    permuted = alpha(points[order], squared=True)
    for dim in range(len(expected.pairings)):
        assert sorted(permuted.pairings[dim]) == sorted(expected.pairings[dim])
    moved = points * 4.0
    if points.shape[1] == 2:
        moved = moved[:, ::-1] * np.array([1.0, -1.0])
    scaled = alpha(moved, squared=True)

    def longer_than_rounding(bars, factor):
        return sorted((b * factor, d * factor) for b, d in bars if d - b > 1e-12 * abs(d))

    for dim in range(len(expected.pairings)):
        ours, theirs = longer_than_rounding(expected.pairings[dim], 16), longer_than_rounding(scaled.pairings[dim], 1)
        assert len(ours) == len(theirs)
        for (b, d), (b2, d2) in zip(ours, theirs):
            assert math.isclose(b, b2, rel_tol=1e-9, abs_tol=1e-300) and math.isclose(d, d2, rel_tol=1e-9)


def test_larger_clouds_match_gudhi():
    """Thousands of points, in the plane and in space, and a grid with lots of cospherical points."""
    rng = np.random.default_rng(0)
    grid = np.array(list(itertools.product(range(6), range(6), range(5))), dtype=float)
    for points in [rng.random((3000, 2)), rng.random((2000, 3)), grid]:
        _, tree = gudhi_complex(points)
        tolerance = scale(points)
        barcode = alpha(points, squared=True)
        for dim in range(points.shape[1]):
            assert same(diagram(barcode, dim, tolerance), gudhi_diagram(tree, dim, tolerance), tolerance), dim


def test_default_columns():
    resolved = Algorithm.serial().resolve("clouds.alpha")
    assert (resolved.columns, resolved.cache, resolved.store_r) == ("pivot", False, True)


def test_input_validation():
    for bad, message in [
        (np.zeros((3, 4)), "2 or 3 coordinates"),
        (np.zeros((3, 1)), "2 or 3 coordinates"),
        (np.zeros((0, 2)), "no points"),
        (np.array([[0.0, np.inf], [1.0, 1.0]]), "finite"),
        ([[0.0, 1.0], [1.0]], "same length"),
    ]:
        with pytest.raises(ValueError, match=message):
            alpha(bad)
    # Repeated points: cells name the first copy
    barcode = alpha([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]], representatives=True)
    assert barcode.pairings == [[(0.0, 1.5)], []]
    assert barcode.essential_representatives[0] == [[(0,)]]
