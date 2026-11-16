"""Benchmark phlite.clouds.rips on sparse distance matrices against ripser and giotto-ph.

Vietoris–Rips persistence of `n` points uniformly random in the unit cube, keeping only the edges up to a radius
chosen so that each point has `degree` neighbours on average: given as a scipy sparse matrix (built untimed), whose
missing entries never appear. Too many points for a dense matrix at the largest sizes. `phlite (points)` starts from
the points instead (with the radius as its threshold), so includes computing the distances, in float64 rather than
the references' float32; its barcode is checked by the fuzz tests rather than against the references here.
"""

import functools

import numpy as np
import pytest
import scipy.sparse
import scipy.spatial
from common import THREADS, fingerprint, run, sweep

# Import everything up front, so that import costs are not counted
from gph import ripser_parallel
from phlite.core import Algorithm
from phlite.clouds import rips
from ripser import ripser

INF = float("inf")
REFERENCE = "ripser"

SUITE = {
    "title": "Sparse Rips: `phlite.clouds.rips` on sparse matrices vs ripser and giotto-ph",
    "description": "Vietoris–Rips persistence in dimensions up to `dim` of `n` points uniformly random in the unit "
        "cube, keeping only the edges up to the radius at which each point has `degree` neighbours on average, given "
        "to every implementation as the same scipy sparse matrix (missing entries never appear). `phlite (points)` "
        "starts from the points instead, so includes computing the distances.",
    "order": 11,
    "summary": ["ripser", "giotto-ph (1 thread)", "giotto-ph (parallel)", "phlite (serial)", "phlite (parallel)", "phlite (points)"],
    "default": "phlite (serial)",
    "light_references": ["ripser"],
}


def phlite(algorithm=Algorithm.serial):
    def compute(case):
        matrix, _points, _radius, max_dim = case
        barcode = rips(matrix, max_dim, algorithm=algorithm())
        return lambda: fingerprint([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])

    return compute


def phlite_points(case):
    _matrix, points, radius, max_dim = case
    barcode = rips(points=points, threshold=radius, max_dim=max_dim)
    return lambda: fingerprint([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])


def ripser_py(case):
    matrix, _points, _radius, max_dim = case
    dgms = ripser(matrix, distance_matrix=True, maxdim=max_dim)["dgms"]
    return lambda: fingerprint([[(float(b), float(d)) for b, d in dgm] for dgm in dgms])


def giotto(threads):
    def compute(case):
        matrix, _points, _radius, max_dim = case
        dgms = ripser_parallel(matrix, maxdim=max_dim, metric="precomputed", n_threads=threads)["dgms"]
        return lambda: fingerprint([[(float(b), float(d)) for b, d in dgm] for dgm in dgms])

    return compute


IMPLEMENTATIONS = {
    "ripser": ripser_py,
    "giotto-ph (1 thread)": giotto(1),
    "giotto-ph (parallel)": giotto(THREADS),
    "phlite (serial)": phlite(),
    "phlite (parallel)": phlite(lambda: Algorithm.parallel(THREADS)),
    "phlite (serial, merged)": phlite(lambda: Algorithm.serial().with_columns("merged").with_cache(True)),
    "phlite (points)": phlite_points,
}
IMPLEMENTATIONS.update(sweep(phlite))

# (n, degree, max_dim)
CASES = [(2000, 20, 2), (5000, 20, 2), (10000, 10, 2), (20000, 20, 1)]
QUICK = [(2000, 20, 2)]
LARGE = [(50000, 20, 1), (20000, 20, 2)]


def label(n, degree, max_dim):
    return f"sparse rips n={n} degree={degree} dim≤{max_dim}"


@functools.cache
def case(n, degree):
    """The points, the radius, and the sparse matrix (upper triangle, float32-rounded) of the edges up to it."""
    points = np.random.default_rng(0).uniform(size=(n, 3))
    # A ball of radius r holds n * (4/3) pi r^3 points on average
    radius = float(np.float32((3 * degree / (4 * np.pi * n)) ** (1 / 3)))
    pairs = scipy.spatial.cKDTree(points).query_pairs(radius, output_type="ndarray")
    d = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1).astype(np.float32).astype(np.float64)
    keep = d <= radius
    rows, cols = pairs[keep].min(axis=1), pairs[keep].max(axis=1)
    matrix = scipy.sparse.coo_matrix((d[keep], (rows, cols)), shape=(n, n))
    return matrix, points, radius


@functools.cache
def reference(n, degree, max_dim):
    matrix, points, radius = case(n, degree)
    return IMPLEMENTATIONS[REFERENCE]((matrix, points, radius, max_dim))()


def params():
    for params in CASES + LARGE:
        n, degree, max_dim = params
        marks = [pytest.mark.quick] if params in QUICK else [pytest.mark.large] if params in LARGE else []
        for implementation in IMPLEMENTATIONS:
            yield pytest.param(n, degree, max_dim, implementation, marks=marks, id=f"{n}-{degree}-{max_dim}-{implementation}")


@pytest.mark.parametrize("n, degree, max_dim, implementation", params())
def test_rips_sparse(benchmark, n, degree, max_dim, implementation):
    run(
        benchmark,
        group=label(n, degree, max_dim),
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: (*case(n, degree), max_dim),
        reference=(
            None
            if implementation in (REFERENCE, "phlite (points)")
            else functools.partial(reference, n, degree, max_dim)
        ),
        rounds=1 if (n, degree, max_dim) in LARGE else None,
    )
