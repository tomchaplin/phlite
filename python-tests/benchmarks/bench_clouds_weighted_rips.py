"""Benchmark weighted Rips (DTM-Rips) in phlite.clouds.rips against GUDHI's WeightedRipsComplex.

DTM-Rips persistence in dimensions up to `dim` of `n` points: a noisy circle (or, `sphere`, a noisy 2-sphere) with 10%
uniform outliers, weighted by the distance to measure (k = 10) computed once, untimed, and given to both (so their
barcodes agree exactly). GUDHI is timed building its weighted Rips simplex tree and computing its persistence; on
the largest cases (minutes for GUDHI, which builds its edges in Python) only phlite runs, checked against its own
serial barcode.
"""

import functools

import numpy as np
import pytest
from common import THREADS, fingerprint, run
from gudhi.point_cloud.dtm import DistanceToMeasure
from gudhi.weighted_rips_complex import WeightedRipsComplex

from phlite.core import Algorithm
from phlite.clouds import rips

INF = float("inf")
REFERENCE = "gudhi"

SUITE = {
    "title": "Weighted Rips: DTM-Rips by `phlite.clouds.rips` vs GUDHI",
    "description": "DTM-Rips persistence in dimensions up to `dim` of `n` points, a noisy circle (or `sphere`, a noisy "
        "2-sphere) with 10% uniform outliers, weighted by the distance to measure (k = 10), computed once and given to "
        "both: GUDHI's WeightedRipsComplex (building its simplex tree and computing persistence) and "
        "`phlite.clouds.rips(distances, weights=...)`.",
    "order": 14,
    "summary": ["gudhi", "phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": ["gudhi"],
}


def phlite(algorithm=Algorithm.serial):
    def compute(case):
        distances, weights, max_dim = case
        barcode = rips(distances, max_dim, weights=weights, algorithm=algorithm())
        return lambda: fingerprint([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])

    return compute


def gudhi_weighted(case):
    distances, weights, max_dim = case
    tree = WeightedRipsComplex(distances, weights=list(weights)).create_simplex_tree(max_dim + 1)
    tree.compute_persistence(min_persistence=-1, persistence_dim_max=True)
    bars = [[(float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(dim)] for dim in range(max_dim + 1)]
    return lambda: fingerprint(bars)


IMPLEMENTATIONS = {
    "gudhi": gudhi_weighted,
    "phlite (serial)": phlite(),
    "phlite (parallel)": phlite(lambda: Algorithm.parallel(THREADS)),
}

# (shape, n, max_dim)
CASES = [("circle", 200, 1), ("circle", 500, 1), ("sphere", 150, 2)]
QUICK = [("circle", 200, 1)]
LARGE = [("circle", 1000, 1), ("sphere", 300, 2)]


@functools.cache
def case(shape, n):
    rng = np.random.default_rng(0)
    signal = n - n // 10
    if shape == "circle":
        t = rng.uniform(0, 2 * np.pi, signal)
        points = np.stack([np.cos(t), np.sin(t)], axis=1) + 0.05 * rng.standard_normal((signal, 2))
        outliers = rng.uniform(-1.5, 1.5, (n - signal, 2))
    else:
        v = rng.standard_normal((signal, 3))
        points = v / np.linalg.norm(v, axis=1, keepdims=True) + 0.05 * rng.standard_normal((signal, 3))
        outliers = rng.uniform(-1.5, 1.5, (n - signal, 3))
    points = np.concatenate([points, outliers])
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    weights = DistanceToMeasure(10, metric="precomputed").fit_transform(distances)
    return distances, weights


def reference_name(shape, n, max_dim):
    return "phlite (serial)" if (shape, n, max_dim) in LARGE else REFERENCE


@functools.cache
def reference(shape, n, max_dim):
    return IMPLEMENTATIONS[reference_name(shape, n, max_dim)]((*case(shape, n), max_dim))()


def params():
    for params in CASES + LARGE:
        shape, n, max_dim = params
        marks = [pytest.mark.quick] if params in QUICK else [pytest.mark.large] if params in LARGE else []
        for implementation in IMPLEMENTATIONS:
            if params in LARGE and implementation == "gudhi":
                continue
            yield pytest.param(shape, n, max_dim, implementation, marks=marks, id=f"{shape}-{n}-{max_dim}-{implementation}")


@pytest.mark.parametrize("shape, n, max_dim, implementation", params())
def test_weighted_rips(benchmark, shape, n, max_dim, implementation):
    run(
        benchmark,
        group=f"dtm-rips {shape} n={n} dim≤{max_dim}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: (*case(shape, n), max_dim),
        reference=(
            None
            if implementation == reference_name(shape, n, max_dim)
            else functools.partial(reference, shape, n, max_dim)
        ),
        rounds=1 if (shape, n, max_dim) in LARGE else None,
    )
