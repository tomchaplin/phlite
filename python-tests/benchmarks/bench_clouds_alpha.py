"""Benchmark phlite.clouds.alpha against GUDHI's alpha complexes.

Alpha persistence (squared radii, every dimension) of `n` points uniformly random in the unit square or cube, or noisy
samples of a torus in R³ (long bars). GUDHI is timed building its alpha complex's simplex tree (with its default,
'safe', precision) and computing its persistence. Bars shorter than rounding (relative 1e-9) are left out of every
fingerprint, since the two compute radii differently (GUDHI exactly, phlite to a few units in the last place).
"""

import functools

import gudhi
import numpy as np
import pytest
from common import THREADS, fingerprint, run
import datasets

from phlite.core import Algorithm
from phlite.clouds import alpha

INF = float("inf")
REFERENCE = "gudhi"

SUITE = {
    "title": "Alpha: `phlite.clouds.alpha` vs GUDHI",
    "description": "Alpha persistence (squared radii) of `n` points uniformly random in the unit square (`plane`) or "
        "cube (`space`), or a noisy torus in R³ (`torus`, long bars), from the points: Delaunay triangulation, alpha "
        "values and reduction. GUDHI builds its alpha complex (default 'safe' precision) and computes persistence.",
    "order": 12,
    "summary": ["gudhi", "phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": ["gudhi"],
}


def summary(dims):
    """The fingerprint, without bars shorter than rounding."""
    return fingerprint([[(b, d) for b, d in bars if d - b > 1e-9 * max(1.0, abs(d) if d != INF else 1.0)] for bars in dims])


def phlite(algorithm=Algorithm.serial):
    def compute(case):
        points = case
        barcode = alpha(points, squared=True, algorithm=algorithm())
        return lambda: summary([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])

    return compute


def gudhi_alpha(case):
    points = case
    tree = gudhi.AlphaComplex(points=points).create_simplex_tree()
    tree.compute_persistence(min_persistence=-1)
    dims = points.shape[1]
    out = [[(float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(dim)] for dim in range(dims)]
    return lambda: summary(out)


IMPLEMENTATIONS = {
    "gudhi": gudhi_alpha,
    "phlite (serial)": phlite(),
    "phlite (parallel)": phlite(lambda: Algorithm.parallel(THREADS)),
    "phlite (serial, merged)": phlite(lambda: Algorithm.serial().with_columns("merged").with_store_r(False)),
}

# (shape, n)
CASES = [("plane", 10_000), ("plane", 100_000), ("space", 10_000), ("space", 50_000), ("torus", 20_000)]
QUICK = [("plane", 10_000)]
LARGE = [("plane", 1_000_000), ("space", 300_000)]


@functools.cache
def points(shape, n):
    rng = np.random.default_rng(0)
    if shape == "plane":
        return rng.uniform(size=(n, 2))
    if shape == "space":
        return rng.uniform(size=(n, 3))
    return datasets.points(shape, n)


@functools.cache
def reference(shape, n):
    return IMPLEMENTATIONS[REFERENCE](points(shape, n))()


def params():
    for case in CASES + LARGE:
        shape, n = case
        marks = [pytest.mark.quick] if case in QUICK else [pytest.mark.large] if case in LARGE else []
        for implementation in IMPLEMENTATIONS:
            yield pytest.param(shape, n, implementation, marks=marks, id=f"{shape}-{n}-{implementation}")


@pytest.mark.parametrize("shape, n, implementation", params())
def test_alpha(benchmark, shape, n, implementation):
    run(
        benchmark,
        group=f"alpha {shape} n={n}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: points(shape, n),
        reference=None if implementation == REFERENCE else functools.partial(reference, shape, n),
        rounds=1 if (shape, n) in LARGE else None,
    )
