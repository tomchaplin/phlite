"""Benchmark phlite_rips against ripser and giotto-ph on point clouds.

Vietoris–Rips persistence of `n` points, in dimensions up to `max_dim`, from a distance matrix: points uniformly in
the unit cube (`cube`, whose bars are all short), or noisy samples of shapes with long bars (`torus`, `sphere`, a wedge
of `circles` or of `spheres`; see datasets.py). Sorted columns aren't run on the shapes, where they take hours. giotto-ph is ripser with parallelism. Cases marked `quick` are the ones
`just bench-check` uses. The large cases are timed once each.
"""

import functools

import numpy as np
import pytest
from common import THREADS, fingerprint, run
import datasets

# Import everything up front, so that import costs are not counted
from gph import ripser_parallel
from phlite_rips import Algorithm, rips
from ripser import ripser

INF = float("inf")
REFERENCE = "ripser"


def phlite(algorithm=Algorithm.serial, method="cohomology", representatives=False):
    """`algorithm` builds the `Algorithm`."""

    def compute(case):
        distances, max_dim = case
        barcode = rips(distances, max_dim, method=method, algorithm=algorithm(), representatives=representatives)
        return lambda: fingerprint([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])

    return compute


def ripser_py(case):
    distances, max_dim = case
    dgms = ripser(distances, distance_matrix=True, maxdim=max_dim)["dgms"]
    return lambda: fingerprint([[(float(b), float(d)) for b, d in dgm] for dgm in dgms])


def giotto(threads):
    def compute(case):
        distances, max_dim = case
        dgms = ripser_parallel(distances, maxdim=max_dim, metric="precomputed", n_threads=threads)["dgms"]
        return lambda: fingerprint([[(float(b), float(d)) for b, d in dgm] for dgm in dgms])

    return compute


IMPLEMENTATIONS = {
    "ripser": ripser_py,
    "giotto-ph (1 thread)": giotto(1),
    "giotto-ph (parallel)": giotto(THREADS),
    "phlite (serial)": phlite(),
    "phlite (parallel)": phlite(lambda: Algorithm.parallel(THREADS)),
    "phlite (serial, reps)": phlite(representatives=True),
    # Every column representation, cached or not, named explicitly (the presets' choice differs between packages)
    **{
        f"phlite (serial, {columns}{'' if cache else ', uncached'})": phlite(
            lambda columns=columns, cache=cache: Algorithm.serial().with_columns(columns).with_cache(cache)
        )
        for columns in ["merged", "heap", "sorted"]
        for cache in [True, False]
    },
    "phlite (serial, store R)": phlite(lambda: Algorithm.serial().with_store_r()),
    "phlite (parallel, lock-free)": phlite(lambda: Algorithm.parallel(THREADS).with_storage("lock-free")),
}

# (shape, n, max_dim)
CASES = [
    ("cube", 100, 2),
    ("cube", 200, 2),
    ("cube", 300, 2),
    ("cube", 400, 2),
    ("torus", 400, 2),
    ("sphere", 400, 2),
    ("spheres", 400, 2),
    ("circles", 1000, 1),
]
QUICK = [("cube", 100, 2)]
LARGE = [("cube", 1000, 1), ("cube", 1000, 2), ("cube", 2000, 1), ("torus", 1000, 2), ("circles", 2000, 1)]
# The implementations run on the large cases: all but representatives (whose memory grows with the barcode)


def no_ripser(n, max_dim):
    """ripser runs out of memory (more than 7 GB) on these, so they are checked against giotto-ph instead."""
    return n >= 1000 and max_dim >= 2


def reference_name(n, max_dim):
    return "giotto-ph (1 thread)" if no_ripser(n, max_dim) else REFERENCE


def label(shape, n, max_dim):
    return f"rips n={n} dim≤{max_dim}" if shape == "cube" else f"rips {shape} n={n} dim≤{max_dim}"


@functools.cache
def distances(shape, n):
    """The distance matrix of `n` points of `shape`, rounded to float32 so every implementation sees the same values."""
    if shape == "cube":
        points = np.random.default_rng(0).uniform(size=(n, 3))
    else:
        points = datasets.points(shape, n)
    return datasets.distance_matrix(points)


@functools.cache
def reference(shape, n, max_dim):
    return IMPLEMENTATIONS[reference_name(n, max_dim)]((distances(shape, n), max_dim))()


def params():
    for case in CASES + LARGE:
        shape, n, max_dim = case
        large = case in LARGE
        marks = [pytest.mark.quick] if case in QUICK else [pytest.mark.large] if large else []
        for implementation in IMPLEMENTATIONS:
            if large and implementation == "phlite (serial, reps)":
                continue
            if no_ripser(n, max_dim) and implementation == "ripser":
                continue
            if shape != "cube" and "sorted" in implementation:
                # Hours where the others take seconds: the columns added reach far below the pivot
                continue
            id = f"{n}-{max_dim}-{implementation}" if shape == "cube" else f"{shape}-{n}-{max_dim}-{implementation}"
            yield pytest.param(shape, n, max_dim, implementation, marks=marks, id=id)


@pytest.mark.parametrize("shape, n, max_dim, implementation", params())
def test_rips(benchmark, shape, n, max_dim, implementation):
    run(
        benchmark,
        group=label(shape, n, max_dim),
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: (distances(shape, n), max_dim),
        reference=(
            None
            if implementation == reference_name(n, max_dim)
            else functools.partial(reference, shape, n, max_dim)
        ),
        rounds=1 if (shape, n, max_dim) in LARGE else None,
    )
