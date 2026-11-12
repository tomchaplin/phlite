"""Benchmark phlite_rips against ripser and giotto-ph on random point clouds.

Vietoris–Rips persistence of `n` points uniformly in the unit cube, in dimensions up to `max_dim`, from a distance
matrix. giotto-ph is ripser with parallelism. Cases marked `quick` are the ones `just bench-check` uses. The large
cases (1000 points) run only the main implementations, timed once each.
"""

import functools

import numpy as np
import pytest
from common import THREADS, fingerprint, run

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

# (n, max_dim)
CASES = [(100, 2), (200, 2), (300, 2), (400, 2)]
QUICK = [(100, 2)]
LARGE = [(1000, 1), (1000, 2), (2000, 1)]
# The implementations run on the large cases: all but representatives (whose memory grows with the barcode)
# ripser runs out of memory (more than 7 GB) on these, so they are checked against giotto-ph instead
NO_RIPSER = [(1000, 2)]


def reference_name(n, max_dim):
    return "giotto-ph (1 thread)" if (n, max_dim) in NO_RIPSER else REFERENCE


@functools.cache
def distances(n):
    """Points uniformly in the unit cube, with distances rounded to float32 so every implementation sees the same values."""
    rng = np.random.default_rng(0)
    points = rng.uniform(size=(n, 3))
    d = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return d.astype(np.float32).astype(np.float64)


@functools.cache
def reference(n, max_dim):
    return IMPLEMENTATIONS[reference_name(n, max_dim)]((distances(n), max_dim))()


def params():
    for n, max_dim in CASES + LARGE:
        large = (n, max_dim) in LARGE
        marks = [pytest.mark.quick] if (n, max_dim) in QUICK else [pytest.mark.large] if large else []
        for implementation in IMPLEMENTATIONS:
            if large and implementation == "phlite (serial, reps)":
                continue
            if (n, max_dim) in NO_RIPSER and implementation == "ripser":
                continue
            yield pytest.param(n, max_dim, implementation, marks=marks, id=f"{n}-{max_dim}-{implementation}")


@pytest.mark.parametrize("n, max_dim, implementation", params())
def test_rips(benchmark, n, max_dim, implementation):
    run(
        benchmark,
        group=f"rips n={n} dim≤{max_dim}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: (distances(n), max_dim),
        reference=None if implementation == reference_name(n, max_dim) else functools.partial(reference, n, max_dim),
        rounds=1 if (n, max_dim) in LARGE else None,
    )
