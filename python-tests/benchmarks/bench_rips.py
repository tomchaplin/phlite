"""Benchmark phlite_rips against ripser and giotto-ph on random point clouds.

Vietoris–Rips persistence of `n` points uniformly in the unit cube, in dimensions up to 2, from a distance matrix.
giotto-ph is ripser with parallelism. Cases marked `quick` are the ones `just bench-check` uses.
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
MAX_DIM = 2


def phlite(algorithm=Algorithm.serial, method="cohomology", representatives=False):
    """`algorithm` builds the `Algorithm`."""

    def compute(distances):
        barcode = rips(distances, MAX_DIM, method=method, algorithm=algorithm(), representatives=representatives)
        return lambda: fingerprint([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])

    return compute


def ripser_py(distances):
    dgms = ripser(distances, distance_matrix=True, maxdim=MAX_DIM)["dgms"]
    return lambda: fingerprint([[(float(b), float(d)) for b, d in dgm] for dgm in dgms])


def giotto(threads):
    def compute(distances):
        dgms = ripser_parallel(distances, maxdim=MAX_DIM, metric="precomputed", n_threads=threads)["dgms"]
        return lambda: fingerprint([[(float(b), float(d)) for b, d in dgm] for dgm in dgms])

    return compute


IMPLEMENTATIONS = {
    "ripser": ripser_py,
    "giotto-ph (1 thread)": giotto(1),
    "giotto-ph (parallel)": giotto(THREADS),
    "phlite (serial)": phlite(),
    "phlite (parallel)": phlite(lambda: Algorithm.parallel(THREADS)),
    "phlite (serial, reps)": phlite(representatives=True),
    "phlite (parallel, lock-free)": phlite(lambda: Algorithm.parallel(THREADS).with_storage("lock-free")),
}

CASES = [100, 200, 300, 400]
QUICK = [100]


@functools.cache
def distances(n):
    """Points uniformly in the unit cube, with distances rounded to float32 so every implementation sees the same values."""
    rng = np.random.default_rng(0)
    points = rng.uniform(size=(n, 3))
    d = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return d.astype(np.float32).astype(np.float64)


@functools.cache
def reference(n):
    return IMPLEMENTATIONS[REFERENCE](distances(n))()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("n", [pytest.param(n, marks=pytest.mark.quick) if n in QUICK else n for n in CASES])
def test_rips(benchmark, n, implementation):
    run(
        benchmark,
        group=f"rips n={n} dim≤{MAX_DIM}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: distances(n),
        reference=None if implementation == REFERENCE else functools.partial(reference, n),
    )
