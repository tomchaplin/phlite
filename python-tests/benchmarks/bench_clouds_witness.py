"""Benchmark phlite.clouds.witness against GUDHI's witness complexes.

Persistence (dimensions 0 and 1) of the witness complex of `m` random landmarks among `n` random points in the unit
square or cube (every point a witness), relaxed (`EuclideanWitnessComplex`) or strong (`EuclideanStrongWitnessComplex`).
"""

import functools
import math

import gudhi
import numpy as np
import pytest
from common import THREADS, fingerprint, run

from phlite.clouds import witness
from phlite.core import Algorithm

SUITE = {
    "title": "Witness complexes: `phlite.clouds.witness` vs GUDHI",
    "description": "Persistence in dimensions 0 and 1 of the witness complex (values α² up to about 3/m, simplices up to "
        "dimension 2) of `m` random landmarks among `n` random points in the unit square or cube, each point a witness: GUDHI's "
        "`EuclideanWitnessComplex` / `EuclideanStrongWitnessComplex` (building the simplex tree and computing "
        "persistence) against `witness(landmarks, points, 1, strong=...)`.",
    "order": 16,
    "summary": ["GUDHI", "phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": ["GUDHI"],
}


def bound(landmarks):
    """The largest α² listed: a few times the squared spacing of the landmarks (unbounded, the strong complex grows
    huge)."""
    m, dim = landmarks.shape
    return (3.0 if dim == 2 else 4.0) / m


def make(n, m, dim, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.random((n, dim))
    return points[rng.choice(n, m, replace=False)], points


def gudhi_impl(strong):
    def compute(case):
        landmarks, points = case
        cls = gudhi.EuclideanStrongWitnessComplex if strong else gudhi.EuclideanWitnessComplex
        tree = cls(landmarks=landmarks, witnesses=points).create_simplex_tree(max_alpha_square=bound(landmarks), limit_dimension=2)
        tree.compute_persistence(homology_coeff_field=2, min_persistence=0)
        dims = [[(float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(k)] for k in range(2)]
        return lambda: fingerprint(dims)

    return compute


def phlite_impl(strong, algorithm=Algorithm.serial):
    def compute(case):
        landmarks, points = case
        b = witness(landmarks, points, 1, strong=strong, max_alpha_square=bound(landmarks), algorithm=algorithm())
        return lambda: fingerprint([list(p) + [(x, math.inf) for x in e] for p, e in zip(b.pairings, b.essential)])

    return compute


# (n, m, dim)
CASES = [(500, 30, 2), (2000, 60, 2), (2000, 40, 3), (10000, 100, 2)]
QUICK = [(500, 30, 2)]


@functools.cache
def reference(n, m, dim, strong):
    return gudhi_impl(strong)(make(n, m, dim))()


def params():
    for case in CASES:
        for strong in (False, True):
            for name in ["GUDHI", "phlite (serial)", "phlite (parallel)"]:
                marks = [pytest.mark.quick] if case in QUICK else []
                yield pytest.param(*case, strong, name, marks=marks, id=f"{case[0]}-{case[1]}-{case[2]}-{'strong' if strong else 'relaxed'}-{name}")


@pytest.mark.parametrize("n, m, dim, strong, implementation", params())
def test_witness(benchmark, n, m, dim, strong, implementation):
    impl = {
        "GUDHI": gudhi_impl(strong),
        "phlite (serial)": phlite_impl(strong),
        "phlite (parallel)": phlite_impl(strong, lambda: Algorithm.parallel(THREADS)),
    }[implementation]
    run(
        benchmark,
        group=f"witness {'strong' if strong else 'relaxed'} n={n} m={m} d={dim}",
        implementation=impl,
        make_input=lambda: make(n, m, dim),
        reference=None if implementation == "GUDHI" else functools.partial(reference, n, m, dim, strong),
    )
