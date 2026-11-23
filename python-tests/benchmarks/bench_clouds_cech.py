"""Benchmark phlite.clouds.cech (Čech complexes in any dimension).

In the plane and in space, against GUDHI's Delaunay–Čech complex (by Bauer and Edelsbrunner, the same barcode) and
phlite's own `delaunay_cech`; in more dimensions, phlite alone (no other Python package computes Čech persistence).
"""

import functools
import math

import gudhi
import numpy as np
import pytest
from common import fingerprint, run

from phlite.clouds import cech, delaunay_cech

SUITE = {
    "title": "Čech complexes: `phlite.clouds.cech`",
    "description": "Persistence (dimensions 0 to `max_dim`) of the Čech filtration of `n` random points in the unit "
        "cube of dimension `d`, by `cech(points, max_dim, max_radius=r)` (every simplex of up to `max_dim + 2` points "
        "within the radius, listed and reduced). In the plane and in space the Delaunay–Čech filtration has the same "
        "barcode from a far smaller complex, so GUDHI's `DelaunayCechComplex` and phlite's `delaunay_cech` are there "
        "for scale; in more dimensions phlite is alone.",
    "order": 17,
    "summary": ["GUDHI (Delaunay–Čech)", "phlite (delaunay_cech)", "phlite (serial)"],
    "default": "phlite (serial)",
    "light_references": ["GUDHI (Delaunay–Čech)"],
}


def make(n, d, seed=0):
    return np.random.default_rng(seed).random((n, d))


def bars(b, top):
    return fingerprint([list(b.pairings[k]) + [(x, math.inf) for x in b.essential[k]] for k in range(top + 1)])


def phlite_cech(max_dim, r):
    def compute(points):
        b = cech(points, max_dim, max_radius=r)
        return lambda: bars(b, max_dim)

    return compute


def phlite_dc(max_dim, r):
    def compute(points):
        b = delaunay_cech(points, max_dim)
        # Cut at the radius, as the Čech run is
        cut = [[(x, y if y <= r else math.inf) for x, y in b.pairings[k] if x <= r] + [(x, math.inf) for x in b.essential[k] if x <= r] for k in range(max_dim + 1)]
        return lambda: fingerprint(cut)

    return compute


def gudhi_dc(max_dim, r):
    def compute(points):
        tree = gudhi.DelaunayCechComplex(points=points).create_simplex_tree(output_squared_values=False)
        tree.compute_persistence(homology_coeff_field=2, min_persistence=0)
        cut = []
        for k in range(max_dim + 1):
            cut.append([(float(x), float(y) if y <= r else math.inf) for x, y in tree.persistence_intervals_in_dimension(k) if x <= r])
        return lambda: fingerprint(cut)

    return compute


# (n, d, max_dim, max_radius)
CASES = [(40, 2, 1, 0.3), (80, 2, 1, 0.2), (40, 3, 2, 0.35), (30, 4, 2, 0.5), (25, 6, 2, 0.8)]
QUICK = [(40, 2, 1, 0.3)]


@functools.cache
def reference(n, d, max_dim, r):
    return phlite_cech(max_dim, r)(make(n, d))()


def params():
    for case in CASES:
        n, d, max_dim, r = case
        names = ["phlite (serial)"] + (["GUDHI (Delaunay–Čech)", "phlite (delaunay_cech)"] if d <= 3 else [])
        for name in names:
            marks = [pytest.mark.quick] if case in QUICK else []
            yield pytest.param(*case, name, marks=marks, id=f"{n}-{d}-{max_dim}-{r}-{name}")


@pytest.mark.parametrize("n, d, max_dim, r, implementation", params())
def test_cech(benchmark, n, d, max_dim, r, implementation):
    impl = {"phlite (serial)": phlite_cech, "GUDHI (Delaunay–Čech)": gudhi_dc, "phlite (delaunay_cech)": phlite_dc}[implementation]
    run(
        benchmark,
        group=f"cech n={n} d={d} max_dim={max_dim} r={r}",
        implementation=impl(max_dim, r),
        make_input=lambda: make(n, d),
        reference=None if implementation == "phlite (serial)" else functools.partial(reference, n, d, max_dim, r),
    )
