"""Benchmark phlite.complexes' constructions around persistence: images, kernels and cokernels and the six-pack of
an inclusion, persistence over the integers, and spectral sequences. phlite alone here (phimaker, Kenzo and the other
references are compared in perf-results/, as they need environments of their own)."""

import math

import numpy as np
import pytest
from common import fingerprint, run

from phlite.clouds import rips_complex
from phlite.complexes import FilteredComplex

SUITE = {
    "title": "Maps, integers and spectral sequences: `phlite.complexes`",
    "description": "On the Rips filtration (up to triangles) of `n` random points in the unit square, as an explicit "
        "complex: the image, kernel and cokernel of the inclusion of the left half of the points "
        "(`image_kernel_cokernel`) and its whole six-pack with representatives (`six_pack`); ordinary persistence "
        "over Z/2 for scale; persistence over the integers (`modulus=0`: the barcode over Q and its torsion); and "
        "the spectral sequence of the filtration to its last page (`spectral_sequence`). phlite alone: phimaker, "
        "Kenzo, spectral-sequence-engine and PerMaViss are compared in `perf-results/`.",
    "order": 44,
    "summary": ["persistence (Z/2)", "image/kernel/cokernel", "six-pack", "integers", "spectral sequence"],
    "default": "persistence (Z/2)",
    "light_references": [],
}


def make(n, seed=0):
    points = np.random.default_rng(seed).random((n, 2))
    simplices = rips_complex(points=points, dimension=2, threshold=0.25)
    sub = [s for s, _ in simplices if all(points[v, 0] < 0.5 for v in s)]
    return simplices, sub


def barcode_fingerprint(b, top=1):
    return fingerprint([list(b.pairings[k]) + [(x, math.inf) for x in b.essential[k]] for k in range(top + 1)])


def persistence(case):
    simplices, _ = case
    b = FilteredComplex.from_simplices(simplices).persistence(1)
    return lambda: barcode_fingerprint(b)


def ikc(case):
    simplices, sub = case
    FilteredComplex.from_simplices(simplices).image_kernel_cokernel(sub)
    return lambda: None


def six_pack(case):
    simplices, sub = case
    FilteredComplex.from_simplices(simplices).six_pack(sub, max_dim=1)
    return lambda: None


def integers(case):
    simplices, _ = case
    b = FilteredComplex.from_simplices(simplices, modulus=0).persistence(1)
    return lambda: None


def spectral(case):
    simplices, _ = case
    ss = FilteredComplex.from_simplices(simplices).spectral_sequence(1)
    ss.infinity
    return lambda: None


IMPLEMENTATIONS = {
    "persistence (Z/2)": persistence,
    "image/kernel/cokernel": ikc,
    "six-pack": six_pack,
    "integers": integers,
    "spectral sequence": spectral,
}
CASES = [200, 500, 1000]
QUICK = [200]


def params():
    for n in CASES:
        for name in IMPLEMENTATIONS:
            marks = [pytest.mark.quick] if n in QUICK else []
            yield pytest.param(n, name, marks=marks, id=f"{n}-{name}".replace(" ", "_"))


@pytest.mark.parametrize("n, implementation", params())
def test_maps(benchmark, n, implementation):
    run(
        benchmark,
        group=f"Rips of {n} points (threshold 0.25)",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: make(n),
    )
