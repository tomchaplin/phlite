"""Benchmark phlite.images.cubical against GUDHI's and Cubical Ripser's cubical complexes.

Sublevel persistence of random images in both constructions: uniform noise (the worst case: few cells in apparent
pairs) and smoothed noise (more like a real image), in 2D and 3D, in every dimension.
"""

import functools

import cripser
import gudhi
import numpy as np
import perseus
import pytest
from common import THREADS, fingerprint, perseus_fingerprint, run, sweep

from phlite.core import Algorithm
from phlite.images import cubical

INF = float("inf")
REFERENCE = "gudhi"

SUITE = {
    "title": "Cubical complexes: `phlite.images.cubical` vs GUDHI and Cubical Ripser",
    "description": "Sublevel persistence, in every dimension, of a random image (`noise`: uniform; `smooth`: noise "
        "averaged over a 5-pixel window along each axis, then rounded to float32) of the given shape, in the T- or "
        "V-construction: GUDHI's `CubicalComplex` (`top_dimensional_cells` or `vertices`) and Cubical Ripser "
        "(`computePH_T` or `computePH`) against `cubical(image, construction=...)`. Perseus 6 (`perseus.cubical`, over "
        "Z/2 on one thread) takes top cells, so only runs the T-construction.",
    "order": 30,
    "summary": ["gudhi", "cripser", "perseus (F2)", "phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": ["gudhi"],
}


@functools.cache
def random_image(kind, shape, seed=0):
    rng = np.random.default_rng(seed)
    image = rng.random(shape)
    if kind == "smooth":
        for axis in range(len(shape)):
            image = sum(np.roll(image, shift, axis=axis) for shift in range(-2, 3)) / 5
    image = image.astype(np.float32).astype(np.float64)
    image.setflags(write=False)
    return image


def bars_fingerprint(bars):
    return fingerprint([[(b, d) for b, d in dim if d != b] for dim in bars])


def gudhi_cubical(case):
    image, construction = case
    complex = gudhi.CubicalComplex(**{"top_dimensional_cells" if construction == "T" else "vertices": image})
    complex.compute_persistence(homology_coeff_field=2, min_persistence=-1)
    bars = [[(float(b), float(d)) for b, d in complex.persistence_intervals_in_dimension(dim)] for dim in range(image.ndim)]
    return lambda: bars_fingerprint(bars)


def cripser_cubical(case):
    image, construction = case
    compute = cripser.computePH_T if construction == "T" else cripser.computePH
    pairs = compute(image, maxdim=image.ndim - 1, location="no")
    bars = [[] for _ in range(image.ndim)]
    for dim, b, d in pairs[:, :3]:
        bars[int(dim)].append((float(b), INF if d >= 1e308 else float(d)))
    return lambda: bars_fingerprint(bars)


def phlite(algorithm=Algorithm.serial):
    def compute(case):
        image, construction = case
        barcode = cubical(image, construction=construction, algorithm=algorithm())
        return lambda: bars_fingerprint(
            [list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)]
        )

    return compute


def perseus_cubical(case):
    """Perseus 6 over Z/2, one thread: top cells only, so the T-construction (see `params`)."""
    image, construction = case
    assert construction == "T"
    result = perseus.cubical(image, coeff=2, threads=1)
    return lambda: perseus_fingerprint(result, image.ndim - 1)


IMPLEMENTATIONS = {
    "gudhi": gudhi_cubical,
    "perseus (F2)": perseus_cubical,
    "cripser": cripser_cubical,
    "phlite (serial)": phlite(),
    "phlite (parallel)": phlite(lambda: Algorithm.parallel(THREADS)),
}
IMPLEMENTATIONS.update(sweep(phlite))

# (kind, shape, construction)
CASES = [
    ("noise", (256, 256), "T"),
    ("noise", (256, 256), "V"),
    ("smooth", (1024, 1024), "T"),
    ("smooth", (1024, 1024), "V"),
    ("noise", (32, 32, 32), "T"),
    ("smooth", (64, 64, 64), "V"),
]
QUICK = [("noise", (256, 256), "T")]
LARGE = [("noise", (1024, 1024), "T"), ("smooth", (1080, 1920), "T"), ("noise", (64, 64, 64), "V"), ("smooth", (128, 128, 128), "T")]


@functools.cache
def reference(kind, shape, construction):
    return IMPLEMENTATIONS[REFERENCE]((random_image(kind, shape), construction))()


def params():
    for case in CASES + LARGE:
        kind, shape, construction = case
        marks = [pytest.mark.quick] if case in QUICK else [pytest.mark.large] if case in LARGE else []
        size = "x".join(map(str, shape))
        for implementation in IMPLEMENTATIONS:
            if implementation.startswith("perseus") and construction != "T":
                continue
            yield pytest.param(kind, shape, construction, implementation, marks=marks, id=f"{kind}-{size}-{construction}-{implementation}")


@pytest.mark.parametrize("kind, shape, construction, implementation", params())
def test_cubical(benchmark, kind, shape, construction, implementation):
    run(
        benchmark,
        group=f"cubical {kind} {'x'.join(map(str, shape))} {construction}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: (random_image(kind, shape), construction),
        reference=None if implementation == REFERENCE else functools.partial(reference, kind, shape, construction),
        rounds=1 if (kind, shape, construction) in LARGE else None,
    )
