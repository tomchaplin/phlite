"""Benchmark phlite.images.cubical with periodic boundaries against GUDHI's PeriodicCubicalComplex, and the binary
image filtrations (phlite.images.filtrations)."""

import functools
import math

import gudhi
import numpy as np
import pytest
from common import fingerprint, run

from phlite.images import cubical, filtrations

SUITE = {
    "title": "Periodic cubical complexes and binary filtrations: `phlite.images`",
    "description": "Persistence of a random image (smoothed noise) wrapped round on every axis, as a torus: GUDHI's "
        "`PeriodicCubicalComplex` against `cubical(image, periodic=True)`, T-construction. And the binary filtrations "
        "of `phlite.images.filtrations` (as giotto-tda's), each followed by `cubical`, on the image cut at its median.",
    "order": 31,
    "summary": ["GUDHI", "phlite (serial)", "phlite (dilation)", "phlite (signed distance)", "phlite (density)"],
    "default": "phlite (serial)",
    "light_references": ["GUDHI"],
}


def make(shape, seed=0):
    rng = np.random.default_rng(seed)
    image = rng.random(shape)
    # A little smoothing (a few box blurs along each axis, wrapping round), so features have some size
    for _ in range(3):
        for axis in range(len(shape)):
            image = (image + np.roll(image, 1, axis) + np.roll(image, -1, axis)) / 3
    return image


def bars_of(b, top):
    return fingerprint([list(b.pairings[k]) + [(x, math.inf) for x in b.essential[k]] if k < len(b.pairings) else [] for k in range(top + 1)])


def phlite_periodic(image):
    b = cubical(image, periodic=True)
    return lambda: bars_of(b, image.ndim)


def gudhi_periodic(image):
    c = gudhi.PeriodicCubicalComplex(top_dimensional_cells=image, periodic_dimensions=[True] * image.ndim)
    c.compute_persistence(homology_coeff_field=2, min_persistence=0)
    return lambda: fingerprint([[(float(x), float(y)) for x, y in c.persistence_intervals_in_dimension(k)] for k in range(image.ndim + 1)])


def binary(kind):
    def compute(image):
        mask = image > np.median(image)
        values = {"dilation": filtrations.dilation, "signed distance": filtrations.signed_distance,
                  "density": lambda m: filtrations.density(m, radius=3)}[kind](mask)
        cubical(np.where(np.isfinite(values), values, np.nanmax(np.where(np.isfinite(values), values, np.nan)) + 1))
        return lambda: None

    return compute


IMPLEMENTATIONS = {
    "GUDHI": gudhi_periodic,
    "phlite (serial)": phlite_periodic,
    "phlite (dilation)": binary("dilation"),
    "phlite (signed distance)": binary("signed distance"),
    "phlite (density)": binary("density"),
}
CASES = [(128, 128), (512, 512), (1024, 1024), (48, 48, 48), (96, 96, 96)]
QUICK = [(128, 128)]


@functools.cache
def reference(shape):
    return gudhi_periodic(make(shape))()


def params():
    for shape in CASES:
        for name in IMPLEMENTATIONS:
            if name == "phlite (density)" and (len(shape) == 3 or shape[0] > 512):
                continue  # a radius-3 ball sum per pixel: slow on big inputs, and not what's being measured
            marks = [pytest.mark.quick] if shape in QUICK else []
            yield pytest.param(shape, name, marks=marks, id=f"{'x'.join(map(str, shape))}-{name}")


@pytest.mark.parametrize("shape, implementation", params())
def test_periodic(benchmark, shape, implementation):
    run(
        benchmark,
        group=f"periodic {'×'.join(map(str, shape))}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: make(shape),
        reference=functools.partial(reference, shape) if implementation == "phlite (serial)" else None,
    )
