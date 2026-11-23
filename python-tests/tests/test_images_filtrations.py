"""phlite.images.filtrations against brute force: every pixel's value from its definition."""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from phlite.images import cubical, filtrations


@st.composite
def binary_images(draw):
    """A random binary image of 1 to 3 axes, sparse to dense."""
    shape = draw(st.sampled_from([(7,), (5, 6), (4, 3, 5), (1, 8), (6, 6)]))
    p = draw(st.sampled_from([0.1, 0.3, 0.5, 0.8, 1.0]))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    return rng.random(shape) < p


def brute_distance(sources, metric="manhattan"):
    pts = list(itertools.product(*[range(n) for n in sources.shape]))
    src = [p for p in pts if sources[p]]
    out = np.full(sources.shape, np.inf)
    for p in pts:
        for q in src:
            d = sum(abs(a - b) for a, b in zip(p, q))
            out[p] = min(out[p], d)
    return out


@given(image=binary_images())
def test_dilation_erosion_signed_distance(image):
    if True:
        to_active, to_inactive = brute_distance(image), brute_distance(~image)
        assert np.array_equal(filtrations.dilation(image), to_active)
        assert np.array_equal(filtrations.erosion(image), to_inactive)
        assert np.array_equal(filtrations.signed_distance(image), np.where(image, to_inactive - 1, -to_active))
        capped = filtrations.dilation(image, n_iterations=1)
        assert np.array_equal(capped, np.where(to_active <= 1, to_active, np.inf))
        last = filtrations.dilation(image, n_iterations=1, unreached="last")
        assert np.isfinite(last).all()


@given(image=binary_images())
def test_height_radial_density(image):
    if True:
        coords = np.stack(np.meshgrid(*[np.arange(n) for n in image.shape], indexing="ij"), -1).astype(float)
        direction = np.arange(1, image.ndim + 1, dtype=float) * (-1) ** np.arange(image.ndim)
        unit = direction / np.linalg.norm(direction)
        shifted = coords.copy()
        for a, n in enumerate(image.shape):
            if unit[a] < 0:
                shifted[..., a] = n - 1 - shifted[..., a]
        h = filtrations.height(image, direction)
        assert np.allclose(h[image], (shifted @ np.abs(unit))[image]) and np.isinf(h[~image]).all()
        center = np.array(image.shape) / 2
        r = filtrations.radial(image, center, radius=2.0)
        dist = np.sqrt(((coords - center) ** 2).sum(-1))
        assert np.allclose(np.where(image & (dist <= 2), dist, np.inf), r)
        d = filtrations.density(image, radius=1.5)
        pts = list(itertools.product(*[range(n) for n in image.shape]))
        for p in pts:
            count = sum(1 for q in pts if image[q] and math.dist(p, q) <= 1.5)
            assert d[p] == count


def test_through_cubical():
    image = np.zeros((20, 20), dtype=bool)
    image[5:15, 5] = image[5:15, 14] = image[5, 5:15] = image[14, 5:15] = True  # a square loop
    # The loop's hole: one H_1 bar in the dilation filtration (born at 0, filled when the dilations meet)
    barcode = cubical(filtrations.dilation(image))
    assert len(barcode.pairings[1]) + len(barcode.essential[1]) == 1
    barcode = cubical(filtrations.signed_distance(image), construction="V")
    assert any(d - b > 1 for b, d in barcode.pairings[1])
    with pytest.raises(ValueError):
        filtrations.height(image, [0, 0])
