"""Filtrations of binary images, as giotto-tda's `gtda.images` filtrations: each turns a binary image (which pixels are
*active*) into a greyscale one, whose sublevel sets `phlite.images.cubical` then filters by.

```python
from phlite.images import cubical, filtrations
barcode = cubical(filtrations.signed_distance(binary))
```

Coordinates are in the image's own axis order (numpy's). By default pixels a filtration doesn't reach are given
`inf`, so that they never enter (`cubical` leaves them out); `unreached="last"` gives them one more than the largest
value instead, so that they enter last, as giotto-tda does (with its `max_value_`).
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np

__all__ = ["height", "radial", "dilation", "erosion", "signed_distance", "density"]

Unreached = Literal["inf", "last"]


def _binary(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim < 1 or image.ndim > 4:
        raise ValueError(f"images have between 1 and 4 axes, got {image.ndim}")
    return image.astype(bool)


def _finish(values: np.ndarray, unreached: Unreached) -> np.ndarray:
    if unreached == "inf":
        return values
    if unreached != "last":
        raise ValueError(f"unreached must be 'inf' or 'last', got {unreached!r}")
    finite = values[np.isfinite(values)]
    last = (finite.max() if finite.size else 0.0) + 1.0
    return np.where(np.isfinite(values), values, last)


def _coordinates(shape) -> np.ndarray:
    """Each pixel's coordinates, an array of shape `shape + (len(shape),)`."""
    return np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).astype(float)


def height(image, direction: Optional[Sequence[float]] = None, *, unreached: Unreached = "inf") -> np.ndarray:
    """The height filtration: each active pixel at its height along `direction` (its coordinates' dot product with the
    unit vector, measured from the corner the direction points away from, so heights start at 0), as giotto-tda's
    `HeightFiltration`.

    Args:
        image: A binary image (anything nonzero is active), with 1 to 4 axes.
        direction: The direction (default: all ones), one component per axis.
        unreached: `'inf'` (inactive pixels never enter) or `'last'` (they enter after everything else).
    """
    image = _binary(image)
    d = np.ones(image.ndim) if direction is None else np.asarray(direction, dtype=float)
    if d.shape != (image.ndim,) or not np.any(d):
        raise ValueError("direction needs one component per axis, not all zero")
    d = d / np.linalg.norm(d)
    coords = _coordinates(image.shape)
    # Measure from the corner the direction points away from
    for a, n in enumerate(image.shape):
        if d[a] < 0:
            coords[..., a] = (n - 1) - coords[..., a]
    values = coords @ np.abs(d)
    return _finish(np.where(image, values, np.inf), unreached)


def radial(
    image,
    center: Optional[Sequence[float]] = None,
    radius: float = np.inf,
    *,
    metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
    unreached: Unreached = "inf",
) -> np.ndarray:
    """The radial filtration: each active pixel within `radius` of `center` at its distance from it, as giotto-tda's
    `RadialFiltration`.

    Args:
        image: A binary image, with 1 to 4 axes.
        center: The centre, in pixel coordinates (default: the first pixel, all zeros).
        radius: Pixels further than this aren't reached.
        metric: `'euclidean'`, `'manhattan'` or `'chebyshev'`.
        unreached: `'inf'` or `'last'`, for inactive pixels and those beyond the radius.
    """
    image = _binary(image)
    c = np.zeros(image.ndim) if center is None else np.asarray(center, dtype=float)
    if c.shape != (image.ndim,):
        raise ValueError("center needs one coordinate per axis")
    delta = np.abs(_coordinates(image.shape) - c)
    if metric == "euclidean":
        values = np.sqrt((delta**2).sum(-1))
    elif metric in ("manhattan", "cityblock"):
        values = delta.sum(-1)
    elif metric == "chebyshev":
        values = delta.max(-1)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    values = np.where(image & (values <= radius), values, np.inf)
    return _finish(values, unreached)


def _manhattan_distance(sources: np.ndarray) -> np.ndarray:
    """The Manhattan (L1) distance from every pixel to the nearest source pixel (`inf` if there is none): exactly, as the
    L1 distance transform separates into one-dimensional ones, each a forward and a backward pass along an axis."""
    out = np.where(sources, 0.0, np.inf)
    for axis in range(out.ndim):
        out = np.moveaxis(out, axis, 0)
        for i in range(1, out.shape[0]):
            out[i] = np.minimum(out[i], out[i - 1] + 1)
        for i in range(out.shape[0] - 2, -1, -1):
            out[i] = np.minimum(out[i], out[i + 1] + 1)
        out = np.moveaxis(out, 0, axis)
    return out


def dilation(image, n_iterations: Optional[int] = None, *, unreached: Unreached = "inf") -> np.ndarray:
    """The dilation filtration: active pixels at 0, and every other pixel at its Manhattan distance to the active ones
    (the number of 4-neighbour dilations that reach it), up to `n_iterations` (default: all), as giotto-tda's
    `DilationFiltration`.
    """
    image = _binary(image)
    values = _manhattan_distance(image)
    if n_iterations is not None:
        values = np.where(values <= n_iterations, values, np.inf)
    return _finish(values, unreached)


def erosion(image, n_iterations: Optional[int] = None, *, unreached: Unreached = "inf") -> np.ndarray:
    """The erosion filtration: inactive pixels at 0, and every active pixel at its Manhattan distance to the inactive
    ones (the number of erosions that remove it), up to `n_iterations` (default: all), as giotto-tda's
    `ErosionFiltration`. An image with no inactive pixel is never eroded (all `unreached`).
    """
    image = _binary(image)
    values = _manhattan_distance(~image)
    if n_iterations is not None:
        values = np.where(values <= n_iterations, values, np.inf)
    return _finish(values, unreached)


def signed_distance(image, n_iterations: Optional[int] = None, *, unreached: Unreached = "inf") -> np.ndarray:
    """The signed distance filtration, as giotto-tda's `SignedDistanceFiltration`: an active pixel at its Manhattan
    distance to the inactive ones minus 1 (so 0 on the boundary of an active region), an inactive pixel at minus its
    distance to the active ones; distances beyond `n_iterations` (default: all) aren't reached.
    """
    image = _binary(image)
    inside = _manhattan_distance(~image) - 1
    outside = -_manhattan_distance(image)
    values = np.where(image, inside, outside)
    if n_iterations is not None:
        values = np.where(np.abs(np.where(image, inside + 1, outside)) <= n_iterations, values, np.inf)
    return _finish(np.where(np.isnan(values), np.inf, values), unreached)


def density(
    image,
    radius: float = 3.0,
    *,
    metric: Literal["euclidean", "manhattan", "chebyshev"] = "euclidean",
) -> np.ndarray:
    """The density filtration: every pixel at the number of active pixels within `radius` of it (itself included),
    as giotto-tda's `DensityFiltration`. Use `cubical(..., filtration='superlevel')` for dense regions first.
    """
    image = _binary(image)
    reach = int(np.floor(radius))
    offsets = np.stack(np.meshgrid(*[np.arange(-reach, reach + 1)] * image.ndim, indexing="ij"), axis=-1).reshape(
        -1, image.ndim
    )
    if metric == "euclidean":
        lengths = np.sqrt((offsets**2).sum(-1))
    elif metric in ("manhattan", "cityblock"):
        lengths = np.abs(offsets).sum(-1)
    elif metric == "chebyshev":
        lengths = np.abs(offsets).max(-1)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    padded = np.pad(image.astype(float), reach)
    out = np.zeros(image.shape)
    for offset in offsets[lengths <= radius]:
        index = tuple(slice(reach + o, reach + o + n) for o, n in zip(offset, image.shape))
        out += padded[index]
    return out
