"""Filtrations of binary images, as giotto-tda's `gtda.images` filtrations: each turns a binary image (which pixels are
*active*) into a greyscale one, whose sublevel sets `phlite.images.cubical` then filters by.

```python
from phlite.images import cubical, filtrations
barcode = cubical(filtrations.signed_distance(binary))
```

Coordinates are in the image's own axis order (numpy's). By default pixels a filtration doesn't reach are given
`inf`, so that they never enter (`cubical` leaves them out); `unreached="last"` gives them one more than the largest
value instead, so that they enter last, as giotto-tda does (with its `max_value_`).

The filtrations are computed in Rust (`phlite_images::filtrations`, which the playground uses too), for any number of
axes.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np

from ._images import _binary_filtration

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
    return _finish(_binary_filtration("height", image, direction=d.tolist()), unreached)


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
    values = _binary_filtration("radial", image, center=c.tolist(), radius=float(radius), metric=metric)
    return _finish(values, unreached)


def dilation(image, n_iterations: Optional[int] = None, *, unreached: Unreached = "inf") -> np.ndarray:
    """The dilation filtration: active pixels at 0, and every other pixel at its Manhattan distance to the active ones
    (the number of 4-neighbour dilations that reach it), up to `n_iterations` (default: all), as giotto-tda's
    `DilationFiltration`.
    """
    image = _binary(image)
    return _finish(_binary_filtration("dilation", image, n_iterations=n_iterations), unreached)


def erosion(image, n_iterations: Optional[int] = None, *, unreached: Unreached = "inf") -> np.ndarray:
    """The erosion filtration: inactive pixels at 0, and every active pixel at its Manhattan distance to the inactive
    ones (the number of erosions that remove it), up to `n_iterations` (default: all), as giotto-tda's
    `ErosionFiltration`. An image with no inactive pixel is never eroded (all `unreached`).
    """
    image = _binary(image)
    return _finish(_binary_filtration("erosion", image, n_iterations=n_iterations), unreached)


def signed_distance(image, n_iterations: Optional[int] = None, *, unreached: Unreached = "inf") -> np.ndarray:
    """The signed distance filtration, as giotto-tda's `SignedDistanceFiltration`: an active pixel at its Manhattan
    distance to the inactive ones minus 1 (so 0 on the boundary of an active region), an inactive pixel at minus its
    distance to the active ones; distances beyond `n_iterations` (default: all) aren't reached.
    """
    image = _binary(image)
    return _finish(_binary_filtration("signed_distance", image, n_iterations=n_iterations), unreached)


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
    if metric not in ("euclidean", "manhattan", "cityblock", "chebyshev"):
        raise ValueError(f"unknown metric {metric!r}")
    return _binary_filtration("density", image, radius=float(radius), metric=metric)
