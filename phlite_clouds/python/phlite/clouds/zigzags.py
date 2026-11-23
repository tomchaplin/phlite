"""Zigzag persistence of point clouds that change: `cloud_zigzag` and `sliding_windows`.

```python
from phlite.clouds import cloud_zigzag, sliding_windows

# Windows of 30 consecutive points of a time-delay embedding, each joined to the next by the union of their complexes
bars = cloud_zigzag(sliding_windows(points, 30), threshold=0.4, max_dim=1)
# Points that move: frame t is an (n, d) array, point j labelled j throughout; Čech complexes, joined by intersections
bars = cloud_zigzag([positions[t] for t in range(T)], 0.2, filtration="cech", join="intersection")
```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional

import numpy as np
from phlite.core import ZigzagBar, ZigzagBarcode

from ._clouds import _cloud_zigzag

__all__ = ["cloud_zigzag", "sliding_windows"]


def sliding_windows(points, window: int, step: int = 1) -> list[tuple[list[int], np.ndarray]]:
    """Windows of `window` consecutive points, `step` apart, as frames for `cloud_zigzag`: `(labels, points)` pairs,
    each point labelled by its index."""
    points = np.asarray(points, dtype=float)
    if step < 1:
        raise ValueError("step must be positive")
    window = min(window, len(points))
    return [(list(range(s, s + window)), points[s : s + window]) for s in range(0, len(points) - window + 1, step)]


def _frames(frames, distances):
    out = []
    for frame in frames:
        if distances is not None and not isinstance(frame, tuple):
            out.append(([int(v) for v in frame], None))
        elif isinstance(frame, tuple):
            labels, pts = frame
            out.append(([int(v) for v in labels], None if pts is None else np.asarray(pts, dtype=float)))
        else:
            pts = np.asarray(frame, dtype=float)
            out.append((list(range(len(pts))), pts))
    return out


def cloud_zigzag(
    frames,
    threshold: float,
    max_dim: int = 1,
    *,
    filtration: Literal["rips", "cech", "alpha", "delaunay_cech", "witness"] = "rips",
    metric: str = "euclidean",
    join: Literal["union", "intersection"] = "union",
    distances=None,
    witnesses=None,
    strong: bool = False,
    algorithm=None,
) -> ZigzagBarcode:
    """The zigzag barcode of a sequence of point clouds, each one's complex cut at `threshold` and joined to the
    next by their union (`K_0 -> K_0 ∪ K_1 <- K_1 -> ...`) or intersection (`K_0 <- K_0 ∩ K_1 -> K_1 <- ...`).

    Points are named by labels, so that a point in several frames is the same vertex in each: the complexes are
    on labels, and a feature persists from one frame to the next while it overlaps itself (with unions) or while what
    it shares with the next frame carries it (with intersections). Computed by FastZigzag (Dey and Hou).

    Args:
        frames: The frames, each either an `(n, d)` array of points (labelled `0, ..., n - 1`: the same points
            moving), or a `(labels, points)` tuple (as `sliding_windows` gives); with `distances`, each a sequence of
            labels.
        threshold: The scale: the largest distance within a simplex for Rips, the radius for Čech, alpha and
            Delaunay–Čech, alpha squared for witness complexes.
        max_dim: The largest homology dimension (simplices are listed up to one more).
        filtration: How each frame's complex is built: `'rips'` (in `metric`, or from `distances`), `'cech'`,
            `'alpha'` and `'delaunay_cech'` (points in the plane or space), or `'witness'` (each frame's points as
            landmarks, witnessed by `witnesses`).
        metric: For Rips: `'euclidean'`, `'manhattan'` or `'chebyshev'`.
        join: `'union'` or `'intersection'`.
        distances: For Rips: a square distance matrix indexed by label, instead of the frames' points.
        witnesses: For `'witness'`: an `(m, d)` array of witnesses.
        strong: For `'witness'`: strong witnessing rather than relaxed.
        algorithm: The reduction algorithm (a `phlite.core.Algorithm`, default `Algorithm.serial()`).
            Pipeline defaults (`'clouds.cloud_zigzag'`): columns='pivot', cache=False, store_r=True.

    Returns:
        A `phlite.core.ZigzagBarcode` over levels: frame `i` is level `2 i` and the join of frames `i` and `i + 1`
        level `2 i + 1`, each interval the levels where its class is alive (both ends inclusive).
    """
    frames = _frames(frames, distances)
    if not frames:
        return ZigzagBarcode([], 0)
    bars = _cloud_zigzag(
        frames, float(threshold), max_dim, filtration=filtration, metric=metric, join=join,
        distances=None if distances is None else np.asarray(distances, dtype=float),
        witnesses=None if witnesses is None else np.asarray(witnesses, dtype=float), strong=strong,
        algorithm=algorithm,
    )
    return ZigzagBarcode((ZigzagBar(d, b, e, True, True) for d, b, e in bars), 2 * len(frames) - 1)
