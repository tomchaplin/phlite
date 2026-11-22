"""Zigzag persistence of videos: `video_zigzag`.

A video's frames, each cut at a threshold (its sublevel or superlevel set, as a cubical complex), make a zigzag
`K_0 -> K_0 ∪ K_1 <- K_1 -> ...`, whose barcode follows features (components, loops, voids) from frame to frame: a
blob that drifts keeps its bar, two blobs that merge end one, a ring that closes starts an H1 bar. A feature keeps its
bar from one frame to the next as long as it overlaps itself (the unions are of the complexes).

```python
from phlite.images import video_zigzag

barcode = video_zigzag(frames, threshold=0.5)     # frames: an array of shape (n_frames, height, width)
barcode.in_dimension(0)                           # ZigzagBar(dimension, birth, death, ...) in levels:
                                                  #   frame i at 2i, frames i and i + 1 together at 2i + 1
barcode.betti_curve(1)[::2]                       # the number of loops in each frame
barcode.plot()
```
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import numpy as np

from phlite.core import ZigzagBar, ZigzagBarcode

from ._images import _video_zigzag

__all__ = ["video_zigzag"]


def video_zigzag(
    video,
    threshold: float,
    max_dim: Optional[int] = None,
    *,
    construction: Literal["T", "V"] = "T",
    filtration: Literal["sublevel", "superlevel"] = "sublevel",
    periodic: Union[bool, Sequence[bool]] = False,
    algorithm=None,
) -> ZigzagBarcode:
    """The zigzag barcode of a video, each frame cut at `threshold`.

    Args:
        video: The frames: an array of shape `(n_frames, *frame_shape)` (frames of 1 to 3 axes), or a sequence of
            frames of one shape. Colour frames should be made greyscale first.
        threshold: Where to cut each frame: its pixels at most `threshold` (sublevel) or at least (superlevel).
        max_dim: The largest homology dimension (default: one less than a frame's axes, or all of them if every axis
            is periodic).
        construction: `'T'` (pixels as squares: diagonal neighbours touch) or `'V'` (pixels as vertices), as for
            `cubical`.
        filtration: `'sublevel'` (dark pixels) or `'superlevel'` (bright pixels).
        periodic: Whether the frames wrap round: `True` for every axis, or one bool per axis.
        algorithm: The reduction algorithm (a `phlite.core.Algorithm`), as for `cubical`.

    Returns:
        A `phlite.core.ZigzagBarcode` over levels: frame `i` is level `2 i` and the union of frames `i` and `i + 1`
        level `2 i + 1`. Each interval is the levels where its class is alive (both ends inclusive).
    """
    frames = np.asarray(video if isinstance(video, np.ndarray) else np.stack([np.asarray(f) for f in video]), dtype=float)
    if frames.ndim < 2:
        raise ValueError("a video is a sequence of frames: give an array with at least two axes")
    if frames.shape[0] == 0:
        return ZigzagBarcode([], 0)
    axes = frames.ndim - 1
    if isinstance(periodic, (bool, np.bool_)):
        periodic = [bool(periodic)] * axes
    periodic = [bool(p) for p in periodic]
    if len(periodic) != axes:
        raise ValueError(f"periodic has {len(periodic)} entries but a frame has {axes} axes")
    if filtration not in ("sublevel", "superlevel"):
        raise ValueError(f"filtration must be 'sublevel' or 'superlevel', got {filtration!r}")
    if max_dim is None:
        max_dim = axes if all(periodic) else axes - 1
    bars = _video_zigzag(
        frames, float(threshold), max_dim, construction, filtration == "superlevel", periodic, algorithm=algorithm
    )
    return ZigzagBarcode((ZigzagBar(d, b, e, True, True) for d, b, e in bars), 2 * frames.shape[0] - 1)
