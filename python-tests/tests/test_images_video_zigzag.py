"""phlite.images.video_zigzag: zigzag persistence of a video's frames cut at a threshold.

At each frame's level the zigzag's Betti numbers are those of the frame's own sublevel set (from `cubical`, the bars
alive at the threshold); over a still stretch (identical frames) every class lives on; and a drifting blob keeps one
bar.
"""

import numpy as np
import pytest

from phlite.core import ZigzagBarcode
from phlite.images import cubical, video_zigzag


def alive(barcode, k, t):
    if k >= len(barcode.pairings):
        return 0
    return sum(1 for b, d in barcode.pairings[k] if b <= t < d) + sum(1 for b in barcode.essential[k] if b <= t)


@pytest.mark.parametrize("construction", ["T", "V"])
@pytest.mark.parametrize("shape", [(9, 11), (5, 6, 7)])
def test_frames_against_cubical(construction, shape):
    rng = np.random.default_rng(len(shape))
    for trial in range(3):
        video = rng.random((6, *shape))
        video[3] = video[2]  # a still pair
        for filtration, threshold in (("sublevel", 0.4), ("superlevel", 0.6)):
            zz = video_zigzag(video, threshold, construction=construction, filtration=filtration)
            assert isinstance(zz, ZigzagBarcode) and zz.n_positions == 11
            for i, frame in enumerate(video):
                barcode = cubical(frame, construction=construction, filtration=filtration)
                for k in range(len(shape)):
                    t = threshold if filtration == "sublevel" else -threshold
                    expected = alive(barcode, k, t) if filtration == "sublevel" else sum(
                        1 for b, d in barcode.pairings[k] if b >= threshold > d) + sum(1 for b in barcode.essential[k] if b >= threshold)
                    assert zz.rank(2 * i, k) == expected, (trial, filtration, i, k)
            # Frames 2 and 3 are the same: every class at level 4 lives through to level 6
            for k in range(len(shape)):
                spanning = sum(1 for b in zz.in_dimension(k) if b.birth <= 4 and b.death >= 6)
                assert spanning == zz.rank(4, k)


def test_drifting_blob_and_options():
    frames = np.ones((6, 12, 12))
    for i in range(6):
        frames[i, 4:8, i : i + 4] = 0.0  # a square blob moving right by one pixel a frame
    zz = video_zigzag(frames, 0.5)
    assert [tuple(b) for b in zz] == [(0, 0, 10, True, True)]
    assert zz.betti_curve(0).tolist() == [1] * 11
    # A list of frames, periodic frames, superlevel
    assert video_zigzag(list(frames), 0.5) == zz
    assert video_zigzag(1 - frames, 0.5, filtration="superlevel") == zz
    torus = video_zigzag(np.zeros((2, 4, 4)), 0.5, periodic=True)
    assert [zz_b.dimension for zz_b in torus] == [0, 1, 1, 2]
    with pytest.raises(ValueError):
        video_zigzag(frames, 0.5, periodic=[True])
