# phlite.images: images

Persistent homology of images: the `phlite-images` package, built from the `phlite_images` crate (the Rust
implementation, and its bindings in `src/python`).

| Function | Filtration | Chapter |
|---|---|---|
| `phlite.images.cubical` | the cubical complex of an image or volume (T- or V-construction, sublevel or superlevel sets) | [Cubical complexes](images/cubical.md) |

Every function returns a `phlite.core.Barcode`, and takes a [`phlite.core.Algorithm`](algorithms.md) whose unset
options it fills in with its own defaults (the pipeline names are `images.<function>`, e.g. `images.cubical`).

To compute persistence of a cubical complex you have built yourself (or any other filtered cell complex), use
`phlite.complexes`, as the [image notebook](notebooks.md#an-image-phlitecomplexes) does.

## The complex itself

`phlite.images.cubical_complex(image)` lists the cubical complex as an explicit chain complex with its oriented
boundary (coefficients ±1), as arrays for `phlite.complexes.ChainComplex.from_csc`, with each cell's value and
coordinates: for your own computations, such as persistence over the integers (`modulus=0`: the barcode over Q and the
primes of the torsion, see [explicit complexes](complexes.md#over-the-integers)).

## Periodic images

`cubical(image, periodic=True)` (or one bool per axis) wraps the image round, as GUDHI's `PeriodicCubicalComplex`
and Cubical Ripser's periodic option: along a periodic axis the first and last pixels are neighbours, so a constant
image is a torus (and by default the top dimension is computed too). The periodic complex is built explicitly and
reduced as `phlite.complexes` does, with union-find at both ends: exact, in any number of axes, but slower than the
non-periodic complex on large volumes, and without representatives. It has the same barcodes as GUDHI's
`PeriodicCubicalComplex` and is 1.1–1.3× faster (`perf-results/gudhi-filtrations-2026-11-22.md`: 84 ms against 90 ms
on 256×256 noise, 2.0 s against 2.6 s on 1024×1024, 1.3 s against 1.5 s on 64³).

## Filtrations of binary images

`phlite.images.filtrations` turns a binary image into a greyscale one to filter, as giotto-tda's `gtda.images`
filtrations: `height`, `radial`, `dilation`, `erosion`, `signed_distance` (Manhattan distances, exactly) and
`density`. For example `cubical(filtrations.signed_distance(binary))`. Pixels a filtration doesn't reach are `inf` (they
never enter), or with `unreached="last"` enter after everything else, as giotto-tda's do.

## Videos: zigzag persistence

`video_zigzag(video, threshold)` follows the features of a video (components, loops, voids) from frame to frame: each
frame cut at `threshold` (its sublevel set, or with `filtration='superlevel'` its superlevel set) is a cubical complex,
and the frames make the zigzag `K_0 -> K_0 ∪ K_1 <- K_1 -> ...`, computed by FastZigzag (see
[zigzag persistence](zigzag.md)). A blob that drifts keeps its bar so long as it overlaps itself from one frame to the
next; two that merge end one; a ring that closes starts an H1 bar.

```python
from phlite.images import video_zigzag

barcode = video_zigzag(frames, threshold=0.5)    # frames: (n_frames, height, width), or volumes over time
barcode.betti_curve(0)[::2]                      # the number of components in each frame (frame i is level 2i)
barcode.plot()
```

It takes the T- or V-construction and periodic frames as `cubical` does, and frames of 1 to 3 axes; the result is a
`phlite.core.ZigzagBarcode` over levels. In Rust, `phlite_images::zigzag::video_zigzag`. Checked against `cubical` of
each frame (the Betti numbers at each frame's level) and on still stretches (every class lives on).
