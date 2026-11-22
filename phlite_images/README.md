# phlite_images

Persistent homology of images and volumes (cubical complexes), in Rust and Python (`pip install phlite-images`, or
`pip install phlite` for every package). Part of [phlite](https://github.com/tomchaplin/phlite), which brings the
techniques of Ripser and its relatives to any filtration; see [the book](https://phlite.tomchaplin.xyz/book/) and
the [in-browser playground](https://phlite.tomchaplin.xyz/playground/), where you can paint an image and see its
barcode.

```python
import numpy as np
from phlite.images import cubical

image = np.random.default_rng(0).random((512, 512))
barcode = cubical(image)                              # T-construction (pixels as squares), sublevel sets
barcode = cubical(image, construction="V")            # pixels as vertices
barcode = cubical(image, filtration="superlevel")     # bright regions first
barcode.pairings[1]                                   # (birth, death) of the finite H1 bars
```

## What it does

`cubical` takes a numpy array with 1 to 4 axes (an image, a volume, a video of volumes) and computes the persistent
homology of its cubical complex:

* in the **T-construction** (GUDHI's `top_dimensional_cells`, Cubical Ripser's `computePH_T`) or the
  **V-construction** (GUDHI's `vertices`, Cubical Ripser's `computePH`);
* through sublevel or superlevel sets; NaN pixels are left out;
* with representatives, on the doubled-grid coordinates GUDHI uses;
* **periodic** along any axes (`periodic=`), as GUDHI's `PeriodicCubicalComplex`;
* exported with signed boundaries (`cubical_complex`), for `phlite.complexes` (over the integers, spectral sequences).

`video_zigzag` computes the zigzag persistence of a video (each frame cut at a threshold, consecutive frames joined by
their union or intersection), following features from frame to frame; in Rust, `zigzag::CubicalCut` is a cubical
complex for any zigzag (`phlite_complexes::zigzag::ComplexZigzag`).

`phlite.images.filtrations` turns binary images into greyscale ones as giotto-tda's `gtda.images` filtrations do:
height, radial, dilation, erosion, signed distance and density.

Nothing is stored but the image: cells are numbered densely and computed on demand, emergent pairs end most columns
at once (as in Cubical Ripser), cells in zero-persistence pairs are left out, and the two ends (dimension 0 and the
top dimension) are found by union-find, as Cubical Ripser does, so only the dimensions between are reduced.
[Book chapter.](https://phlite.tomchaplin.xyz/book/python/images/cubical.html)

## Compared with other packages

The techniques for images come from [Cubical Ripser](https://github.com/shizuo-kaji/CubicalRipser_3dim) (Kaji, Sudo
and Ahara), and the conventions from [GUDHI](https://gudhi.inria.fr); phlite is tested against both.

* **[Cubical Ripser](https://github.com/shizuo-kaji/CubicalRipser_3dim)** (`cripser`), the fastest we know: in 2D
  phlite is level with it on large images (between 1.16× slower and 1.14× faster) and 2–2.3× faster on small ones,
  with 25–45% less memory; **in 3D Cubical Ripser is 1.5–1.7× faster**.
* **[GUDHI](https://gudhi.inria.fr)** (`CubicalComplex`): phlite is 8–15× faster in 2D and 3.5–7× faster in 3D; on
  periodic images (`PeriodicCubicalComplex`), 1.1–1.3× faster, with identical barcodes.

All of the numbers are in [the benchmarks](https://github.com/tomchaplin/phlite/blob/main/python-tests/BENCHMARKS.md).
