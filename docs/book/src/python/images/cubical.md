# Cubical complexes (`phlite.images.cubical`)

```python
import numpy as np
from phlite.images import cubical

image = np.random.default_rng(0).random((512, 512))
barcode = cubical(image)                                # T-construction, sublevel sets
barcode = cubical(image, construction="V")              # pixels as vertices
barcode = cubical(image, filtration="superlevel")       # bright regions first
barcode.pairings[1]                                     # (birth, death) of the finite H1 bars
```

An image (a numpy array with 1 to 4 axes: an image, a volume, a video of volumes) gives a filtered cubical complex in
one of two ways:

* the **T-construction** (the default, as GUDHI's `top_dimensional_cells` and Cubical Ripser's `computePH_T`): pixels
  are the top-dimensional cells (squares, cubes), and every other cell enters with the smallest value of the top cells
  around it. Diagonally adjacent pixels touch at a corner, so are connected (8-connectivity in 2D).
* the **V-construction** (as GUDHI's `vertices` and Cubical Ripser's `computePH`): pixels are the vertices, and every
  other cell enters with the largest value of its vertices (the lower-star filtration). Only pixels sharing an edge are
  connected (4-connectivity in 2D).

Either way the complex grows through the image's sublevel sets, or with `filtration='superlevel'` through its
superlevel sets, reported in the image's own values: so superlevel bars run downwards, from a larger birth to a smaller
death. NaN pixels are left out of the complex, as are pixels that would enter at infinity (`+inf` for sublevel sets,
`-inf` for superlevel sets): the complex is then that of the remaining pixels, and its bars that would die at infinity
are essential.

Cells (births, deaths and representatives) are given by their coordinates on the doubled grid, as GUDHI does, in the
order of the image's axes (`(row, column)` for a 2D array): a coordinate is even where the cell is a single vertex
position along that axis and odd where it spans an interval, so a cell's dimension is its number of odd coordinates.
In the T-construction pixel `p` is the top cell `2 p + 1`; in the V-construction it is the vertex `2 p`.

## How

As for Vietoris–Rips, nothing is stored but the image (the crate's `phlite_images::cubical`):

* Cells are numbered densely by a `u32`, type by type (a type is the set of axes along which a cell spans an
  interval), by dimension and then type, and within a type in C order. So every face comes before its cofaces even
  when they have the same value (which in images they very often do), and a cell's faces and cofaces (a step either
  way along each axis) come out in order of number, with no sorting. A cell's value is computed from the at most
  `2^D` pixels around it whenever it is needed.
* The coboundary and boundary oracles override `phlite`'s performance hooks: no coface enters before its cell, and
  cofaces come in order, so the first coface at the cell's own value is its oldest (an *emergent pair*, as in Cubical
  Ripser), and similarly for faces.
* The bases leave out every cell in a zero-persistence apparent pair, which in smooth images is most cells, and never
  hold the top dimension in full.
* By cohomology (the default), the two ends are found by union-find, as Cubical Ripser does, and only the dimensions
  in between (dimension 1 of a volume) are reduced: dimension 0 by joining pixels along the edges in filtration order
  (the elder rule), and the top dimension `D - 1` by duality, joining top cells (and the outside of the grid) along the
  cells of dimension `D - 1` in decreasing order. The edges that kill components are left out of dimension 1, as
  clearing would. (Reducing dimension 0 of a large image by cohomology took 3–4 times as long as the rest.)
* The union-finds need every edge (or cell of dimension `D - 1`) in filtration order. Every cell's value is one of its
  pixels', so the pixels are ranked once, and a cell's rank is the min or max of its pixels' ranks: a walk over each
  type of cell in C order (no decoding), which also finds each cell's two neighbours, then a radix sort on the ranks,
  stable so that ties stay in order of number, exactly as the reductions break them.
* Representatives come from the union-finds too: a component's is its birth pixel plus the oldest pixel of the
  component it joins; a void's (the top dimension), the boundary of the top cells that fill it. In the middle
  dimensions they come by the involution, unless the image has missing pixels (only then can there be essential
  classes there, whose representatives need dimension 0 reduced), when every dimension is reduced. By homology, every
  dimension is reduced.
* A cell's value in the matrices is the min (T) or max (V) over the pixels at fixed offsets for its type, precomputed,
  with the image padded with `+inf` for the T-construction so that every cell has the same pattern.

Pipeline defaults (`'images.cubical'`): merged columns, not cached, storing R (by measurement: storing R makes reducing
the middle dimension of a volume up to 3× faster).

Times, in seconds, by `phlite.images.cubical` (sublevel sets, every dimension) and Cubical Ripser (`computePH_T` or
`computePH`), on smoothed noise (noise averaged over 5 pixels along each axis) and uniform noise:

| Image | Construction | phlite (serial) | phlite (parallel) | phlite (with representatives) | Cubical Ripser |
|---|---|---|---|---|---|
| 1024², smoothed | T | 0.39 | 0.38 | 1.06 | 0.27 |
| 1024², smoothed | V | 0.34 | 0.32 | 1.05 | 0.26 |
| 256², noise | T | 0.012 | 0.011 | 0.062 | 0.009 |
| 64³, smoothed | T | 0.28 | 0.17 | 0.58 | 0.14 |
| 64³, smoothed | V | 0.22 | 0.12 | 0.72 | 0.11 |
| 32³, noise | T | 0.054 | 0.051 | 0.23 | 0.026 |

GUDHI takes 3.6 s on the 1024² image and 1.7 s on the 64³ volume (`python-tests/benchmarks/bench_images_cubical.py`).

## Checked against

GUDHI's and Cubical Ripser's cubical complexes (both constructions, 1 to 3 axes), and every cubical complex built by
brute force and reduced by LoPHAT (both constructions, sublevel and superlevel sets, 1 to 4 axes, NaN and infinite
pixels, every algorithm, representatives checked from first principles); and invariants: the Euler characteristic of
every sublevel set, superlevel sets as the negated image's sublevel sets, flipping and permuting the axes, and
increasing maps of the values (`python-tests/tests/test_images_cubical.py`).

Cubical Ripser's V-construction (`computePH`) treats a 2D image with an axis of length 1 as disconnected (`[[0, 0]]`
has two essential classes in dimension 0), so the tests drop such axes before giving it the image.
