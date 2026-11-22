# Zigzag persistence

A *zigzag* is a sequence of complexes in which each one is included in the next or contains it:
`K_0 -> K_1 <- K_2 -> ...`. Sliding a window along a time series, growing and pruning a complex, or comparing
consecutive samples all give zigzags. Their homology still decomposes into intervals (Carlsson and de Silva), and
`phlite.complexes` computes them over Z/2, by **FastZigzag** (Dey and Hou, *Fast computation of zigzag persistence*,
ESA 2022) on top of phlite's ordinary reductions.

## In Python

Build a zigzag by chaining operations, or from a sequence of complexes:

```python
from phlite.complexes import Zigzag

barcode = (
    Zigzag()
    .insert((0,), (1,), (2,), (0, 1), (1, 2), (0, 2))   # positions 1-6: the boundary of a triangle
    .delete((0, 2))                                     # position 7
    .barcode()
)
print(barcode)               # ZigzagBarcode(10 intervals over 8 positions; H0: 9, H1: 1)
barcode.in_dimension(1)      # [ZigzagBar(dimension=1, birth=6, death=6, birth_closed=True, death_closed=True)]
barcode.betti_curve(0)       # array([0, 1, 2, 3, 2, 1, 1, 1])

# Insert a simplex with its missing faces; delete one with its cofaces
Zigzag().insert((0, 1, 2), with_faces=True).delete((0,), with_cofaces=True)

# The zigzag K_0 -> K_0 ∪ K_1 <- K_1 -> ... of a sequence of complexes (e.g. windows of a time series)
zz = Zigzag.from_complexes([window_0, window_1, window_2])
zz.barcode(levels=True)      # intervals over levels: K_i at 2i, K_i ∪ K_{i+1} at 2i + 1
```

For a point cloud, `Zigzag.rips_windows(points, windows, threshold)` is the zigzag of the Rips complexes (at the scale
`threshold`) of windows of its points: give the windows as lists of point indices, or an int `w` for windows of `w`
consecutive points sliding by one, as along a time series:

```python
barcode = Zigzag.rips_windows(points, 30, threshold=0.4).barcode(levels=True)   # window i at level 2i
barcode.betti_curve(1)[::2]      # the first Betti number of each window's Rips complex
```

`phlite.complexes.zigzag(operations)` takes the operations directly, as `(simplex, insert)` pairs; or, for cells of
any kind (cubes, cells of a CW complex, ...), as `(cell, insert, faces)` triples, with any hashable cells and each
cell's faces listed when it is inserted (over Z/2 only the faces matter, not orientations):

```python
zigzag([("a", True, []), ("b", True, []), ("ab", True, ["a", "b"]), ("ab", False)])
```

**Positions.** A cell-wise zigzag `K_0 <-> K_1 <-> ... <-> K_m` starts from the empty complex `K_0`, and `K_{i+1}` is
the result of operation `i`. An interval `ZigzagBar(dimension, birth, death, birth_closed, death_closed)` is homology
alive in `K_birth, ..., K_death`; its birth is closed if made by an insertion (open if by a deletion), its death closed
if followed by a deletion (or the end) and open if by an insertion. Intervals are named tuples, so they compare and
unpack as tuples. A `ZigzagBarcode` has `in_dimension`, `diagram(dim)`, `rank(position, dim)`, `betti_curve(dim)` and
`to_dionysus()` (Dionysus's convention: `(birth, death + 1)`, with `inf` for intervals alive at the end).

## How

FastZigzag turns a zigzag of `m` operations into an ordinary filtration of about `2m` cells, whose barcode maps back to
the zigzag barcode interval by interval (`phlite_complexes::zigzag`):

1. Each insertion makes a new cell (a copy, if a simplex is inserted again after being deleted).
2. The *up-down* filtration: all the insertions in order, then all the deletions in order. It has the same barcode,
   up to a reindexing of the ends (Dey and Hou, Proposition 15).
3. Its *extended* filtration, by coning the deletions off: a cone point first, the insertions, then the cones on the
   deleted cells, last deleted first. This is an ordinary filtration, reduced like any other (cohomology, clearing,
   union-find in dimension 0, in parallel if you like), and its bars are the up-down filtration's, ordinary, relative
   or extended (Proposition 19).

So zigzag persistence gets every optimisation of ordinary persistence in phlite; the cost is one barcode of a complex
twice the zigzag's length. Zigzags that don't end empty are emptied (cofaces first), and intervals reaching those
deletions are cut at the last position.

In Rust, `phlite_complexes::zigzag::zigzag_barcode` takes `Operation`s on cells of any hashable type, with their faces,
and an `Algorithm`.

## Checked against

* **Dionysus 2** (`zigzag_homology_persistence`), on random zigzags with simplices inserted again after deletion:
  identical barcodes.
* **Linear algebra**: at every position the intervals containing it number the Betti numbers of `K_i`, and over every
  run of insertions (or deletions) the intervals spanning it number the rank of the map; with insertions only, the
  barcode is ordinary persistence's. For `from_complexes`, the same at every level and over every arrow.

`python-tests/tests/test_complexes_zigzag.py`.

## Performance

Sliding windows of Rips complexes (points on a noisy circle arriving one at a time, each new point bringing its
simplices with the window's points and the oldest point leaving with its simplices), against Dionysus 2 (its
`zigzag_homology_persistence`, phlite's `zigzag` from the operations, best of three):

| zigzag | operations | intervals | phlite | Dionysus 2 | agree |
|---|---:|---:|---:|---:|:-:|
| sliding Rips, 300 points, window 20 | 6611 | 3350 | 3.5 ms | 60.1 ms | ✓ |
| sliding Rips, 1000 points, window 30 | 126182 | 63316 | 144.9 ms | 12.18 s | ✓ |
| sliding Rips, 5000 points, window 40 | 1674242 | 839034 | 4.94 s | 1900.53 s | ✓ |

phlite's time grows about linearly with the number of operations, Dionysus's faster than that: 17× faster on the
smallest zigzag, 84× at 126 thousand operations and 385× at 1.7 million (4.9 s against 32 minutes).
