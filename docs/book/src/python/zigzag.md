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

### Zigzags of point clouds, networks and images

Most zigzags are a sequence of complexes `K_0, K_1, ..., K_s` (windows of a time series, frames of a video, snapshots
of a network), each joined to the next by their **union** `K_0 -> K_0 ∪ K_1 <- K_1 -> ...` (the default: a feature
lasts while it overlaps itself from one complex to the next) or their **intersection** `K_0 <- K_0 ∩ K_1 -> K_1 <- ...`
(`join="intersection"`: while what it shares with the next complex carries it). Each package builds the complexes in
Rust and returns a `ZigzagBarcode` over *levels*: `K_i` at level `2 i`, the join of `K_i` and `K_{i+1}` at `2 i + 1`.

```python
from phlite.clouds import cloud_zigzag, sliding_windows
from phlite.networks import network_zigzag
from phlite.images import video_zigzag

# Point clouds: Rips (any metric, or a distance matrix), Čech, alpha, Delaunay–Čech or witness complexes
bars = cloud_zigzag(sliding_windows(points, 30), threshold=0.4)            # windows of a time series
bars = cloud_zigzag([positions[t] for t in range(T)], 0.2, filtration="alpha", join="intersection")  # moving points
bars = cloud_zigzag([[0, 1, 2], [1, 2, 3]], 0.5, distances=D)              # frames as labels into a distance matrix

# Networks on shared nodes: clique complexes, or directed flag complexes (directed cliques, as flagser)
bars = network_zigzag([G_monday, G_tuesday, G_wednesday])                  # networkx graphs, or edge lists
bars = network_zigzag(digraphs, complex="directed_flag")

# Videos: each frame cut at a threshold, as a cubical complex (T- or V-construction, periodic or not)
bars = video_zigzag(frames, threshold=0.5, join="intersection")

bars.betti_curve(1)[::2]      # the first Betti number of each frame's complex
```

A frame of a point-cloud zigzag is an `(n, d)` array (its points labelled `0, ..., n - 1`, so the same points moving)
or a `(labels, points)` pair, as `sliding_windows` gives: points are matched from frame to frame by label. See
[images](images.md#videos-zigzag-persistence) for videos. For complexes you build yourself,
`Zigzag.from_complexes(complexes, join=...)` does the same in Python, and `Zigzag.rips_windows(points, windows,
threshold)` is the Rips case.

**Path homology** has no zigzag: its chains, the ∂-invariant paths `Ω(G)`, have no basis of cells, and `Ω` doesn't
respect unions (`Ω(G ∪ H)` can be bigger than `Ω(G) + Ω(H)`), so FastZigzag's reduction (which rests on
Mayer–Vietoris diamonds of subcomplexes) doesn't apply. The directed flag complex is its nearest cellular relative.

### Cell by cell

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
and an `Algorithm`. Above it, the trait `ZigzagComplex` is what can be a term of a zigzag: a complex that lists its
cells (each with a key naming it in every complex, and its dimension) and each cell's faces. `ComplexZigzag` takes
complexes one at a time (`push`), joined by `Join::Union` or `Join::Intersection`, works out the operations (inserting
faces first, deleting cofaces first) and gives the barcode in levels (`LevelBar`). Implementations:

| complex | crate | key |
|---|---|---|
| `SimplicialComplex<V>`, `BTreeSet<Vec<V>>` (sorted simplices, or ordered ones for directed flag complexes) | `phlite_complexes::zigzag` | the vertices |
| `CubicalCut` (an image or volume cut at a threshold, T or V, periodic or not) | `phlite_images::zigzag` | position on the doubled grid |
| point clouds: any `PointFiltration` (`Rips`, `RipsDistances`, `Cech`, `Alpha`, `DelaunayCech`, `WitnessFiltration`, or a closure) cut at a threshold, by `cloud_zigzag` | `phlite_clouds::zigzag` | the points' labels |
| networks: `network_complex` (clique or directed flag) of each `Snapshot`, by `network_zigzag` | `phlite_networks::zigzag` | the nodes' labels |

A new kind of complex needs only `cells` and `faces`:

```rust
use phlite_complexes::zigzag::{ComplexZigzag, Join, ZigzagComplex};

impl ZigzagComplex for MyComplex {
    type Key = MyCell;                                  // Hash + Ord + Clone, the same cell in every complex
    fn cells(&self) -> Vec<(MyCell, usize)> { ... }     // every cell and its dimension
    fn faces(&self, cell: &MyCell) -> Vec<MyCell> { ... }
}

let bars = ComplexZigzag::of(Join::Union, &complexes).barcode(Algorithm::default())?;
```

## Checked against

* **Dey and Hou's FastZigzag** (`fzz`, the C++ reference implementation) and **Dionysus 2**
  (`zigzag_homology_persistence`), on random zigzags with simplices inserted again after deletion: identical barcodes
  (thousands of cases each).
* **Linear algebra**: at every position the intervals containing it number the Betti numbers of `K_i`, and over every
  run of insertions (or deletions) the intervals spanning it number the rank of the map; with insertions only, the
  barcode is ordinary persistence's. For `from_complexes`, the same at every level and over every arrow, joined by
  unions and by intersections (and in Rust, `ComplexZigzag` on random sequences of complexes).
* **The Rust builders against the Python one**: `cloud_zigzag` (Rips, Čech, alpha, Delaunay–Čech; unions and
  intersections) and `network_zigzag` (clique complexes) give the barcodes of `Zigzag.from_complexes` on the same
  complexes listed by `rips_complex`, `cech_complex`, `alpha_complex`, ...; `video_zigzag`'s frames have their own
  cubical complexes' Betti numbers.

`python-tests/tests/test_complexes_zigzag.py`, `python-tests/tests/test_clouds_networks_zigzag.py`,
`python-tests/tests/test_images_video_zigzag.py`.

## Performance

Sliding windows of Rips complexes (points on a noisy circle arriving one at a time, each new point bringing its
simplices with the window's points and the oldest point leaving with its simplices), against Dey and Hou's own
FastZigzag (`fzz`, C++ on PHAT, timing its computation only, not reading the file) and Dionysus 2
(`zigzag_homology_persistence`); phlite's `zigzag` from the operations, best of three:

Each reduction is run both ways, by cohomology and by homology: phlite's `method=` option, and fzz as released
(homology, PHAT's twist reduction) and patched to reduce the dual (`compute_persistence_pairs_dualized`, cohomology).

| zigzag | operations | intervals | phlite (cohomology) | fzz (cohomology) | phlite (homology) | fzz (homology) | Dionysus 2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sliding Rips, 300 points, window 20 | 6,611 | 3,350 | 4.5 ms | 2.3 ms | 4.6 ms | 1.5 ms | 62 ms |
| sliding Rips, 1000 points, window 30 | 126,182 | 63,316 | 158 ms | 99 ms | 631 ms | 458 ms | 13.3 s |
| sliding Rips, 2000 points, window 35 | 461,837 | 231,291 | 0.88 s | 0.66 s | 321 s | 324 s | — |
| sliding Rips, 5000 points, window 40 | 1,674,242 | 839,034 | 4.9 s | 3.9 s | > 10 min | > 10 min | 32 min |

Every barcode is identical. What matters is the direction of the reduction, not the implementation: by cohomology
(with clearing) both take about a second for half a million operations, by homology both take five minutes. Reducing
the coned filtration by homology fills in long columns (each cone over a deleted simplex is born late and bounds
everything in its link), which cohomology avoids. Like for like, fzz is 1.25 to 1.6 times faster than phlite: phlite
takes the operations from Python and builds a general explicit complex (about 30 per cent of its time at 126k
operations). Union-find in dimension 0 makes no measurable difference here (158 ms with, 156 ms without:
`union_find=False`). phlite reduces by cohomology by default; fzz as released by homology.
(`perf-results/maps-zigzag-2026-11-22.md`.)
