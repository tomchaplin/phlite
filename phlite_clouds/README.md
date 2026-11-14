# phlite_clouds

Persistent homology of point clouds and distance matrices with `phlite`. So far one filtration:

* `phlite_clouds::alpha`: filtrations of the (exact) Delaunay triangulation of points in the plane or in space, as
  GUDHI computes them: alpha complexes, and Delaunay–Čech complexes (each simplex at the radius of its minimal
  enclosing ball), with `delaunay_barcode` their persistence. In Python, `phlite.clouds.alpha(points)` and
  `delaunay_cech(points)` (and `alpha_complex` and `delaunay_cech_complex` for the filtered complexes themselves).
* `phlite_clouds::rips`: the Vietoris-Rips filtration, à la Ripser, of dense or sparse distance matrices or points,
  optionally weighted (`rips::Weighted`, as GUDHI's weighted Rips, and with `rips::distance_to_measure` DTM-Rips).
  It is also a familiar example for those new to `phlite`, and a tutorial (see below).

With the feature `python`, this crate is also the Python package `phlite-clouds`, `phlite.clouds` (its bindings live
in `src/python`, one file per pipeline). It was called `phlite_rips` (crate and Python package).

## Vietoris-Rips

The crate has a CLI, `phlite_clouds_rips`, which accepts a distance matrix on STDIN in CSV format and then computes VR
persistent homology. The typical usage looks like
```
cat my_distance_matrix.csv | cargo run --release --bin phlite_clouds_rips 3
```
if you want to compute PH for the distance matrix in `my_distance_matrix.csv` up to maximum dimension 3.
If you do not specify max dimension, it is assumed to be 1.

If you want to quickly test this out for say 100 points on a circle (using intrinsic distance) you can try
```
python circle_distance_matrix.py | cargo run --release --bin phlite_clouds_rips
```

### In Python

```python
import numpy as np
from phlite.clouds import rips

points = np.random.default_rng(0).normal(size=(50, 2))
distances = np.linalg.norm(points[:, None] - points[None], axis=-1)

barcode = rips(distances, max_dim=1, representatives=True)
barcode.pairings[1]                 # (birth, death) of the finite H_1 bars
barcode.pairing_representatives[1]  # a cycle (list of edges (a, b)) for each of them
```

`rips` takes the same keyword arguments (`method`, `algorithm` and `representatives`) and returns the same
`phlite.core.Barcode` as [`phlite.networks.grpph`](../phlite_networks/README.md#in-python).
`algorithm` is a `phlite.core.Algorithm`, built as described [there](../phlite_networks/README.md#choosing-an-algorithm), for
example `rips(distances, algorithm=Algorithm.parallel(threads=4))`.
By default (the pipeline `clouds.rips`) columns are represented with a heap and not cached: Rips columns are cheap to recompute, and caching them
would take most of the memory (about 4× the peak, see [the benchmarks](../python-tests/BENCHMARKS.md)).
On point clouds with long bars (noisy samples of spheres or wedges of circles), merged, cached columns
(`Algorithm.serial().with_columns("merged").with_cache(True)`) are up to 1.9× faster, at about twice the memory.
Avoid `'sorted'` columns there: they can take hours where the others take seconds.
Representatives are lists of simplices, each a tuple of point indices in increasing order.
An infinite distance means that edge never appears. See `examples/circle.py` for a full example.

Instead of a dense matrix, `rips` also takes a scipy sparse matrix (a missing entry is an edge that never appears, so
memory and time grow with the number of edges), or points: `rips(points=X)`, with `metric='euclidean'` (the default),
`'manhattan'` or `'chebyshev'`. With a `threshold`, points are turned straight into a sparse matrix of the distances up
to it. In Rust, `rips_barcode` takes any `rips::Distances`: a dense `Vec<Vec<NotNan<f64>>>`, a `rips::SparseDistances`
(from edges, or from points with `SparseDistances::from_points`), or a reference to either.

Pass `threshold=t` to only build the Rips complex up to diameter `t` (bars still alive at `t` are then essential).
Simplices above the enclosing radius `min_i max_j d(i, j)` are always skipped, since beyond it the complex is a cone and nothing changes, so this never affects the result.

To build it, see [python-tests](../python-tests/README.md) (`just build` there builds every phlite package).

### Optimisations

`phlite_clouds::rips` implements the main optimisations of [Ripser](https://github.com/Ripser/ripser)
(Bauer, "Ripser: efficient computation of Vietoris–Rips persistence barcodes", J. Appl. Comput. Topol. 2021)
and [giotto-ph](https://github.com/giotto-ai/giotto-ph)
(Pérez et al., "giotto-ph: A python library for high-performance computation of persistent homology of Vietoris-Rips filtrations", arXiv:2107.05412),
from which they are adapted:
* clearing (Ripser), and a parallel lock-free reduction (giotto-ph, after Morozov and Nigmetov);
* apparent pairs (Ripser): simplices in zero-persistence apparent pairs (nearly all of them) are never even put in the column basis,
  which is built in parallel (giotto-ph);
* emergent pairs (Ripser): a column whose pivot is unclaimed is finished as soon as its pivot is found, and pivots are found by
  enumerating cofacets in order and stopping at the first with the simplex's own diameter;
* truncation at the enclosing radius (Ripser and giotto-ph), and an optional threshold.

It does not (yet) support sparse distance matrices, edge collapse, or union-find for H_0.

### `phlite` tutorial

The `rips` module also acts as a tutorial for the `phlite` crate.
There are two entry points for this tutorial:
* `src/rips/mod.rs` - To see how to start implementing the core `phlite` traits.
* `src/bin/rips.rs` - To see how to use such an implementation to actually compute persistent homology.
