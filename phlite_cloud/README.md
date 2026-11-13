# phlite_cloud

Persistent homology of point clouds and distance matrices with `phlite`. So far one filtration:

* `phlite_cloud::rips`: the Vietoris-Rips filtration, à la Ripser. It is also a familiar example for those new to
  `phlite`, and a tutorial (see below).

In Python, this crate is the `phlite.cloud` submodule of the `phlite` package (its bindings live in
`phlite_py/src/python/cloud`). It was called `phlite_rips` (crate and Python package) until the packages merged.

## Vietoris-Rips

The crate has a CLI, `phlite_cloud_rips`, which accepts a distance matrix on STDIN in CSV format and then computes VR
persistent homology. The typical usage looks like
```
cat my_distance_matrix.csv | cargo run --release --bin phlite_cloud_rips 3
```
if you want to compute PH for the distance matrix in `my_distance_matrix.csv` up to maximum dimension 3.
If you do not specify max dimension, it is assumed to be 1.

If you want to quickly test this out for say 100 points on a circle (using intrinsic distance) you can try
```
python circle_distance_matrix.py | cargo run --release --bin phlite_cloud_rips
```

### In Python

```python
import numpy as np
from phlite.cloud import rips

points = np.random.default_rng(0).normal(size=(50, 2))
distances = np.linalg.norm(points[:, None] - points[None], axis=-1)

barcode = rips(distances, max_dim=1, representatives=True)
barcode.pairings[1]                 # (birth, death) of the finite H_1 bars
barcode.pairing_representatives[1]  # a cycle (list of edges (a, b)) for each of them
```

`rips` takes the same keyword arguments (`method`, `algorithm` and `representatives`) and returns the same
`phlite.ValueBarcode` as [`phlite.network.grpph`](../phlite_network/README.md#in-python).
`algorithm` is a `phlite.Algorithm`, built as described [there](../phlite_network/README.md#choosing-an-algorithm), for
example `rips(distances, algorithm=Algorithm.parallel(threads=4))`.
By default (the pipeline `cloud.rips`) columns are represented with a heap and not cached: Rips columns are cheap to recompute, and caching them
would take most of the memory (about 4× the peak, see [the benchmarks](../python-tests/BENCHMARKS.md)).
On point clouds with long bars (noisy samples of spheres or wedges of circles), merged, cached columns
(`Algorithm.serial().with_columns("merged").with_cache(True)`) are up to 1.9× faster, at about twice the memory.
Avoid `'sorted'` columns there: they can take hours where the others take seconds.
Representatives are lists of simplices, each a tuple of point indices in increasing order.
An infinite distance means that edge never appears. See `phlite_py/examples/cloud_circle.py` for a full example.

Pass `threshold=t` to only build the Rips complex up to diameter `t` (bars still alive at `t` are then essential).
Simplices above the enclosing radius `min_i max_j d(i, j)` are always skipped, since beyond it the complex is a cone and nothing changes, so this never affects the result.

To build the `phlite` package, see [`phlite_py`](../phlite_py/README.md).

### Optimisations

`phlite_cloud::rips` implements the main optimisations of [Ripser](https://github.com/Ripser/ripser)
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
