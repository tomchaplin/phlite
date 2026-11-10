# phlite_rips

This crate is a implementation of the Vietoris-Rips filtration in `phlite`.
This is mainly meant as a proof-of-concept for the main library and as a familiar example for those new to `phlite`.

The main output of the crate is a CLI which accepts a distance matrix on STDIN in CSV format and then computes VR persistent homology.
The typical usage looks like
```
cat my_distance_matrix.csv | cargo run --release 3
```
if you want to compute PH for the distance matrix in `my_distance_matrix.csv` up to maximum dimension 3.
If you do not specify max dimension, it is assumed to be 1.

If you want to quickly test this out for say 100 points on a circle (using intrinsic distance) you can try
```
python circle_distance_matrix.py | cargo run --release
```

## Python bindings

The crate also builds a Python module, `phlite_rips` (with the `python` feature, via [maturin](https://www.maturin.rs/)):

```python
import numpy as np
from phlite_rips import rips

points = np.random.default_rng(0).normal(size=(50, 2))
distances = np.linalg.norm(points[:, None] - points[None], axis=-1)

barcode = rips(distances, max_dim=1, representatives=True)
barcode.pairings[1]                 # (birth, death) of the finite H_1 bars
barcode.pairing_representatives[1]  # a cycle (list of edges (a, b)) for each of them
```

`rips` takes the same keyword arguments (`method`, `algorithm`, `representatives` and `threads`) and returns the same `Barcode` type as [`phlite_grpph`](../phlite_grpph/README.md#usage).
Representatives are lists of simplices, each a tuple of point indices in increasing order.
An infinite distance means that edge never appears. See `examples/circle.py` for a full example.

Pass `threshold=t` to only build the Rips complex up to diameter `t` (bars still alive at `t` are then essential).
Simplices above the enclosing radius `min_i max_j d(i, j)` are always skipped, since beyond it the complex is a cone and nothing changes, so this never affects the result.

To build it you need [`just`](https://github.com/casey/just) and [`uv`](https://github.com/astral-sh/uv): run `just develop` and then `uv run --no-sync python ...`, or `just` to see all recipes.

## Optimisations

`phlite_rips` implements the main optimisations of [Ripser](https://github.com/Ripser/ripser)
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

## `phlite` tutorial

This crate also acts as a tutorial for the `phlite` crate.
There are two entry points for this tutorial:
* `src/lib.rs` - To see how to start implementing the core `phlite` traits.
* `src/main.rs` - To see how to use such an implementation to actually compute persistent homology.
