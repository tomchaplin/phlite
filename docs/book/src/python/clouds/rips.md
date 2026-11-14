# Vietoris–Rips (`phlite.clouds.rips`)

```python
import numpy as np
from phlite.core import Algorithm
from phlite.clouds import rips

points = np.random.default_rng(0).normal(size=(100, 2))

barcode = rips(points=points, max_dim=1, representatives=True)
barcode.pairings[1]                  # [(birth, death), ...] of the finite H1 bars
barcode.essential[0]                 # births of the infinite H0 bars
barcode.pairing_representatives[1]   # a cycle (a list of edges (a, b)) for each finite H1 bar
```

It takes a distance matrix (the first argument), dense or sparse, or points:

```python
distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
rips(distances)                                   # a dense distance matrix
rips(scipy.sparse.coo_matrix(...))                # a sparse one: missing entries never appear
rips(points=points, metric="chebyshev")           # points, with a metric
rips(points=points, threshold=0.5)                # only the edges up to 0.5, never storing the rest
```

| Argument | | Default |
|---|---|---|
| `distance_matrix` | a square numpy array (float64 or float32), nested list, or scipy sparse matrix or array; `inf`, or a missing sparse entry, means the edge never appears | |
| `max_dim` | the highest homology dimension | `1` |
| `points` | instead of a distance matrix, one point per row | `None` |
| `metric` | for `points`: `"euclidean"`, `"manhattan"` (or `"cityblock"`) or `"chebyshev"` | `"euclidean"` |
| `weights` | a weight per point, for [weighted Rips](weighted.md) (with `dtm(...)`, DTM-Rips) | `None` |
| `threshold` | only build simplices up to this diameter (bars alive then become essential) | `None` |
| `method` | `"cohomology"` or `"homology"` | `"cohomology"` |
| `algorithm` | an [`Algorithm`](../algorithms.md) | `Algorithm.serial()` (heap columns, uncached) |
| `representatives` | compute a cycle for every bar | `False` |

## What it does

It implements the optimisations of Ripser and giotto-ph on `phlite`'s oracles:

- **Simplices as integers.** A simplex's vertices are the digits of a base-\\((n+1)\\) number, so a simplex is one
  `usize`, and its cofacets and facets are computed arithmetically by inserting or removing a digit.
- **Clearing** and **cohomology** by default.
- **Emergent pairs**: cofacets are enumerated in order, and the first with the simplex's own diameter is its pivot,
  so most columns are finished after a step or two.
- **Zero-length apparent pairs are left out of the basis entirely** (they are nearly all of them), and the basis is
  built in parallel for the parallel algorithm.
- **Truncation at the enclosing radius** (beyond it the complex is a cone), and an optional threshold.
- **Sparse distance matrices**, as in Ripser's sparse mode: a cofacet can only add a neighbour of each of the
  simplex's vertices, so only the neighbours of the vertex with fewest are tried, and memory and time grow with the
  number of edges rather than the square of the number of points. Points with a threshold are turned straight into a
  sparse matrix, without storing every distance. In Rust, `rips_barcode` takes any `Distances`: a dense matrix, a
  `SparseDistances`, or a reference to either.

A sparse matrix may give each edge once, either way round, or both ways round (with the same distance; it is an error
otherwise), and its diagonal is ignored: every point enters at 0. (ripser.py reads only a sparse matrix's upper
triangle, ignoring entries below the diagonal, and reads the diagonal as the points' births.)
- **Heap columns, uncached** by default: Rips columns are cheap to recompute, and caching them dominates the memory.
  On point clouds with long bars (noisy spheres, wedges of circles), merged cached columns
  (`Algorithm.serial().with_columns("merged").with_cache(True)`) are up to 1.9× faster at about twice the memory.
  Avoid `'sorted'` columns on such inputs: they can take hours.

See the [notebooks](../notebooks.md) for a circle, a torus and representatives drawn on the points.

The Rust implementation is the `rips` module of the `phlite_clouds` crate (`phlite_clouds/src/rips`, with its
bindings in `src/python/rips.rs`); this was the `phlite_rips` crate and Python package.
