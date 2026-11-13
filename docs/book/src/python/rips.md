# phlite_rips: Vietoris–Rips

```python
import numpy as np
from phlite_rips import rips, Algorithm

points = np.random.default_rng(0).normal(size=(100, 2))
distances = np.linalg.norm(points[:, None] - points[None], axis=-1)

barcode = rips(distances, max_dim=1, representatives=True)
barcode.pairings[1]                  # [(birth, death), ...] of the finite H1 bars
barcode.essential[0]                 # births of the infinite H0 bars
barcode.pairing_representatives[1]   # a cycle (a list of edges (a, b)) for each finite H1 bar
```

| Argument | | Default |
|---|---|---|
| `distance_matrix` | a square numpy array (float64 or float32) or nested list; `inf` means the edge never appears | |
| `max_dim` | the highest homology dimension | `1` |
| `threshold` | only build simplices up to this diameter (bars alive then become essential) | `None` |
| `method` | `"cohomology"` or `"homology"` | `"cohomology"` |
| `algorithm` | an [`Algorithm`](algorithms.md) | `Algorithm.serial()` (heap columns, uncached) |
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
- **Heap columns, uncached** by default: Rips columns are cheap to recompute, and caching them dominates the memory.
  On point clouds with long bars (noisy spheres, wedges of circles), merged cached columns
  (`Algorithm.serial().with_columns("merged").with_cache(True)`) are up to 1.9× faster at about twice the memory.
  Avoid `'sorted'` columns on such inputs: they can take hours.

See the [notebooks](notebooks.md) for a circle, a torus and representatives drawn on the points.
