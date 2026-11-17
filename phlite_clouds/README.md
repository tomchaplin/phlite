# phlite_clouds

Persistent homology of point clouds and distance matrices, in Rust and Python (`pip install phlite-clouds`, or
`pip install phlite` for every package). Part of [phlite](https://github.com/tomchaplin/phlite), which brings the
techniques of Ripser and its relatives to any filtration; see [the book](https://phlite.tomchaplin.xyz/book/) and
the [in-browser playground](https://phlite.tomchaplin.xyz/playground/), where you can draw a point cloud and watch
its barcode change.

```python
from phlite.clouds import rips, alpha, dtm

barcode = rips(points=X, max_dim=2)                    # Vietoris–Rips of points
barcode = rips(D, max_dim=1, representatives=True)     # of a distance matrix, with cycles
barcode = rips(points=X, weights=dtm(points=X, k=10))  # DTM-Rips: robust to outliers
barcode = alpha(X)                                     # the alpha complex, in 2D or 3D
barcode.pairings[1]                                    # (birth, death) of the finite H1 bars
```

## Filtrations

* **Vietoris–Rips** (`rips`) of a dense or sparse distance matrix, or of points: Ripser's algorithm (cohomology,
  clearing, apparent and emergent pairs, union-find in dimension 0, truncation at the enclosing radius) with
  giotto-ph's parallel reduction. A sparse matrix that isn't a metric (the edges of a weighted graph, say) is
  listed and reduced as an explicit complex, which is far faster for those. [Book chapter.](https://phlite.tomchaplin.xyz/book/python/clouds/rips.html)
* **Weighted Rips and DTM-Rips** (`rips(..., weights=...)`, `dtm`), as GUDHI's `WeightedRipsComplex` and
  `DistanceToMeasure`. [Book chapter.](https://phlite.tomchaplin.xyz/book/python/clouds/weighted.html)
* **Alpha and Delaunay–Čech complexes** (`alpha`, `delaunay_cech`) of points in the plane or in space, from an exact
  Delaunay triangulation, filtered as GUDHI filters them. [Book chapter.](https://phlite.tomchaplin.xyz/book/python/clouds/alpha.html)

Every function takes `algorithm=` (a `phlite.core.Algorithm`: serial or parallel, and how columns are represented),
`method=` (cohomology or homology) and `representatives=`, and returns a `phlite.core.Barcode`. In Rust,
`rips::rips_barcode` takes any `rips::Distances` (dense, sparse, weighted, or your own), and `alpha::delaunay_barcode`
any Delaunay filtration. A command-line tool, `phlite_clouds_rips`, reads a distance matrix as CSV.

## Compared with other packages

phlite's Vietoris–Rips is Ripser's algorithm, with giotto-ph's parallelism; credit for both goes to their authors
([Ripser](https://github.com/Ripser/ripser), [giotto-ph](https://github.com/giotto-ai/giotto-ph)). On one thread, on
random points in the cube and on noisy samples of shapes with long bars:

* **[ripser.py](https://github.com/scikit-tda/ripser.py)**: phlite is 1.4–7× faster, and uses less memory on
  random points; ripser runs out of memory (over 7 GB) on 1000 points in dimension 2.
* **[giotto-ph](https://github.com/giotto-ai/giotto-ph)**: phlite is 1–2.2× faster (1.5–2.2× on random points in
  dimension 2), and on sparse matrices from 7% slower to 1.3× faster. But on point clouds with long bars phlite's default (heap
  columns, recomputed) uses far more memory: 546 MB against giotto-ph's 7 MB on a noisy sphere of 400 points. Merged
  columns, cached (`Algorithm.serial().with_columns("merged").with_cache(True)`), are faster there and take 25 MB.
* **[GUDHI](https://gudhi.inria.fr)**: alpha and Delaunay–Čech complexes are level with GUDHI's or up to 2.4× faster,
  up to 100,000 points in the plane and 300,000 in space, but **7–8× slower on a million points in the plane** (167 s against 23 s), which we
  have yet to look into. DTM-Rips is 50–400× faster than GUDHI's `WeightedRipsComplex`.

What phlite adds is the rest of the package: sparse and non-metric inputs, weights, representatives by cohomology,
a choice of algorithm, and the same API as phlite's other filtrations. If you only need Vietoris–Rips barcodes of
point clouds, ripser.py and giotto-ph are mature and excellent. All of the numbers, with memory, are in
[the benchmarks](https://github.com/tomchaplin/phlite/blob/main/python-tests/BENCHMARKS.md).

## For Rust developers

`phlite_clouds::rips` is also phlite's most complete example of implementing a filtration with every optimisation:
a lazy coboundary oracle over simplices encoded as integers, apparent pairs with early-stopping searches, and a basis
that leaves out zero-persistence pairs. The [book's tutorial](https://phlite.tomchaplin.xyz/book/tutorial/index.html)
builds up the same ideas step by step.
