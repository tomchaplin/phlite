# Weighted and Čech-like filtrations

## Weighted Rips and DTM-Rips (`phlite.clouds.rips` with `weights`)

```python
from phlite.clouds import rips, dtm

rips(points=X, weights=w)                       # weighted Rips: point i appears at 2 w[i]
rips(points=X, weights=dtm(points=X, k=10))     # DTM-Rips: weighted by the distance to measure
```

Weighted Rips (the weighted Rips filtration of Anai et al.'s *DTM-based filtrations*, with \\(p = 1\\), and doubled to
match Rips, as GUDHI's `WeightedRipsComplex`) gives each point \\(i\\) a weight \\(w_i\\): the point appears at
\\(2 w_i\\), and the edge between \\(i\\) and \\(j\\) at \\(\\max(2 w_i, 2 w_j, d(i, j) + w_i + w_j)\\); a
simplex appears with its last edge, as in Rips. Every input `rips` takes works with weights: dense or sparse distance
matrices, or points (with a `threshold`, built sparse, keeping the edges up to it, since a weighted edge is never
shorter than its distance), and every algorithm.

With the *distance to measure* as weights, it is **DTM-Rips**: `dtm(distance_matrix, k=..., q=2)` (or
`dtm(points=..., k=...)`) is, for each point, \\(\\left(\\frac{1}{k}\\sum d^q\\right)^{1/q}\\) over its \\(k\\) nearest
points (itself included): small in dense regions and large for outliers, which then appear late and barely matter, so
the barcode is robust to them.

In Rust, `rips::Weighted::new(distances, &weights)` wraps any `Distances` (and `rips::distance_to_measure` computes the
weights). The Rips oracles take each point's birth into account wherever they compute a simplex's value; for
unweighted distances (`Distances::HAS_BIRTHS` false) that costs nothing.

Checked against GUDHI's `WeightedRipsComplex`, value for value (`python-tests/tests/test_clouds_weighted_rips.py`,
with every method and algorithm, thresholds, sparse input and representatives), and `dtm` against its
`DistanceToMeasure` (to rounding: the two sum the \\(k\\) distances in different orders) and `DTMRipsComplex`.
Weighted alpha complexes (GUDHI's `AlphaComplex(weights=...)`) need a weighted (regular) Delaunay triangulation, which
neither triangulator here provides, so are not (yet) here.

## Delaunay–Čech (`phlite.clouds.delaunay_cech`)

```python
from phlite.clouds import delaunay_cech, delaunay_cech_complex

barcode = delaunay_cech(points)          # points in the plane or in space
delaunay_cech(points, squared=True)      # squared radii, as GUDHI gives
delaunay_cech_complex(points)            # every simplex with the radius at which it appears
```

The Delaunay–Čech filtration (Bauer and Edelsbrunner) puts each simplex of the Delaunay triangulation at the radius of
its *minimal enclosing ball*: the radius at which the balls about its vertices first all meet, as in the Čech
filtration. It has the Čech filtration's barcode, like the alpha filtration, with as few simplices as the alpha complex,
but its simplices enter at their Čech times rather than when their Voronoi-restricted balls meet. It takes the same
arguments as [`alpha`](alpha.md), and shares its triangulation (exact) and its reduction; its pipeline is
`clouds.delaunay_cech`, with the same defaults.

For at most four points, the minimal enclosing ball is the smallest circumsphere of some subset of them which encloses
them all: of the subsets whose spheres enclose every vertex (decided exactly, as alpha's in-sphere tests are), the
smallest. Every value is at most the simplex's alpha value (an enclosing ball, but not necessarily an empty one), and an
edge's is half its length.

Checked against GUDHI's `DelaunayCechComplex` (with exact arithmetic) by `python-tests/tests/test_clouds_delaunay_cech.py`,
on the alpha tests' point clouds, with every algorithm, and representatives checked from first principles. The
[precision note](alpha.md#precision-compared-with-gudhi) for alpha complexes applies here too.
