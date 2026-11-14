# Alpha complexes (`phlite.clouds.alpha`)

```python
import numpy as np
from phlite.clouds import alpha, alpha_complex

points = np.random.default_rng(0).uniform(size=(1000, 3))   # in the plane (2 columns) or in space (3)

barcode = alpha(points, representatives=True)
barcode.pairings[1]                  # [(birth, death), ...] of the finite H1 bars, as radii
barcode.pairing_representatives[1]   # a cycle (a list of edges (a, b)) for each

alpha(points, squared=True)          # squared radii, as GUDHI gives
alpha_complex(points)                # every simplex with the radius at which it appears, in filtration order
```

| Argument | | Default |
|---|---|---|
| `points` | one point per row, 2 or 3 finite coordinates each | |
| `max_dim` | the highest homology dimension | one less than the number of coordinates |
| `squared` | squared radii (GUDHI's convention) rather than radii | `False` |
| `method` | `"cohomology"` or `"homology"` | `"cohomology"` |
| `algorithm` | an [`Algorithm`](../algorithms.md) | `Algorithm.serial()` (pivot columns, storing R) |
| `representatives` | compute a cycle for every bar | `False` |

## What it computes

The alpha complex at radius \\(r\\) is the nerve of the balls of radius \\(r\\) about the points, each cut down to its
point's Voronoi cell. It has the homotopy type of the union of the balls, so the same barcode as the Čech filtration,
but it lives inside the Delaunay triangulation: in space about 6.5 tetrahedra per point, however large the radius,
where a Čech or Rips complex grows without bound. So it handles hundreds of thousands of points where Rips is limited
to a few thousand.

## How

1. **Delaunay triangulation**, exactly. In the plane by [spade](https://github.com/Stoeoef/spade), and in space by
   [axiolid-tetrahedralize](https://axiolid.github.io/kernel/), whose orientation and in-sphere tests are exact, with
   ties (cospherical points, as in any grid) broken by a symbolic perturbation that doesn't depend on the order of the
   points. Repeated points are merged into the first; points spanning fewer dimensions than they have coordinates
   (collinear points, or coplanar points in space) are triangulated within their own span.
2. **Alpha values** (as Edelsbrunner defines them, and GUDHI computes them), from the top dimension down: a simplex
   appears at the squared radius of its smallest circumsphere if no vertex of the triangulation is strictly inside
   that sphere (it is *Gabriel*: checking the vertices opposite it in its cofacets is enough), and otherwise when its
   first cofacet does. The in-sphere test is exact: floating point when a bound on its rounding error allows, big
   integers otherwise. Radii are accurate to a few units in the last place, computed exactly for thin simplices whose
   floating point formula would cancel badly, and a simplex whose radius exactly equals its cofacet's (a right angle)
   gets exactly the same value, so the bar between them has length zero, as it should.
3. **Reduction** of the filtered complex as an explicit complex, by `phlite.complexes`' machinery: cells numbered by
   dimension (so a face enters before a coface with the same value), with its defaults (pivot columns, storing R).

The only differences from GUDHI's exact computation are bars shorter than the rounding of the values (a few units in
the last place), which may round to length zero or not, and, for cospherical points, which of the equally valid
Delaunay triangulations is used (the same barcode, but the representatives may use different simplices).

## Checked against

[GUDHI](https://gudhi.inria.fr)'s `AlphaComplex` (with exact arithmetic) by the fuzz tests in
`python-tests/tests/test_clouds_alpha.py`: random points, and points with small integer coordinates (repeated,
collinear, coplanar and cospherical), in the plane and in space, with every algorithm, and representatives checked
from first principles; values in the plane also against [alphalpha](https://crates.io/crates/alphalpha) (in Rust).

The Rust implementation is the `alpha` module of the `phlite_clouds` crate (`phlite_clouds/src/alpha`, with its
bindings in `src/python/alpha.rs`). axiolid-tetrahedralize is MPL-2.0 licensed and needs Rust 1.99, so
`phlite_clouds` does too.
