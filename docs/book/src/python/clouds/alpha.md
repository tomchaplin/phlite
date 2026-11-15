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
   points. Repeated points are merged into the first. Points spanning fewer dimensions than they have coordinates
   are triangulated exactly within their own span (as GUDHI does too): collinear points (decided by exact orientation
   tests) as a path in order along their line, and coplanar points in space by adding an apex off their plane and
   triangulating in space: the sphere through the apex and a triangle meets the plane in that triangle's
   circumcircle, so the triangles opposite the apex are exactly the points' Delaunay triangulation in their plane.
   Coordinates negligible next to the largest (below \\(2^{-90}\\) of it) are taken as zero, since the exact
   predicates need a bounded range of exponents.
2. **Alpha values** (as Edelsbrunner defines them, and GUDHI computes them), from the top dimension down: a simplex
   appears at the squared radius of its smallest circumsphere if no vertex of the triangulation is strictly inside
   that sphere (it is *Gabriel*: checking the vertices opposite it in its cofacets is enough), and otherwise when its
   first cofacet does. The in-sphere test is exact: floating point when a bound on its rounding error allows, then
   double-double (about 32 digits), then big integers. Radii are accurate to an ulp or two, the same way (very thin
   simplices, whose floating point formulas cancel badly, get the more precise arithmetic), and a simplex whose
   radius exactly equals its cofacet's (a right angle) gets exactly the same value, so the bar between them has
   length zero, as it should.
3. **Reduction** of the filtered complex as an explicit complex, by `phlite.complexes`' machinery: cells numbered by
   dimension (so a face enters before a coface with the same value), with its defaults (pivot columns, storing R).

How close this comes to GUDHI's exact computation is set out [below](#precision-compared-with-gudhi).

## Precision, compared with GUDHI

Triangulations and in-sphere tests are exact, as in GUDHI. What remains:

- **Values to an ulp or two.** phlite's radii are within an ulp or two of the exact values, not correctly rounded, so
  a bar shorter than that (exactly) may come out with length zero or a few ulps. GUDHI's `precision="exact"` is no
  closer: it computes exactly but converts to a double only to within an ulp. Against exact rational arithmetic, on
  400 random simplices, its values were the nearest double for 198, one ulp below for 103 and one ulp above for 99.
  So no implementation matches GUDHI bit for bit, and comparisons with it (as in the tests) allow an ulp or two.
  Correctly rounding every value (double-double arithmetic, with an exact fallback near ties, for every radius,
  perhaps 10–20% slower) would make phlite's values exactly invariant under relabelling, quarter turns and scaling
  by powers of two, with a bar of zero length exactly when its two exact values round to the same double. It isn't
  done (yet).
- **Coordinates below \\(2^{-90}\\) of the largest are taken as zero.** The exact predicates need every non-zero
  coordinate within a bounded range of exponents (axiolid accepts magnitudes from \\(2^{-100}\\) to
  \\(2^{100}\\)), and no scaling or translation can fit points whose differences span more than that. GUDHI
  accepts any doubles. Raising an error for such inputs instead would keep results exact or refused.
- **Cospherical points** (any grid) have several Delaunay triangulations, and phlite and GUDHI may pick different
  ones. The barcode is the same; only the representatives may use different simplices.

## Checked against

[GUDHI](https://gudhi.inria.fr)'s `AlphaComplex` (with exact arithmetic) by the fuzz tests in
`python-tests/tests/test_clouds_alpha.py`: random points, and points with small integer coordinates (repeated,
collinear, coplanar and cospherical), in the plane and in space, with every algorithm, and representatives checked
from first principles; values in the plane also against [alphalpha](https://crates.io/crates/alphalpha) (in Rust).

The Rust implementation is the `alpha` module of the `phlite_clouds` crate (`phlite_clouds/src/alpha`, with its
bindings in `src/python/alpha.rs`). axiolid-tetrahedralize is MPL-2.0 licensed and needs Rust 1.99, so
`phlite_clouds` does too.
