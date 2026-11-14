# Weighted and Čech-like filtrations

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
