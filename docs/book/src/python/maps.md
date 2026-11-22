# Maps: images, kernels, cokernels and six-packs

A map of filtrations `f: L -> K` (a subcomplex included in a complex, a simplicial map, a subsample of a point cloud,
a projection) induces maps `f_*: H(L_t) -> H(K_t)` at every value `t`. Their **image**, **kernel** and **cokernel**
are persistence modules with barcodes of their own (Cohen-Steiner, Edelsbrunner, Harer and Morozov), which say what of
`L`'s homology survives in `K`, what of it dies there, and what of `K`'s homology doesn't come from `L`. With the
barcodes of `L` and `K` themselves and that of the map's **relative** homology (its mapping cone; the pair `(K, L)` for
an inclusion), they make up the map's *six-pack*.

phlite computes six-packs for any map you can describe, with a representative for every bar, and image, kernel and
cokernel barcodes faster than [phimaker](https://github.com/tomchaplin/phimaker), the reference implementation.

## In Python

Every six-pack is a `phlite.core.SixPack`: a mapping from `'domain'`, `'codomain'`, `'image'`, `'kernel'`,
`'cokernel'` and `'relative'` to barcodes, each a sequence of bars (`dimension`, `birth`, `death`, `representative`):

```python
from phlite.complexes import ChainComplex, FilteredComplex, FilteredMap

K = FilteredComplex.from_simplices([((0,), 0), ((1,), 0), ((2,), 0), ((0, 1), 1), ((1, 2), 2), ((0, 2), 3)])

pack = K.six_pack([(0,), (1,), (2,)])     # the inclusion of the three vertices
print(pack)
# SixPack, bars by dimension:
#               H0     H1
#   domain       3      0
#   codomain     3      1
#   image        3      0
#   kernel       2      0
#   cokernel     0      1
#   relative     0      3

pack.kernel.diagram(0)                    # array([[1., inf], [2., inf]]): two classes of L die in K, at 1 and 2
bar = pack.kernel[0]
bar.representative                        # [((1,), 1), ((0,), 1)]: the cycle (0) + (1) of L, a boundary in K from 1
pack.ranks(2.5, dim=0)                    # {'domain': 3, 'codomain': 1, 'image': 1, 'kernel': 2, ...}
for name, barcode in pack.items():
    print(name, barcode.to_gudhi())
```

Each barcode has `diagram(dim)`, `diagrams()`, `in_dimension(dim)`, `essential()`, `finite()`, `rank(t, dim)`,
`representatives(dim)` and `to_gudhi()`. A kernel bar's dimension is that of its classes (a class of `H_k` is born at
the `(k + 1)`-cell that makes it a boundary in `K`).

**Representatives.** Cycles of `L` for the domain, image and kernel (an image bar's cycle maps to the image class; a
kernel bar's cycle maps to a boundary in `K` from the bar's birth), cycles of `K` for the codomain and cokernel, and
relative cycles for the relative barcode: chains of `K` outside `L` with boundary in `L` (for
`FilteredComplex.six_pack`), or cycles of the mapping cone, on cells of `K` and cones `phlite.core.Cone(σ)` on cells of
`L` (for a `FilteredMap`). Representatives are on by default; `representatives=False` is faster (see below).

### Maps of complexes

`FilteredMap(L, K, images)` is a chain map between two filtered complexes over the same field: `images` gives each cell
of `L` a chain of `K` (a dict `{cell: coefficient}`, or a list of cells). `FilteredMap.simplicial(L, K, vertex_map)`
makes the simplicial map of a map of vertices (a simplex goes to the simplex on its vertices' images, with the sign of
the permutation sorting them, or to zero if two vertices meet). Both check that the map commutes with the boundaries,
preserves dimensions and never makes a cell enter earlier.

```python
f = FilteredMap.simplicial(L, K, {0: 0, 1: 1, 2: 2, 3: 2})   # collapse an edge of a square onto a triangle
f.six_pack(max_dim=1)
```

A map that is the inclusion of a subcomplex (each cell to one cell, with the same value) is recognised, and computed
without the mapping cylinder.

### Point clouds

* `phlite.clouds.rips_six_pack(points=..., subset=...)`: the inclusion of the Rips filtration of a subset of the points
  into that of all of them.
* `phlite.clouds.rips_map_six_pack(domain, codomain, vertex_map)`: a map of point clouds that never increases a
  distance (`d(f x, f y) <= d(x, y)`) is a simplicial map of Rips filtrations at every scale: a subsample, a
  projection, a contraction, a map onto a coarser sample. Both run on the lazy Rips boundaries; give a `threshold` for
  more than a few hundred points.

### Composing with other filtrations

`FilteredComplex.from_simplices` takes the `(simplex, value)` pairs that every exporter returns
(`phlite.clouds.rips_complex`, `alpha_complex`, `delaunay_cech_complex`, `cech_complex`, `witness_complex`), so any of
these filtrations has six-packs of its subcomplexes and maps:

```python
from phlite import clouds
K = FilteredComplex.from_simplices(clouds.alpha_complex(points))
inside = [s for s, _ in clouds.alpha_complex(points) if all(points[v][0] < 0.5 for v in s)]
K.six_pack(inside).image
```

`FilteredComplex.image_kernel_cokernel(subcomplex)` (and `clouds.rips_image_kernel`) is the older, leaner interface:
the image, kernel, cokernel and relative barcodes only, as `ComplexBarcode`s, with no representatives.

## How

For the inclusion of a subcomplex `L` in `K` (with the restricted filtration), phlite follows phimaker, which follows
Cohen-Steiner et al., reading the barcodes off a few reductions ([`phlite::image_kernel`](https://docs.rs/phlite/latest/phlite/image_kernel/)):

* `D_f` and `D_g`, the boundaries of `K` and `L`, by phlite's usual clearing (with apparent pairs where the oracle has
  them): which cells are positive or negative, and a cycle for every positive cell, from `V`;
* `D_im`, `D_f` with the rows of `L` first: **only its columns negative in `K`** need reducing, since whether a column
  reduces to zero doesn't depend on the order of the rows;
* `D_ker`, the cycles of `K` with the rows of `L` first: any cycle basis compatible with the filtration has the same
  pivots, so the cycles from the reduction of `K` do, and only the dimensions with a kernel death, up to the last
  death, are reduced;
* `D_cok`, `D_f` with the columns positive in `L` replaced by `L`'s cycles: the columns positive in `K` and not
  positive in `L` reduce to zero (their boundaries lie in the span of earlier columns), so they are left out;
* the quotient `K / L` for the relative barcode, by cohomology with clearing (or by homology, for representatives).

The four reductions after the first two are independent, and run in parallel. Representatives come from the same
reductions: the columns of `R = D V` of each reduced matrix for bars that die, columns of `V` for bars that don't, and
for a kernel bar dying at `j`, `∂` of the part outside `L` of the reduced cycle `R_ker,j`.

For a general map `f: L -> K` the six-pack is that of the inclusion of `L` in the map's algebraic **mapping
cylinder**, a lazy oracle (`phlite::six_pack::MappingCylinder`) with a cell `Domain(σ)` and a cell `Cylinder(σ)` (one
dimension up) for every cell `σ` of `L` and a cell `Codomain(τ)` for every cell of `K`, where
`∂ Cylinder(σ) = Codomain(f σ) - Domain(σ) - Cylinder(∂σ)`. At every value the cylinder deformation retracts onto
`K` (by `Domain(σ) -> f σ`, `Cylinder(σ) -> 0`), so `H(L) -> H(Cyl)` is `f_*`, and `H(Cyl, L)` is the homology of the
mapping cone. Codomain and cokernel representatives are carried back to `K` along the retraction.

### In Rust

A map is anything implementing [`phlite::six_pack::FilteredChainMap`](https://docs.rs/phlite/latest/phlite/six_pack/):
two boundary oracles (any of phlite's, lazy or explicit) over the same field and filtration values, their cells by
dimension, and `apply`, the image of a cell of `L` as a chain of `K`. Its `six_pack(columns, representatives,
max_dim)` goes through the cylinder; `Inclusion` overrides it to reduce the pair directly. The implementations in the
packages are `phlite_complexes::maps::ComplexMap` (explicit complexes) and `phlite_clouds::rips::image::RipsMap`
(lazy Rips complexes).

## Checked against

* **Linear algebra**, over Z/2, at every filtration value of random complexes, random subcomplexes and random
  simplicial maps: the number of bars of each barcode alive there is the dimension of `H(L_t)`, `H(K_t)`, the image,
  kernel and cokernel of `H(L_t) -> H(K_t)`, and the relative homology (for a map, `coker_k + ker_{k-1}`, from the long
  exact sequence of the mapping cone); and the representatives of those bars form a basis there (cycles of the right
  complex, present at `t`, independent modulo the right subspace; kernel cycles bounding in `K`).
* **phimaker**, on random complexes with induced subcomplexes: identical image, kernel and cokernel barcodes. (Its
  relative diagram is of the quotient space `K / L`, not the pair, so isn't compared.)
* The lazy Rips routes against the explicit complexes, and the cylinder against the direct route on inclusions.

`python-tests/tests/test_complexes_six_pack.py` and `test_complexes_image_kernel.py`.

## Performance

Rips complexes of random points in the plane with half of them as the subset, dimensions 0 and 1
(`python-tests/benchmarks/ikc_zigzag_compare.py`, one machine; phimaker's time is its `compute_ensemble`, phlite's the
call on a prebuilt complex):

| complex | simplices | phlite | phlite six-pack + representatives | phlite (lazy Rips, from the points) | phimaker | agree |
|---|---:|---:|---:|---:|---:|:-:|
| Rips, 100 points, r ≤ 0.25 | 2951 | 5.4 ms | 8.4 ms | 11.2 ms | 8.1 ms | ✓ |
| Rips, 200 points, r ≤ 0.18 | 7997 | 16.3 ms | 25.7 ms | 34.2 ms | 19.8 ms | ✓ |
| Rips, 400 points, r ≤ 0.13 | 20702 | 45.8 ms | 67.2 ms | 102.3 ms | 56.4 ms | ✓ |
| Rips, 800 points, r ≤ 0.09 | 37942 | 75.7 ms | 121.6 ms | 184.2 ms | 108.2 ms | ✓ |
| Rips, 1600 points, r ≤ 0.065 | 82093 | 215.7 ms | 298.6 ms | 454.4 ms | 315.1 ms | ✓ |
| torus mesh 30×30, lower star | 5400 | 6.9 ms | 11.0 ms | — | 12.5 ms | ✓ |
| torus mesh 60×60, lower star | 21600 | 30.7 ms | 48.5 ms | — | 53.4 ms | ✓ |

phlite computes the image, kernel and cokernel barcodes 1.2–1.8× faster than phimaker, and the whole six-pack with a
representative for every bar (which phimaker doesn't give) in about the time phimaker takes for its barcodes. The
lazy Rips route's time includes building the Rips complex from the points. A general map, through the mapping
cylinder, costs about twice an inclusion of the same size.
