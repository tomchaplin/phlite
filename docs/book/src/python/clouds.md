# phlite.clouds: point clouds

Persistent homology of point clouds and distance matrices: the `phlite-clouds` package, built from the
`phlite_clouds` crate (the Rust implementation, and its bindings in `src/python`).

| Function | Filtration | Chapter |
|---|---|---|
| `phlite.clouds.rips` | Vietoris–Rips of a distance matrix (dense or sparse) or points; weighted with `weights` | [Vietoris–Rips](clouds/rips.md), [weighted](clouds/weighted.md) |
| `phlite.clouds.dtm` | the distance to measure, the weights of DTM-Rips | [Weighted and Čech-like](clouds/weighted.md) |
| `phlite.clouds.alpha` | alpha complexes of points in the plane or in space (and `alpha_complex`, the filtered complex) | [Alpha complexes](clouds/alpha.md) |
| `phlite.clouds.delaunay_cech` | Delaunay–Čech complexes (and `delaunay_cech_complex`) | [Weighted and Čech-like](clouds/weighted.md) |

Every function returns a `phlite.core.Barcode`, and takes a [`phlite.core.Algorithm`](algorithms.md) whose unset
options it fills in with its own defaults (the pipeline names are `clouds.<function>`, e.g. `clouds.rips`).

## The complexes themselves

`alpha_complex`, `delaunay_cech_complex` and `rips_complex` list the complexes as `(simplex, value)` pairs, for
drawing or your own computations: `phlite.complexes.ChainComplex.from_simplices(simplices, modulus=0)` with the values
gives their persistence over the integers (see [explicit complexes](complexes.md#over-the-integers)). Unlike `rips`,
which stops at the enclosing radius, `rips_complex` lists every simplex up to its `threshold`.
