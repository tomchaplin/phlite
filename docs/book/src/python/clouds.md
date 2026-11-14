# phlite.clouds: point clouds

Persistent homology of point clouds and distance matrices: the `phlite-clouds` package, built from the
`phlite_clouds` crate (the Rust implementation, and its bindings in `src/python`).

| Function | Filtration | Chapter |
|---|---|---|
| `phlite.clouds.rips` | Vietoris–Rips of a distance matrix (dense or sparse) or points | [Vietoris–Rips](clouds/rips.md) |
| `phlite.clouds.alpha` | alpha complexes of points in the plane or in space (and `alpha_complex`, the filtered complex) | [Alpha complexes](clouds/alpha.md) |
| `phlite.clouds.delaunay_cech` | Delaunay–Čech complexes (and `delaunay_cech_complex`) | [Weighted and Čech-like](clouds/weighted.md) |

Every function returns a `phlite.core.Barcode`, and takes a [`phlite.core.Algorithm`](algorithms.md) whose unset
options it fills in with its own defaults (the pipeline names are `clouds.<function>`, e.g. `clouds.rips`).
