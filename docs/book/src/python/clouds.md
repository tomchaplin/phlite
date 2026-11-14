# phlite.clouds: point clouds

Persistent homology of point clouds and distance matrices: the `phlite-clouds` package, built from the
`phlite_clouds` crate (the Rust implementation, and its bindings in `src/python`).

| Function | Filtration | Chapter |
|---|---|---|
| `phlite.clouds.rips` | Vietoris–Rips of a distance matrix | [Vietoris–Rips](clouds/rips.md) |

Every function returns a `phlite.core.Barcode`, and takes a [`phlite.core.Algorithm`](algorithms.md) whose unset
options it fills in with its own defaults (the pipeline names are `clouds.<function>`, e.g. `clouds.rips`).
