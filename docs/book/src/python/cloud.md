# phlite.cloud: point clouds

Persistent homology of point clouds and distance matrices, implemented in the `phlite_cloud` crate.

| Function | Filtration | Chapter |
|---|---|---|
| `phlite.cloud.rips` | Vietoris–Rips of a distance matrix | [Vietoris–Rips](cloud/rips.md) |

Every function returns a `phlite.ValueBarcode`, and takes an [`Algorithm`](algorithms.md) whose unset options it fills
in with its own defaults (the pipeline names are `cloud.<function>`, e.g. `cloud.rips`).
