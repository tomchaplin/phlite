# phlite.networks: networks

Persistent homology of networks (weighted digraphs): the `phlite-networks` package, built from the
`phlite_networks` crate (the Rust implementation, and its bindings in `src/python`).

| Function | Filtration | Chapter |
|---|---|---|
| `phlite.networks.grpph` | grounded persistent path homology | [Grounded path homology](networks/grpph.md) |

Every function returns a `phlite.core.Barcode`, and takes a [`phlite.core.Algorithm`](algorithms.md) whose unset
options it fills in with its own defaults (the pipeline names are `networks.<function>`, e.g. `networks.grpph`).
