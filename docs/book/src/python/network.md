# phlite.network: networks

Persistent homology of networks (weighted digraphs), implemented in the `phlite_network` crate.

| Function | Filtration | Chapter |
|---|---|---|
| `phlite.network.grpph` | grounded persistent path homology | [Grounded path homology](network/grpph.md) |

Every function returns a `phlite.ValueBarcode`, and takes an [`Algorithm`](algorithms.md) whose unset options it fills
in with its own defaults (the pipeline names are `network.<function>`, e.g. `network.grpph`).
