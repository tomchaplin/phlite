# phlite.networks: networks

Persistent homology of networks, directed and undirected: the `phlite-networks` package, built from the
`phlite_networks` crate (the Rust implementation, and its bindings in `src/python`).

| Function | Filtration | Chapter |
|---|---|---|
| `phlite.networks.grpph` | grounded persistent path homology | [Grounded path homology](networks/grpph.md) |
| `phlite.networks.pph` | persistent path homology (Chowdhury and Mémoli), any dimension | [Path homology](networks/pph.md) |
| `phlite.networks.directed_flag` | directed flag complexes (as flagser), grounded or not | [Directed flag complexes](networks/directed_flag.md) |
| `phlite.networks.dowker` | Dowker sink and source complexes | [Dowker complexes](networks/dowker.md) |
| `phlite.networks.ordered_tuples` | ordered-tuple complexes (Turner's directed Rips) | [Directed Rips](networks/ordered_tuples.md) |
| `phlite.networks.flag` | flag complexes of **undirected** graphs, by edge weights, node values (lower-star) or shortest paths | [Flag complexes of graphs](networks/flag.md) |
| `phlite.networks.magnitude`, `magnitude_path_spectral_sequence` | magnitude homology (Hepworth and Willerton) and the magnitude-path spectral sequence | [Magnitude homology](networks/magnitude.md) |

Every function returns a `phlite.core.Barcode`, and takes a [`phlite.core.Algorithm`](algorithms.md) whose unset
options it fills in with its own defaults (the pipeline names are `networks.<function>`, e.g. `networks.grpph`).

Every function for digraphs takes one the same way: a networkx `DiGraph` (weights from the `weight` attribute, default 1;
cells come back as tuples of node labels), or `(n_vertices, edges)` with `(source, target, weight)` edges. Weights must
be non-negative, and may be infinite (the edge never appears); self-loops are refused. Which time an edge enters at is
the `filtration` argument where it applies: `'weight'`, its own weight, or `'shortest_path'`, the length of the shortest
directed path (making the weights a quasi-metric). `flag` takes an undirected networkx `Graph`, or
`(n_vertices, edges)` with `(u, v)` or `(u, v, weight)` edges, and node values (`node_weight`: an attribute name, or one
value per node). For the [extended persistence](extended.md) of a function on a graph's nodes, see that chapter.

The complexes other than (Gr)PPH are built on `phlite_networks::lazy`: a complex described by listing any simplex's
cofacets (in increasing order) and facets, from which come coboundary and boundary oracles with Ripser's optimisations
(emergent and apparent pairs, clearing, the top dimension never stored), as for Vietoris–Rips. They are checked against
references (grpphati, flagser) and, all of them, against the complex built by brute force and reduced by LoPHAT, as well
as invariants of each filtration (`python-tests/tests/test_networks_complexes.py`).
