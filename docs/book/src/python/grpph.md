# phlite_grpph: path homology of digraphs

*Grounded persistent path homology* (Chaplin, Harrington and Tillmann) is a persistence theory for weighted digraphs
that sees directed cycles: a directed square, a long directed loop, a bidirected edge. `phlite_grpph` computes it, with
representatives, much faster than the reference implementation grpphati.

```python
import networkx as nx
from phlite_grpph import grpph

G = nx.DiGraph()
G.add_weighted_edges_from([("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0), ("d", "a", 1.0)])

barcode = grpph(G, representatives=True)
barcode.pairings[1]                  # [(0.0, 2.0)]: the directed 4-cycle, filled in at distance 2
barcode.pairing_representatives[1]   # [[('d', 'a'), ('c', 'd'), ('b', 'c'), ('a', 'b')]]
barcode.essential[0]                 # [0.0]: one component
```

or with integer nodes: `grpph(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)])`.

| Argument | | Default |
|---|---|---|
| `graph`, `edges` | a networkx `DiGraph` / `MultiDiGraph`, or a node count and `(source, target, weight)` edges | |
| `weight` | the edge attribute holding weights (missing means 1) | `"weight"` |
| `method` | `"cohomology"` or `"homology"` (homology needs every 2-cell up front: far more memory) | `"cohomology"` |
| `algorithm` | an [`Algorithm`](algorithms.md) | `Algorithm.serial()` (merged columns, cached) |
| `representatives` | a cycle for every bar, as a list of nodes (H0) or edges (H1) | `False` |

## What it does

The filtration is the **grounded** path complex: edges of the graph are present from time 0 (so the cycle above is
born at 0, not at 1), and the regular path complex of the shortest-path digraph grows with distance. Its coboundary enumerates the allowed paths extending a
path, which is relatively expensive, so:

- columns of the coboundary are **cached** (sorted) once per dimension: about 3× faster;
- **merged columns** keep the working column small and never read below the pivot: 3–4× faster than heap columns;
- paths are compared through a **packed 128-bit key** instead of a nested enum.

The cost grows with the number of *nodes* (the shortest-path digraph is complete), so keep graphs to a few hundred
nodes; edges are cheap.

See the [notebooks](notebooks.md) for a worked example with directed cycles.
