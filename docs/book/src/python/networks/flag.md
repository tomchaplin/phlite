# Flag complexes of graphs (`phlite.networks.flag`)

```python
import networkx as nx
from phlite.networks import flag

flag(G, max_dim=2)                                       # by edge weights, nodes at 0
flag(G, max_dim=2, weight=None, node_weight="height")    # the lower-star filtration of the node values
flag(G, max_dim=2, node_weight="height")                 # both
flag(G, max_dim=2, filtration="shortest_path")           # Vietoris–Rips of the shortest-path metric
flag(G, max_dim=2, weight=None, filtration="shortest_path")  # ... of the hop distance
```

The **flag complex** (or clique complex) of an undirected graph has a simplex for every clique: every set of nodes
pairwise joined by an edge. Each edge \\(uv\\) has a weight \\(w(uv)\\) and each node \\(v\\) a value \\(f(v)\\) (0 by
default), and a simplex enters at the largest of them:

\\[ \\operatorname{value}(\\sigma) = \\max\\Big(\\max_{uv \\subseteq \\sigma} w(uv),\\ \\max_{v \\in \\sigma} f(v)\\Big). \\]

So a node enters at its value, and an edge at the largest of its weight and its nodes' values. This one filtration
covers:

| Arguments | Filtration |
|---|---|
| `flag(G)` | the weighted flag complex, every node at 0: `phlite.clouds.rips` of the graph's sparse distance matrix |
| `weight=None, node_weight=...` | the **lower-star** filtration of the node values: each simplex enters with its latest node |
| both | each simplex no earlier than its edges' weights or its nodes' values (GUDHI's `SimplexTree` with vertex and edge values, made non-decreasing) |

With `filtration='shortest_path'`, every two nodes joined by a path are joined by an edge at the length of the
shortest one, so the flag complex is the **Vietoris–Rips filtration of the graph's shortest-path metric**. Edge weights
are then lengths, so must be non-negative; with `weight=None` each edge has length 1 (the hop distance). Node values
still apply, and nodes in different components are never joined.

Weights may be negative (they are only values, unless they're lengths). An infinite weight is an edge that never
appears; an edge given twice keeps its smallest weight. Node values must be finite. Representatives come back as
tuples of node labels.

## How it's computed

Two ways, chosen automatically (or by `explicit=True` or `False`):

- **Explicitly**, by edge weight, when the complex has at most a million simplices (up to dimension `max_dim + 1`,
  the threshold and the enclosing radius, past which the complex is a cone): every clique is listed, and the complex
  reduced as `phlite.complexes` reduces one, with its defaults (`'networks.flag'`: pivot columns, storing R). Each
  column is built once, which is much the fastest for sparse graphs and for weights that aren't distances: on a
  random graph of 300 nodes and 4,500 edges, about 10 ms in dimensions up to 2, against GUDHI's 115 ms.
- **Implicitly**, for bigger complexes and for shortest paths: `phlite_networks::undirected::Graph` stores the graph
  as a sparse symmetric matrix of the values at which edges enter, and implements `phlite_clouds`' `Distances` with
  the node values as births, so the Rips pipeline computes the barcode unchanged: union-find in dimension 0, apparent
  and emergent pairs, and cofacets enumerated through the neighbours of a simplex's sparsest vertex, with nothing
  stored but the graph (`'networks.flag_implicit'`, Rips' defaults). On a metric, such as shortest paths, apparent
  pairs settle nearly every column without reducing it; on arbitrary weights they settle few, and each column is
  recomputed every time it's added, which is why such graphs are listed explicitly when they can be.

Asking for union-find (`Algorithm.serial().with_union_find()`) chooses the implicit computation, the only one with it.
Shortest paths are found by Dijkstra's algorithm from every node.

## Checked against

GUDHI's `SimplexTree`, built from the vertices and edges with their values and expanded, for every method and
algorithm (with integer values, so lots of ties, and floats, some negative); shortest paths against networkx's
Dijkstra and GUDHI, and Rips of the matrix of path lengths; the graph without node values against Rips of its sparse
distance matrix; thresholds; and representatives from first principles
(`python-tests/tests/test_networks_flag.py`).
