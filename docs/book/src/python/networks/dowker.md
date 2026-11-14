# Dowker complexes (`phlite.networks.dowker`)

```python
from phlite.networks import dowker

dowker(G, max_dim=2)                   # the sink complex
dowker(G, max_dim=2, side="source")    # the source complex: the same barcode
```

With `ω(a, b)` the weight of the edge `a -> b` (infinite if there is none, and `ω(a, a) = 0`), a set of nodes is a
simplex of the Dowker **sink** complex (Chowdhury and Mémoli) once some node `x`, perhaps one of them, is within `t`
of each of them, `ω(v, x) <= t`: it enters at \\(\\min_x \\max_{v \\in \\sigma} \\omega(v, x)\\). The
**source** complex uses `ω(x, v)` instead. By Dowker duality the two have the same barcode. `filtration='shortest_path'`
takes `ω` to be the shortest paths' lengths instead of the edges' weights.

It is built lazily, as the directed flag complex is: simplices are sets of nodes in increasing order, so their cofacets
come in increasing order by adding nodes in increasing order, each found from the simplex's own
`max_v ω(v, x)` for every `x` in `O(n)`. Every set of nodes with a common sink is a simplex, so Dowker complexes are
dense: they suit small digraphs (up to a few hundred nodes) and low dimensions.

## Checked against

The complex built by brute force and reduced by LoPHAT (sink and source, every algorithm, representatives from first
principles), and Dowker duality: the sink and source complexes' barcodes agree, and the sink complex of a digraph is
the source complex of its reverse.
