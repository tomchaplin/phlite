# Directed Rips: ordered tuples (`phlite.networks.ordered_tuples`)

```python
from phlite.networks import ordered_tuples

ordered_tuples(G, max_dim=2)
```

Turner's ordered-tuple complex (*Rips filtrations for quasimetric spaces and asymmetric functions*; the survey
*Persistence homology of networks: methods and applications*' GVR-b) is the [directed flag complex](directed_flag.md)
with repeated nodes allowed, so long as no node is next to itself: `(x_0, ..., x_k)` with an edge `x_i -> x_j` whenever
`i < j` and `x_i != x_j`. So `(a, b, a)` is a 2-simplex as soon as `a -> b` and `b -> a` are both edges: a pair of
opposite edges is filled in at once, rather than making a loop as in the directed flag complex. Faces with a node next
to itself are degenerate and left out of the boundary (the normalised chain complex), so the boundary of `(a, b, a)` is
`(b, a) + (a, b)`. It takes `filtration` (`'weight'`, the default, or `'shortest_path'`) as `directed_flag` does, and is
built the same way. It grows quickly with the dimension.

## Checked against

The complex built by brute force and reduced by LoPHAT (any dimension, either filtration, every algorithm,
representatives from first principles), and the invariants of the directed flag complex. Not against grpphati: its
`OrderedTuplesHomology` keeps a 2-cell only when `edge_time((source, target))` is finite, which for a double edge
`(a, b, a)` asks for `edge_time((a, a))`, never finite, so it leaves every double edge out and computes the directed
flag complex's homology instead.
