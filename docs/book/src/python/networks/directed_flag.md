# Directed flag complexes (`phlite.networks.directed_flag`)

```python
from phlite.networks import directed_flag

directed_flag(G, max_dim=2)                          # as flagser: each edge at its weight
directed_flag(G, filtration="shortest_path")         # each pair at its shortest path's length
directed_flag(G, grounded=True)                      # grounded at the digraph's edges, dimensions 0 and 1
```

The directed flag complex of a digraph (as [flagser](https://github.com/luetge/flagser) computes it) has a simplex for
each sequence of distinct nodes `(x_0, ..., x_k)` with an edge `x_i -> x_j` whenever `i < j`: a directed clique,
ordered by its edges. A simplex enters when its last edge does (flagser's `max` filtration; nodes at 0), and the faces
of a simplex are its subsequences.

**Grounded**, as grounded persistent path homology (and as grpphati's grounded pipeline with
`DirectedFlagComplexHomology`): the digraph's own edges are present from time 0 (even those of infinite weight), and
the rest is the shortest-path filtration's directed flag complex, so that the bars in dimension 1 are the digraph's own
cycles, born at 0, until they become boundaries. Only dimensions 0 and 1 make sense there.

## How

Simplices are packed into a `u128` (the dimension, then each node in as few bits as the number of nodes needs, the
first most significant), so that they are ordered by dimension and then lexicographically, and adding or removing a
node is a few shifts. A simplex's cofacets are listed in that order without sorting: inserting `w` at position `p`
needs `w` to be an out-neighbour of `x_{p-1}` and an in-neighbour of `x_p`, so the candidates are the shorter of those
two sorted adjacency lists, and the order is: for `p = 0, ..., k` the candidates below `x_p`; then every candidate
appended last; then for `p = k, ..., 0` those above `x_p`. With cofacets in order and each at least the simplex's own
value, the first at that value is the oldest, which is Ripser's emergent pairs; facets give apparent pairs; and the bases
leave out zero-persistence apparent pairs and never store the top dimension in full.

## Checked against

flagser (`pyflagser.flagser_weighted`) up to dimension 3; grpphati in dimension 1, with the shortest-path filtration
and grounded; and, with either filtration and grounded, the complex built by brute force and reduced by LoPHAT, with
representatives checked from first principles. Invariants: reversing every edge (which reverses each simplex), relabelling
the nodes and doubling the weights; grounded, every H1 bar is born at 0, and there are as many as the digraph's cycle
rank when every weight is positive.
