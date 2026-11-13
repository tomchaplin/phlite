# 4. Apparent pairs

The boundary is the coboundary's transpose, and attaching it costs one line:

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:step4}}
```

With the transpose the reduction can ask "which is the first column with an entry in row \\(r\\)?" (the smallest entry
of column \\(r\\) of the transpose), and so recognise [apparent pairs](../algorithms/apparent-pairs.md): pairs that
need no reduction and no storage. In a lower-star filtration most pairs are apparent, and most of those have length
zero (an edge and its brighter end appear together), which clearing then doesn't even record.

But measure it:

```text
A 40³ volume: 493039 cells

homology, standard algorithm                    5375.1 ms
cohomology, clearing                             156.7 ms
cohomology, clearing, apparent pairs             186.4 ms
```

**Here it is 20% slower.** Our columns have at most six entries, so reducing a column is cheap anyway, and the
transpose's generic answer (scan the column of the transpose, computing every entry's filtration value from up to eight
voxels) costs more than it saves. Apparent pairs pay off when a column is expensive to reduce, or, as in the next two
steps, when they can be recognised more cheaply and used to avoid listing cells at all.

This is a general lesson: every optimisation in `phlite` has a cost, and the only way to know whether it pays on
your matrix is to time it.
