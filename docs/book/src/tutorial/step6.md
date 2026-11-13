# 6. Leaving cells out

This is where a lazy oracle pulls ahead of a stored matrix. A cell in a **zero-length apparent pair** needs no
reduction and gives no bar, and the reduction can recognise it again as an owner when it needs it (through
`first_column_in_row`). So it needn't be in the basis at all:

```rust,ignore
{{#include ../../../../phlite/examples/volume_lazy.rs:reduction_basis}}
```

A cell is in a zero-length apparent pair as a *birth* if its oldest coface has the same value and has it as youngest
face, or as a *death* if its youngest face has the same value and has it as oldest coface. Both checks are local: a
few voxel lookups. Cells are streamed, so only those kept are ever stored.

```rust,ignore
{{#include ../../../../phlite/examples/volume_lazy.rs:step6}}
```

On a 96³ volume (7 million cells, 6.1 million of them up to dimension 2):

```text
build every cell up to dimension 2                    2965.6 ms
cohomology, clearing, hooks                           3017.3 ms
build the reduction basis                             2211.4 ms
cohomology, clearing, reduction basis                 2020.2 ms
    basis: 338183 cells instead of 6110496 (5.5%)
    peak memory: 78 MB instead of 326 MB
```

The basis is 18× smaller, the reduction 1.5× faster, and peak memory a quarter. Building the basis is now the
largest cost, and it is embarrassingly parallel (`phlite_rips` builds its basis on all cores).

For comparison, storing this complex explicitly, as PHAT or the `phlite` Python package do, takes about 36 bytes per
cell and 8 per boundary entry, roughly 400 MB here, before any reduction.

## Trade-offs

- **Only the clearing algorithms** can leave cells out: the standard algorithm needs every column.
- The test must be **exactly right**: leave out a cell that isn't in a zero-length apparent pair and its bar is lost.
- It pays when most pairs are zero-length apparent pairs (lower-star filtrations of smooth data, Vietoris–Rips). On
  noise, where few pairs have length zero, the test costs more than it saves.
