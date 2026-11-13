# 3. Cohomology and clearing

The **coboundary** lists each cell's cofaces. Its oracle mirrors the boundary's:

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:coboundary}}
```

and now [cohomology](../background/cohomology.md) with [clearing](../algorithms/clearing.md) is available:

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:step3}}
```

Two things changed:

- **The basis stops at the squares.** For \\(H_0\\), \\(H_1\\) and \\(H_2\\) by cohomology the columns are vertices,
  edges and squares; cubes only appear as *rows*, produced by the oracle as cofaces of squares. They are never listed.
- **Clearing.** `barcode::cohomology` reverses the coboundary (the anti-transpose) and reduces it dimension by
  dimension, skipping every column that the previous dimension's pairs show to be zero. The basis must be split by
  dimension, which `SplitBasis` is.

On the 40³ volume this takes **157 ms**, against 5.4 s for the standard algorithm on the boundary: 34× faster.

**You can now** use every algorithm in `phlite` that doesn't need extra information about the matrix.
