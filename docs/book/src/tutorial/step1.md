# 1. A boundary matrix

The least `phlite` needs is a **label type** for cells and a **matrix oracle**.

## The label

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:cell}}
```

`BasisElement` marks a type as a row or column label. It must be `Ord` (the order breaks ties between cells with the
same filtration value) and `Clone`, and it is cloned a lot, so it should be small: here one `u64`.

## The oracle

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:boundary}}
```

Three associated types say what the matrix is over (`Z2`, whose one non-zero element is zero-sized) and what labels
its columns and rows; `column` returns an iterator over a column's entries. The boundary is computed from the cell's
number each time: the oracle holds just a reference to the volume.

## Running it

To reduce a matrix, `phlite` also needs the **order on rows** and **which columns to reduce, in that order**. With
no filtration, `with_trivial_filtration()` puts every cell at time 0, so rows are ordered by label alone, and the
basis must list the columns in the same order:

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:basis_by_number}}
```

```rust,ignore
{{#include ../../../../phlite/examples/volume_basic.rs:step1}}
```

`barcode::homology` reduces the boundary with the [standard algorithm](../algorithms/standard.md). With every cell at
time 0, every finite bar has length zero and is dropped, which leaves the essential bars: the Betti numbers of a solid
cube.

```text
No filtration: 1 bars in H0, 0 in H1 and 0 in H2
```

> **The basis must be in the rows' order.** The first version of this example sorted the basis by voxel value while
> the rows, with no filtration, were ordered by number. The reduction then pairs the wrong cells, and this example
> reported thousands of loops and cavities in a solid cube. `phlite` can't check the order for you without visiting
> every column, so this is worth a test of its own.

**You can now** compute the homology of anything you can describe cell by cell.
