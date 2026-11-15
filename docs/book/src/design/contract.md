# The contract: what your matrix must satisfy

phlite trusts your matrix. It reads the columns, filtration values and hints you give it, and never checks them while
reducing (that would cost as much as the reduction). If one of the requirements below fails, the result is usually a
**wrong barcode, not an error**. This page lists every requirement, and which algorithms and options rely on it.
[`phlite::matrices::contract`](#checking-a-matrix) checks them all on a small example, for your tests.

The pipelines in `phlite_complexes`, `phlite_clouds`, `phlite_networks` and `phlite_images` meet the contract
themselves, and are tested against other libraries (see [Testing](../appendix/testing.md)). This page is for writing
your own matrix.

## Every algorithm

Write \\(D\\) for your (co)boundary matrix, with rows and columns indexed by cells of type `C`. The ordering phlite
uses on cells is **by filtration value, then by `C`'s own `Ord`**. Call it the *filtration order*. It's a total order,
since `C: Ord`.

1. **Columns**: [`column(c)`](../design/oracles.md) is defined for every cell `c` in the column basis, and gives the
   non-zero entries \\((a, r)\\) of column \\(c\\). It must give the same column each time it is called. A row may
   appear more than once (its coefficients are summed), but no coefficient may be zero. The order of the entries
   doesn't matter, though descending is fastest. An oracle may panic when asked for a column outside its basis:
   phlite never asks for one.
2. **A filtration**: every cell enters no earlier than its faces. For a boundary matrix, each row \\(r\\) of column
   \\(c\\) comes before \\(c\\) in the filtration order. For a coboundary matrix, each row comes after it.
   Equal filtration values are fine, as long as `C`'s order puts faces first among cells of equal value. Without
   this, \\(D\\) isn't triangular, and pivots aren't persistence pairs.
3. **A chain complex**: \\(D^2 = 0\\) over your coefficient field. If faces of faces don't cancel, there is no
   homology to compute.
4. **The basis order**: the column basis lists every cell, and each dimension's cells in increasing filtration
   order: **the same order the rows are compared in**, by `filtration_value` and then `C`'s `Ord`. Columns have no
   order of their own; the reduction assumes that the order it visits columns in is the order of those cells as
   rows. So ties in filtration value must be broken by `C`'s `Ord` in the basis too, not by any other rule (a basis
   sorted by value alone, with ties in insertion order, breaks this). A basis split by dimension (which the barcode functions require, for clearing) lists the dimensions one
   after another, so it isn't sorted overall; nor need it be, since columns of different dimensions never share a
   row, so never meet in a reduction. Each column's rows are cells of one dimension lower (boundary) or higher
   (coboundary).
5. **Square**: rows and columns are the same kind of cell (`ColT = RowT`), as the barcode functions require
   (`Reducible`). Rows may be cells outside the column basis: a coboundary computed up to dimension \(d\) has rows
   in dimension \(d + 1\).

## Cohomology and representatives

6. **The coboundary is the transpose of the boundary**, with the same coefficients. `barcode::cohomology` reports
   bars in homology conventions by assuming it.
7. **The attached transpose is right.** With
   [`with_transpose`](../design/oracles.md), the transpose must be exactly \\(D^\top\\). Representatives by the
   [involution](../algorithms/involution.md) are computed from it, and so are apparent pairs. In debug builds, the
   involution checks that each representative has its pivot at the birth cell, and panics with "do `boundary` and
   `coboundary` agree?" if not.

## Optional hooks: if you implement them, they must be right

Each of these has a safe default. Override one only if your override is exact.

| Hook | Requirement | Used by |
|---|---|---|
| `column_pivot` / `column_min` | the largest / smallest entry in the filtration order (after summing repeats), or `None` for a zero column | every algorithm |
| `sorted_column` / `sorted_column_ascending` | every entry, each row once, no zero coefficients, in decreasing / increasing filtration order | merged columns |
| `first_column_in_row` / `last_column_in_row` | of the columns with a non-zero entry in the row, the first / last in the filtration order; `None` if unknown | apparent pairs (clearing) |
| `same_filtration_value` | true only if the two rows enter at the same value of the filtration whose bars you report (it may be coarser than `FiltrationT`, when that is a position); a wrong `true` drops a bar, a wrong `false` only skips less | skipping zero-length apparent pairs |
| `row_ranks` | the rank of each row in the filtration order (indexed by `dense_index`) | pivot columns |
| `BasisElement::DENSE_INDEX` | `dense_index` injective and small; `from_dense_index` its inverse | per-column bookkeeping in vectors, storing R |

## Options with conditions

- **Skipping zero-length apparent pairs** (on by default for clearing). The column basis may leave out the cells of
  apparent pairs of length zero (`phlite_clouds`' Rips does). The standard algorithm doesn't look for apparent
  pairs, so it needs every cell.
- **Storing R** (`Clearing::store_r`): at most \\(2^{32}\\) columns reduced in each dimension.
- **Pivot columns** need `row_ranks` (and so a dense index).
- **The parallel algorithm** needs your matrix to be `Send + Sync` (the `parallel` feature's `MaybeSync`). Its
  `column` must be safe to call from several threads at once.

## Coefficients

- `Z2` and `ZP<P>` are exact for every prime `P`. That `P` is prime is checked at compile time.
- `Q` panics if a numerator or denominator overflows, rather than giving a wrong answer.

## What phlite checks, and what panics

phlite returns errors for bad *data*, and panics only for misuse it documents:
- **Errors for bad data.** The pipelines check their input and return typed errors: `ComplexError`, `CloudError`,
  `NetworkError` and `CubicalError`. That covers NaN or negative weights, self-loops, out-of-range indices, ragged
  points, and so on.
- **Documented panics.** Each is in the function's `# Panics` section:
  - an index out of range (`VecVecMatrix::column`);
  - malformed `SplitBasis::from_parts` offsets;
  - sizes beyond what a cell type can encode. Each such limit has a check you can run first: `RipsIndex::fits`,
    `lazy::fits` and `pph::Network::max_dimension`.
- **Invariant panics.** Any other panic is an internal invariant failing, which is a bug. Please
  [report it](https://github.com/tomchaplin/phlite/issues).

## Checking a matrix

`phlite::matrices::contract` checks requirements 1–4 and 6–7, and every hook you implement but `same_filtration_value`
(which compares values the matrix can't see), on a matrix small
enough to look at every column. It returns the first violation it finds. Run it in your tests, on a few small
examples of your filtration:

```rust,ignore
use phlite::matrices::contract::{check_boundary, check_coboundary, check_split_basis, check_transposes, Direction};

check_boundary(&boundary)?;                    // requirements 1–4 and the hooks
check_split_basis(&boundary, Direction::Boundary)?; // requirement 4, by dimension (for clearing)
check_coboundary(&coboundary)?;
check_transposes(&boundary, &coboundary)?;     // requirements 6–7
```

The checks are exhaustive (at least quadratic in the number of cells), so they're for tests, not production.
