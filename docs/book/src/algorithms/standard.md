# The standard algorithm

```text
for j in columns, in filtration order:
    while column j is non-zero and some earlier column k has the same pivot:
        add (a multiple of) column k to column j       # and record it in V
    if column j is non-zero: column j owns its pivot
```

This is the textbook algorithm (Edelsbrunner, Letscher and Zomorodian), and `phlite::reduction::Standard` implements
it with no optimisations: it builds every column in full and needs the full column basis.

```rust,ignore
let reduction = Standard::new().reduce(&matrix);
```

## How it works in phlite

- A **pivot table** maps each owned pivot (a row label) to its column (`low_inverse`).
- The column being reduced is a **working column** with two operations, *pop the pivot* and *add a column*; how it is
  stored is a [column representation](../columns/overview.md).
- Only \\(V\\) is stored, as sparse columns. To add column \\(k\\) of \\(R = DV\\), the algorithm adds the column of
  \\(D\\) for every term of \\(V_k\\) (see [storing R](store-r.md) for the alternative).

## When to use it

- **As a reference.** It is the simplest and the easiest to trust, and the tests compare everything else against it.
- **Small matrices**, where the bookkeeping of the cleverer algorithms costs more than it saves.
- **When the basis can't be split by dimension**, since clearing needs that.

It is slow on anything large: it reduces every column, including the many that clearing would skip, and it can't
use apparent pairs, so it needs every cell in the basis (`phlite_rips` gives it the full basis for that reason).
