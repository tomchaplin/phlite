# Choosing a representation

Start with the defaults: each package picks the one that won its benchmarks. If you are writing your own oracle:

```mermaid
flowchart TD
    Q1{"can the matrix give<br/>row ranks?"}
    Q1 -- yes --> P["PivotColumns + store_r,<br/>cache off if columns come sorted"]
    Q1 -- no --> Q2{"are columns of D<br/>expensive to compute?"}
    Q2 -- yes --> M["MergedColumns, cached"]
    Q2 -- no --> Q3{"short columns, few<br/>added per column?"}
    Q3 -- yes --> H["HeapColumns, uncached<br/>(least memory)"]
    Q3 -- no --> MU["MergedColumns, uncached"]
```

| Matrix | Best | Package default |
|---|---|---|
| Vietoris–Rips cohomology | heap or merged, uncached | `phlite_rips`: heap, uncached |
| GrPPH cohomology | merged, cached | `phlite_grpph`: merged, cached |
| explicit cubical / dense / cell complexes | pivot, store R, uncached | `phlite`: pivot, store R, uncached |
| homology of small simplicial complexes | sorted | |

All combinations are compared in the [benchmarks](../appendix/benchmarks.md), and every combination is checked to
give the same barcode by the [tests](../appendix/testing.md).
