# Summary

[Introduction](introduction.md)

# Background

- [Persistent homology in one chapter](background/persistence.md)
- [Homology or cohomology?](background/cohomology.md)

# How phlite is built

- [Matrix oracles](design/oracles.md)
- [From a matrix to a barcode](design/barcodes.md)

# Reduction algorithms

- [The standard algorithm](algorithms/standard.md)
- [Clearing](algorithms/clearing.md)
- [Apparent and emergent pairs](algorithms/apparent-pairs.md)
- [Storing R, or recomputing it from V](algorithms/store-r.md)
- [Parallel clearing](algorithms/parallel.md)
- [Representatives and the involution](algorithms/involution.md)

# Column representations

- [The working column](columns/overview.md)
- [Heap columns](columns/heap.md)
- [Merged columns](columns/merged.md)
- [Sorted columns](columns/sorted.md)
- [Pivot columns](columns/pivot.md)
- [Caching columns](columns/caching.md)
- [Choosing](columns/choosing.md)

# Inside phlite

- [Bookkeeping: pivot tables and maps](internals/bookkeeping.md)
- [Explicit complexes (phlite_py)](internals/explicit.md)
- [Where the time goes](internals/performance.md)

# Tutorial: your own filtration in Rust

- [Overview](tutorial/index.md)
- [1. A boundary matrix](tutorial/step1.md)
- [2. A filtration and a basis](tutorial/step2.md)
- [3. Cohomology and clearing](tutorial/step3.md)
- [4. Apparent pairs](tutorial/step4.md)
- [5. Cheaper pivots](tutorial/step5.md)
- [6. Dense indices and pivot columns](tutorial/step6.md)
- [7. Representatives and threads](tutorial/step7.md)

# Python

- [The three packages](python/index.md)
- [phlite: explicit complexes](python/phlite.md)
- [phlite_rips: Vietoris–Rips](python/rips.md)
- [phlite_grpph: path homology of digraphs](python/grpph.md)
- [Choosing an algorithm](python/algorithms.md)
- [Example notebooks](python/notebooks.md)

# Appendix

- [Testing](appendix/testing.md)
- [Benchmarks](appendix/benchmarks.md)
- [References](appendix/references.md)
