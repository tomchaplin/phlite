# phlite_complexes

Persistent homology of chain complexes you build yourself, with any filtration, in Rust and Python
(`pip install phlite-complexes`, or `pip install phlite` for every package). Part of
[phlite](https://github.com/tomchaplin/phlite), which brings the techniques of Ripser and its relatives to any
filtration; see [the book](https://phlite.tomchaplin.xyz/book/) and the
[in-browser playground](https://phlite.tomchaplin.xyz/playground/), where you can edit a chain complex and its
filtration and watch the reduction.

```python
from phlite.complexes import ChainComplex

# A hollow tetrahedron, as (dimension, boundary) pairs, as PHAT and LoPHAT take them
pairs = [
    (0, []), (0, []), (0, []), (0, []),
    (1, [0, 1]), (1, [0, 2]), (1, [1, 2]), (1, [0, 3]), (1, [1, 3]), (1, [2, 3]),
    (2, [4, 7, 8]), (2, [5, 7, 9]), (2, [6, 8, 9]), (2, [4, 5, 6]),
]
chain = ChainComplex.from_pairs(pairs)

barcode = chain.filtration([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]).persistence(representatives=True)
barcode.diagram(1)                  # [[1, 3], [2, 3], [2, 3]]
barcode.representative(6)           # the cycle born at edge 6, as [(cell, coefficient), ...]
```

## What it does

* **Build a complex once**, from integer columns (`from_pairs`, `from_columns`), CSC arrays straight from numpy or
  scipy (`from_csc`), or **your own Python objects** as cells (`from_cells`: anything with a dimension and a boundary,
  such as grpphati's columns); over Z/2 or Z/p.
* **Filter it as often as you like** (`filtration`): values as an array, a mapping, a function of a cell, or on the
  vertices only (`extend=True` gives the lower-star filtration).
* **Compute persistence** (`persistence`) by homology or cohomology, serially or in parallel, with representatives,
  and get results back in terms of your own cells.
* **Extended persistence** (`extended_persistence`) of any filtration, going back down by the least value of each cell
  and its faces (for values on vertices: upper-star, the classical case).
  [Book chapter.](https://phlite.tomchaplin.xyz/book/python/extended.html)

The reduction is phlite's, with the techniques of PHAT for explicit matrices: pivot columns (a bit tree over the rows)
and storing R. Each filtered complex is stored compactly, renumbered in filtration order with sorted columns and its
transpose, so pivots and apparent pairs cost nothing to find.
[Book chapter.](https://phlite.tomchaplin.xyz/book/python/complexes.html)

## Compared with other packages

phlite's reduction of explicit complexes owes its pivot columns and stored R to
[PHAT](https://github.com/blazs/phat) (Bauer, Kerber, Reininghaus and Wagner), and its parallel reduction to
Morozov and Nigmetov, as does [LoPHAT](https://github.com/tomchaplin/lophat).

* **[PHAT](https://github.com/blazs/phat)** (twist reduction, given the same matrix, on one thread): phlite is 11–26×
  faster on Rips complexes (by cohomology, which PHAT doesn't do) and 1.1–2× faster on sparse cell complexes; level on
  triangulated surfaces; **PHAT is up to 1.4× faster on cubical complexes and dense boundaries**. phlite uses more
  memory, up to 4.5× on Rips complexes, as it also stores the transpose.
* **[LoPHAT](https://github.com/tomchaplin/lophat)** (lock-free, parallel): on these inputs, on one thread or four,
  phlite's serial reduction is 3–180× faster. (phlite's own parallel reduction doesn't pay off on explicit complexes
  either, where its serial default with pivot columns is fastest.)
* **[GUDHI](https://gudhi.inria.fr)** (extended persistence, by `SimplexTree.extended_persistence`): phlite is
  2.5–5.5× faster on random graphs and flag complexes, building the complex included, and handles any filtration, not
  only lower-star ones.

What phlite adds is the Python interface: cells of your own type, many filtrations of one complex, representatives in
terms of your cells, and the same `Algorithm` as phlite's other packages. All of the numbers are in
[the benchmarks](https://github.com/tomchaplin/phlite/blob/main/python-tests/BENCHMARKS.md).
