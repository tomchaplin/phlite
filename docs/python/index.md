# phlite for Python

Persistent homology from Rust, in three Python packages:

| Package | Computes | Reference |
|---|---|---|
| `phlite` | persistence of any filtered chain complex you build: from integer columns, CSC arrays, or your own Python objects as cells | {doc}`api/phlite/index` |
| `phlite_rips` | Vietoris–Rips persistence of a distance matrix | {doc}`api/phlite_rips/index` |
| `phlite_grpph` | grounded persistent path homology of a weighted digraph | {doc}`api/phlite_grpph/index` |

All three return barcodes with optional representative cycles, compute by cohomology (default) or homology, release
the GIL while they work, and share one `Algorithm` class for choosing how the reduction is done.

```python
import phlite

chain = phlite.ChainComplex.from_pairs([(0, []), (0, []), (1, [0, 1])])
barcode = chain.filtration([0.0, 1.0, 2.0]).persistence(representatives=True)
barcode.pairings      # [(1, 2)]
barcode.diagram(0)    # array([[0., inf], [1., 2.]])
```

These pages are the API reference, generated from the packages' docstrings. For guides, worked examples (with
notebooks) and how everything works inside, see **<a href="../book/index.html">the phlite book</a>**.

```{toctree}
:maxdepth: 2
:caption: API reference

api/phlite/index
api/phlite_rips/index
api/phlite_grpph/index
```
