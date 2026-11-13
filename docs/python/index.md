# phlite for Python

Persistent homology from Rust, in one Python package with a submodule per kind of data:

| Module | Computes | Reference |
|---|---|---|
| `phlite` | persistence of any filtered chain complex you build: from integer columns, CSC arrays, or your own Python objects as cells | {doc}`api/phlite/index` |
| `phlite.cloud` | persistence of point clouds and distance matrices: Vietoris–Rips | {doc}`api/phlite/cloud/index` |
| `phlite.network` | persistence of networks: grounded persistent path homology of a weighted digraph | {doc}`api/phlite/network/index` |
| `phlite.image` | persistence of images (nothing yet) | {doc}`api/phlite/image/index` |

Every pipeline returns a barcode with optional representative cycles, computes by cohomology (default) or homology,
releases the GIL while it works, and takes the one `Algorithm` class for choosing how the reduction is done
(`phlite.defaults()` lists each pipeline's defaults).

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
```
