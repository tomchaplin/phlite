# phlite for Python

Persistent homology from Rust, in Python packages sharing the namespace `phlite`, one per kind of input
(`pip install phlite` installs them all):

| Module | Package | Computes | Reference |
|---|---|---|---|
| `phlite.complexes` | `phlite-complexes` | persistence of any filtered chain complex you build: from integer columns, CSC arrays, or your own Python objects as cells | {doc}`api/phlite/complexes/index` |
| `phlite.clouds` | `phlite-clouds` | persistence of point clouds and distance matrices: Vietoris–Rips | {doc}`api/phlite/clouds/index` |
| `phlite.networks` | `phlite-networks` | persistence of networks: grounded persistent path homology of a weighted digraph | {doc}`api/phlite/networks/index` |
| `phlite.images` | `phlite-images` | persistence of images (nothing yet) | {doc}`api/phlite/images/index` |
| `phlite.core` | `phlite-core` | what they share: `Algorithm`, `Barcode` and `defaults()` | {doc}`api/phlite/core/index` |

Every pipeline returns a barcode with optional representative cycles, computes by cohomology (default) or homology,
releases the GIL while it works, and takes the one `phlite.core.Algorithm` class for choosing how the reduction is done
(`phlite.core.defaults()` lists each pipeline's defaults).

```python
from phlite.complexes import ChainComplex

chain = ChainComplex.from_pairs([(0, []), (0, []), (1, [0, 1])])
barcode = chain.filtration([0.0, 1.0, 2.0]).persistence(representatives=True)
barcode.pairings      # [(1, 2)]
barcode.diagram(0)    # array([[0., inf], [1., 2.]])
```

These pages are the API reference, generated from the packages' docstrings. For guides, worked examples (with
notebooks) and how everything works inside, see **<a href="../book/index.html">the phlite book</a>**.

```{toctree}
:maxdepth: 2
:caption: API reference

api/phlite/complexes/index
api/phlite/clouds/index
api/phlite/networks/index
api/phlite/images/index
api/phlite/core/index
```
