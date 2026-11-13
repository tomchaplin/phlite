# The three Python packages

| Package | Crate | Input | Typical use |
|---|---|---|---|
| [`phlite`](phlite.md) | `phlite_py` | a chain complex you build: integer columns, CSC arrays, or your own cell objects; any filtration | cubical complexes, simplicial complexes, cell complexes, anything PHAT or LoPHAT would take |
| [`phlite_rips`](rips.md) | `phlite_rips` | a distance matrix | Vietoris–Rips persistence, like ripser and giotto-ph |
| [`phlite_grpph`](grpph.md) | `phlite_grpph` | a weighted digraph (networkx or edge list) | grounded persistent path homology, like grpphati |

The full **[API reference](../../api/index.html)** (every class, method and argument) is generated from the packages'
docstrings with Sphinx (`docs/python`). These chapters are the guide.

All three share:

- an [`Algorithm`](algorithms.md) class, with the same options, to choose the reduction;
- `method="cohomology"` (default) or `"homology"`;
- `representatives=True` for a cycle representing each bar;
- release of the GIL during the computation, so other Python threads keep running.

## Installing

The packages are built from source with [maturin](https://www.maturin.rs/). From a crate's directory:

```sh
just develop            # builds the release extension into the current virtual environment
```

or with maturin directly: `maturin develop --release -m phlite_py/Cargo.toml`.

## Barcodes at a glance

```python
import phlite

chain = phlite.ChainComplex.from_pairs([(0, []), (0, []), (1, [0, 1])])
barcode = chain.filtration([0.0, 1.0, 2.0]).persistence()
barcode.pairings      # [(1, 2)]: the component born at cell 1 dies at cell 2
barcode.essential     # [0]
barcode.diagram(0)    # array([[0., inf], [1., 2.]])
```

```python
from phlite_rips import rips
barcode = rips(distances, max_dim=1, representatives=True)
barcode.pairings[1]                  # (birth, death) values of the finite H1 bars
barcode.pairing_representatives[1]   # a cycle (list of edges) for each
```

`phlite`'s barcode is in terms of *cells* (with `diagram` for values); `phlite_rips`' and `phlite_grpph`'s are lists
per dimension of *values*, with representatives as lists of simplices or paths.
