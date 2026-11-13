# The phlite package

The `phlite` Python package has the general pipeline at its top level, and one submodule per kind of data:

| Module | Input | Pipelines | Rust crate |
|---|---|---|---|
| [`phlite`](phlite.md) | a chain complex you build: integer columns, CSC arrays, or your own cell objects; any filtration | `FilteredComplex.persistence`: cubical, simplicial and cell complexes, anything PHAT or LoPHAT would take | `phlite_py` |
| [`phlite.cloud`](cloud.md) | a point cloud or distance matrix | [`rips`](cloud/rips.md): Vietoris–Rips, like ripser and giotto-ph | `phlite_cloud` |
| [`phlite.network`](network.md) | a weighted digraph (networkx or edge list) | [`grpph`](network/grpph.md): grounded persistent path homology, like grpphati | `phlite_network` |
| [`phlite.image`](image.md) | an image | none yet | `phlite_image` |

The full **[API reference](../../api/index.html)** (every class, method and argument) is generated from the package's
docstrings with Sphinx (`docs/python`). These chapters are the guide.

Every pipeline shares:

- the [`Algorithm`](algorithms.md) class, to choose the reduction, with defaults tuned per pipeline
  (`phlite.defaults()` lists them);
- `method="cohomology"` (default) or `"homology"`;
- `representatives=True` for a cycle representing each bar;
- release of the GIL during the computation, so other Python threads keep running.

The package is one extension module, built from the `phlite_py` crate, which also holds the bindings of the other
crates (in `phlite_py/src/python/<module>`); `phlite_cloud`, `phlite_network` and `phlite_image` are plain Rust. Until
they merged, `phlite.cloud` and `phlite.network` were the separate packages `phlite_rips` and `phlite_grpph`, each with
its own `Algorithm` class.

## Installing

The package is built from source with [maturin](https://www.maturin.rs/). From `phlite_py`:

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
from phlite.cloud import rips
barcode = rips(distances, max_dim=1, representatives=True)
barcode.pairings[1]                  # (birth, death) values of the finite H1 bars
barcode.pairing_representatives[1]   # a cycle (list of edges) for each
barcode.algorithm                    # the algorithm that ran, every option filled in
```

`FilteredComplex.persistence` gives a `Barcode` in terms of *cells* (with `diagram` for values); the submodules'
pipelines give a `ValueBarcode`: lists per dimension of *values*, with representatives as lists of cells, each a tuple
of vertices.
