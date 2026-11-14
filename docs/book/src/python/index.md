# The Python packages

phlite's Python packages share the namespace `phlite`: one package per kind of input, and `phlite.core`, which they
all use. `pip install phlite` installs them all.

| Package | Module | Input | Pipelines | Rust crate |
|---|---|---|---|---|
| `phlite-complexes` | [`phlite.complexes`](complexes.md) | a chain complex you build: integer columns, CSC arrays, or your own cell objects; any filtration | `FilteredComplex.persistence`: cubical, simplicial and cell complexes, anything PHAT or LoPHAT would take | `phlite_complexes` |
| `phlite-clouds` | [`phlite.clouds`](clouds.md) | a point cloud or distance matrix | [`rips`](clouds/rips.md): Vietoris–Rips, like ripser and giotto-ph | `phlite_clouds` |
| `phlite-networks` | [`phlite.networks`](networks.md) | a weighted digraph (networkx or edge list) | [`grpph`](networks/grpph.md): grounded persistent path homology, like grpphati | `phlite_networks` |
| `phlite-images` | [`phlite.images`](images.md) | an image | none yet | `phlite_images` |
| `phlite-core` | `phlite.core` | | [`Algorithm`](algorithms.md), `Barcode` and `defaults()`, shared by every package | `phlite_bindings` |

```python
from phlite.core import Algorithm
from phlite.clouds import rips
from phlite.networks import grpph
from phlite.complexes import ChainComplex
```

The full **[API reference](../../api/index.html)** (every class, method and argument) is generated from the packages'
docstrings with Sphinx (`docs/python`). These chapters are the guide.

Every pipeline shares:

- the [`Algorithm`](algorithms.md) class (`phlite.core.Algorithm`), to choose the reduction, with defaults tuned per
  pipeline (`phlite.core.defaults()` lists them);
- `method="cohomology"` (default) or `"homology"`;
- `representatives=True` for a cycle representing each bar;
- release of the GIL during the computation, so other Python threads keep running.

## How the packages fit together

Each package is its own extension module, built from its own crate, which holds both its Rust implementation and its
Python bindings (in `src/python`, behind the feature `python`). `phlite` is a
[namespace package](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) (as `google.cloud`
and `azure` are): no package has a `phlite/__init__.py`, each provides only its own `phlite/<name>`, so they install,
upgrade and uninstall independently. `phlite`, the package you `pip install`, has no files at all: it only depends on
the others.

Separate extension modules can't share Rust classes, so `phlite.core` owns the shared ones, and the other packages
reach them through Python (with the helpers of the `phlite_bindings` crate):

- an `Algorithm` reaches a pipeline as plain data (its private `_spec()`), which the pipeline resolves against its own
  defaults;
- a pipeline builds its `phlite.core.Barcode` through a private constructor, recording the algorithm that ran;
- each package registers its pipelines' defaults with `phlite.core` when it is imported, and `phlite.core` imports
  every installed package when asked for them (`defaults()`, `Algorithm.resolve`).

Every package is released on the same version, pins `phlite-core` to exactly it, and refuses to import against any
other, saying how to fix it. `phlite.clouds` and `phlite.networks` used to be the packages `phlite_rips` and
`phlite_grpph`, each with its own `Algorithm` class, and `phlite.complexes` was the package `phlite`; there are no
aliases for the old names.

## Installing

The packages are built from source with [maturin](https://www.maturin.rs/), `phlite-core` first (the others depend
on it). In python-tests, `just build` builds them all into its environments; or, into your own environment:

```sh
maturin develop --release -m phlite_bindings/Cargo.toml   # phlite-core
maturin develop --release -m phlite_clouds/Cargo.toml     # phlite-clouds, and so on
```

## Barcodes at a glance

```python
from phlite.complexes import ChainComplex

chain = ChainComplex.from_pairs([(0, []), (0, []), (1, [0, 1])])
barcode = chain.filtration([0.0, 1.0, 2.0]).persistence()
barcode.pairings      # [(1, 2)]: the component born at cell 1 dies at cell 2
barcode.essential     # [0]
barcode.diagram(0)    # array([[0., inf], [1., 2.]])
```

```python
from phlite.clouds import rips
barcode = rips(distances, max_dim=1, representatives=True)
barcode.pairings[1]                  # (birth, death) values of the finite H1 bars
barcode.pairing_representatives[1]   # a cycle (list of edges) for each
barcode.algorithm                    # the algorithm that ran, every option filled in
```

`phlite.complexes` gives a `ComplexBarcode`, in terms of *cells* (with `diagram` for values); the other packages'
pipelines give a `phlite.core.Barcode`: lists per dimension of *values*, with representatives as lists of cells, each a
tuple of vertices.
