# phlite_bindings (`phlite-core`)

What [phlite](https://github.com/tomchaplin/phlite)'s Python packages share: the Python package `phlite-core`
(`phlite.core`, built with the feature `extension`), which every other phlite package depends on, and the Rust
helpers their bindings use. You don't usually install it yourself: `pip install phlite` (or any one package, such as
`phlite-clouds`) brings it in. See [the book](https://phlite.tomchaplin.xyz/book/) and the
[Python API reference](https://phlite.tomchaplin.xyz/api/).

## `phlite.core`

* **`Algorithm`** chooses how a barcode is computed, the same way for every pipeline of every package:
  `Algorithm.serial()` (the default) or `Algorithm.parallel(threads)`, adjusted with `with_columns`, `with_cache`,
  `with_store_r` and so on. Options whose best value depends on the pipeline are left to it.
  ```python
  from phlite.core import Algorithm
  from phlite.clouds import rips

  rips(points=X, max_dim=2, algorithm=Algorithm.parallel(threads=8))
  ```
  [Choosing an algorithm](https://phlite.tomchaplin.xyz/book/python/algorithms.html) explains every option and when it
  helps, and `phlite.core.defaults()` lists each pipeline's defaults.
* **`Barcode`**, what most pipelines return: `pairings` (finite bars) and `essential` (births of infinite ones) per
  dimension, their representatives if asked for, and the algorithm that computed it.
* **`ExtendedBarcode`**, what extended persistence returns: ordinary, relative and extended diagrams, as GUDHI gives
  them.

## For developers of the bindings

phlite's Python packages are separate wheels sharing the namespace package `phlite`, each built from its own crate.
Separate extension modules can't share Rust classes, so `phlite.core` owns the shared ones, and the others reach them
through Python with the helpers here:

* a pipeline reads an `Algorithm` as plain data (`Algorithm._spec()`) with `algorithm_from_py`, filling in its own
  `AlgorithmDefaults`;
* `SortedBars::into_python` builds a `Barcode` through its private constructor `Barcode._from_parts`, recording the
  algorithm that ran (and `extended_into_python` an `ExtendedBarcode`);
* `init_package`, called when a package's module is initialised, checks that `phlite-core` has the package's own
  version (every phlite package is released in lockstep) and registers its pipelines' defaults.

The crate's documentation (`cargo doc -p phlite_bindings --open`) says how to add a pipeline to a package.
