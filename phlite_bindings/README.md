# phlite_bindings

What phlite's Python packages share, and the package `phlite-core` (`phlite.core`, with the feature `extension`).

phlite's Python packages are separate wheels sharing the namespace package `phlite`, each built from its own crate
(`phlite.complexes` from `phlite_complexes`, `phlite.clouds` from `phlite_clouds`, ...). Separate extension modules
can't share Rust classes, so `phlite.core` owns the shared ones, and the others reach them through Python with the
helpers here:

* `phlite.core.Algorithm` chooses the reduction. Options whose best value depends on the pipeline (columns, caching,
  storing R) are left unset by the presets; a pipeline reads the algorithm as plain data (`Algorithm._spec()`) with
  `algorithm_from_py`, filling in its own `AlgorithmDefaults`.
* `phlite.core.Barcode` is what most pipelines return: `SortedBars::into_python` builds one through its private
  constructor `Barcode._from_parts`, recording the algorithm that ran.
* `init_package`, called when a package's module is initialised, checks that `phlite-core` has the package's own
  version (every phlite package is released in lockstep) and registers its pipelines' defaults, which
  `phlite.core.defaults()` and `Algorithm.resolve(pipeline)` report.

The crate's documentation (`cargo doc -p phlite_bindings --open`) says how to add a pipeline to a package.
