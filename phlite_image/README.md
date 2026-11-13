# phlite_image

Persistent homology of images (cubical complexes) with [`phlite`](../phlite/README.md). **Not implemented yet**: this
crate is a placeholder for a dedicated cubical matrix oracle, with the V- and T-constructions, in 2D and 3D.

In Python it will be the `phlite.image` submodule of the [`phlite`](../phlite_py/README.md) package (the bindings
live in `phlite_py/src/python/image`). Until then, build the cubical complex explicitly with
`phlite.ChainComplex`, as in the image notebook (`docs/notebooks/phlite_image.py`).
