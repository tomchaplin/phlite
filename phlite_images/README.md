# phlite_images

Persistent homology of images (cubical complexes) with [`phlite`](../phlite/README.md). **Not implemented yet**: this
crate is a placeholder for a dedicated cubical matrix oracle, with the V- and T-constructions, in 2D and 3D.

With the feature `python`, it is also the Python package `phlite-images`, `phlite.images` (the bindings live in
`src/python`, one file per pipeline), so far empty. Until then, build the cubical complex explicitly with
`phlite.complexes.ChainComplex`, as in the image notebook (`docs/notebooks/phlite_image.py`).
