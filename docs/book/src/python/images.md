# phlite.images: images

Persistent homology of images: the `phlite-images` package, built from the `phlite_images` crate (the Rust
implementation, and its bindings in `src/python`).

| Function | Filtration | Chapter |
|---|---|---|
| `phlite.images.cubical` | the cubical complex of an image or volume (T- or V-construction, sublevel or superlevel sets) | [Cubical complexes](images/cubical.md) |

Every function returns a `phlite.core.Barcode`, and takes a [`phlite.core.Algorithm`](algorithms.md) whose unset
options it fills in with its own defaults (the pipeline names are `images.<function>`, e.g. `images.cubical`).

To compute persistence of a cubical complex you have built yourself (or any other filtered cell complex), use
`phlite.complexes`, as the [image notebook](notebooks.md#an-image-phlitecomplexes) does.
