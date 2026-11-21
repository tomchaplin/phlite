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

## The complex itself

`phlite.images.cubical_complex(image)` lists the cubical complex as an explicit chain complex with its oriented
boundary (coefficients ±1), as arrays for `phlite.complexes.ChainComplex.from_csc`, with each cell's value and
coordinates: for your own computations, such as persistence over the integers (`modulus=0`: the barcode over Q and the
primes of the torsion, see [explicit complexes](complexes.md#over-the-integers)).
