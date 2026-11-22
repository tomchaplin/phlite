# Over the integers: torsion

Persistent homology over a field forgets torsion: the projective plane has the homology of a point over Q and of a
circle and a sphere over Z/2. Over the integers a filtration has no barcode (its persistence module needn't decompose
into intervals), but everything around one can be computed exactly, and phlite does it for **any filtration**: build
the complex with `modulus=0`.

```python
from phlite.complexes import ChainComplex, FilteredComplex
from phlite import clouds

# Any simplicial filtration, with the signs filled in: here the Rips complex of some points
K = FilteredComplex.from_simplices(clouds.rips_complex(points=points, dimension=3, threshold=1.0), modulus=0)

K.persistence()              # the barcode over Q
K.torsion()                  # {p: barcode over F_p} for the primes p whose barcode differs from Q's
K.persistent_group(1, 0.3, 0.5)   # (rank, torsion) of the image of H_1 at 0.3 in H_1 at 0.5, e.g. (2, [2])
K.spectral_sequence()        # its spectral sequence over Z, a page at a time (see the next chapter)
```

`ChainComplex.from_simplices(..., modulus=0)` and `FilteredComplex.from_simplices(..., modulus=0)` orient simplices
in the usual way; `ChainComplex.from_pairs`, `from_columns` and `from_csc` take your own signed coefficients (building
the complex checks that `∂∂ = 0`). Every filtration in phlite can be exported with its signs:
`clouds.rips_complex`, `alpha_complex`, `delaunay_cech_complex`, `cech_complex`, `witness_complex` (simplices with
their values) and `images.cubical_complex` (a signed CSC matrix for `ChainComplex.from_csc`).

## The barcode over Q and the bad primes

`persistence()` over the integers is the barcode over Q, and `torsion()` is every prime `p` whose barcode over `F_p`
differs from it, with that barcode. These are the primes of the torsion, whether in the homology of one step or only in
the maps between steps: a loop `x`, then a loop `y` and a cell with boundary `p y - x`, has no torsion at any step, but
over `F_p` the first bar dies where over Q it doesn't.

They are found as **Perseus** (Mischaikow and Nanda) does: a fraction-free reduction over Z (each column reduced by
integer combinations and divided by its content) has the pivots of the reduction over Q; a prime dividing none of the
pivots used and none of the contents divided out sees the same reduction mod p, so the same barcode. Only the primes
dividing them are candidates, and each is checked by reducing mod p. The arithmetic is checked `i128`: an overflow is
an error, never a wrong answer.

## Groups: persistent groups and spectral sequences

`persistent_group(dim, start, end)` is the image of `H_dim(K_start)` in `H_dim(K_end)` as `(rank, torsion)`, and the
[spectral sequence](spectral.md) of the filtration gives its pages as groups, with generators. Both are exact quotients
of lattices, by integer elimination and Smith normal forms, on the cells of the window of steps an entry depends on:
fine up to a few thousand cells per degree.

## In Rust

`phlite::integral_persistence::integral_persistence_of(&boundary, &cells, degree)` takes any boundary oracle with
`phlite::fields::Integer` coefficients (lazy or explicit) and its cells in filtration order;
`integral_persistence(columns, degrees)` takes explicit columns. Groups come from `phlite::integral`
(`quotient`, `persistent_group`, `page_entry`, generators and classification), and spectral sequences of lazy
complexes from `phlite::spectral::LocalSpectral` over any `StepFiltered` oracle.

## Checked against

* The barcode over `F_p`, for every prime `p ≤ 13`, of random complexes with random integer coefficients: equal to
  Q's barcode exactly when `p` isn't among the bad primes, and equal to `torsion()[p]` when it is.
* Known homology: projective spaces, lens spaces, the torus and the Klein bottle, products (Künneth), Hopf
  fibrations, and every filtration phlite exports, brought over Z.
* [Kenzo](https://github.com/miguelmarco/kenzo), for spectral sequences over Z (next chapter).

`python-tests/tests/test_spectral_sequences.py` and `test_complexes_spectral.py`.
