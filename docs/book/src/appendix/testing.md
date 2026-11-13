# Testing

Every algorithm, representation and option must give the same barcode, and that is what most of the tests check,
from three directions: against reference libraries, against invariants that need no reference, and component by
component.

## Against references (Python, Hypothesis)

`python-tests/` generates random inputs with [Hypothesis](https://hypothesis.readthedocs.io/), which shrinks a failure
to a minimal example.

- **Explicit complexes** (`test_sparse.py`): random cellular complexes over Z/2 (multi-edges, loops, empty and long
  boundaries, few distinct values so many ties) against PHAT and LoPHAT, for **every configuration** in the grid
  (algorithm × columns × caching × skipping × dimension order × store R × threads × storage), both methods, every input
  format.
- **Vietoris–Rips** (`test_rips.py`): random distance matrices (point clouds, small-integer matrices with ties, infinite
  entries) against ripser and giotto-ph.
- **GrPPH** (`test_grpph.py`): random weighted digraphs against grpphati.

## Invariants (no reference needed)

`test_invariants.py` uses cellular complexes over Z/2, Z/3 and Z/5 with cells listed **out of filtration order**, and
checks properties every barcode must have, with exact linear algebra over Z/p (`tests/modp.py`):

- **The rank invariant**: for values \\(s \le t\\), the bars alive from \\(s\\) to \\(t\\) number
  \\(\dim(Z_d(K_s) + B_d(K_t)) - \dim B_d(K_t)\\). This alone determines the barcode. (Also for Rips.)
- **Betti numbers and the Euler characteristic** from the essential bars.
- **Representatives** are cycles in the right dimension, led by the birth cell, that become boundaries exactly at the
  death.
- **Invariance**: relabelling cells, strictly increasing changes of values (and of distances for Rips, weights for
  GrPPH) change the bars accordingly.
- **Bar structure**, and agreement between the barcode's accessors.

## Components (Rust, proptest)

- Every **column representation**, driven through the whole column API by random operations, against a plain map from
  row to coefficient (Z/2 and Z/3, with and without row ranks, through `reverse()`).
- `ElementMap` and `ElementSet` against `BTreeMap` and `BTreeSet`, for every storage.
- phlite_py's **renumbering by rank** and every clearing configuration on random simplicial complexes.

## Running them

```sh
just test            # everything, briefly (100 examples per Hypothesis test)
just fuzz 20000      # 20000 cases per property test and examples per Hypothesis test (~40 minutes)
```

Every test has a time limit (`pytest-timeout`), so a bug that makes a reduction loop forever fails the run rather than
hanging it.

## Were they worth it?

The fuzz tests have found real bugs: a panic for dimension orders beyond the matrix, births of skipped pairs reported
as essential under some dimension orders, and wrong representatives when a restricted basis met the single-lookup pivot
table (see [full basis](../internals/bookkeeping.md#full-basis)). Deliberately reintroduced bugs in the bit tree, the
"same value" check and coefficient cancellation are all caught.
