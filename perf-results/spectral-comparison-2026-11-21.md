# Spectral sequences: phlite against other software (2026-11-21)

Harnesses: `python-tests/benchmarks/spectral_compare.py` (Kenzo, spectral-sequence-engine) and
`python-tests/benchmarks/permaviss_compare.py` (PerMaViss). Same machine as the other tables.

## Pages E^1..E^4, every entry

Over Z: phlite (`ChainComplex(..., modulus=0)...spectral_sequence(integer_steps=True)`) against Kenzo
(miguelmarco/kenzo with Ana Romero's spectral sequences, `eff-spsq-basis-dvs`, SBCL 2.6.10). Over F_3: phlite against
spectral-sequence-engine 0.5.0 (pure Python). ✓: every entry of every page agrees. phlite and engine times are from a
quiet run; Kenzo's from a run in which PerMaViss was also busy on three other cores (Kenzo is single-threaded), with a
one-hour limit for all its cases together, which the last three didn't fit in.

| complex | cells | torsion entries | phlite Z | Kenzo Z | agree | phlite F_3 | engine F_3 | agree |
|---|---:|---:|---:|---:|:-:|---:|---:|:-:|
| lens L(5), 7-dim, by skeleta | 8 | 9 | 1.1 ms | 1.0 ms | ✓ | 1.0 ms | 3.3 ms | ✓ |
| Hopf S^1 → S^9 → CP^4 | 10 | 0 | 0.6 ms | 1.0 ms | ✓ | 0.5 ms | 2.7 ms | ✓ |
| RP^3 × RP^3 (Künneth) | 16 | 22 | 0.3 ms | 1.0 ms | ✓ | 0.3 ms | 3.1 ms | ✓ |
| L(3) × RP^4 (Künneth) | 30 | 30 | 1.0 ms | 2.0 ms | ✓ | 0.4 ms | 7.8 ms | ✓ |
| random over Z, 40 pieces | 57 | 26 | 1.1 ms | 27.0 ms | ✓ | 0.4 ms | 24.1 ms | ✓ |
| random over Z, 150 pieces | 218 | 64 | 5.0 ms | 1.07 s | ✓ | 0.6 ms | 235.9 ms | ✓ |
| torus mesh 6×6, lower star | 216 | 0 | 8.8 ms | 1.34 s | ✓ | 0.5 ms | 267.5 ms | ✓ |
| torus mesh 10×10, lower star | 600 | 0 | 80.5 ms | 26.90 s | ✓ | 0.9 ms | 1.88 s | ✓ |
| random over Z, 500 pieces | 737 | 117 | 48.6 ms | 46.72 s | ✓ | 1.6 ms | 2.80 s | ✓ |
| random over Z, 2000 pieces | 2890 | 185 | 1.31 s | (limit) | — | 3.3 ms | — | — |
| torus mesh 16×16, lower star | 1536 | 0 | 933.9 ms | (limit) | — | 1.6 ms | 13.52 s | ✓ |
| torus mesh 24×24, lower star | 3456 | 0 | 14.75 s | (limit) | — | 3.4 ms | — | — |

(The engine was only given complexes of up to 2,500 cells.)

Notes:
* Kenzo needed two workarounds to run: ignoring SBCL's type warnings when compiling (ASDF otherwise fails the build),
  and reloading `src/anromero/spectral-sequences.lisp` after loading Kenzo, because the finite-spaces module, loaded
  later, redefines `SUBMATRIX` with a different arity, which breaks the spectral sequences.
* Kenzo's matrices are `fixnum` arrays: on a random complex whose input coefficients reached 1.6×10⁷ (an earlier,
  denser version of the random generator) its Smith normal form overflowed (entries past 10²⁰) with a type error.
  phlite checks for overflow in `i128` and picks least-modulus pivots; the table uses inputs with coefficients below
  10⁴, which Kenzo handles.
* phlite over a field reads the pages off one barcode, so the F_3 column is near-constant; over Z each entry is a
  dense lattice computation in its window of steps, which is what grows: on lower-star meshes with only ten levels
  the windows hold most of the complex.

## PerMaViss (persistence Mayer-Vietoris spectral sequence)

PerMaViss 0.0.2 computes Vietoris-Rips persistence through the persistence Mayer-Vietoris spectral sequence of a
cubical cover; compared with phlite's `clouds.rips` (serial) over Z/2 up to `max_r`.

| point cloud | filtration | bars per dimension | PerMaViss | phlite | agree |
|---|---|---|---:|---:|:-:|
| cube, 130 points (PerMaViss's own example) | r < 0.36, dims 0–2 | [130, 49, 6] | 203.2 s | 2.1 ms | ✓ |
| cube, 300 points | r < 0.3, dims 0–2 | | > 45 min (stopped) | | |

## Not compared

* Macaulay2's SpectralSequences (filtered complexes of modules over polynomial rings): Macaulay2 isn't installed.
* Sage has no spectral sequences of its own, only an interface to Kenzo (not installed in Sage here).
