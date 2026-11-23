# Spectral sequences

A filtration `F_0 C ⊆ F_1 C ⊆ ...` of a chain complex has a spectral sequence: pages `E^r_p` (in each degree `k`)
with differentials `d^r: E^r_p -> E^r_{p-r}` (degree `k - 1`), each page the homology of the last, converging to
`H(C)`. `E^1` is the homology of the steps `F_p / F_{p-1}`; later pages see further across the filtration. phlite
computes them for any filtered complex, over a field or the integers, **one page at a time**, with groups, generators
and differentials on elements.

```python
from phlite.complexes import FilteredComplex

ss = FilteredComplex.from_simplices(pairs).spectral_sequence()   # or modulus=0 for the integers
page = ss.next()           # E^1; printing it shows a table (steps down, degrees across) and its differentials
page = ss.next()           # E^2
ss.page(3)                 # E^3 (computing the pages before it if need be)
ss.infinity()              # E^∞
ss.degenerates_at          # the first page after which nothing changes, once known
for page in ss: ...        # every page, until it degenerates

page[p, k]                 # the entry at step p in degree k: its dimension, or (rank, torsion) over the integers
g = page.group(p, k)       # as a group: str(g) ("Z ⊕ Z/2"), g.rank, g.torsion, g.order, g.is_zero
g.generators               # representing chains (cell -> coefficient), with their orders
g.classify(chain)          # the coordinates of a chain's class in the generators
g.d(chain)                 # d^r of its class, in the generators of the target
page.plot()                # the page as a grid (matplotlib), arrows for the differentials
```

In a notebook a page shows as an HTML table. Steps are positions among the distinct filtration values, so `E^r`
concerns differentials across `r` values.

Other spectral sequences in phlite:

* `phlite.networks.magnitude_path_spectral_sequence(G, l_max)`: the magnitude-path spectral sequence of a digraph
  (Asao; Hepworth and Roff), whose `E^1` is [magnitude homology](networks/magnitude.md) and whose `E^2` diagonal is
  path homology, laid out as gramag lays it out (lengths down, degrees across).
* Double complexes, cellular Serre and Atiyah–Hirzebruch spectral sequences, filtrations by skeleta: anything given
  by a filtered chain complex, through `ChainComplex` (the total complex, filtered).

## How

**Over a field** the whole spectral sequence is determined by the persistence barcode (a bar `[b, d)` of degree `k`,
counted in steps, is a generator of `E^r` at step `b` in degree `k` and at step `d` in degree `k + 1` for every
`r <= d - b`, and `d^{d-b}` cancels the two; an essential bar lives on every page), so phlite computes the barcode
once, with all of its usual optimisations, and each entry is a lookup. Generators are computed exactly, when first
asked for, from the cells of the entry's window of steps.

**Over the integers** there is no barcode. `E^r_p` depends only on the quotient complex `F_{p+r-1} / F_{p-r}`, a
window of `2r - 1` steps, so each entry is computed exactly from its window alone, as a quotient of lattices
`Z^r_p / (Z^{r-1}_{p-1} + d Z^{r-1}_{p+r-1})`, by integer elimination and Smith normal forms in checked `i128`
(`phlite::integral`). Entries of one page are independent, and are computed in parallel.

### In Rust

`phlite::spectral::SpectralBackend` answers for one entry at a time (`E^r_p` in degree `k`, and the image of `d^r`
out of it), so a spectral sequence can be driven a page at a time from anywhere:

* `BarcodeSpectral`: over a field, from a barcode computed by any reduction of any boundary oracle;
* `LocalSpectral`: over the integers (or anywhere without a barcode), for any `StepFiltered` complex: a boundary oracle
  that can list its cells by degree and step. Lazy complexes work, since only the cells of each window are listed.

Every spectral sequence over a field with finite pages is that of a filtered complex, so these cover those given by a
filtered chain complex; those known only from an exact couple are out of reach.

## Checked against

* **Known spectral sequences**: projective spaces and lens spaces by skeleta (`E^1` the cellular chains), Hopf
  fibrations `S^1 -> S^{2n+1} -> CP^n`, the torus and the Klein bottle as circle bundles over the circle, the
  Künneth spectral sequence of `RP^2 × RP^2`, both filtrations of a double complex converging to the same homology;
  Hepworth and Roff's computations of the magnitude-path spectral sequence.
* **Properties of every spectral sequence**, fuzzed over Z and Z/p: each page is the homology of the last
  (`rank E^{r+1} = rank E^r - rank d^r out - rank d^r in`), `E^∞` sums to `H(C)` (and its order is `|H_k|` when
  finite), universal coefficients on `E^1`, `d^r ∘ d^r = 0` on generators, invariance under reparametrising the
  filtration and reordering cells.
* **[Kenzo](https://github.com/miguelmarco/kenzo)** (Ana Romero's spectral sequences, over Z) and
  **[spectral-sequence-engine](https://pypi.org/project/spectral-sequence-engine/)** (over F_p): every entry of pages
  1–4 agrees, on lens spaces, Hopf models, Künneth products, random complexes with torsion and lower-star meshes.
  **[PerMaViss](https://github.com/atorras1618/PerMaViss)** (Mayer–Vietoris spectral sequences of Rips complexes):
  the same barcodes on its own example.

`python-tests/tests/test_spectral_sequences.py`, `test_complexes_spectral.py`; the comparison harnesses are
`python-tests/benchmarks/spectral_compare.py` and `permaviss_compare.py`.

## Performance

Pages `E^1`–`E^4`, every entry (from `perf-results/spectral-comparison-2026-11-21.md`):

| complex | cells | phlite over Z | Kenzo | phlite over F_3 | spectral-sequence-engine |
|---|---:|---:|---:|---:|---:|
| RP^3 × RP^3 (Künneth) | 16 | 0.3 ms | 1.0 ms | 0.3 ms | 3.1 ms |
| random over Z, 150 pieces | 218 | 5.0 ms | 1.07 s | 0.6 ms | 235.9 ms |
| torus mesh 10×10, lower star | 600 | 80.5 ms | 26.90 s | 0.9 ms | 1.88 s |
| random over Z, 500 pieces | 737 | 48.6 ms | 46.72 s | 1.6 ms | 2.80 s |
| torus mesh 16×16, lower star | 1536 | 933.9 ms | over an hour | 1.6 ms | 13.52 s |

PerMaViss took 203 s on its example, against phlite's 2.1 ms for the same barcodes. What Kenzo has and phlite doesn't:
effective homology, for spectral sequences of infinite objects (Serre spectral sequences of fibrations,
Eilenberg–Moore), and the extension problem (the filtration of `H(C)`). phlite's weak point is large windows over Z,
which are eliminated densely.
