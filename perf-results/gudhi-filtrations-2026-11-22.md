# Witness and periodic cubical complexes against GUDHI 3.13 (2026-11-22)

`python-tests/benchmarks/gudhi_filtrations_compare.py`, 8-core laptop, best of three, from the input arrays. phlite's
witness complexes enumerate witnesses in parallel (rayon); everything else is on one thread. The witness rows were run
while the playground was compiling in the background, which slowed GUDHI's relaxed complex too (13.1 s on the first
row, against 9.9 s on a quiet run). GUDHI's strong witness complex is far slower than its relaxed one, so its cases are
small.

## Witness complexes: phlite against GUDHI (dimensions 0 and 1, Z/2; ✓: identical barcodes)

| complex | simplices | phlite | GUDHI | agree |
|---|---:|---:|---:|:-:|
| witness, 50 of 2000 points in R^3, α² ≤ inf | 20875 | 501.6 ms | 13.10 s | ✓ |
| witness, 100 of 10000 points in R^3, α² ≤ 0.02 | 3040 | 15.0 ms | 96.3 ms | ✓ |
| witness, 200 of 20000 points in R^3, α² ≤ 0.01 | 5526 | 43.6 ms | 189.4 ms | ✓ |
| strong witness, 15 of 200 points in R^3, α² ≤ inf | 575 | 2.8 ms | 1.91 s | ✓ |
| strong witness, 20 of 300 points in R^3, α² ≤ inf | 1350 | 8.1 ms | 145.11 s | ✓ |

## Periodic cubical complexes: phlite against GUDHI's PeriodicCubicalComplex (every dimension)

| image | phlite | GUDHI | agree |
|---|---:|---:|:-:|
| periodic 256×256, uniform noise, T | 84.0 ms | 90.4 ms | ✓ |
| periodic 1024×1024, uniform noise, T | 2.02 s | 2.55 s | ✓ |
| periodic 32×32×32, uniform noise, T | 151.4 ms | 175.7 ms | ✓ |
| periodic 64×64×64, uniform noise, T | 1.28 s | 1.49 s | ✓ |
