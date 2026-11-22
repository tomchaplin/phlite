# Images, kernels, cokernels and zigzag persistence against phimaker and Dionysus (2026-11-22)

`python-tests/benchmarks/ikc_zigzag_compare.py`, 8-core laptop. Times exclude building inputs and passing them between processes; phlite best of three, serial; phimaker's `compute_ensemble` (its threads), best of three; Dionysus once. The 20000-point zigzag was stopped (Dionysus past the two-hour limit).

## Images, kernels, cokernels (dimensions 0 and 1; ✓: identical barcodes)

| complex | simplices | phlite | phlite six-pack + reps | phlite (lazy Rips, from points) | phimaker | agree |
|---|---:|---:|---:|---:|---:|:-:|
| Rips, 100 points, r ≤ 0.25 | 2951 | 5.4 ms | 8.4 ms | 11.2 ms | 8.1 ms | ✓ |
| Rips, 200 points, r ≤ 0.18 | 7997 | 16.3 ms | 25.7 ms | 34.2 ms | 19.8 ms | ✓ |
| Rips, 400 points, r ≤ 0.13 | 20702 | 45.8 ms | 67.2 ms | 102.3 ms | 56.4 ms | ✓ |
| Rips, 800 points, r ≤ 0.09 | 37942 | 75.7 ms | 121.6 ms | 184.2 ms | 108.2 ms | ✓ |
| Rips, 1600 points, r ≤ 0.065 | 82093 | 215.7 ms | 298.6 ms | 454.4 ms | 315.1 ms | ✓ |
| torus mesh 30×30, lower star | 5400 | 6.9 ms | 11.0 ms | — | 12.5 ms | ✓ |
| torus mesh 60×60, lower star | 21600 | 30.7 ms | 48.5 ms | — | 53.4 ms | ✓ |

## Zigzag: phlite (FastZigzag) against Dionysus 2

| zigzag | operations | intervals | phlite | Dionysus | agree |
|---|---:|---:|---:|---:|:-:|
| sliding Rips, 300 points, window 20 | 6611 | 3350 | 3.5 ms | 60.1 ms | ✓ |
| sliding Rips, 1000 points, window 30 | 126182 | 63316 | 144.9 ms | 12.18 s | ✓ |
| sliding Rips, 5000 points, window 40 | 1674242 | 839034 | 4.94 s | 1900.53 s | ✓ |

## Zigzag: phlite against FastZigzag (fzz), by cohomology and by homology

fzz as released reduces by homology (PHAT twist); "fzz (cohomology)" is fzz patched to call
`compute_persistence_pairs_dualized<twist_reduction>`. fzz timed on its computation only. 10-minute cap.

| zigzag | operations | phlite coh. | fzz coh. | phlite hom. | fzz hom. | agree |
|---|---:|---:|---:|---:|---:|:-:|
| sliding Rips, 300 points, window 20 | 6611 | 4.5 ms | 2.3 ms | 4.6 ms | 1.5 ms | ✓ |
| sliding Rips, 1000 points, window 30 | 126182 | 157.5 ms | 98.8 ms | 630.8 ms | 458.1 ms | ✓ |
| sliding Rips, 2000 points, window 35 | 461837 | 876.5 ms | 656.2 ms | 320.85 s | 324.48 s | ✓ |
| sliding Rips, 5000 points, window 40 | 1674242 | 4.91 s | 3.94 s | > 10 min | > 10 min | ✓ (coh.) |

Union-find in dimension 0 (phlite, cohomology, 126k operations): 158 ms with, 156 ms without.
