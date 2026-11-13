# Benchmarks

The full, regenerated results are in
[`python-tests/BENCHMARKS.md`](https://github.com/tomchaplin/phlite/blob/main/python-tests/BENCHMARKS.md), with time
and peak memory for every configuration. The suites:

| Suite | phlite package | Against | Inputs |
|---|---|---|---|
| `bench_rips.py` | `phlite_rips` | ripser, giotto-ph | random points in a cube; noisy torus, sphere, wedges of circles and spheres |
| `bench_grpph.py` | `phlite_grpph` | grpphati | random digraphs; rings and lattices |
| `bench_sparse.py` | `phlite` | PHAT, LoPHAT | Rips complexes, cubical complexes of noise and of an image of rings, a meshed torus, sparse and dense cell complexes |

Each benchmark checks its barcode against the reference before timing, records peak memory (with memray, so Rust
allocations are counted), and runs its first round in a forked process with a time limit, so one pathological
configuration can't stall a whole session.

Headline results, one thread:

- **Vietoris–Rips**: on a par with ripser and giotto-ph; faster in dimension 2 on random points. The default (heap
  columns, uncached) favours memory; merged cached columns are up to 1.9× faster on point clouds with long bars, at
  about twice the memory.
- **GrPPH**: 25–100× faster than grpphati.
- **Explicit complexes**, against PHAT's twist reduction with a prebuilt matrix: 7–13× faster on Rips complexes,
  1.1–2× faster on sparse cell complexes, level on dense boundaries and meshes, PHAT 1.1–1.3× faster on cubical
  complexes.

```sh
just bench           # every benchmark
just bench-check     # compare the quick cases with the saved baselines
just bench-docs      # rerun everything and regenerate BENCHMARKS.md
```
