# Benchmarks

The full, regenerated results are in
[`python-tests/BENCHMARKS.md`](https://github.com/tomchaplin/phlite/blob/main/python-tests/BENCHMARKS.md), with time
and peak memory for every configuration. The suites:

| Suite | Pipeline | Against | Inputs |
|---|---|---|---|
| `bench_clouds_rips.py` | `phlite.clouds.rips` | ripser, giotto-ph | random points in a cube; noisy torus, sphere, wedges of circles and spheres |
| `bench_networks_grpph.py` | `phlite.networks.grpph` | grpphati | random digraphs; rings and lattices |
| `bench_complexes.py` | `phlite.complexes` | PHAT, LoPHAT | Rips complexes, cubical complexes of noise and of an image of rings, a meshed torus, sparse and dense cell complexes |

Each package's pipelines get a suite `bench_<package>_<pipeline>.py` (and fuzz tests `test_<package>_<pipeline>.py`),
which the justfile and the report pick up without other changes (see `python-tests/README.md`).

Each benchmark checks its barcode against the reference before timing, records peak memory (with memray, so Rust
allocations are counted), and runs its first round in a forked process with a time limit, so one pathological
configuration can't stall a whole session.

Headline results, one thread:

- **Vietoris–Rips**: 1–2.2× faster than giotto-ph and 1.4–7× faster than ripser; on sparse matrices from 7% slower
  to 1.3× faster than giotto-ph. On point clouds with long bars the default (heap columns, uncached) uses far more
  memory than giotto-ph (546 MB against 7 MB on a noisy sphere of 400 points), where merged cached columns are faster
  and take 25 MB; on random points the default uses the least.
- **Weighted (DTM) Rips**: 50–400× faster than GUDHI.
- **Alpha and Delaunay–Čech**: level with GUDHI or up to 2.4× faster up to 100,000 points in the plane and 300,000
  in space, but 7–8× slower on a million points in the plane.
- **Directed flag complexes**: 1.4–1.9× faster than flagser, with 2–4× more memory.
- **GrPPH and path homology**: 24–105× and 50–95× faster than grpphati.
- **Flag complexes of graphs**: 3–18× faster than GUDHI; **extended persistence**: 2.5–5.5× faster than GUDHI.
- **Cubical complexes**: against Cubical Ripser, level on large 2D images and 2–2.3× faster on small ones, 1.5–1.7×
  slower in 3D; 3.5–15× faster than GUDHI.
- **Explicit complexes**, against PHAT's twist reduction with a prebuilt matrix: 11–26× faster on Rips complexes,
  1.1–2× faster on sparse cell complexes, level on meshes, PHAT up to 1.4× faster on cubical complexes and dense
  boundaries.
- **Perseus 6** (over Z/2, one thread), in the Rips, flag and cubical suites: up to 1.6× faster than phlite on small
  random Rips complexes, level at 1000 points or more, and phlite 2–3× faster with long bars; on images Perseus is
  6–13× faster in 2D and 1.2–1.6× in 3D (after phlite's filtered discrete gradient for volumes); on random
  (Linial–Meshulam) complexes phlite is 5–7× faster. (`perf-results/benchmarks-2026-11-21.md`,
  `perf-results/perseus-plan-log.md`.)
- **Images, kernels and cokernels**: 1.2–1.8× faster than phimaker ([Maps](../python/maps.md#performance)).
- **Zigzag persistence**: 17–84× faster than Dionysus 2 on sliding-window Rips zigzags
  ([Zigzag](../python/zigzag.md#performance)).
- **Spectral sequences**: over Z, 1–1000× faster than Kenzo; over F_3, milliseconds against seconds for
  spectral-sequence-engine ([Spectral sequences](../python/spectral.md#performance)).
- **Magnitude homology**: 8–100× faster than gramag, level with Perseus ([Magnitude](../python/networks/magnitude.md#performance)).

```sh
just bench           # every benchmark
just bench-check     # compare the quick cases with the saved baselines
just bench-docs      # rerun everything and regenerate BENCHMARKS.md
```

## Profile-guided optimisation

**Only the Linux x86_64 wheels on PyPI** (`manylinux`, which is what `pip install phlite` gets on a 64-bit Intel or
AMD Linux machine) are built with **profile-guided optimisation** (PGO). Every other wheel (macOS, Windows, Linux on
ARM and other architectures, and `musllinux`) is a plain release build: PGO needs a training run on the platform being
built for, which the release workflow only does for Linux x86_64.

To build them, every package is first built instrumented, `python-tests/pgo_train.py` runs every pipeline on
mid-sized random inputs, and the wheels are rebuilt with the profiles, so that LLVM lays out and inlines the hot paths
it saw. On the medium benchmark cases (inputs different from the training run's) this made the wheels about 4%
faster overall (geometric mean), and up to 20% faster for the networks pipelines:

| Suite | PGO time / plain build |
|---|---|
| `networks.pph` | 0.82 |
| `networks.grpph` | 0.84 |
| `networks.dowker`, `networks.ordered_tuples` | 0.90 |
| `networks.directed_flag` | 0.97 |
| `clouds.rips` | 0.97 (0.88–0.90 on structured clouds, 1.03–1.04 on random ones) |
| `images.cubical` | 0.98 |
| `clouds.alpha` | 1.00 |
| explicit complexes (`phlite.complexes`) | 1.07 (cases under 0.1 s; up to 1.18 on small Rips complexes) |

The gains are largest for the lazy oracles (GrPPH, PPH, Dowker), whose coboundaries branch a lot; explicit complexes'
tight loops are already well optimised, and come out slightly slower. How much PGO helps depends on the training run:
one on the benchmarks' and tests' inputs (which need every reference implementation, so can't run in the release
workflow) gave 5% overall and 20–25% on GrPPH and PPH.

### Optimised wheels for other platforms

On macOS, Windows or any other platform, you can build PGO wheels for your own machine from a clone of the repository.
You need [Rust](https://rustup.rs) (with its `llvm-tools` component), [uv](https://docs.astral.sh/uv/) and
[just](https://just.systems); on Windows, run it from Git Bash.

```sh
rustup component add llvm-tools
just pgo-wheels               # instrumented build, training run (about a minute), then optimised wheels in dist/
pip install dist/*.whl        # into your own environment (replacing the wheels from PyPI)
```

`just pgo-profile` does only the profiling, leaving the profile in `pgo/merged.profdata`. To optimise for a different
workload, edit `python-tests/pgo_train.py` to run what you run: LLVM favours whatever the training run does.
