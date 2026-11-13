# python-tests

Fuzz tests ([Hypothesis](https://hypothesis.readthedocs.io/)), benchmarks
([pytest-benchmark](https://pytest-benchmark.readthedocs.io/)) and performance regression checks for the Python bindings
(`phlite_grpph`, `phlite_rips` and `phlite_py`) against reference implementations:
[ripser.py](https://github.com/scikit-tda/ripser.py) and [giotto-ph](https://github.com/giotto-ai/giotto-ph) for Rips,
[grpphati](https://github.com/tomchaplin/grpphati) for GrPPH,
and [PHAT](https://github.com/blazs/phat) and [LoPHAT](https://github.com/tomchaplin/lophat) for explicit boundary matrices.
The Rust reduction itself is benchmarked with [criterion](https://bheisler.github.io/criterion.rs/book/), in
`phlite_rips/benches` and `phlite_grpph/benches`. The latest results of both are in [BENCHMARKS.md](BENCHMARKS.md).

You need [`just`](https://github.com/casey/just), [`uv`](https://github.com/astral-sh/uv) and a C++ compiler (for PHAT).
The `justfile` at the root of the repository runs everything, Rust and Python:

```sh
just test                 # all tests: Rust, then Python (fuzzing briefly, 100 examples per test)
just fuzz 20000           # fuzz for longer (20000 cases per Rust property test and examples per Python test)
just bench                # every benchmark: Rust (criterion), then Python (pytest-benchmark)
just bench-py -k grid     # Python benchmarks, with pytest options
just bench-rust 'rips'    # Rust benchmarks, with criterion options (a filter, --baseline, ...)
just bench-check          # check for performance regressions against the saved baselines (a few minutes)
just check                # `test` then `bench-check`
just bench-baseline       # save new baselines for bench-check
just bench-docs           # rerun every benchmark and regenerate BENCHMARKS.md
```

This folder's `justfile` has the Python side of these (`test`, `fuzz`, `bench`, `bench-check`, `bench-baseline`,
`bench-docs`).

### Two environments

The references need different Pythons, so there are two uv projects here, and every recipe first builds the extension
modules (in release mode) into the environment that uses them:

| Environment | Python | References | Tests | Benchmark suite |
|---|---|---|---|---|
| `.` | 3.11 | ripser, giotto-ph, grpphati | `test_rips.py`, `test_grpph.py`, phlite_py's `test_real_grpphati` | `rips`, `grpph` |
| `latest/` | 3.13 | PHAT, LoPHAT | `test_sparse.py`, `test_invariants.py`, phlite_py's own tests | `sparse` |

giotto-ph has no wheels after 3.12 and grpphati pins `numpy<1.25` (no wheels after 3.11), while LoPHAT 0.14 needs 3.12 or later.
PHAT's only PyPI release is an sdist whose metadata disagrees with its version, which uv and pip refuse,
so `just venv` builds it from the unpacked sdist instead (`just install-phat`, checking its SHA-256).

## Fuzz tests

The tests use [Hypothesis](https://hypothesis.readthedocs.io/) to generate inputs, and on failure shrink them to a minimal example.
Every algorithm configuration (`tests/grid.py`: algorithm, column representation, caching, skipping, dimension order,
storing R, threads and storage) is checked in all three packages. Each test has a time limit (`pytest-timeout`, 300 s
plus half a second per example, or `PHLITE_TEST_TIMEOUT`), so a bug that makes a reduction loop forever fails the run
instead of hanging it.

* `tests/test_rips.py` generates distance matrices: Euclidean point clouds, and small-integer (often non-metric) matrices with lots of ties, sometimes with infinite entries.
  Entries are rounded to float32, since that is what ripser and giotto-ph use internally, so all barcodes can be compared exactly.
  * Every `method` × `algorithm` must give exactly the barcode of ripser (which must agree with giotto-ph), and the
    whole configuration grid must agree.
  * The rank invariant: for diameters s ≤ t, the bars alive from s to t count the cycles at s that are not yet boundaries at t.
  * Permuting the points, or changing distances by a strictly increasing function, changes the bars accordingly.
  * Every representative is checked from first principles with linear algebra over Z/2 (see `tests/mod2.py`):
    it is a cycle, its latest simplex appears at the birth time, it is not a boundary before the death time, and it is a boundary at the death time.
* `tests/test_grpph.py` generates weighted digraphs with half-integer weights (so shortest paths are exact), sometimes with infinite weights.
  * Every `method` × `algorithm` must agree, and the dimension 1 barcode must match grpphati's.
  * H₁ representatives must be cycles made of edges present at the birth time;
    H₀ representatives must pick out the weakly connected components.
  * Passing a networkx graph (with mixed, shuffled node labels) must agree with the integer API, including representatives.
  * Renaming nodes changes nothing, and doubling the weights doubles the bars.
* `tests/test_sparse.py` generates random filtered cellular chain complexes over Z/2, up to dimension 3: each k-cell's boundary is a random (k-1)-cycle,
  so there are multi-edges, loops, zero boundaries and long boundaries. Cells are in a random valid filtration order, with few distinct values (so many ties and zero-length pairs).
  * PHAT (with four reduction/representation pairs) and LoPHAT (with and without clearing and anti-transposition, on 1 and 2 threads) must agree.
  * With the index filtration, every `method` × `Algorithm` configuration (standard, clearing and parallel; merged, heap and sorted columns, cached or not; skipping on and off;
    detected, ascending and descending dimension orders; locking and lock-free storage) must give exactly PHAT's pairs and essential cells.
  * With filtration values (as a list or numpy array, via `from_pairs`, `from_columns` or `from_csc`, with and without `max_dim`), phlite must give PHAT's pairs of positive length.
  * Representatives must be cycles in decreasing filtration order starting at the birth, a boundary at the death and not before, and never a boundary if essential.
  * A few larger complexes (~1600 cells) give the parallel algorithms something to race on.
* `tests/test_invariants.py` generates random cellular complexes over Z/2, Z/3 and Z/5 with cells listed out of
  filtration order (so a cell's index is not its position), and checks properties needing no other library
  (with linear algebra over Z/p, see `tests/modp.py`):
  * every configuration agrees with a reference reduction, and representatives are cycles, led by the birth, that
    become boundaries exactly at the death;
  * relabelling cells and strictly increasing changes of values change the bars accordingly;
  * bars pair a cell with a later one of the dimension above; the essential bars give the Betti numbers and the Euler
    characteristic; and the rank invariant holds;
  * the barcode's accessors agree with each other, and with `max_dim` and `representatives`;
  * cubical complexes of ~20k cells (three levels of the pivot columns' bit tree) agree with PHAT.

The Rust crates have their own property tests ([proptest](https://proptest-rs.github.io/proptest/)): every column
representation against a simple model, `ElementMap` against `BTreeMap`, and phlite_py's renumbering and every clearing
configuration on random complexes over Z/2 and Z/3.

## Benchmarks

### Python (pytest-benchmark)

`benchmarks/bench_*.py` are pytest-benchmark suites, each comparing a binding with its references:

* `bench_rips.py`: `phlite_rips` vs ripser and giotto-ph, on random point clouds, up to 2000 points, and on noisy samples of a torus,
  spheres and a wedge of circles (from `benchmarks/datasets.py`, using tadasets), which have long bars.
* `bench_grpph.py`: `phlite_grpph` vs grpphati, on random weighted digraphs, a ring with chords and a lattice with holes
  (on the largest of each, only phlite runs, since grpphati needs over 5 GB or minutes).
* `bench_sparse.py`: `phlite` vs PHAT and LoPHAT, on explicit boundary matrices given to every implementation as the
  same list of columns in filtration order: Rips complexes, cubical grids with a lower-star filtration, and sparse and
  dense random cellular complexes; and, with long bars, Rips complexes of a noisy torus and wedge of circles, images
  of rings, and a triangulated torus filtered by height. Plain times include loading the columns from Python; `(reduce)` variants build the
  matrix before the clock starts. Variants cover every column representation (including `pivot`, the default), cached
  or not, with and without storing R.

Each suite's largest cases are marked `large` and timed once each (deselect them with `-m "not large"`); every
implementation runs on them, except as noted, so that they show which phlite configuration is best at scale.

Each benchmark is one implementation on one case, so the usual pytest options select them, e.g.
`just bench-py -k "grid and phlite"` or `just bench-py -m quick`. Each one (see `benchmarks/common.py`):

1. builds its input, untimed, and runs once to check its barcode against the reference implementation's (a
   fingerprint: the number of bars and total persistence in each dimension) and to measure peak memory with
   [memray](https://bloomberg.github.io/memray/): the most heap memory allocated at once during the run, including
   native allocations (such as Rust's, on any thread), stored in the results as `extra_info["peak_mb"]`. Since every
   benchmark runs in the same process, this is more reliable than resident memory, which doesn't grow when a run
   reuses memory freed by an earlier one (memray works on Linux and macOS). This first run is in a forked child
   process, killed after `PHLITE_BENCH_TIMEOUT` seconds (default 600; 0 for no limit), when the benchmark is skipped
   rather than holding up the session: the time is spent in native code, which a Python timeout couldn't interrupt;
2. is timed over `PHLITE_BENCH_ROUNDS` runs (default 3) with `benchmark.pedantic`, building a fresh input before each,
   untimed, since some implementations reduce their input in place.

Parallel implementations use `PHLITE_BENCH_THREADS` threads (default 4, or fewer cores).
`benchmarks/conftest.py` leaves the hostname and the CPU's current clock speed out of the saved machine info.

### Rust (criterion)

`phlite_rips/benches/reduction.rs` and `phlite_grpph/benches/reduction.rs` time the clearing reduction alone on a Rips
and a GrPPH coboundary matrix: serial with merged, heap and sorted columns, each cached or not, and storing R, and
parallel with locking and lock-free storage on 1, 2 and 4 threads. Run them with `just bench-rust` (any arguments go to criterion, e.g. a name filter); the reports
are in `target/criterion`.

### Regressions and BENCHMARKS.md

* `just bench-check` runs the Python benchmarks marked `quick` with pytest-benchmark's
  `--benchmark-compare --benchmark-compare-fail=min:50%`, against the saved runs in `benchmarks/baselines/` (one per
  environment, since pytest-benchmark keys saved runs by platform and Python version), failing if any is more than 50%
  slower. It then runs the criterion benchmarks with `--baseline-lenient quick`, which reports how each changed against
  the saved criterion baseline (criterion doesn't fail on a regression). Peak memory is recorded but not checked.
* `just bench-baseline` saves new baselines for both, after an intended change in performance or on a new machine.
  Baselines are only meaningful on the machine they were recorded on; the committed Python ones are from the machine in
  BENCHMARKS.md, and criterion's are local (in `target/criterion`).
* `just bench-docs` runs every benchmark, saves the Python results in `benchmarks/results/` (pytest-benchmark JSON,
  which `pytest-benchmark compare` can compare), and writes BENCHMARKS.md from them and criterion's results with
  `benchmarks/report.py`.
