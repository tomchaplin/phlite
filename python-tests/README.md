# python-tests

Fuzz tests and benchmarks for the Python bindings (`phlite_grpph` and `phlite_rips`) against reference implementations:
[ripser.py](https://github.com/scikit-tda/ripser.py) and [giotto-ph](https://github.com/giotto-ai/giotto-ph) for Rips,
and [grpphati](https://github.com/tomchaplin/grpphati) for GrPPH.

This is a separate uv project with its own Python 3.11 environment, because giotto-ph has no wheels after 3.12 and grpphati pins `numpy<1.25` (no wheels after 3.11).
Every recipe first builds both extension modules into that environment in release mode.
You need [`just`](https://github.com/casey/just) and [`uv`](https://github.com/astral-sh/uv).

```sh
just test                 # quick fuzz run (100 examples per test)
just fuzz 20000           # longer fuzz run (20000 examples per test)
just bench-rips           # phlite_rips vs ripser and giotto-ph
just bench-grpph          # phlite_grpph vs grpphati
```

## Fuzz tests

The tests use [Hypothesis](https://hypothesis.readthedocs.io/) to generate inputs, and on failure shrink them to a minimal example.

* `tests/test_rips.py` generates distance matrices: Euclidean point clouds, and small-integer (often non-metric) matrices with lots of ties, sometimes with infinite entries.
  Entries are rounded to float32, since that is what ripser and giotto-ph use internally, so all barcodes can be compared exactly.
  * Every `method` × `algorithm` must give exactly the barcode of ripser (which must agree with giotto-ph).
  * Every representative is checked from first principles with linear algebra over Z/2 (see `tests/mod2.py`):
    it is a cycle, its latest simplex appears at the birth time, it is not a boundary before the death time, and it is a boundary at the death time.
* `tests/test_grpph.py` generates weighted digraphs with half-integer weights (so shortest paths are exact), sometimes with infinite weights.
  * Every `method` × `algorithm` must agree, and the dimension 1 barcode must match grpphati's.
  * H₁ representatives must be cycles made of edges present at the birth time;
    H₀ representatives must pick out the weakly connected components.
  * Passing a networkx graph (with mixed, shuffled node labels) must agree with the integer API, including representatives.

## Benchmarks

Each case runs in a fresh subprocess, so the time and peak memory of one implementation cannot affect another.
Libraries are imported, and the input built, before measuring.
"Peak MB" is the peak resident memory during the computation, minus what was in use when it started.
On Linux this is exact (we reset the kernel's high-water mark `VmHWM` first); elsewhere it falls back to `getrusage`, which under-reports runs that stay below the memory used during setup.
"agrees" checks a fingerprint of the barcode (number of bars and total persistence per dimension) against the reference implementation.

Run either script with `--help` for options, for example:

```sh
just bench-rips --sizes 100 200 300 --max-dim 1 --repeats 5
just bench-rips --all --only phlite      # include the slower phlite variants (standard, homology, lock-free)
just bench-grpph --graph cycle --sizes 100 200 400
just bench-grpph --sizes 50 100 --p 0.3 --json results.json
```

The Rust-level comparison of serial and parallel reduction lives in `scripts/bench_parallel.sh`.
