"""Shared helpers for the pytest-benchmark suites (`bench_*.py`), which import this module.

Every benchmark is one implementation on one case. `run(benchmark, ...)`:

1. builds the input (untimed) and runs the implementation once under memray, recording the peak heap memory of that
   run in the benchmark's `extra_info["peak_mb"]`, and checking its barcode against the reference implementation's;
2. times `ROUNDS` more runs with `benchmark.pedantic`, building a fresh input before each (untimed), since some
   implementations reduce their input in place.

Environment variables: `PHLITE_BENCH_THREADS` (threads for parallel implementations, default min(4, cores)) and
`PHLITE_BENCH_ROUNDS` (timed runs per benchmark, default 3).
"""

import math
import os
import tempfile

import memray

THREADS = int(os.environ.get("PHLITE_BENCH_THREADS", min(4, os.cpu_count() or 1)))
ROUNDS = int(os.environ.get("PHLITE_BENCH_ROUNDS", 3))


# ====== Peak memory ======================================================


def peak_heap_mb(function):
    """Call `function()`, returning its result and the peak of the memory it had allocated at once, in MB.

    Measured with memray, which also sees native allocations (so those made in Rust, on any thread). Unlike resident
    memory, this isn't hidden when a run reuses memory the process already holds from earlier benchmarks.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "memray.bin")
        with memray.Tracker(path):
            result = function()
        peak = memray.FileReader(path).metadata.peak_memory
    return result, peak / 1e6


# ====== Barcodes =========================================================


def fingerprint(dims):
    """Summarise a barcode (a list, per dimension, of (birth, death) bars) so implementations can be compared:
    the number of finite and infinite bars, and the total finite persistence, ignoring zero-length bars."""
    out = []
    for bars in dims:
        bars = [(float(b), float(d)) for b, d in bars if b < d]
        finite = [d - b for b, d in bars if d != math.inf]
        out.append([len(finite), len(bars) - len(finite), round(sum(finite), 6)])
    return out


def _resolve(result):
    """Implementations may return a function computing the fingerprint, so that isn't timed."""
    return result() if callable(result) else result


# ====== Running a benchmark ==============================================


def run(benchmark, *, group, implementation, make_input, reference=None, rounds=None):
    """Benchmark `implementation(input)` on inputs from `make_input()`, checking it agrees with `reference()`.

    `reference` returns the reference fingerprint (and should be cached, since every benchmark of a case asks for it);
    leave it out for the reference implementation itself. `rounds` overrides `ROUNDS`, for cases too slow to repeat.
    """
    benchmark.group = group
    benchmark.extra_info["threads"] = THREADS

    data = make_input()
    result, peak = peak_heap_mb(lambda: implementation(data))
    benchmark.extra_info["peak_mb"] = round(peak, 1)
    result = _resolve(result)
    del data
    if reference is not None:
        assert result == reference(), "barcode differs from the reference implementation's"
    benchmark.extra_info["fingerprint"] = result

    rounds = ROUNDS if rounds is None else min(rounds, ROUNDS)
    benchmark.pedantic(implementation, setup=lambda: ((make_input(),), {}), rounds=rounds, iterations=1)

