"""Shared helpers for the pytest-benchmark suites (`bench_*.py`), which import this module.

Every benchmark is one implementation on one case. `run(benchmark, ...)`:

1. builds the input (untimed) and runs the implementation once under memray, recording the peak heap memory of that
   run in the benchmark's `extra_info["peak_mb"]`, and checking its barcode against the reference implementation's.
   This run is in a forked child process, which is killed if it takes longer than `TIMEOUT` seconds, and the
   benchmark skipped, so that one pathological case (say, a column representation that is quadratic on some input)
   can't hold up the whole session. A Python timeout couldn't stop it: the time is spent in native code;
2. times `ROUNDS` more runs with `benchmark.pedantic`, building a fresh input before each (untimed), since some
   implementations reduce their input in place.

Environment variables: `PHLITE_BENCH_THREADS` (threads for parallel implementations, default min(4, cores)),
`PHLITE_BENCH_ROUNDS` (timed runs per benchmark, default 3) and `PHLITE_BENCH_TIMEOUT` (the time limit on the first run
of each benchmark, in seconds, default 600; 0 for none).
"""

import math
import os
import pickle
import select
import signal
import sys
import tempfile
import time
import traceback

import pytest

import memray

THREADS = int(os.environ.get("PHLITE_BENCH_THREADS", min(4, os.cpu_count() or 1)))
ROUNDS = int(os.environ.get("PHLITE_BENCH_ROUNDS", 3))
TIMEOUT = float(os.environ.get("PHLITE_BENCH_TIMEOUT", 600))


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


# ====== Time limit =======================================================


def call_with_timeout(function, timeout):
    """`function()`, called in a forked child process (its result sent back pickled), or `None` if it doesn't finish
    within `timeout` seconds, when the child is killed. Called directly if `timeout` is 0 or the OS can't fork.

    The child shares the parent's memory copy-on-write, so the input is built only once. A pool of worker threads or
    processes kept by the parent wouldn't work in the child (which has only the forking thread), and could hang it.
    The bindings build a fresh thread pool for each parallel run, so keep none; see `_before_fork` for the others.
    """
    if not timeout or not hasattr(os, "fork"):
        return function()
    _before_fork()
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:
        # The child: never return into pytest
        status = 0
        try:
            os.close(read)
            try:
                out = (True, function())
            except BaseException:
                out = (False, traceback.format_exc())
            with os.fdopen(write, "wb") as f:
                pickle.dump(out, f)
        except BaseException:
            status = 1
        finally:
            os._exit(status)
    os.close(write)
    chunks, deadline = [], time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([read], [], [], remaining)[0]:
                os.kill(pid, signal.SIGKILL)
                return None
            chunk = os.read(read, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(read)
        os.waitpid(pid, 0)
    if not chunks:
        raise RuntimeError("the benchmark's child process died without a result")
    ok, out = pickle.loads(b"".join(chunks))
    if not ok:
        raise RuntimeError(f"the first run failed, in a child process:\n{out}")
    return out


def _before_fork():
    """Shut down joblib's reusable process pool (used by grpphati's parallel pipelines), if one is running: its
    management threads wouldn't exist in a forked child, which would then wait on it forever. The child, and the timed
    runs after it, start a new one."""
    if "joblib" not in sys.modules:
        return
    from joblib.externals.loky import reusable_executor

    if reusable_executor._executor is not None:
        reusable_executor._executor.shutdown(wait=True)
        reusable_executor._executor = reusable_executor._executor_kwargs = None


# ====== Running a benchmark ==============================================


def run(benchmark, *, group, implementation, make_input, reference=None, rounds=None):
    """Benchmark `implementation(input)` on inputs from `make_input()`, checking it agrees with `reference()`.

    `reference` returns the reference fingerprint (and should be cached, since every benchmark of a case asks for it);
    leave it out for the reference implementation itself. `rounds` overrides `ROUNDS`, for cases too slow to repeat.
    """
    benchmark.group = group
    benchmark.extra_info["threads"] = THREADS

    data = make_input()

    def first_run():
        result, peak = peak_heap_mb(lambda: implementation(data))
        return _resolve(result), peak

    out = call_with_timeout(first_run, TIMEOUT)
    del data
    if out is None:
        pytest.skip(f"over the time limit of {TIMEOUT:g} s (PHLITE_BENCH_TIMEOUT)")
    result, peak = out
    benchmark.extra_info["peak_mb"] = round(peak, 1)
    if reference is not None:
        assert result == reference(), "barcode differs from the reference implementation's"
    benchmark.extra_info["fingerprint"] = result

    rounds = ROUNDS if rounds is None else min(rounds, ROUNDS)
    benchmark.pedantic(implementation, setup=lambda: ((make_input(),), {}), rounds=rounds, iterations=1)

