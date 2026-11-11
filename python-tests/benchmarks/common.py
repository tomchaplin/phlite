"""Shared helpers for the pytest-benchmark suites (`bench_*.py`), which import this module.

Every benchmark is one implementation on one case. `run(benchmark, ...)`:

1. builds the input (untimed) and runs the implementation once, recording the peak memory of that run in the
   benchmark's `extra_info["peak_mb"]` and checking its barcode against the reference implementation's;
2. times `ROUNDS` more runs with `benchmark.pedantic`, building a fresh input before each (untimed), since some
   implementations reduce their input in place.

Environment variables: `PHLITE_BENCH_THREADS` (threads for parallel implementations, default min(4, cores)) and
`PHLITE_BENCH_ROUNDS` (timed runs per benchmark, default 3).
"""

import math
import os
import resource
import sys

THREADS = int(os.environ.get("PHLITE_BENCH_THREADS", min(4, os.cpu_count() or 1)))
ROUNDS = int(os.environ.get("PHLITE_BENCH_ROUNDS", 3))


# ====== Peak memory ======================================================


def _proc_status_mb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024  # reported in kB
    raise KeyError(field)


def reset_peak_rss():
    """Start measuring peak memory from now, returning the current memory use in MB.

    On Linux we reset the kernel's high-water mark (VmHWM), so the peak is that of what runs next. Elsewhere we can
    only use the lifetime peak, so runs that stay below an earlier peak report ~0.
    """
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return _proc_status_mb("VmRSS")
    except OSError:
        return peak_rss_mb()


def peak_rss_mb():
    try:
        return _proc_status_mb("VmHWM")
    except OSError:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


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


def run(benchmark, *, group, implementation, make_input, reference=None):
    """Benchmark `implementation(input)` on inputs from `make_input()`, checking it agrees with `reference()`.

    `reference` returns the reference fingerprint (and should be cached, since every benchmark of a case asks for it);
    leave it out for the reference implementation itself.
    """
    benchmark.group = group
    benchmark.extra_info["threads"] = THREADS

    data = make_input()
    baseline = reset_peak_rss()
    result = _resolve(implementation(data))
    benchmark.extra_info["peak_mb"] = round(max(peak_rss_mb() - baseline, 0.0), 1)
    del data
    if reference is not None:
        assert result == reference(), "barcode differs from the reference implementation's"
    benchmark.extra_info["fingerprint"] = result

    benchmark.pedantic(implementation, setup=lambda: ((make_input(),), {}), rounds=ROUNDS, iterations=1)

