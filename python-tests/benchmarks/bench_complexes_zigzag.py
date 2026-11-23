"""Benchmark phlite.complexes.zigzag (FastZigzag on phlite's reductions) against Dey and Hou's fzz and Dionysus 2.

Sliding windows of Rips complexes over points arriving one at a time along a noisy circle: each new point brings its
simplices with the window's points (up to triangles), and the oldest leaves with its simplices. The references run
only if given: `FZZ_BIN` (the fzz binary, timed as a whole process, reading and writing its files) and
`DIONYSUS_PYTHON` (a Python with Dionysus 2, timed as a subprocess, so including its start-up and parsing).
"""

import functools
import json
import os
import subprocess
import tempfile

import pytest
from common import fingerprint, run

from phlite.complexes import timed_zigzag, zigzag

from ikc_zigzag_compare import DIONYSUS_RUNNER, sliding_rips

FZZ = os.environ.get("FZZ_BIN")
DIONYSUS = os.environ.get("DIONYSUS_PYTHON")

SUITE = {
    "title": "Zigzag persistence: `phlite.complexes.zigzag` vs fzz and Dionysus 2",
    "description": "Sliding windows of Rips complexes (points along a noisy circle arriving one at a time, each "
        "bringing its simplices with the window's points, the oldest leaving with its simplices): phlite's `zigzag` "
        "on the operations and `timed_zigzag` on the same simplices given lifetimes, against Dey and Hou's fzz (the C++ "
        "binary, as released: homology on PHAT, timed as a whole process with its file input and output) and Dionysus "
        "2's `zigzag_homology_persistence` (in a subprocess). fzz and Dionysus run when `FZZ_BIN` and "
        "`DIONYSUS_PYTHON` are set; see `perf-results/maps-zigzag-2026-11-22.md` for fzz reducing by cohomology too.",
    "order": 46,
    "summary": ["Dionysus 2", "fzz", "phlite (serial)", "phlite (timed)"],
    "default": "phlite (serial)",
    "light_references": [],
}


@functools.cache
def operations(n, window, threshold):
    return sliding_rips(n, window, threshold, 0)


def make(n, window, threshold):
    ops = operations(n, window, threshold)
    # The same zigzag as lifetimes: operation i at time i + 1
    events = {}
    for i, (s, _) in enumerate(ops):
        events.setdefault(tuple(s), []).append(i + 1)
    cells = {s: [(ts[j], ts[j + 1] if j + 1 < len(ts) else None) for j in range(0, len(ts), 2)] for s, ts in events.items()}
    return ops, cells


def by_dim(bars):
    out = {}
    for k, b, e in bars:
        out.setdefault(k, []).append((b, e))
    return fingerprint([out.get(k, []) for k in range(max(out, default=-1) + 1)])


def phlite_ops(case):
    ops, _ = case
    zz = zigzag(ops)
    return lambda: by_dim([(b.dimension, b.birth, b.death) for b in zz])


def phlite_timed(case):
    ops, cells = case
    m = len(ops)
    zz = timed_zigzag(cells)
    return lambda: by_dim([(b.dimension, b.birth, b.death - 1 if b.death != float("inf") else m) for b in zz])


def fzz(case):
    ops, _ = case
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "filt")
        with open(path, "w") as f:
            for s, insert in ops:
                f.write(("i " if insert else "d ") + " ".join(map(str, sorted(s))) + "\n")
        subprocess.run([FZZ, path], capture_output=True, check=True, cwd=tmp)
        with open(os.path.join(tmp, "filt_pers")) as f:
            bars = [tuple(map(int, line.split())) for line in f if line.strip()]
    return lambda: by_dim(bars)


def dionysus(case):
    ops, _ = case
    m = len(ops)
    out = subprocess.run([DIONYSUS, "-I", "-c", DIONYSUS_RUNNER], input=json.dumps([[list(s), i] for s, i in ops]),
                         capture_output=True, text=True, check=True)
    bars = [(k, int(b), m if e < 0 else int(e) - 1) for k, b, e in json.loads(out.stdout)["bars"] if e < 0 or e > b]
    return lambda: by_dim(bars)


IMPLEMENTATIONS = {"phlite (serial)": phlite_ops, "phlite (timed)": phlite_timed}
if FZZ:
    IMPLEMENTATIONS["fzz"] = fzz
if DIONYSUS:
    IMPLEMENTATIONS["Dionysus 2"] = dionysus

# (points, window, threshold): about 6.6k, 126k, 462k and 1.7M operations
CASES = [(300, 20, 0.35), (1000, 30, 0.3), (2000, 35, 0.27), (5000, 40, 0.25)]
QUICK = [(300, 20, 0.35)]
# Dionysus takes 13 s at 126k operations and half an hour at 1.7M: only the first two
DIONYSUS_CASES = CASES[:2]


@functools.cache
def reference(case):
    return phlite_ops(make(*case))()


def params():
    for case in CASES:
        for name in IMPLEMENTATIONS:
            if name == "Dionysus 2" and case not in DIONYSUS_CASES:
                continue
            marks = [pytest.mark.quick] if case in QUICK else []
            yield pytest.param(case, name, marks=marks, id=f"{case[0]}-{case[1]}-{name}")


@pytest.mark.parametrize("case, implementation", params())
def test_zigzag(benchmark, case, implementation):
    n, window, threshold = case
    run(
        benchmark,
        group=f"sliding Rips n={n} window={window} ({len(operations(*case))} operations)",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: make(*case),
        reference=None if implementation == "phlite (serial)" else functools.partial(reference, case),
        rounds=1 if n >= 2000 else None,
    )
