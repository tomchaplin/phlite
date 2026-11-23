"""Benchmark the zigzags built in Rust from point clouds, networks and videos: phlite.clouds.cloud_zigzag,
phlite.networks.network_zigzag and phlite.images.video_zigzag. No other package builds these; serial against
parallel reductions, for scale."""

import pytest
from common import THREADS, fingerprint, run

import numpy as np
from phlite.clouds import cloud_zigzag, sliding_windows
from phlite.core import Algorithm
from phlite.images import video_zigzag
from phlite.networks import network_zigzag

SUITE = {
    "title": "Zigzags of point clouds, networks and videos: `cloud_zigzag`, `network_zigzag`, `video_zigzag`",
    "description": "Zigzags whose complexes phlite builds in Rust from the data: sliding windows and moving points "
        "(Rips, alpha and Čech complexes of each frame, joined by unions), snapshots of a random temporal network "
        "(clique and directed flag complexes of the edges active in each window of time), and videos (each frame of "
        "smoothed noise cut at its median, as a cubical complex). phlite alone: serial and parallel reductions.",
    "order": 47,
    "summary": ["phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": [],
}


def bars(zz):
    out = {}
    for b in zz:
        out.setdefault(b.dimension, []).append((b.birth, b.death))
    return fingerprint([out.get(k, []) for k in range(max(out, default=-1) + 1)])


def circle(n, seed=0, turns=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi * turns, n)
    return np.c_[np.cos(t), np.sin(t)] + 0.1 * rng.standard_normal((n, 2))


def make_case(kind, size):
    rng = np.random.default_rng(0)
    if kind.startswith("windows"):
        return ("cloud", sliding_windows(circle(size), 40, step=2))
    if kind.startswith("moving"):
        points = circle(size, turns=1)
        frames, current = [], points
        for _ in range(30):
            frames.append((list(range(size)), current))
            current = current + 0.02 * rng.standard_normal(current.shape)
        return ("cloud", frames)
    if kind.startswith("network"):
        n = size
        times = rng.random((n, n)) * 100
        present = rng.random((n, n)) < 8 / n
        edges = [(a, b, times[a, b]) for a in range(n) for b in range(n) if a != b and present[a, b]]
        snapshots = [[(a, b) for a, b, t in edges if s <= t < s + 10] for s in range(0, 100, 2)]
        return ("network", [(list(range(n)), e) for e in snapshots])
    video = rng.random((40, size, size))
    for _ in range(3):
        for axis in (1, 2):
            video = (video + np.roll(video, 1, axis) + np.roll(video, -1, axis)) / 3
    return ("video", video)


# (kind, size, how)
CASES = [
    ("windows rips", 1000, "rips"), ("windows alpha", 1000, "alpha"), ("windows cech", 400, "cech"),
    ("moving rips", 300, "rips"), ("moving alpha", 2000, "alpha"),
    ("network clique", 200, "clique"), ("network directed flag", 200, "directed_flag"),
    ("video", 64, "T"), ("video", 128, "T"),
]
QUICK = [("windows rips", 1000, "rips"), ("video", 64, "T")]
THRESHOLDS = {"rips": 0.3, "alpha": 0.15, "cech": 0.15}


def make_implementation(kind, how, algorithm):
    def compute(case):
        what, data = case
        if what == "cloud":
            zz = cloud_zigzag(data, THRESHOLDS[how], 1, filtration=how, algorithm=algorithm())
        elif what == "network":
            zz = network_zigzag(data, 1, complex=how, algorithm=algorithm())
        else:
            zz = video_zigzag(data, float(np.median(data)), construction=how, algorithm=algorithm())
        return lambda: bars(zz)

    return compute


ALGORITHMS = {"phlite (serial)": Algorithm.serial, "phlite (parallel)": lambda: Algorithm.parallel(THREADS)}
_references = {}


def reference(kind, size, how):
    key = (kind, size, how)
    if key not in _references:
        _references[key] = make_implementation(kind, how, Algorithm.serial)(make_case(kind, size))()
    return _references[key]


def params():
    for case in CASES:
        for name in ALGORITHMS:
            marks = [pytest.mark.quick] if case in QUICK else []
            yield pytest.param(*case, name, marks=marks, id=f"{case[0]}-{case[1]}-{name}".replace(" ", "_"))


@pytest.mark.parametrize("kind, size, how, implementation", params())
def test_zigzag(benchmark, kind, size, how, implementation):
    run(
        benchmark,
        group=f"{kind} {size}",
        implementation=make_implementation(kind, how, ALGORITHMS[implementation]),
        make_input=lambda: make_case(kind, size),
        reference=None if implementation == "phlite (serial)" else (lambda: reference(kind, size, how)),
    )
