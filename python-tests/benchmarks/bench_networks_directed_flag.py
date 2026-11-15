"""Benchmark phlite.networks.directed_flag against flagser (pyflagser).

Persistent homology in dimensions up to `dim` of the directed flag complex of a random weighted digraph on `n` nodes,
each edge present with probability `p` and weighted uniformly in [0, 1] (flagser's `max` filtration).
"""

import functools

import pytest
import pyflagser
from common import THREADS, fingerprint, run
from digraphs import adjacency, random_digraph

from phlite.core import Algorithm
from phlite.networks import directed_flag

INF = float("inf")
REFERENCE = "flagser"

SUITE = {
    "title": "Directed flag complexes: `phlite.networks.directed_flag` vs flagser",
    "description": "Persistent homology in dimensions up to `dim` of the directed flag complex of a random weighted "
        "digraph on `n` nodes, each edge present with probability `p`, weights uniform in [0, 1]: flagser (by "
        "pyflagser, from a dense adjacency matrix) against `directed_flag(n, edges, dim)`.",
    "order": 21,
    "summary": ["flagser", "phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": ["flagser"],
}


def phlite(algorithm=Algorithm.serial):
    def compute(case):
        n, edges, max_dim = case
        barcode = directed_flag(n, edges, max_dim, algorithm=algorithm())
        return lambda: fingerprint([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])

    return compute


def flagser(case):
    n, edges, max_dim = case
    dgms = pyflagser.flagser_weighted(adjacency(n, edges), max_dimension=max_dim)["dgms"]
    bars = [[(float(b), float(d)) for b, d in (dgms[dim] if dim < len(dgms) else [])] for dim in range(max_dim + 1)]
    return lambda: fingerprint(bars)


IMPLEMENTATIONS = {
    "flagser": flagser,
    "phlite (serial)": phlite(),
    "phlite (parallel)": phlite(lambda: Algorithm.parallel(THREADS)),
    "phlite (serial, merged)": phlite(lambda: Algorithm.serial().with_columns("merged").with_cache(True)),
}

# (n, p, max_dim)
CASES = [(100, 0.2, 2), (200, 0.1, 2), (60, 0.3, 3), (500, 0.05, 2)]
QUICK = [(100, 0.2, 2)]
LARGE = [(1000, 0.05, 2), (100, 0.3, 3)]


@functools.cache
def reference(n, p, max_dim):
    return IMPLEMENTATIONS[REFERENCE]((n, random_digraph(n, p), max_dim))()


def params():
    for case in CASES + LARGE:
        n, p, max_dim = case
        marks = [pytest.mark.quick] if case in QUICK else [pytest.mark.large] if case in LARGE else []
        for implementation in IMPLEMENTATIONS:
            yield pytest.param(n, p, max_dim, implementation, marks=marks, id=f"{n}-{p}-{max_dim}-{implementation}")


@pytest.mark.parametrize("n, p, max_dim, implementation", params())
def test_directed_flag(benchmark, n, p, max_dim, implementation):
    run(
        benchmark,
        group=f"directed flag n={n} p={p} dim≤{max_dim}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: (n, random_digraph(n, p), max_dim),
        reference=None if implementation == REFERENCE else functools.partial(reference, n, p, max_dim),
        rounds=1 if (n, p, max_dim) in LARGE else None,
    )
