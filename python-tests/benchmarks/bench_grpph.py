"""Benchmark phlite_grpph against grpphati on random weighted digraphs.

Grounded persistent path homology (dimension 1, all grpphati computes) of random digraphs on `n` nodes, each edge
present with probability 0.1, with integer weights 1–5. `GrPPH_par_wedge` is grpphati's parallel pipeline over wedge
components. Cases marked `quick` are the ones `just bench-check` uses.
"""

import functools

import numpy as np
import pytest
from common import THREADS, fingerprint, run

# Import everything up front, so that import costs are not counted
import networkx as nx
from grpphati.pipelines import grounded
from phlite_grpph import Algorithm, grpph

INF = float("inf")
REFERENCE = "grpphati (GrPPH)"
P = 0.1


def phlite(algorithm=Algorithm.serial, method="cohomology", representatives=False):
    """`algorithm` builds the `Algorithm`."""

    def compute(graph):
        n, edges = graph
        barcode = grpph(n, edges, method=method, algorithm=algorithm(), representatives=representatives)
        return lambda: fingerprint([list(barcode.pairings[1]) + [(b, INF) for b in barcode.essential[1]]])

    return compute


def grpphati(pipeline_name):
    def compute(graph):
        n, edges = graph
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_weighted_edges_from(edges)
        result = getattr(grounded, pipeline_name)(g)
        return lambda: fingerprint([[(float(b), float(d)) for b, d in result.barcode]])

    return compute


IMPLEMENTATIONS = {
    "grpphati (GrPPH)": grpphati("GrPPH"),
    "grpphati (GrPPH_par_wedge)": grpphati("GrPPH_par_wedge"),
    "phlite (serial)": phlite(),
    "phlite (parallel)": phlite(lambda: Algorithm.parallel(THREADS)),
    "phlite (serial, reps)": phlite(representatives=True),
    "phlite (parallel, lock-free)": phlite(lambda: Algorithm.parallel(THREADS).with_storage("lock-free")),
}

CASES = [25, 50, 100]
QUICK = [50]


@functools.cache
def graph(n):
    """A random digraph with integer weights in 1..5."""
    rng = np.random.default_rng(0)
    mask = rng.uniform(size=(n, n)) < P
    np.fill_diagonal(mask, False)
    pairs = list(zip(*np.nonzero(mask)))
    weights = rng.integers(1, 6, size=len(pairs))
    return n, [(int(i), int(j), float(w)) for (i, j), w in zip(pairs, weights)]


@functools.cache
def reference(n):
    return IMPLEMENTATIONS[REFERENCE](graph(n))()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("n", [pytest.param(n, marks=pytest.mark.quick) if n in QUICK else n for n in CASES])
def test_grpph(benchmark, n, implementation):
    run(
        benchmark,
        group=f"grpph n={n} p={P}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: graph(n),
        reference=None if implementation == REFERENCE else functools.partial(reference, n),
    )
