"""Benchmark phlite.networks.pph against grpphati's PPH pipeline.

Persistent path homology (dimensions 0 and 1) of a random weighted digraph on `n` nodes, each edge present with
probability `p` and weighted uniformly in [0, 1]. On the largest cases only phlite runs (grpphati takes minutes), checked
against its own serial barcode.
"""

import functools

import networkx as nx
import pytest
from common import THREADS, fingerprint, run, sweep
from digraphs import random_digraph
from grpphati.pipelines.standard import PPH

from phlite.core import Algorithm
from phlite.networks import pph

INF = float("inf")
REFERENCE = "grpphati (PPH)"

SUITE = {
    "title": "PPH: `phlite.networks.pph` vs grpphati",
    "description": "Persistent path homology (dimensions 0 and 1, on the shortest-path filtration) of a random weighted "
        "digraph on `n` nodes, each edge present with probability `p`, weights uniform in [0, 1]: grpphati's `PPH` "
        "pipeline against `pph(n, edges, filtration='shortest_path')`. On the largest cases only phlite runs.",
    "order": 22,
    "summary": ["grpphati (PPH)", "phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": ["grpphati (PPH)"],
}


def phlite(algorithm=Algorithm.serial):
    def compute(case):
        n, edges = case
        barcode = pph(n, edges, filtration="shortest_path", algorithm=algorithm())
        return lambda: fingerprint([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])

    return compute


def grpphati(case):
    n, edges = case
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(edges)
    result = PPH(graph)
    bars = [(float(b), float(d)) for b, d in result.barcode]
    # grpphati gives dimension 1 only
    return lambda: fingerprint([bars])


def phlite_h1(algorithm=Algorithm.serial):
    """phlite's dimension 1 only, to compare with grpphati's fingerprint."""

    def compute(case):
        n, edges = case
        barcode = pph(n, edges, filtration="shortest_path", algorithm=algorithm())
        return lambda: fingerprint([list(barcode.pairings[1]) + [(b, INF) for b in barcode.essential[1]]])

    return compute


IMPLEMENTATIONS = {
    "grpphati (PPH)": grpphati,
    "phlite (serial)": phlite_h1(),
    "phlite (parallel)": phlite_h1(lambda: Algorithm.parallel(THREADS)),
}
IMPLEMENTATIONS.update(sweep(phlite_h1))

# (n, p)
CASES = [(30, 0.2), (60, 0.1), (60, 0.3)]
QUICK = [(30, 0.2)]
LARGE = [(200, 0.1), (300, 0.3)]


def reference_name(n, p):
    return "phlite (serial)" if (n, p) in LARGE else REFERENCE


@functools.cache
def reference(n, p):
    return IMPLEMENTATIONS[reference_name(n, p)]((n, random_digraph(n, p)))()


def params():
    for case in CASES + LARGE:
        n, p = case
        marks = [pytest.mark.quick] if case in QUICK else [pytest.mark.large] if case in LARGE else []
        for implementation in IMPLEMENTATIONS:
            if case in LARGE and implementation == REFERENCE:
                continue
            yield pytest.param(n, p, implementation, marks=marks, id=f"{n}-{p}-{implementation}")


@pytest.mark.parametrize("n, p, implementation", params())
def test_pph(benchmark, n, p, implementation):
    run(
        benchmark,
        group=f"pph n={n} p={p}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: (n, random_digraph(n, p)),
        reference=None if implementation == reference_name(n, p) else functools.partial(reference, n, p),
        rounds=1 if (n, p) in LARGE else None,
    )
