"""Benchmark phlite.networks.flag (flag complexes of undirected graphs) against GUDHI's SimplexTree.

Random graphs G(n, p) with weights uniform in [0, 1] and node values uniform in [0, 1]: the flag complex by edge
weights (`weight`), the lower-star filtration of the node values (`lower star`), and Rips of the shortest-path metric
(`shortest path`).
"""

import functools
import math
import random

import gudhi
import networkx as nx
import pytest
from common import THREADS, fingerprint, run

from phlite.core import Algorithm
from phlite.networks import flag

INF = math.inf

SUITE = {
    "title": "Flag complexes of graphs: `phlite.networks.flag`",
    "description": "Persistent homology in dimensions up to `dim` of the flag complex of a random graph G(n, p), "
        "weights and node values uniform in [0, 1]: by edge weights (`weight`), by the node values' lower-star "
        "filtration (`lower star`), and Rips of the shortest-path metric (`shortest path`). The reference is GUDHI's "
        "SimplexTree (with the edges and values inserted, and expanded).",
    "order": 24,
    "summary": ["gudhi", "phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": ["gudhi"],
}


@functools.cache
def random_graph(n, p, seed=0):
    """`(n, edges, values)`: G(n, p) with weights and node values uniform in [0, 1]."""
    rng = random.Random(seed)
    edges = [(u, v, rng.random()) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    values = [rng.random() for _ in range(n)]
    return n, edges, values


def setup(kind, n, edges, values):
    """The arguments of `flag` for each kind, and the edges and node values GUDHI is given."""
    if kind == "weight":
        return dict(weight="weight"), edges, [0.0] * n
    if kind == "lower star":
        return dict(weight=None, node_weight=values), [(u, v, max(values[u], values[v])) for u, v, _ in edges], values
    # Shortest paths: every connected pair at its path length
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(edges)
    paths = [(u, v, d) for u, row in nx.all_pairs_dijkstra_path_length(graph) for v, d in row.items() if u < v]
    return dict(filtration="shortest_path"), paths, [0.0] * n


def gudhi_flag(kind):
    def compute(case):
        n, edges, values, max_dim = case
        _, gudhi_edges, gudhi_values = setup(kind, n, edges, values)
        tree = gudhi.SimplexTree()
        for v in range(n):
            tree.insert([v], gudhi_values[v])
        for u, v, w in gudhi_edges:
            tree.insert([u, v], w)
        tree.expansion(max_dim + 1)
        tree.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
        bars = [[tuple(map(float, b)) for b in tree.persistence_intervals_in_dimension(d)] for d in range(max_dim + 1)]
        return lambda: fingerprint(bars)

    return compute


def phlite_flag(kind, algorithm=Algorithm.serial):
    def compute(case):
        n, edges, values, max_dim = case
        kwargs, _, _ = setup(kind, n, edges, values)
        barcode = flag(n, edges, max_dim, algorithm=algorithm(), **kwargs)
        return lambda: fingerprint([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])

    return compute


IMPLEMENTATIONS = {
    kind: {
        "gudhi": gudhi_flag(kind),
        "phlite (serial)": phlite_flag(kind),
        "phlite (parallel)": phlite_flag(kind, lambda: Algorithm.parallel(THREADS)),
    }
    for kind in ["weight", "lower star", "shortest path"]
}

# (kind, n, p, max_dim)
CASES = [
    ("weight", 300, 0.1, 2),
    ("weight", 3000, 0.003, 2),
    ("lower star", 300, 0.1, 2),
    ("shortest path", 120, 0.03, 1),
]
QUICK = [("weight", 300, 0.1, 2)]
LARGE = [("weight", 1000, 0.1, 2), ("shortest path", 300, 0.02, 1)]


@functools.cache
def reference(kind, n, p, max_dim):
    return IMPLEMENTATIONS[kind]["gudhi"]((*random_graph(n, p), max_dim))()


def params():
    for case in CASES + LARGE:
        kind, n, p, max_dim = case
        marks = [pytest.mark.quick] if case in QUICK else [pytest.mark.large] if case in LARGE else []
        for implementation in IMPLEMENTATIONS[kind]:
            yield pytest.param(
                kind, n, p, max_dim, implementation, marks=marks, id=f"{kind}-{n}-{p}-{max_dim}-{implementation}"
            )


@pytest.mark.parametrize("kind, n, p, max_dim, implementation", params())
def test_flag(benchmark, kind, n, p, max_dim, implementation):
    run(
        benchmark,
        group=f"{kind} n={n} p={p} dim≤{max_dim}",
        implementation=IMPLEMENTATIONS[kind][implementation],
        make_input=lambda: (*random_graph(n, p), max_dim),
        reference=None if implementation == "gudhi" else functools.partial(reference, kind, n, p, max_dim),
        rounds=1 if (kind, n, p, max_dim) in LARGE else None,
    )
