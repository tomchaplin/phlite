"""Benchmark phlite_grpph against grpphati on weighted digraphs.

Grounded persistent path homology (dimension 1, all grpphati computes), with integer weights 1–5, of
* random digraphs on `n` nodes, each edge present with probability 0.1;
* `ring`: a directed cycle on `n` nodes with random forward chords, so one long bar (the cycle) among short ones;
* `lattice`: an `n` x `n` grid with edges right and down, with rectangular holes, so a long bar around each hole.

GrPPH works on the complete digraph of shortest-path distances, so the cost grows quickly with the number of nodes
even when the digraph is sparse. `GrPPH_par_wedge` is grpphati's parallel pipeline over wedge components. Cases marked
`quick` are the ones `just bench-check` uses. Cases marked `large` are timed once each, and only phlite runs on them
(grpphati needs over 5 GB on random digraphs of 100 nodes, and minutes on the others), checked against phlite's default.
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
    # Every column representation, cached or not, named explicitly (the presets' choice differs between packages)
    **{
        f"phlite (serial, {columns}{'' if cache else ', uncached'})": phlite(
            lambda columns=columns, cache=cache: Algorithm.serial().with_columns(columns).with_cache(cache)
        )
        for columns in ["merged", "heap", "sorted"]
        for cache in [True, False]
    },
    "phlite (serial, store R)": phlite(lambda: Algorithm.serial().with_store_r()),
    "phlite (parallel, lock-free)": phlite(lambda: Algorithm.parallel(THREADS).with_storage("lock-free")),
}

# (kind, n)
CASES = [("random", 25), ("random", 50), ("random", 100), ("ring", 100), ("lattice", 15)]
QUICK = [("random", 50)]
LARGE = [("random", 200), ("ring", 400), ("lattice", 30)]
# grpphati needs over 5 GB at 100 random nodes, so these are checked against phlite's default instead
NO_GRPPHATI = LARGE
LARGE_REFERENCE = "phlite (serial)"


def reference_name(case):
    return LARGE_REFERENCE if case in NO_GRPPHATI else REFERENCE


def label(kind, n):
    if kind == "random":
        return f"grpph n={n} p={P}"
    return f"grpph ring n={n}" if kind == "ring" else f"grpph lattice {n}x{n}"


@functools.cache
def graph(kind, n):
    """A digraph of the given kind (see the module docs) with integer weights in 1..5, as (nodes, edges)."""
    rng = np.random.default_rng(0)
    if kind == "random":
        mask = rng.uniform(size=(n, n)) < P
        np.fill_diagonal(mask, False)
        pairs = list(zip(*np.nonzero(mask)))
        weights = rng.integers(1, 6, size=len(pairs))
        return n, [(int(i), int(j), float(w)) for (i, j), w in zip(pairs, weights)]
    if kind == "ring":
        edges = [(i, (i + 1) % n) for i in range(n)]
        # Chords jumping 2 to 4 nodes ahead
        edges += [(i, (i + int(rng.integers(2, 5))) % n) for i in range(n) if rng.uniform() < 0.5]
        nodes = n
    else:
        # Holes: a few rectangles of nodes removed, each a quarter to a fifth of the side
        hole = np.zeros((n, n), dtype=bool)
        for _ in range(4):
            h, w = (int(x) for x in rng.integers(n // 5, n // 4 + 1, size=2))
            i, j = (int(x) for x in rng.integers(1, n - max(h, w) - 1, size=2))
            hole[i : i + h, j : j + w] = True
        node = lambda i, j: i * n + j  # noqa: E731
        edges = []
        for i in range(n):
            for j in range(n):
                if hole[i, j]:
                    continue
                if j + 1 < n and not hole[i, j + 1]:
                    edges.append((node(i, j), node(i, j + 1)))
                if i + 1 < n and not hole[i + 1, j]:
                    edges.append((node(i, j), node(i + 1, j)))
        nodes = n * n
    weights = rng.integers(1, 6, size=len(edges))
    return nodes, [(i, j, float(w)) for (i, j), w in zip(edges, weights)]


@functools.cache
def reference(case):
    return IMPLEMENTATIONS[reference_name(case)](graph(*case))()


def params():
    for case in CASES + LARGE:
        kind, n = case
        marks = [pytest.mark.quick] if case in QUICK else [pytest.mark.large] if case in LARGE else []
        for implementation in IMPLEMENTATIONS:
            if case in NO_GRPPHATI and not implementation.startswith("phlite"):
                continue
            id = f"{n}-{implementation}" if kind == "random" else f"{kind}-{n}-{implementation}"
            yield pytest.param(kind, n, implementation, marks=marks, id=id)


@pytest.mark.parametrize("kind, n, implementation", params())
def test_grpph(benchmark, kind, n, implementation):
    case = (kind, n)
    run(
        benchmark,
        group=label(kind, n),
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: graph(kind, n),
        reference=None if implementation == reference_name(case) else functools.partial(reference, case),
        rounds=1 if case in LARGE else None,
    )
