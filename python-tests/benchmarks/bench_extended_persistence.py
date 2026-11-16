"""Benchmark phlite.complexes' FilteredComplex.extended_persistence against GUDHI's SimplexTree.extend_filtration and
extended_persistence: the lower-star filtration of random vertex values on a random graph G(n, p) (`graph`) and on its
flag complex up to triangles (`flag`).

(GUDHI isn't in the `latest` environment, so this file runs in the main one, despite benchmarking phlite.complexes.)
"""

import functools
import itertools
import random

import gudhi
import pytest
from common import THREADS, run

from phlite.complexes import ChainComplex
from phlite.core import Algorithm

SUITE = {
    "title": "Extended persistence: `phlite.complexes.FilteredComplex.extended_persistence`",
    "description": "Extended persistence of the lower-star filtration of vertex values uniform in [0, 1] on a random "
        "graph G(n, p) (`graph`), or its flag complex up to triangles (`flag`). The reference is GUDHI's SimplexTree "
        "(the simplices inserted, extend_filtration, extended_persistence).",
    "order": 25,
    "summary": ["gudhi", "phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": ["gudhi"],
}


@functools.cache
def random_complex(kind, n, p, seed=0):
    """`(simplices, values)`: the vertices and edges of G(n, p) (and its triangles, for `flag`), and vertex values."""
    rng = random.Random(seed)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    simplices = [(v,) for v in range(n)] + edges
    if kind == "flag":
        present = set(edges)
        neighbours = {v: set() for v in range(n)}
        for u, v in edges:
            neighbours[u].add(v)
            neighbours[v].add(u)
        simplices += [(u, v, w) for u, v in edges for w in sorted(neighbours[u] & neighbours[v]) if w > v]
    values = [rng.random() for _ in range(n)]
    return simplices, values


def fingerprint(bars):
    """Per kind, the number of bars longer than rounding, and their total length."""
    out = []
    for diagram in bars:
        long = [abs(d - b) for b, d in diagram if abs(d - b) > 1e-9]
        out.append([len(long), round(sum(long), 6)])
    return out


def gudhi_extended(case):
    simplices, values = case
    tree = gudhi.SimplexTree()
    for s in simplices:
        tree.insert(list(s), max(values[v] for v in s))
    tree.extend_filtration()
    diagrams = tree.extended_persistence(homology_coeff_field=2)
    return lambda: fingerprint([[b for _, b in d] for d in diagrams])


def phlite_extended(algorithm=Algorithm.serial):
    def compute(case):
        simplices, values = case
        chain = ChainComplex.from_columns(
            [[] if len(s) == 1 else [index[f] for f in itertools.combinations(s, len(s) - 1)] for s in simplices]
            if (index := {s: i for i, s in enumerate(simplices)})
            else [],
            [len(s) - 1 for s in simplices],
        )
        filtered = chain.filtration([max(values[v] for v in s) for s in simplices])
        barcode = filtered.extended_persistence(algorithm=algorithm())
        return lambda: fingerprint([[b for _, b in d] for d in barcode.to_gudhi()])

    return compute


IMPLEMENTATIONS = {
    "gudhi": gudhi_extended,
    "phlite (serial)": phlite_extended(),
    "phlite (parallel)": phlite_extended(lambda: Algorithm.parallel(THREADS)),
}

# (kind, n, p)
CASES = [("graph", 2000, 0.002), ("graph", 300, 0.05), ("flag", 300, 0.05)]
QUICK = [("graph", 2000, 0.002)]
LARGE = [("graph", 20000, 0.0002), ("flag", 1000, 0.02)]


@functools.cache
def reference(kind, n, p):
    return IMPLEMENTATIONS["gudhi"](random_complex(kind, n, p))()


def params():
    for case in CASES + LARGE:
        kind, n, p = case
        marks = [pytest.mark.quick] if case in QUICK else [pytest.mark.large] if case in LARGE else []
        for implementation in IMPLEMENTATIONS:
            yield pytest.param(kind, n, p, implementation, marks=marks, id=f"{kind}-{n}-{p}-{implementation}")


@pytest.mark.parametrize("kind, n, p, implementation", params())
def test_extended(benchmark, kind, n, p, implementation):
    run(
        benchmark,
        group=f"{kind} n={n} p={p}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: random_complex(kind, n, p),
        reference=None if implementation == "gudhi" else functools.partial(reference, kind, n, p),
        rounds=1 if (kind, n, p) in LARGE else None,
    )
