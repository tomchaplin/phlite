"""Benchmark phlite.networks.dowker (and ordered_tuples) against their complexes built in Python.

No other library computes Dowker or ordered-tuple complexes, so the references build each complex by brute force in
Python, then reduce it: the Dowker sink complex in a GUDHI simplex tree (its persistence by GUDHI), the ordered-tuple
complex (not simplicial) as boundary columns for LoPHAT. Random weighted digraphs on `n` nodes, each edge present with
probability `p`, weights uniform in [0, 1].
"""

import functools
import itertools

import gudhi
import lophat
import pytest
from common import THREADS, fingerprint, run
from digraphs import random_digraph

from phlite.core import Algorithm
from phlite.networks import dowker, ordered_tuples

INF = float("inf")

SUITE = {
    "title": "Dowker complexes and ordered tuples: `phlite.networks.dowker` and `ordered_tuples`",
    "description": "Persistent homology in dimensions up to `dim` of the Dowker sink complex (`dowker`) and the "
        "ordered-tuple complex (`tuples`) of a random weighted digraph on `n` nodes, each edge present with probability "
        "`p`, weights uniform in [0, 1]. No other library computes these, so the references build the complex in "
        "Python: the Dowker complex in a GUDHI simplex tree, the ordered tuples as columns for LoPHAT.",
    "order": 23,
    "summary": ["brute force", "phlite (serial)", "phlite (parallel)"],
    "default": "phlite (serial)",
    "light_references": ["brute force"],
}


def dowker_brute_force(case):
    n, edges, max_dim = case
    omega = {(a, b): w for a, b, w in edges}
    tree = gudhi.SimplexTree()
    for k in range(1, max_dim + 3):
        for s in itertools.combinations(range(n), k):
            value = min(max(0.0 if v == x else omega.get((v, x), INF) for v in s) for x in range(n))
            if value != INF:
                tree.insert(list(s), value)
    tree.compute_persistence(min_persistence=-1, persistence_dim_max=True)
    bars = [[(float(b), float(d)) for b, d in tree.persistence_intervals_in_dimension(dim)] for dim in range(max_dim + 1)]
    return lambda: fingerprint(bars)


def tuples_brute_force(case):
    n, edges, max_dim = case
    d = {(a, b): w for a, b, w in edges}
    cells = {(v,): 0.0 for v in range(n)}
    current = [(v,) for v in range(n)]
    for _ in range(max_dim + 1):
        longer = []
        for s in current:
            for w in range(n):
                if w == s[-1]:
                    continue
                t = s + (w,)
                values = [d.get((t[i], t[j])) for i, j in itertools.combinations(range(len(t)), 2) if t[i] != t[j]]
                if None not in values:
                    cells[t] = max(values)
                    longer.append(t)
        current = longer
    order = sorted(cells, key=lambda c: (cells[c], len(c), c))
    index = {c: i for i, c in enumerate(order)}

    def faces(c):
        out = []
        for i in range(len(c)):
            f = c[:i] + c[i + 1 :]
            if len(c) > 1 and all(f[k] != f[k + 1] for k in range(len(f) - 1)):
                out.append(index[f])
        return sorted(out)

    columns = [(len(c) - 1, faces(c)) for c in order]
    diagram = lophat.compute_pairings(columns, True, lophat.LoPhatOptions(num_threads=1))
    bars = [[] for _ in range(max_dim + 1)]
    paired = set()
    for b, e in diagram.paired:
        paired.update((b, e))
        if len(order[b]) - 1 <= max_dim:
            bars[len(order[b]) - 1].append((cells[order[b]], cells[order[e]]))
    for i, c in enumerate(order):
        if i not in paired and len(c) - 1 <= max_dim:
            bars[len(c) - 1].append((cells[c], INF))
    return lambda: fingerprint(bars)


def phlite(function, algorithm=Algorithm.serial):
    def compute(case):
        n, edges, max_dim = case
        barcode = function(n, edges, max_dim, algorithm=algorithm())
        return lambda: fingerprint([list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)])

    return compute


IMPLEMENTATIONS = {
    "dowker": {
        "brute force": dowker_brute_force,
        "phlite (serial)": phlite(dowker),
        "phlite (parallel)": phlite(dowker, lambda: Algorithm.parallel(THREADS)),
    },
    "tuples": {
        "brute force": tuples_brute_force,
        "phlite (serial)": phlite(ordered_tuples),
        "phlite (parallel)": phlite(ordered_tuples, lambda: Algorithm.parallel(THREADS)),
    },
}

# (complex, n, p, max_dim)
CASES = [("dowker", 20, 0.3, 2), ("dowker", 40, 0.2, 1), ("tuples", 20, 0.3, 2), ("tuples", 60, 0.2, 1)]
QUICK = [("dowker", 20, 0.3, 2)]
LARGE = [("dowker", 60, 0.2, 2), ("tuples", 40, 0.3, 2)]


@functools.cache
def reference(kind, n, p, max_dim):
    return IMPLEMENTATIONS[kind]["brute force"]((n, random_digraph(n, p), max_dim))()


def params():
    for case in CASES + LARGE:
        kind, n, p, max_dim = case
        marks = [pytest.mark.quick] if case in QUICK else [pytest.mark.large] if case in LARGE else []
        for implementation in IMPLEMENTATIONS[kind]:
            yield pytest.param(kind, n, p, max_dim, implementation, marks=marks, id=f"{kind}-{n}-{p}-{max_dim}-{implementation}")


@pytest.mark.parametrize("kind, n, p, max_dim, implementation", params())
def test_dowker_and_tuples(benchmark, kind, n, p, max_dim, implementation):
    run(
        benchmark,
        group=f"{kind} n={n} p={p} dim≤{max_dim}",
        implementation=IMPLEMENTATIONS[kind][implementation],
        make_input=lambda: (n, random_digraph(n, p), max_dim),
        reference=None if implementation == "brute force" else functools.partial(reference, kind, n, p, max_dim),
        rounds=1 if (kind, n, p, max_dim) in LARGE else None,
    )
