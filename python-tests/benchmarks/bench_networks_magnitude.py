"""Benchmark phlite.networks.magnitude (magnitude homology) against gramag, and the magnitude-path spectral sequence.

Magnitude homology ranks over Z/2 of a random digraph on `n` nodes (each edge with probability `p`; magnitude
homology ignores the weights), lengths up to `l_max`: gramag's `MagGraph(...).populate_paths(l_max)` and
`rank_homology()` against `magnitude(n, edges, l_max)`. phlite alone on the spectral sequence (`magnitude_path_spectral_sequence`)
and over the integers.
"""

import functools

import gramag
import pytest
from common import run
from digraphs import random_digraph

from phlite.networks import magnitude, magnitude_path_spectral_sequence

REFERENCE = "gramag"

SUITE = {
    "title": "Magnitude homology: `phlite.networks.magnitude` vs gramag",
    "description": "The ranks of magnitude homology MH_{k,l} over Z/2, for lengths l up to `l_max`, of a random digraph "
        "on `n` nodes, each edge present with probability `p`: gramag (`MagGraph.populate_paths`, `rank_homology`) "
        "against `magnitude(n, edges, l_max)`. Also phlite over the integers (Smith forms: ranks and torsion), and the "
        "magnitude-path spectral sequence to its last page (`magnitude_path_spectral_sequence`).",
    "order": 25,
    "summary": ["gramag", "phlite (serial)", "phlite (integers)", "phlite (spectral sequence)"],
    "default": "phlite (serial)",
    "light_references": ["gramag"],
}


def edges_of(n, p):
    return [(a, b) for a, b, _ in random_digraph(n, p)]


def ranks_fingerprint(table):
    return [list(row) for row in table]


def phlite_ranks(case):
    n, edges, l_max = case
    mh = magnitude(n, edges, l_max)
    return lambda: ranks_fingerprint(mh)


def gramag_ranks(case):
    n, edges, l_max = case
    mg = gramag.MagGraph(edges)
    mg.populate_paths(l_max=l_max)
    ranks = mg.rank_homology()
    # gramag leaves out isolated nodes; phlite counts them in MH_{0,0}
    isolated = n - len({v for e in edges for v in e})
    return lambda: ranks_fingerprint([[r + (isolated if (l, k) == (0, 0) else 0) for k, r in enumerate(row)] for l, row in enumerate(ranks)])


def phlite_integral(case):
    n, edges, l_max = case
    mh = magnitude(n, edges, l_max, integral=True)
    return lambda: None  # a different answer (torsion too): timed, not compared


def phlite_mpss(case):
    n, edges, l_max = case
    ss = magnitude_path_spectral_sequence(n, edges, l_max)
    return lambda: None


IMPLEMENTATIONS = {
    "gramag": gramag_ranks,
    "phlite (serial)": phlite_ranks,
    "phlite (integers)": phlite_integral,
    "phlite (spectral sequence)": phlite_mpss,
}

# (n, p, l_max)
CASES = [(15, 0.2, 4), (25, 0.15, 4), (40, 0.1, 4), (30, 0.2, 5)]
QUICK = [(15, 0.2, 4)]


@functools.cache
def reference(n, p, l_max):
    return gramag_ranks((n, edges_of(n, p), l_max))()


def params():
    for case in CASES:
        for implementation in IMPLEMENTATIONS:
            marks = [pytest.mark.quick] if case in QUICK else []
            yield pytest.param(*case, implementation, marks=marks, id=f"{case[0]}-{case[1]}-{case[2]}-{implementation}")


@pytest.mark.parametrize("n, p, l_max, implementation", params())
def test_magnitude(benchmark, n, p, l_max, implementation):
    run(
        benchmark,
        group=f"magnitude n={n} p={p} l_max={l_max}",
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: (n, edges_of(n, p), l_max),
        reference=functools.partial(reference, n, p, l_max) if implementation == "phlite (serial)" else None,
    )
