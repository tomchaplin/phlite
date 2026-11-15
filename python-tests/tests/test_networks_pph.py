"""Fuzz phlite.networks.pph (persistent path homology, Chowdhury and Mémoli) in any dimension: against a brute force
(the persistent Betti numbers computed from the spaces of ∂-invariant paths Ω_p(G_t) by linear algebra, in
phlite_networks/matlab_compare/brute.py), against grpphati (shortest-path filtration, dimension 1, in
test_networks_complexes.py), and against invariants. Against pph-matlab under Octave: see
phlite_networks/matlab_compare/."""

import itertools
import math
import os
import sys

import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st

from grid import algorithm_grid
from phlite.core import Algorithm
from phlite.networks import pph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "phlite_networks", "matlab_compare"))
import brute  # noqa: E402

INF = math.inf
METHODS = ["cohomology", "homology"]


@st.composite
def networks(draw, max_nodes=5):
    """Small weighted digraphs: integer weights (lots of ties), some edges missing or infinite."""
    n = draw(st.integers(1, max_nodes))
    density = draw(st.sampled_from([0.3, 0.6, 0.9]))
    edges = []
    for a, b in itertools.permutations(range(n), 2):
        if draw(st.floats(0, 1)) < density:
            edges.append((a, b, draw(st.one_of(st.integers(0, 3).map(float), st.just(INF)))))
    return n, edges


def diagram(barcode, dim):
    return sorted(list(barcode.pairings[dim]) + [(b, INF) for b in barcode.essential[dim]])


def diagrams(barcode):
    return [diagram(barcode, dim) for dim in range(len(barcode.pairings))]


def shortest_paths(n, edges):
    d = [[0.0 if a == b else INF for b in range(n)] for a in range(n)]
    for a, b, w in edges:
        d[a][b] = min(d[a][b], w)
    for k, i, j in itertools.product(range(n), repeat=3):
        d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return {(a, b): d[a][b] for a in range(n) for b in range(n) if a != b and d[a][b] != INF}


def rank_function(barcode, dim, times):
    """The persistent Betti numbers: for s <= t, the number of bars [b, d) with b <= s < ... d > t."""
    bars = diagram(barcode, dim)
    return {(s, t): sum(1 for b, d in bars if b <= s and d > t) for s in times for t in times if t >= s}


@given(network=networks(), filtration=st.sampled_from(["weight", "shortest_path"]))
def test_matches_brute_force(network, filtration):
    n, edges = network
    weights = {(a, b): w for a, b, w in edges if w != INF}
    if filtration == "shortest_path":
        weights = shortest_paths(n, edges)
    times = sorted(set(weights.values()) | {0.0})
    max_dim = 2 if n <= 4 else 1
    note(f"weights={weights}")
    for method in METHODS:
        barcode = pph(n, edges, max_dim, filtration=filtration, method=method)
        for dim in range(max_dim + 1):
            assert rank_function(barcode, dim, times) == brute.brute(n, weights, dim, 2, times), f"{method} dim {dim}"


def allow_time(path, weights):
    return brute.ptime(tuple(path), weights)


@given(network=networks(), max_dim=st.integers(0, 2))
def test_representatives_and_algorithms(network, max_dim):
    """Every algorithm agrees; each representative is a cycle of regular paths (irregular faces dropped, over Z/2),
    allowed by the bar's birth, one of them exactly then."""
    n, edges = network
    weights = {(a, b): w for a, b, w in edges if w != INF}
    expected = pph(n, edges, max_dim)
    for method, algorithm in itertools.product(METHODS, algorithm_grid(Algorithm, row_ranks=False)):
        barcode = pph(n, edges, max_dim, method=method, algorithm=algorithm, representatives=True)
        assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings), f"{method} {algorithm!r}"
        for dim in range(max_dim + 1):
            bars = list(zip(barcode.pairings[dim], barcode.pairing_representatives[dim]))
            bars += [((b, INF), rep) for b, rep in zip(barcode.essential[dim], barcode.essential_representatives[dim])]
            for (birth, _), rep in bars:
                paths = [tuple(p) for p in rep]
                assert all(len(p) == dim + 1 for p in paths)
                assert max(allow_time(p, weights) for p in paths) == birth
                if dim == 0:
                    continue
                boundary = {}
                for p in paths:
                    for f, c in brute.bd(p, 2).items():
                        boundary[f] = (boundary.get(f, 0) + c) % 2
                assert not any(boundary.values()), f"not a cycle: {paths}"


reversed_edges = lambda edges: [(b, a, w) for a, b, w in edges]  # noqa: E731


@given(network=networks(max_nodes=6), seed=st.integers(0, 2**32 - 1))
def test_invariants(network, seed):
    """Reversing every edge (which reverses every path), relabelling and doubling the weights; and in the
    shortest-path filtration, dimensions 0 and 1 (computed by GrPPH's matrices) agree with the general computation."""
    n, edges = network
    perm = np.random.default_rng(seed).permutation(n)
    expected = diagrams(pph(n, edges, 2))
    assert diagrams(pph(n, reversed_edges(edges), 2)) == expected
    assert diagrams(pph(n, [(int(perm[a]), int(perm[b]), w) for a, b, w in edges], 2)) == expected
    doubled = diagrams(pph(n, [(a, b, 2 * w) for a, b, w in edges], 2))
    assert doubled == [[(2 * b, 2 * d) for b, d in dim] for dim in expected]
    low = diagrams(pph(n, edges, 1, filtration="shortest_path"))
    high = diagrams(pph(n, edges, 2, filtration="shortest_path"))
    assert high[:2] == low


def test_cycle_networks():
    """Chowdhury and Mémoli's cycle networks: n nodes, a -> b at (b - a) mod n; Dgm_1 = {(1, ceil(n / 2))}."""
    for n in range(3, 9):
        edges = [(a, b, float((b - a) % n)) for a in range(n) for b in range(n) if a != b]
        barcode = pph(n, edges, 1)
        assert diagram(barcode, 1) == [(1.0, float(math.ceil(n / 2)))]
