"""Fuzz phlite.networks.extended (extended persistence of node values on a graph or its flag complex) against
GUDHI's SimplexTree.extend_filtration and extended_persistence.

GUDHI rescales the values into [-2, -1] going up and [1, 2] going down, so its values are within rounding of the
exact ones, and a pair whose two values are equal can come back a rounding error apart (which GUDHI keeps, and phlite
drops as a bar of length zero): the comparison allows for both."""

import itertools
import math

import gudhi
import networkx as nx
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from grid import algorithm_grid
from phlite.core import Algorithm
from phlite.networks import extended

KINDS = ["ordinary", "relative", "extended_plus", "extended_minus"]


@st.composite
def graphs(draw, max_nodes=8):
    """Graphs `(n, edges, values)`, with small integer values (lots of ties) or floats."""
    n = draw(st.integers(1, max_nodes))
    density = draw(st.sampled_from([0.3, 0.6, 0.95]))
    edges = [(u, v) for u, v in itertools.combinations(range(n), 2) if draw(st.floats(0, 1)) < density]
    if draw(st.booleans()):
        values = [float(draw(st.integers(-3, 3))) for _ in range(n)]
    else:
        values = [draw(st.floats(-10, 10, allow_nan=False, width=32)) for _ in range(n)]
    return n, edges, values


def gudhi_extended(n, edges, values, dim, modulus):
    """GUDHI's four diagrams, as lists by dimension of sorted (birth, death), without pairs of length within rounding
    of zero."""
    tree = gudhi.SimplexTree()
    for v in range(n):
        tree.insert([v], filtration=values[v])
    for u, v in edges:
        tree.insert([u, v], filtration=max(values[u], values[v]))
    tree.expansion(dim)
    tree.make_filtration_non_decreasing()
    tree.extend_filtration()
    diagrams = tree.extended_persistence(homology_coeff_field=modulus, min_persistence=-1)
    scale = max([abs(x) for x in values] + [1.0])
    out = []
    for diagram in diagrams:
        by_dim = [[] for _ in range(dim + 2)]
        for d, (b, e) in diagram:
            if abs(b - e) > 1e-9 * scale:
                by_dim[d].append((b, e))
        out.append([sorted(bars) for bars in by_dim])
    return out, scale


def close(ours, theirs, scale):
    """Equal within rounding, after sorting each by values rounded to it (values a rounding error apart, like 0 and
    1e-38, which GUDHI can't tell apart, could otherwise sort differently)."""
    key = lambda bar: tuple(round(x / (1e-7 * scale)) for x in bar)
    ours, theirs = sorted(ours, key=key), sorted(theirs, key=key)
    return len(ours) == len(theirs) and all(
        math.isclose(a, c, abs_tol=1e-9 * scale) and math.isclose(b, d, abs_tol=1e-9 * scale)
        for (a, b), (c, d) in zip(ours, theirs)
    )


def check(barcode, expected, scale):
    """Our diagrams against GUDHI's, both without pairs shorter than rounding (so a pair of length 1e-45, say, which
    is exact here but can't be told from rounding in GUDHI's, is left out of both)."""
    for kind, theirs in zip(KINDS, expected):
        ours = getattr(barcode, kind)
        assert len(ours) == len(theirs), kind
        for dim, (o, t) in enumerate(zip(ours, theirs)):
            o = [(b, d) for b, d in o if abs(b - d) > 1e-9 * scale]
            assert close(o, t, scale), (kind, dim, o, t)


@given(graph=graphs(), dim=st.integers(1, 2), modulus=st.sampled_from([2, 3]))
def test_extended_matches_gudhi(graph, dim, modulus):
    n, edges, values = graph
    note(f"n={n} edges={edges} values={values}")
    expected, scale = gudhi_extended(n, edges, values, dim, modulus)
    for method in ["cohomology", "homology"]:
        barcode = extended(n, edges, node_weight=values, dim=dim, modulus=modulus, method=method)
        check(barcode, expected, scale)


@given(graph=graphs(max_nodes=6), dim=st.integers(1, 2))
def test_every_algorithm_agrees(graph, dim):
    n, edges, values = graph
    expected, scale = gudhi_extended(n, edges, values, dim, 2)
    for algorithm in algorithm_grid(Algorithm):
        check(extended(n, edges, node_weight=values, dim=dim, algorithm=algorithm), expected, scale)


@given(graph=graphs())
def test_invariants(graph):
    """Every class dies: the extended bars of dimension 0 are the components (minus the one of the cone vertex, none
    left out), from each component's minimum to its maximum; and the extended bars of dimension 1 number the graph's
    independent loops, less those whose highest and lowest points are equal (of length zero)."""
    n, edges, values = graph
    barcode = extended(n, edges, node_weight=values)
    graph_ = nx.Graph()
    graph_.add_nodes_from(range(n))
    graph_.add_edges_from(edges)
    ranges = sorted(
        (min(values[v] for v in c), max(values[v] for v in c)) for c in nx.connected_components(graph_)
    )
    assert barcode.extended_plus[0] == [r for r in ranges if r[0] < r[1]]
    assert barcode.extended_minus[0] == []
    loops = len(edges) - n + nx.number_connected_components(graph_)
    assert len(barcode.extended_minus[1]) + len(barcode.extended_plus[1]) <= loops
    for b, d in barcode.extended_minus[1]:
        assert b > d


def test_networkx_and_gudhi_format():
    graph = nx.cycle_graph(4)
    for v, h in zip(graph.nodes, [0.0, 2.0, 1.0, 3.0]):
        graph.nodes[v]["height"] = h
    barcode = extended(graph, node_weight="height")
    assert barcode.ordinary[0] == [(1.0, 2.0)]
    assert barcode.relative[1] == [(2.0, 1.0)]
    assert barcode.extended_plus[0] == [(0.0, 3.0)]
    assert barcode.extended_minus[1] == [(3.0, 0.0)]
    assert barcode.to_gudhi() == [[(0, (1.0, 2.0))], [(1, (2.0, 1.0))], [(0, (0.0, 3.0))], [(1, (3.0, 0.0))]]
    assert "extended_minus=[H1: 1]" in repr(barcode)
    # A sequence of values, in the order of G.nodes, gives the same
    assert extended(graph, node_weight=[0.0, 2.0, 1.0, 3.0]).to_gudhi() == barcode.to_gudhi()


def test_input_validation():
    with pytest.raises(TypeError):
        extended(nx.cycle_graph(3))  # node_weight is required
    with pytest.raises(ValueError, match="no attribute"):
        extended(nx.cycle_graph(3), node_weight="height")
    with pytest.raises(ValueError, match="finite"):
        extended(2, [(0, 1)], node_weight=[0.0, math.nan])
    with pytest.raises(ValueError, match="modulus"):
        extended(2, [(0, 1)], node_weight=[0.0, 1.0], modulus=4)
    with pytest.raises(TypeError, match="directed_flag"):
        extended(nx.DiGraph([(0, 1)]), node_weight=[0.0, 1.0])
