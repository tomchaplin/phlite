"""Fuzz phlite_grpph against grpphati, and check its representatives."""

import itertools
import math

import networkx as nx
from grpphati.pipelines.grounded import GrPPH
from hypothesis import given, note
from hypothesis import strategies as st

from grid import algorithm_grid
from mod2 import Span
import phlite_grpph
from phlite_grpph import grpph

METHODS = ["cohomology", "homology"]
ALGORITHMS = [
    phlite_grpph.Algorithm.standard(),
    phlite_grpph.Algorithm.serial(),
    phlite_grpph.Algorithm.parallel(),
    phlite_grpph.Algorithm.parallel().with_storage("lock-free"),
    phlite_grpph.Algorithm.clearing().with_store_r(),
    # The other representation and caching from this package's default
    phlite_grpph.Algorithm.serial().with_columns("heap").with_cache(False),
]
INF = math.inf


@st.composite
def digraphs(draw, max_nodes=8, allow_infinite=True):
    """Weighted digraphs as (n_vertices, edges).

    Weights are multiples of 1/2 so that shortest-path lengths are computed exactly (and there are lots of ties).
    """
    n = draw(st.integers(1, max_nodes))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    chosen = draw(st.lists(st.sampled_from(pairs), unique=True, max_size=len(pairs))) if pairs else []
    weights = st.integers(0, 12).map(lambda w: w / 2)
    if allow_infinite and draw(st.booleans()):
        weights = st.one_of(weights, st.just(INF))
    return n, [(i, j, draw(weights)) for i, j in chosen]


def diagram(barcode, dim):
    # Keep finite and essential bars distinguishable: a "finite" bar must die at a finite time
    assert all(death != INF for _, death in barcode.pairings[dim]), "finite bar dies at infinity"
    assert all(birth != INF for birth, _ in barcode.pairings[dim]) and INF not in barcode.essential[dim]
    return sorted(list(barcode.pairings[dim]) + [(birth, INF) for birth in barcode.essential[dim]])


def to_networkx(n, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(edges)
    return graph


@given(graph=digraphs())
def test_methods_and_algorithms_agree(graph):
    n, edges = graph
    expected = grpph(n, edges)
    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        note(f"method={method} algorithm={algorithm}")
        barcode = grpph(n, edges, method=method, algorithm=algorithm, representatives=True)
        assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings)
        check_representatives(n, edges, barcode)


@given(graph=digraphs(allow_infinite=False))
def test_grpph_matches_grpphati(graph):
    n, edges = graph
    # grpphati only computes dimension 1
    expected = sorted((float(b), float(d)) for b, d in GrPPH(to_networkx(n, edges)).barcode if b < d)
    assert diagram(grpph(n, edges), 1) == expected


def check_representatives(n, edges, barcode):
    """Check the representatives in both dimensions."""
    finite_graph = to_networkx(n, [e for e in edges if e[2] != INF])
    shortest = dict(nx.all_pairs_dijkstra_path_length(finite_graph))
    in_graph = {(i, j) for i, j, _ in edges}

    def edge_time(a, b):
        # Grounding: edges of the graph are present from time 0, other edges at their shortest-path length
        return 0.0 if (a, b) in in_graph else shortest[a][b]

    # Dimension 0: everything is born at 0 and the graph's edges are too, so nothing finite survives,
    # and the essential classes are the weakly connected components (counting infinite-weight edges).
    assert barcode.pairings[0] == []
    components = list(nx.weakly_connected_components(to_networkx(n, edges)))
    assert len(barcode.essential[0]) == len(components)
    component_of = {v: k for k, comp in enumerate(components) for v in comp}
    # Each representative is a sum of nodes; its image in H_0 is the parity of its intersection with each component.
    # These images should be linearly independent, i.e. span H_0.
    images = Span()
    for rep in barcode.essential_representatives[0]:
        assert all(len(cell) == 1 for cell in rep)
        image = {k for k in range(len(components)) if sum(component_of[v] == k for (v,) in rep) % 2}
        assert image and image not in images, f"{rep} is dependent on earlier representatives"
        images.add(image)

    # Dimension 1: representatives are cycles made of edges that exist at the birth time
    bars = list(zip(barcode.pairings[1], barcode.pairing_representatives[1]))
    bars += [((b, INF), rep) for b, rep in zip(barcode.essential[1], barcode.essential_representatives[1])]
    for (birth, death), rep in bars:
        note(f"bar=({birth}, {death}) rep={rep}")
        assert rep and all(len(cell) == 2 for cell in rep)
        assert len(set(rep)) == len(rep), "repeated edge"
        degree = [0] * n
        for a, b in rep:
            degree[a] += 1
            degree[b] += 1
        assert all(d % 2 == 0 for d in degree), "not a cycle"
        assert max(edge_time(a, b) for a, b in rep) == birth, "not born at birth"


@given(graph=digraphs(), data=st.data())
def test_networkx_input_matches_integer_input(graph, data):
    n, edges = graph
    # Relabel nodes with a mix of label types, in a shuffled order
    labels = data.draw(st.permutations([f"v{i}" if i % 3 == 0 else (i, "x") if i % 3 == 1 else -i for i in range(n)]))
    G = nx.DiGraph()
    G.add_nodes_from(labels[::-1])  # node order differs from label order too
    G.add_weighted_edges_from((labels[i], labels[j], w) for i, j, w in edges)
    index = {label: k for k, label in enumerate(G.nodes)}
    renumbered = [(index[labels[i]], index[labels[j]], w) for i, j, w in edges]

    from_graph = grpph(G, representatives=True)
    from_ints = grpph(n, renumbered, representatives=True)
    assert (from_graph.essential, from_graph.pairings) == (from_ints.essential, from_ints.pairings)
    nodes = list(G.nodes)
    for reps_graph, reps_ints in [
        (from_graph.pairing_representatives, from_ints.pairing_representatives),
        (from_graph.essential_representatives, from_ints.essential_representatives),
    ]:
        relabelled = [[[tuple(nodes[v] for v in cell) for cell in rep] for rep in dim] for dim in reps_ints]
        assert reps_graph == relabelled


def test_default_columns():
    """Each package's presets use its own column representation and caching."""
    for algorithm in [phlite_grpph.Algorithm.serial(), phlite_grpph.Algorithm.parallel(), phlite_grpph.Algorithm.standard()]:
        assert not algorithm.store_r
        assert (algorithm.columns, algorithm.cache) == ('merged', True)


@given(graph=digraphs())
def test_every_algorithm_agrees(graph):
    """Every algorithm and option of the full grid (as for phlite_py) gives the default's barcode."""
    n, edges = graph
    expected = grpph(n, edges)
    for method, algorithm in itertools.product(METHODS, algorithm_grid(phlite_grpph.Algorithm)):
        note(f"method={method} algorithm={algorithm!r}")
        barcode = grpph(n, edges, method=method, algorithm=algorithm)
        assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings)
