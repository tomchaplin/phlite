"""Fuzz phlite.networks.flag (flag complexes of undirected graphs, by edge weights, node values or both) against
GUDHI's SimplexTree, and check its representatives from first principles."""

import itertools
import math

import gudhi
import networkx as nx
import numpy as np
import pytest
import scipy.sparse
from hypothesis import given, note
from hypothesis import strategies as st

from grid import algorithm_grid
from mod2 import Span, boundary, chain_sum
from phlite.clouds import rips
from phlite.core import Algorithm
from phlite.networks import flag

INF = math.inf
METHODS = ["cohomology", "homology"]


@st.composite
def graphs(draw, max_nodes=8):
    """Undirected graphs `(n, edges, node_values, weighted)`: small integer weights and values (lots of ties) or
    floats, weights sometimes infinite or (with node values) negative, node values sometimes absent, and sometimes no
    edge weights at all (the lower-star filtration of the node values)."""
    n = draw(st.integers(1, max_nodes))
    density = draw(st.sampled_from([0.3, 0.6, 0.95]))
    weighted = draw(st.sampled_from([True, True, False]))
    has_values = (not weighted) or draw(st.booleans())
    if draw(st.booleans()):
        number = st.integers(-3, 3).map(float)
    else:
        number = st.floats(-10, 10, allow_nan=False, width=32)
    weight = number if has_values else number.map(abs)
    if draw(st.booleans()):
        weight = st.one_of(weight, st.just(INF))
    edges = []
    for u, v in itertools.combinations(range(n), 2):
        if draw(st.floats(0, 1)) < density:
            edges.append((u, v, draw(weight)) if weighted else (u, v))
    values = [draw(number) for _ in range(n)] if has_values else None
    return n, edges, values, weighted


def edge_values(n, edges, values, weighted):
    """When each edge enters: max(weight, both node values), or never."""
    f = values or [0.0] * n
    out = {}
    for edge in edges:
        u, v = edge[:2]
        w = edge[2] if weighted else -INF
        if w == INF:
            continue
        out[frozenset((u, v))] = max(w, f[u], f[v])
    return out


def value(n, values, d, simplex):
    """When a simplex enters (INF if it isn't a clique)."""
    f = values or [0.0] * n
    result = max(f[v] for v in simplex)
    for u, v in itertools.combinations(simplex, 2):
        result = max(result, d.get(frozenset((u, v)), INF))
    return result


def gudhi_diagrams(n, edges, values, weighted, max_dim):
    """GUDHI's: a SimplexTree with each node at its value and each edge at max(weight, its nodes'), expanded."""
    d = edge_values(n, edges, values, weighted)
    f = values or [0.0] * n
    tree = gudhi.SimplexTree()
    for v in range(n):
        tree.insert([v], filtration=f[v])
    for e, w in d.items():
        tree.insert(sorted(e), filtration=w)
    tree.expansion(max_dim + 1)
    tree.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
    return [
        sorted((float(b), float(e)) for b, e in tree.persistence_intervals_in_dimension(dim) if b < e)
        for dim in range(max_dim + 1)
    ]


def diagram(barcode, dim):
    return sorted(list(barcode.pairings[dim]) + [(b, INF) for b in barcode.essential[dim]])


def check_representatives(n, values, d, barcode, max_dim):
    """Every representative against the definition of persistent homology, by linear algebra over Z/2."""
    for dim in range(max_dim + 1):
        cofaces = [(value(n, values, d, s), s) for s in itertools.combinations(range(n), dim + 2)]
        cofaces = [(t, s) for t, s in cofaces if t != INF]

        def boundaries_before(time, strict):
            return Span(boundary(s) for t, s in cofaces if (t < time if strict else t <= time))

        bars = list(zip(barcode.pairings[dim], barcode.pairing_representatives[dim]))
        bars += [((b, INF), rep) for b, rep in zip(barcode.essential[dim], barcode.essential_representatives[dim])]
        for (birth, death), rep in bars:
            note(f"dim={dim} bar=({birth}, {death}) rep={rep}")
            assert len(rep) > 0 and len(set(rep)) == len(rep)
            assert all(len(cell) == dim + 1 and list(cell) == sorted(cell) for cell in rep)
            if dim > 0:
                assert not chain_sum(boundary(cell) for cell in rep), "not a cycle"
            assert max(value(n, values, d, cell) for cell in rep) == birth, "not born at birth"
            chain = set(rep)
            assert chain not in boundaries_before(death, strict=True), "already a boundary before death"
            if death != INF:
                assert chain in boundaries_before(death, strict=False), "not a boundary at death"


def call(n, edges, values, weighted, max_dim, **kwargs):
    return flag(n, edges, max_dim, weight="weight" if weighted else None, node_weight=values, **kwargs)


@given(graph=graphs(), max_dim=st.integers(0, 2))
def test_flag_matches_gudhi(graph, max_dim):
    n, edges, values, weighted = graph
    note(f"n={n} edges={edges} values={values} weighted={weighted}")
    expected = gudhi_diagrams(n, edges, values, weighted, max_dim)
    d = edge_values(n, edges, values, weighted)
    # Listed explicitly, computed implicitly (by Rips' machinery), and chosen automatically
    for method, explicit in itertools.product(METHODS, [None, True, False]):
        barcode = call(n, edges, values, weighted, max_dim, method=method, explicit=explicit, representatives=True)
        assert [diagram(barcode, dim) for dim in range(max_dim + 1)] == expected, (method, explicit)
        check_representatives(n, values, d, barcode, max_dim)


@given(graph=graphs(max_nodes=7), max_dim=st.integers(0, 2))
def test_every_algorithm_agrees(graph, max_dim):
    n, edges, values, weighted = graph
    expected = gudhi_diagrams(n, edges, values, weighted, max_dim)
    # Implicitly: Rips' options (union-find, no pivot columns); explicitly: phlite.complexes' (pivot columns)
    grids = [(False, algorithm_grid(Algorithm, row_ranks=False, union_find=True)), (True, algorithm_grid(Algorithm))]
    for explicit, grid in grids:
        for algorithm in grid:
            for method in METHODS:
                barcode = call(n, edges, values, weighted, max_dim, method=method, algorithm=algorithm, explicit=explicit)
                assert [diagram(barcode, dim) for dim in range(max_dim + 1)] == expected, (algorithm, method, explicit)
    # Asking for union-find chooses the implicit computation (which has it), and can't be had explicitly
    assert [diagram(call(n, edges, values, weighted, max_dim, algorithm=Algorithm.serial().with_union_find()), dim)
            for dim in range(max_dim + 1)] == expected
    with pytest.raises(ValueError):
        call(n, edges, values, weighted, max_dim, algorithm=Algorithm.serial().with_union_find(), explicit=True)


@given(graph=graphs(), max_dim=st.integers(0, 2), data=st.data())
def test_threshold(graph, max_dim, data):
    """Up to a threshold: the full barcode, cut there (bars alive at the threshold become essential)."""
    n, edges, values, weighted = graph
    threshold = data.draw(st.sampled_from([-1.0, 0.0, 1.0, 2.5, 5.0]))
    full = call(n, edges, values, weighted, max_dim)
    for explicit in [True, False]:
        cut = call(n, edges, values, weighted, max_dim, threshold=threshold, explicit=explicit)
        for dim in range(max_dim + 1):
            expected = sorted((b, d if d <= threshold else INF) for b, d in diagram(full, dim) if b <= threshold)
            assert diagram(cut, dim) == expected, explicit


@given(graph=graphs(), max_dim=st.integers(0, 2))
def test_without_node_values_it_is_rips_of_the_sparse_matrix(graph, max_dim):
    n, edges, values, weighted = graph
    if values is not None or not weighted:
        return
    rows, cols, data = [], [], []
    for u, v, w in edges:
        if w != INF:
            rows += [u, v]
            cols += [v, u]
            data += [w, w]
    matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    expected = rips(matrix, max_dim)
    barcode = call(n, edges, values, weighted, max_dim)
    for dim in range(max_dim + 1):
        assert diagram(barcode, dim) == diagram(expected, dim)


@given(graph=graphs(), max_dim=st.integers(0, 2))
def test_shortest_paths_match_gudhi(graph, max_dim):
    """filtration='shortest_path': the flag complex of the complete graph of path lengths (by networkx's Dijkstra),
    as GUDHI computes it; and without node values, Rips of the matrix of path lengths."""
    n, edges, values, weighted = graph
    if weighted and any(edge[2] < 0 for edge in edges):
        with pytest.raises(ValueError, match="non-negative"):
            call(n, edges, values, weighted, max_dim, filtration="shortest_path")
        return
    lengths = nx.Graph()
    lengths.add_nodes_from(range(n))
    for edge in edges:
        w = edge[2] if weighted else 1.0
        if w != INF and not (lengths.has_edge(*edge[:2]) and lengths.edges[edge[:2]]["weight"] <= w):
            lengths.add_edge(edge[0], edge[1], weight=w)
    paths = [(u, v, d) for u, row in nx.all_pairs_dijkstra_path_length(lengths) for v, d in row.items() if u < v]
    note(f"n={n} edges={edges} values={values} weighted={weighted} paths={paths}")
    expected = gudhi_diagrams(n, paths, values, True, max_dim)
    barcode = call(n, edges, values, weighted, max_dim, filtration="shortest_path", representatives=True)
    assert [diagram(barcode, dim) for dim in range(max_dim + 1)] == expected
    check_representatives(n, values, edge_values(n, paths, values, True), barcode, max_dim)
    if values is None:
        matrix = np.full((n, n), INF)
        np.fill_diagonal(matrix, 0.0)
        for u, v, d in paths:
            matrix[u, v] = matrix[v, u] = d
        expected = rips(matrix, max_dim)
        assert [diagram(barcode, dim) for dim in range(max_dim + 1)] == [
            diagram(expected, dim) for dim in range(max_dim + 1)
        ]


def test_networkx_graphs():
    """A networkx graph: weights and node values from attributes, representatives in node labels."""
    graph = nx.Graph()
    # A square a-b-c-d with heights; the diagonal a-c comes late
    for node, height in zip("abcd", [0.0, 1.0, 0.0, 2.0]):
        graph.add_node(node, height=height)
    graph.add_weighted_edges_from([("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0), ("d", "a", 1.0), ("a", "c", 5.0)])
    barcode = flag(graph, max_dim=1, representatives=True)
    assert diagram(barcode, 1) == [(1.0, 5.0)]
    assert sorted(barcode.pairing_representatives[1][0]) == [("a", "b"), ("a", "d"), ("b", "c"), ("c", "d")]
    # With heights: the loop is born when d (height 2) arrives
    assert diagram(flag(graph, max_dim=1, node_weight="height"), 1) == [(2.0, 5.0)]
    # Lower-star of the heights alone: the minima a and c (at 0) are joined at once by the diagonal (at max(0, 0)),
    # and each loop is filled as it appears (a-b-c by its triangle at 1, a-c-d at 2), so one component and no loops
    lower_star = flag(graph, max_dim=1, weight=None, node_weight="height")
    assert diagram(lower_star, 0) == [(0.0, INF)]
    assert diagram(lower_star, 1) == []
    # A sequence of values, in the order of G.nodes, works too
    assert diagram(flag(graph, max_dim=1, node_weight=[0.0, 1.0, 0.0, 2.0]), 1) == [(2.0, 5.0)]


def test_input_validation():
    with pytest.raises(TypeError, match="directed_flag"):
        flag(nx.DiGraph([(0, 1)]))
    with pytest.raises(ValueError, match="self-loop"):
        flag(nx.Graph([(0, 0)]))
    with pytest.raises(ValueError, match="self-loop"):
        flag(2, [(1, 1, 1.0)])
    with pytest.raises(ValueError, match="outside"):
        flag(2, [(0, 2, 1.0)])
    with pytest.raises(ValueError, match="NaN"):
        flag(2, [(0, 1, math.nan)])
    with pytest.raises(ValueError, match="node values for"):
        flag(2, [(0, 1)], node_weight=[0.0])
    with pytest.raises(ValueError, match="finite"):
        flag(2, [(0, 1)], node_weight=[0.0, INF])
    with pytest.raises(ValueError, match="no attribute"):
        flag(nx.path_graph(3), node_weight="height")
    with pytest.raises(TypeError, match="attribute name"):
        flag(2, [(0, 1)], node_weight="height")
    with pytest.raises(TypeError, match="Missing edges"):
        flag(3)
    with pytest.raises(ValueError, match="Unknown filtration"):
        flag(2, [(0, 1)], filtration="geodesic")
    # The hop distance of a 6-cycle: a loop from 1 to 2, as for six points on a circle
    assert diagram(flag(nx.cycle_graph(6), max_dim=1, weight=None, filtration="shortest_path"), 1) == [(1.0, 2.0)]
