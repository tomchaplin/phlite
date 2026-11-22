"""Fuzz phlite.networks' pph, directed_flag (grounded or not), ordered_tuples and dowker: against grpphati, flagser
(pyflagser), and each complex built by brute force and reduced by LoPHAT; and against invariants of each filtration.

grpphati's `OrderedTuplesHomology` leaves out every double edge `(a, b, a)` (it keeps a 2-cell only when
`edge_time((source, target))` is finite, and `edge_time((a, a))` never is), so it computes the directed flag
complex's homology instead; ordered tuples are checked against the brute-force complex only.
"""

import itertools
import math

import lophat
import networkx as nx
import numpy as np
import pyflagser
import pytest
from grpphati.filtrations import ShortestPathFiltration
from grpphati.homologies import DirectedFlagComplexHomology, RegularPathHomology
from grpphati.pipelines.grounded import make_grounded_pipeline
from grpphati.pipelines.standard import make_standard_pipeline
from hypothesis import given, note
from hypothesis import strategies as st

from grid import algorithm_grid
from mod2 import Span, chain_sum
from phlite.core import Algorithm
from phlite.networks import directed_flag, dowker, ordered_tuples, pph

INF = math.inf
METHODS = ["cohomology", "homology"]
ALGORITHMS = [
    Algorithm.standard(),
    Algorithm.serial(),
    Algorithm.parallel(),
    Algorithm.serial().with_columns("merged").with_cache(True),
    Algorithm.clearing().with_store_r(),
]


@st.composite
def digraphs(draw, max_nodes=7):
    """Weighted digraphs without self-loops or parallel edges: small integer weights (lots of ties) or float32 ones,
    sometimes infinite (the edge never appears)."""
    n = draw(st.integers(1, max_nodes))
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    density = draw(st.sampled_from([0.2, 0.5, 0.9]))
    if draw(st.booleans()):
        weight = st.integers(0, 3).map(float)
    else:
        weight = st.floats(0, 10, allow_nan=False, width=32)
    if draw(st.booleans()):
        weight = st.one_of(weight, st.just(INF))
    edges = []
    for a, b in pairs:
        if draw(st.floats(0, 1)) < density:
            edges.append((a, b, draw(weight)))
    return n, edges


def diagram(barcode, dim):
    return sorted(list(barcode.pairings[dim]) + [(b, INF) for b in barcode.essential[dim]])


def diagrams(barcode):
    return [diagram(barcode, dim) for dim in range(len(barcode.pairings))]


# ====== Brute force ========================================================


def distances(n, edges, filtration):
    """`d[(a, b)]`: when the edge a -> b enters ('weight': its weight; 'shortest_path': the shortest path's length)."""
    finite = [(a, b, w) for a, b, w in edges if w != INF]
    if filtration == "weight":
        return {(a, b): w for a, b, w in finite}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(finite)
    return {
        (a, b): d for a, lengths in nx.all_pairs_dijkstra_path_length(graph) for b, d in lengths.items() if a != b
    }


def ordered_complex(n, d, max_vertices, repeats):
    """Every ordered simplex (directed flag, or ordered tuples if `repeats`) with at most `max_vertices` vertices,
    with its value."""
    cells = {(v,): 0.0 for v in range(n)}
    current = [(v,) for v in range(n)]
    for _ in range(max_vertices - 1):
        longer = []
        for s in current:
            for w in range(n):
                if w == s[-1] or (not repeats and w in s):
                    continue
                t = s + (w,)
                value = 0.0
                ok = True
                for i, j in itertools.combinations(range(len(t)), 2):
                    if t[i] == t[j]:
                        continue
                    if (t[i], t[j]) not in d:
                        ok = False
                        break
                    value = max(value, d[(t[i], t[j])])
                if ok:
                    cells[t] = value
                    longer.append(t)
        current = longer
    return cells


def ordered_boundary(cell):
    """The faces of an ordered simplex, leaving out degenerate ones (a vertex next to itself)."""
    faces = []
    for i in range(len(cell)):
        face = cell[:i] + cell[i + 1 :]
        if all(face[k] != face[k + 1] for k in range(len(face) - 1)):
            faces.append(face)
    return faces if len(cell) > 1 else []


def dowker_complex(n, edges, max_vertices, side):
    omega = {(a, b): w for a, b, w in edges if w != INF}

    def toward(v, x):
        if v == x:
            return 0.0
        pair = (v, x) if side == "sink" else (x, v)
        return omega.get(pair, INF)

    cells = {}
    for k in range(1, max_vertices + 1):
        for s in itertools.combinations(range(n), k):
            value = min(max(toward(v, x) for v in s) for x in range(n))
            if value != INF:
                cells[s] = value
    return cells


def set_boundary(cell):
    return [cell[:i] + cell[i + 1 :] for i in range(len(cell))] if len(cell) > 1 else []


def lophat_diagrams(cells, boundary, max_dim):
    """The barcode of the filtered complex `cells` (cell -> value) over Z/2, by LoPHAT, in dimensions 0..max_dim."""
    order = sorted(cells, key=lambda c: (cells[c], len(c), c))
    index = {c: i for i, c in enumerate(order)}
    columns = [(len(c) - 1, sorted(index[f] for f in boundary(c))) for c in order]
    pairing = lophat.compute_pairings(columns, True, lophat.LoPhatOptions())
    bars = [[] for _ in range(max_dim + 1)]
    paired = set()
    for b, d in pairing.paired:
        paired.update((b, d))
        dim = len(order[b]) - 1
        if dim <= max_dim and cells[order[b]] < cells[order[d]]:
            bars[dim].append((cells[order[b]], cells[order[d]]))
    for i, c in enumerate(order):
        if i not in paired and len(c) - 1 <= max_dim:
            bars[len(c) - 1].append((cells[c], INF))
    return [sorted(b) for b in bars]


def check_representatives(cells, boundary, barcode):
    """Each representative is a cycle of cells of the complex, born at its birth, a boundary at its death, and not
    before (over Z/2)."""
    for dim in range(len(barcode.pairings)):
        cofaces = [(v, c) for c, v in cells.items() if len(c) == dim + 2]

        def boundaries_before(time, strict):
            return Span(
                frozenset(boundary(c)) for v, c in cofaces if (v < time if strict else v <= time)
            )

        bars = list(zip(barcode.pairings[dim], barcode.pairing_representatives[dim]))
        bars += [((b, INF), rep) for b, rep in zip(barcode.essential[dim], barcode.essential_representatives[dim])]
        for (birth, death), rep in bars:
            note(f"dim={dim} bar=({birth}, {death}) rep={rep}")
            rep = [tuple(c) for c in rep]
            assert all(c in cells and len(c) == dim + 1 for c in rep), "not a cell of the complex"
            if dim > 0:
                assert not chain_sum(frozenset(boundary(c)) for c in rep), "not a cycle"
            assert max(cells[c] for c in rep) == birth, "not born at birth"
            chain = frozenset(rep)
            assert chain not in boundaries_before(death, strict=True), "already a boundary before death"
            if death != INF:
                assert chain in boundaries_before(death, strict=False), "not a boundary at death"


# ====== Against brute force (LoPHAT) ======================================


@given(graph=digraphs(), max_dim=st.integers(0, 2), filtration=st.sampled_from(["weight", "shortest_path"]))
def test_directed_flag_and_ordered_tuples_match_brute_force(graph, max_dim, filtration):
    n, edges = graph
    d = distances(n, edges, filtration)
    for function, repeats in [(directed_flag, False), (ordered_tuples, True)]:
        cells = ordered_complex(n, d, max_dim + 2, repeats)
        expected = lophat_diagrams(cells, ordered_boundary, max_dim)
        for method, algorithm in itertools.product(METHODS, ALGORITHMS):
            note(f"{function.__name__} method={method} algorithm={algorithm}")
            barcode = function(n, edges, max_dim, filtration=filtration, method=method, algorithm=algorithm, representatives=True)
            assert diagrams(barcode) == expected
            check_representatives(cells, ordered_boundary, barcode)


@given(graph=digraphs(max_nodes=6), max_dim=st.integers(0, 2), side=st.sampled_from(["sink", "source"]))
def test_dowker_matches_brute_force(graph, max_dim, side):
    n, edges = graph
    cells = dowker_complex(n, edges, max_dim + 2, side)
    expected = lophat_diagrams(cells, set_boundary, max_dim)
    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        note(f"method={method} algorithm={algorithm}")
        barcode = dowker(n, edges, max_dim, side=side, method=method, algorithm=algorithm, representatives=True)
        assert diagrams(barcode) == expected
        check_representatives(cells, set_boundary, barcode)


def grounded_cells(n, edges):
    """The grounded directed flag complex up to dimension 2, as grpphati grounds it."""
    sp = distances(n, edges, "shortest_path")
    cells = {(v,): 0.0 for v in range(n)}
    for (a, b), value in sp.items():
        cells[(a, b)] = value
    for a, b, _ in edges:
        cells[(a, b)] = 0.0
    for t, value in ordered_complex(n, sp, 3, False).items():
        if len(t) == 3:
            cells[t] = value
    return cells


@given(graph=digraphs())
def test_grounded_directed_flag_matches_brute_force(graph):
    n, edges = graph
    cells = grounded_cells(n, edges)
    expected = lophat_diagrams(cells, ordered_boundary, 1)
    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        barcode = directed_flag(n, edges, grounded=True, method=method, algorithm=algorithm, representatives=True)
        assert diagrams(barcode) == expected
        check_representatives(cells, ordered_boundary, barcode)


# ====== Against grpphati and flagser ======================================


def networkx_graph(n, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(edges)
    return graph


def grpphati_diagram(n, edges, pipeline, homology):
    finite = [(a, b, w) for a, b, w in edges if w != INF]
    # grpphati grounds at the networkx graph's edges, whatever their weight: give it the infinite ones too
    result = pipeline(ShortestPathFiltration, homology)(networkx_graph(n, finite if pipeline is make_standard_pipeline else edges))
    return sorted((float(b), float(d)) for b, d in result.barcode if d > b)


@given(graph=digraphs())
def test_against_grpphati(graph):
    n, edges = graph
    finite = [(a, b, w) for a, b, w in edges if w != INF]
    if not finite:
        return
    note(f"edges={edges}")
    assert diagram(pph(n, edges, filtration="shortest_path"), 1) == grpphati_diagram(
        n, edges, make_standard_pipeline, RegularPathHomology
    )
    assert diagram(directed_flag(n, edges, filtration="shortest_path"), 1) == grpphati_diagram(
        n, edges, make_standard_pipeline, DirectedFlagComplexHomology
    )
    assert diagram(directed_flag(n, edges, grounded=True), 1) == grpphati_diagram(
        n, edges, make_grounded_pipeline, DirectedFlagComplexHomology
    )


@given(graph=digraphs(), max_dim=st.integers(0, 3))
def test_directed_flag_matches_flagser(graph, max_dim):
    n, edges = graph
    adjacency = np.full((n, n), INF)
    np.fill_diagonal(adjacency, 0.0)
    for a, b, w in edges:
        adjacency[a, b] = w
    dgms = pyflagser.flagser_weighted(adjacency, max_dimension=max_dim)["dgms"]
    barcode = directed_flag(n, edges, max_dim)
    for dim in range(max_dim + 1):
        expected = sorted((float(b), float(d)) for b, d in (dgms[dim] if dim < len(dgms) else []) if d > b)
        assert diagram(barcode, dim) == expected, f"dim {dim}"


# ====== Invariants =========================================================


def reversed_edges(edges):
    return [(b, a, w) for a, b, w in edges]


@given(graph=digraphs(), max_dim=st.integers(0, 2), seed=st.integers(0, 2**32 - 1))
def test_invariants(graph, max_dim, seed):
    """Reversing every edge changes no barcode (it reverses each ordered simplex, and swaps Dowker's sinks and
    sources); relabelling the nodes changes nothing; doubling every weight doubles every value; and Dowker duality:
    the sink and source complexes have the same barcode."""
    n, edges = graph
    perm = np.random.default_rng(seed).permutation(n)
    relabelled = [(int(perm[a]), int(perm[b]), w) for a, b, w in edges]
    doubled = [(a, b, 2 * w) for a, b, w in edges]
    cases = [
        ("pph", lambda e: pph(n, e)),
        ("directed_flag", lambda e: directed_flag(n, e, max_dim)),
        ("directed_flag sp", lambda e: directed_flag(n, e, max_dim, filtration="shortest_path")),
        ("ordered_tuples", lambda e: ordered_tuples(n, e, max_dim)),
        ("dowker", lambda e: dowker(n, e, max_dim)),
    ]
    # Shortest paths sum weights along a path in an order that reversing or relabelling can change, so values may
    # differ in the last bit (2.0000000000000018 against 2.000000000000002): compare them rounded
    def rounded(ds):
        return [sorted((round(b, 9), round(d, 9)) for b, d in dim) for dim in ds]

    for name, compute in cases:
        base = diagrams(compute(edges))
        expected = rounded(base)
        assert rounded(diagrams(compute(reversed_edges(edges)))) == expected, f"{name}: reversed"
        assert rounded(diagrams(compute(relabelled))) == expected, f"{name}: relabelled"
        assert rounded(diagrams(compute(doubled))) == rounded([[(2 * b, 2 * d) for b, d in dim] for dim in base]), f"{name}: doubled"
    sink = diagrams(dowker(n, edges, max_dim))
    assert diagrams(dowker(n, edges, max_dim, side="source")) == sink, "Dowker duality"
    assert diagrams(dowker(n, reversed_edges(edges), max_dim, side="source")) == sink


@given(graph=digraphs())
def test_grounded_cycles(graph):
    """Grounded: every H1 bar is a cycle of the digraph, born at 0, and there are as many as the cycle rank of its
    underlying graph (each directed edge an edge of it, so a pair of opposite edges makes a cycle), if every weight is
    positive (else a cycle can be filled in at 0, a bar of length zero, left out)."""
    n, edges = graph
    barcode = directed_flag(n, edges, grounded=True)
    h1 = diagram(barcode, 1)
    assert all(b == 0.0 for b, _ in h1)
    underlying = nx.MultiGraph()
    underlying.add_nodes_from(range(n))
    underlying.add_edges_from((a, b) for a, b, _ in edges)
    rank = len(edges) - n + nx.number_connected_components(underlying)
    if all(w > 0 for _, _, w in edges):
        assert len(h1) == rank
    else:
        assert len(h1) <= rank


@given(graph=digraphs(max_nodes=5))
def test_every_algorithm_agrees(graph):
    n, edges = graph
    for name, compute in [
        ("directed_flag", lambda a, m: directed_flag(n, edges, 2, algorithm=a, method=m)),
        ("ordered_tuples", lambda a, m: ordered_tuples(n, edges, 2, algorithm=a, method=m)),
        ("dowker", lambda a, m: dowker(n, edges, 2, algorithm=a, method=m)),
        ("pph", lambda a, m: pph(n, edges, algorithm=a, method=m)),
        ("grounded", lambda a, m: directed_flag(n, edges, grounded=True, algorithm=a, method=m)),
    ]:
        expected = compute(Algorithm.serial(), "cohomology")
        for method, algorithm in itertools.product(METHODS, algorithm_grid(Algorithm, row_ranks=False)):
            barcode = compute(algorithm, method)
            assert (barcode.essential, barcode.pairings) == (expected.essential, expected.pairings), f"{name} {algorithm!r}"


@given(graph=digraphs())
def test_networkx_input(graph):
    """A networkx graph with shuffled, mixed labels gives the same barcode, with cells in its labels."""
    n, edges = graph
    labels = [f"n{i}" if i % 2 else i * 10 for i in range(n)]
    g = nx.DiGraph()
    g.add_nodes_from(reversed(labels))
    g.add_weighted_edges_from((labels[a], labels[b], w) for a, b, w in edges)
    for function in [directed_flag, ordered_tuples, dowker, pph]:
        by_index = function(n, edges, representatives=True)
        by_label = function(g, representatives=True)
        assert (by_label.essential, by_label.pairings) == (by_index.essential, by_index.pairings)
        for reps in by_label.pairing_representatives + by_label.essential_representatives:
            for rep in reps:
                assert all(v in labels for cell in rep for v in cell)


def test_validation():
    with pytest.raises(ValueError, match="grounded"):
        directed_flag(2, [(0, 1, 1.0)], 2, grounded=True)
    with pytest.raises(ValueError, match="shortest-path"):
        directed_flag(2, [(0, 1, 1.0)], grounded=True, filtration="weight")
    with pytest.raises(ValueError, match="Unknown filtration"):
        ordered_tuples(2, [(0, 1, 1.0)], filtration="geodesic")
    with pytest.raises(ValueError, match="Unknown side"):
        dowker(2, [(0, 1, 1.0)], side="middle")
    with pytest.raises(ValueError, match="self-loop"):
        directed_flag(2, [(1, 1, 1.0)])
    with pytest.raises(TypeError, match="Missing edges"):
        dowker(3)
    for pipeline in ["networks.pph", "networks.directed_flag", "networks.ordered_tuples", "networks.dowker"]:
        resolved = Algorithm.serial().resolve(pipeline)
        assert resolved.columns in ("heap", "merged")
