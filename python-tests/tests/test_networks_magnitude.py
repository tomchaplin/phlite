"""Fuzz phlite.networks.magnitude (magnitude homology of digraphs) against Perseus 6's integral magnitude homology:
the ranks over Z/p follow from the integral groups by universal coefficients (the free rank, plus the torsion summands
of order divisible by p in degrees k and k - 1). Random small digraphs, and Kaneta and Yoshinaga's graph of RP^2,
which has 2-torsion."""

import itertools
import warnings

import perseus
from hypothesis import given, note, settings
from hypothesis import strategies as st

import networkx as nx
from phlite.networks import magnitude

MODULI = [2, 3, 5, 251]


def perseus_ranks(n, edges, l_max, p, pairs=None):
    """{(k, l): rank over Z/p} from Perseus's integral groups."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = dict(n=n, directed=True, lmax=l_max)
        if pairs:
            kwargs["pairs"] = pairs
        table = perseus.magnitude(edges, **kwargs)["table"]
    divisible = lambda group: sum(m for order, m in group["torsion"].items() if int(order) % p == 0) if group else 0
    ranks = {}
    for (k, l), group in table.items():
        rank = group["free"] + divisible(group) + divisible(table.get((k - 1, l)))
        if rank:
            ranks[(k, l)] = rank
    return ranks


def phlite_ranks(table):
    return {(k, l): r for l, row in enumerate(table) for k, r in enumerate(row) if r}


@st.composite
def digraphs(draw):
    n = draw(st.integers(1, 7))
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    edges = draw(st.lists(st.sampled_from(pairs), unique=True, max_size=len(pairs))) if pairs else []
    return n, edges


@given(digraphs(), st.integers(0, 5))
@settings(deadline=None)
def test_ranks_match_perseus(graph, l_max):
    n, edges = graph
    note(f"{n} {edges} l_max={l_max}")
    if not edges:
        # Perseus wants at least one edge; the homology is then one generator per node in MH_{0,0}
        assert magnitude(n, edges, l_max=l_max) == [[n]] + [[0] * (l + 1) for l in range(1, l_max + 1)]
        return
    together = magnitude(n, edges, l_max=l_max, modulus=MODULI)
    for p in MODULI:
        assert phlite_ranks(together[p]) == perseus_ranks(n, edges, l_max, p), p
        assert magnitude(n, edges, l_max=l_max, modulus=p) == together[p]


@given(digraphs(), st.integers(1, 5), st.data())
@settings(deadline=None)
def test_pairs_and_degrees(graph, l_max, data):
    """Each pair on its own, and fewer degrees, as a slice of the whole."""
    n, edges = graph
    by_pair = magnitude(n, edges, l_max=l_max, by_pair=True)
    s, t = data.draw(st.integers(0, n - 1)), data.draw(st.integers(0, n - 1))
    k_max = data.draw(st.integers(0, l_max))
    one = magnitude(n, edges, l_max=l_max, k_max=k_max, pairs=[(s, t)], by_pair=True)
    want = [row[: k_max + 1] for row in by_pair.get((s, t), [])]
    # (Pairs whose ranks, up to k_max, are all zero are left out)
    if not any(r for row in want for r in row):
        assert one == {}
    else:
        assert one == {(s, t): want}


def kaneta_yoshinaga_rp2():
    """The Hasse diagram of the face poset of the 6-vertex RP^2, with a bottom and a top: 33 nodes."""
    triangles = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1), (1, 2, 4), (2, 3, 5), (3, 4, 1), (4, 5, 2), (5, 1, 3)]
    faces = {frozenset(s) for t in triangles for k in (1, 2, 3) for s in itertools.combinations(t, k)}
    g = nx.Graph()
    for face in faces:
        for v in face:
            if len(face) > 1:
                g.add_edge(face, face - {v})
        if len(face) == 1:
            g.add_edge("bottom", face)
        if len(face) == 3:
            g.add_edge(face, "top")
    return g


def test_rp2_has_2_torsion():
    """MH_{3,4}(bottom, top) of Kaneta and Yoshinaga's graph of RP^2 is H_1(RP^2) = Z/2: rank 1 over Z/2 (and
    MH_{4,4} too, by universal coefficients), 0 over Z/3 and Z/251."""
    g = kaneta_yoshinaga_rp2().to_directed()
    ranks = magnitude(g, l_max=4, pairs=[("bottom", "top")], modulus=[2, 3, 251], by_pair=True)
    assert len(g) == 33
    assert ranks[2][("bottom", "top")][4][3] == 1
    assert ranks[2][("bottom", "top")][4][4] == 1
    assert ranks[3] == {} or ranks[3][("bottom", "top")][4][3:] == [0, 0]
    assert ranks[251] == {} or ranks[251][("bottom", "top")][4][3:] == [0, 0]
