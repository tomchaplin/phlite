"""Grounded persistent path homology (GrPPH) of a weighted digraph, as in grpphati, written on top of phlite.

Cells are plain tagged tuples:
    ("v", a)                vertex a
    ("e", a, b)             the allowed 1-path a -> b
    ("d", a, b)             the double edge a -> b -> a
    ("t", a, m, b)          the directed triangle a -> m -> b, a -> b
    ("s", a, m, n, b)       the long square a -> m -> b minus a -> n -> b
The 2-cells are grpphati's filtration-adapted basis of the regular path complex. Boundaries are signed, so this also
works over Z/p (grpphati itself works over Z/2).

    python examples/grpph.py
"""

import math
import time
from collections import defaultdict

import networkx as nx

import phlite

DIMENSION = {"v": 0, "e": 1, "d": 2, "t": 2, "s": 2}


def boundary(cell):
    kind, *vs = cell
    if kind == "v":
        return {}
    if kind == "e":
        a, b = vs
        return {("v", b): 1, ("v", a): -1}
    if kind == "d":  # a -> b -> a
        a, b = vs
        return {("e", a, b): 1, ("e", b, a): 1}
    if kind == "t":  # a -> m -> b, a -> b
        a, m, b = vs
        return {("e", a, m): 1, ("e", m, b): 1, ("e", a, b): -1}
    if kind == "s":  # a -> m -> b minus a -> n -> b
        a, m, n, b = vs
        return {("e", a, m): 1, ("e", m, b): 1, ("e", a, n): -1, ("e", n, b): -1}
    raise ValueError(f"Unknown cell {cell}")


def grpph_cells(G, weight="weight"):
    """The cells of the grounded shortest-path filtration, with their entrance times."""
    d = {
        a: {b: t for b, t in ds.items() if b != a}
        for a, ds in nx.all_pairs_dijkstra_path_length(G, weight=weight)
    }

    def dist(a, b):
        return d[a].get(b, math.inf)

    # Grounded 0- and 1-cells: G itself from time 0, other pairs at their distance
    time = {("v", v): 0.0 for v in G}
    for a in d:
        for b, t in d[a].items():
            time["e", a, b] = 0.0 if G.has_edge(a, b) else float(t)

    # 2-cells, entering at their (ungrounded) times
    two_paths = sorted(
        ((max(d[a][m], d[m][b]), (a, m, b)) for a in d for m in d[a] for b in d[m]),
        key=lambda tp: tp[0],
    )
    bridges = defaultdict(list)
    for t, (a, m, b) in two_paths:
        if a == b:
            time["d", a, m] = float(t)
        elif dist(a, b) <= t:
            time["t", a, m, b] = float(t)
        else:
            bridges[a, b].append((m, t))
    for (a, b), mids in bridges.items():
        m0 = mids[0][0]
        if dist(a, b) < math.inf:
            time["t", a, m0, b] = float(
                dist(a, b)
            )  # collapses the first bridge onto the shortcut
        for m, t in mids[1:]:
            time["s", a, m0, m, b] = float(t)  # long squares
    return time


def grpph(G, modulus=2, representatives=False):
    """The barcode of GrPPH in dimension 1."""
    time = grpph_cells(G)
    chain = phlite.ChainComplex.from_cells(
        time.keys(),
        dimension=lambda c: DIMENSION[c[0]],
        boundary=boundary,
        modulus=modulus,
    )
    return chain.filtration(time).persistence(
        max_dim=1, representatives=representatives
    )


if __name__ == "__main__":
    # grpphati's README example: a long square, whose GrPPH is [[0, 3]]
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)], weight=3)
    barcode = grpph(G, representatives=True)
    print(barcode.diagram(1).tolist())
    for bar in barcode.bars:
        if bar.dimension == 1:
            print(
                f"  [{bar.birth_value}, {bar.death_value}): cycle {bar.representative}, killed by {bar.death}"
            )

    # grpphati's README benchmark: the wedge of two complete digraphs on 50 nodes, with 4802 bars
    N = 50
    G_1 = nx.relabel_nodes(
        nx.complete_graph(N, create_using=nx.DiGraph), lambda x: (x, 1) if x > 0 else x
    )
    G_2 = nx.relabel_nodes(
        nx.complete_graph(N, create_using=nx.DiGraph), lambda x: (x, 2) if x > 0 else x
    )
    G_wedge = nx.compose(G_1, G_2)
    start = time.perf_counter()
    cells = grpph_cells(G_wedge)
    built = time.perf_counter()
    chain = phlite.ChainComplex.from_cells(
        cells.keys(), dimension=lambda c: DIMENSION[c[0]], boundary=boundary
    )
    filtered = chain.filtration(cells)
    indexed = time.perf_counter()
    barcode = filtered.persistence(max_dim=1)
    reduced = time.perf_counter()
    print(
        f"Wedge of two complete digraphs: {len(barcode.diagram(1))} bars from {len(cells)} cells "
        f"(cells {built - start:.2f}s, ChainComplex + filtration {indexed - built:.2f}s, "
        f"persistence {reduced - indexed:.2f}s)"
    )
