"""Shrink a graph with torsion in its magnitude homology to a locally minimal one: repeatedly delete a node, then an
edge, keeping a deletion whenever some torsion survives (integral groups up to `--l-max`, `phlite.networks.magnitude`).

    python magnitude_torsion_minimise.py graph.edges [--l-max 6] [--directed]

The file has the number of nodes on its first line, then one edge `a b` per line (both directions unless
`--directed`), as Perseus's `mag_families` writes Kaneta-Yoshinaga graphs.
"""

import argparse
import random

import networkx as nx
from phlite.networks import magnitude


def torsion(g, l_max):
    if g.number_of_edges() == 0:
        return {}
    h = nx.convert_node_labels_to_integers(g)
    return magnitude(h.number_of_nodes(), list(h.edges()), l_max=l_max, integral=True)["torsion"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--l-max", type=int, default=6)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    lines = open(args.file).read().split("\n")
    n = int(lines[0])
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for line in lines[1:]:
        if line.strip():
            a, b = map(int, line.split())
            g.add_edge(a, b)
            if not args.directed:
                g.add_edge(b, a)
    rng = random.Random(args.seed)
    print(f"start: {g.number_of_nodes()} nodes, {g.number_of_edges()} arcs, torsion {torsion(g, args.l_max)}", flush=True)
    changed = True
    while changed:
        changed = False
        nodes = list(g.nodes())
        rng.shuffle(nodes)
        for v in nodes:
            h = g.copy()
            h.remove_node(v)
            if torsion(h, args.l_max):
                g, changed = h, True
                print(f"  - node {v}: {g.number_of_nodes()} nodes, {g.number_of_edges()} arcs", flush=True)
        edges = list(g.edges()) if args.directed else list({tuple(sorted(e)) for e in g.edges()})
        rng.shuffle(edges)
        for a, b in edges:
            h = g.copy()
            h.remove_edge(a, b)
            if not args.directed and h.has_edge(b, a):
                h.remove_edge(b, a)
            if torsion(h, args.l_max):
                g, changed = h, True
                print(f"  - edge ({a}, {b}): {g.number_of_nodes()} nodes, {g.number_of_edges()} arcs", flush=True)
    print(f"minimal: {g.number_of_nodes()} nodes, {g.number_of_edges()} arcs, torsion {torsion(g, args.l_max)}")
    print(f"arcs: {sorted(g.edges())}")


if __name__ == "__main__":
    main()
