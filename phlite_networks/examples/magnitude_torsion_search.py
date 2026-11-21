"""Search for graphs and digraphs with torsion in their magnitude homology, with `phlite.networks.magnitude`.

By default the integral groups are computed exactly (`magnitude(..., integral=True)`: each piece collapsed and
eliminated over Z, then Smith forms of what is left), and the torsion reported by pair and bidegree. With
`--fields`, torsion is found instead by comparing ranks over several fields (by universal coefficients, if the rank
of MH_{k,l} over Z/p differs from its rank over Z/251, a stand-in for Q, then MH_{k,l} or MH_{k-1,l} has p-torsion),
and candidates can be confirmed with Perseus 6 (`--confirm`, needs `pip install perseus-tda` or the pinned build in
python-tests).

Magnitude homology is torsion-free in degrees k <= 2 (k = 0, 1 by construction; k = 2: Gomi for metric spaces,
Ivanov and Mukoseev for digraphs) and on the diagonal k = l, so torsion first shows in k = 3, at l >= 4. The known
small examples are Kaneta and Yoshinaga's graphs of triangulated manifolds: the Hasse diagram of the face poset of
RP^2 with a bottom and a top, 33 nodes for the 6-vertex triangulation, 15 for a cellular one (2-torsion in MH_{3,4}).

    python magnitude_torsion_search.py atlas --l-max 6               # every graph on at most 7 nodes
    python magnitude_torsion_search.py random --n 8 12 --count 500   # random G(n, p)
    python magnitude_torsion_search.py digraph --n 5 8 --count 2000  # random digraphs
    python magnitude_torsion_search.py layered --n 8 14 --count 500  # random layered digraphs (Hasse-like)
    python magnitude_torsion_search.py geng --nauty "-c 8"           # every connected graph on 8 nodes (nauty)
    python magnitude_torsion_search.py directg --nauty "-c 5"        # every digraph on them (nauty's directg)

Options: --l-max, --moduli 2,3,5 (compared with 251), --seed, --confirm.
"""

import argparse
import random
import sys
import time

import networkx as nx
from phlite.networks import magnitude

REFERENCE = 251


def torsion_candidates(n, edges, l_max, moduli):
    """[(s, t, k, l, p, rank over Z/p, rank over Z/251)] where the ranks differ."""
    ranks = magnitude(n, edges, l_max=l_max, modulus=moduli + [REFERENCE], by_pair=True)
    out = []
    reference = ranks[REFERENCE]
    for p in moduli:
        for pair, table in ranks[p].items():
            base = reference.get(pair)
            for l, row in enumerate(table):
                for k, r in enumerate(row):
                    r0 = base[l][k] if base else 0
                    if r != r0:
                        out.append((*pair, k, l, p, r, r0))
    return out


def integral_torsion(n, edges, l_max):
    """[(s, t, k, l, {order: multiplicity})] of the integral torsion of each pair, and the bidegrees whose Smith form
    couldn't be computed."""
    groups = magnitude(n, edges, l_max=l_max, integral=True, by_pair=True)
    found, unknown = [], []
    for (s, t), g in groups.items():
        for (k, l), orders in g["torsion"].items():
            found.append((s, t, k, l, orders))
        for l, row in enumerate(g["free"]):
            for k, r in enumerate(row):
                if r is None:
                    unknown.append((s, t, k, l))
    return found, unknown


def confirm(n, edges, l_max, pairs):
    """Perseus's integral torsion, {(k, l): {order: multiplicity}}, over the pairs given."""
    import warnings

    import perseus

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = perseus.magnitude(edges, n=n, directed=True, lmax=l_max, pairs=pairs)["table"]
    return {key: group["torsion"] for key, group in table.items() if group["torsion"]}


def atlas():
    for g in nx.graph_atlas_g():
        if g.number_of_nodes() > 0 and nx.is_connected(g):
            yield g.number_of_nodes(), [(a, b) for a, b in g.edges()] + [(b, a) for a, b in g.edges()]


def random_graphs(rng, n_range, count):
    for _ in range(count):
        n = rng.randint(*n_range)
        p = rng.uniform(0.15, 0.6)
        g = nx.gnp_random_graph(n, p, seed=rng.randrange(1 << 30))
        if nx.is_connected(g):
            yield n, [(a, b) for a, b in g.edges()] + [(b, a) for a, b in g.edges()]


def random_digraphs(rng, n_range, count):
    for _ in range(count):
        n = rng.randint(*n_range)
        p = rng.uniform(0.1, 0.5)
        yield n, [(a, b) for a in range(n) for b in range(n) if a != b and rng.random() < p]


def layered_digraphs(rng, n_range, count):
    """Edges only from one layer to the next, as in a Hasse diagram, with a bottom and a top joined to every node of
    the first and last layers."""
    for _ in range(count):
        n = rng.randint(*n_range)
        inner = n - 2
        layers = rng.randint(2, max(2, min(4, inner)))
        assignment = sorted(rng.randrange(layers) for _ in range(inner))
        by_layer = [[i + 1 for i, a in enumerate(assignment) if a == layer] for layer in range(layers)]
        if any(not layer for layer in by_layer):
            continue
        p = rng.uniform(0.3, 0.8)
        edges = [(0, v) for v in by_layer[0]] + [(v, n - 1) for v in by_layer[-1]]
        for lower, upper in zip(by_layer, by_layer[1:]):
            for a in lower:
                for b in upper:
                    if rng.random() < p:
                        edges.append((a, b))
        yield n, edges


def nauty(geng_args, directed, directg_args=""):
    """Every graph (or, with `directed`, every orientation of every graph: an edge one way, the other, or both) that
    nauty's `geng` lists with these arguments."""
    import shlex
    import subprocess

    geng = subprocess.Popen(["geng", "-q", *shlex.split(geng_args)], stdout=subprocess.PIPE)
    if not directed:
        for line in geng.stdout:
            g = nx.from_graph6_bytes(line.strip())
            yield g.number_of_nodes(), [(a, b) for a, b in g.edges()] + [(b, a) for a, b in g.edges()]
        return
    directg = subprocess.Popen(["directg", "-q", "-T", *shlex.split(directg_args)], stdin=geng.stdout, stdout=subprocess.PIPE)
    for line in directg.stdout:
        numbers = list(map(int, line.split()))
        n, m = numbers[0], numbers[1]
        yield n, [(numbers[2 + 2 * i], numbers[3 + 2 * i]) for i in range(m)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=["atlas", "random", "digraph", "layered", "geng", "directg"])
    parser.add_argument("--nauty", default="-c 6", help="geng's arguments (geng and directg families)")
    parser.add_argument("--directg", default="", help="directg's arguments, e.g. -o (one direction per edge)")
    parser.add_argument("--l-max", type=int, default=6)
    parser.add_argument("--n", type=int, nargs=2, default=[5, 8])
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--moduli", default="2,3,5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirm", action="store_true", help="with --fields: confirm over Z with Perseus")
    parser.add_argument("--fields", action="store_true", help="compare ranks over fields instead of computing over Z")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    moduli = [int(m) for m in args.moduli.split(",")]
    graphs = {
        "atlas": lambda: atlas(),
        "random": lambda: random_graphs(rng, args.n, args.count),
        "digraph": lambda: random_digraphs(rng, args.n, args.count),
        "layered": lambda: layered_digraphs(rng, args.n, args.count),
        "geng": lambda: nauty(args.nauty, False),
        "directg": lambda: nauty(args.nauty, True, args.directg),
    }[args.family]()
    start, checked, found = time.time(), 0, 0
    for n, edges in graphs:
        checked += 1
        if not args.fields:
            found_here, unknown = integral_torsion(n, edges, args.l_max)
            if found_here or unknown:
                found += bool(found_here)
                print(f"n={n} edges={edges}", flush=True)
                for s, t, k, l, orders in found_here:
                    print(f"    pair ({s}, {t}) MH_{{{k},{l}}} torsion {orders}")
                for s, t, k, l in unknown:
                    print(f"    pair ({s}, {t}) MH_{{{k},{l}}}: Smith form not computed")
            continue
        candidates = torsion_candidates(n, edges, args.l_max, moduli)
        if candidates:
            found += 1
            print(f"n={n} edges={edges}", flush=True)
            for c in candidates:
                print(f"    pair ({c[0]}, {c[1]}) MH_{{{c[2]},{c[3]}}}: rank {c[5]} over Z/{c[4]}, {c[6]} over Z/{REFERENCE}")
            if args.confirm:
                pairs = sorted({(c[0], c[1]) for c in candidates})
                print(f"    Perseus, over Z: {confirm(n, edges, args.l_max, pairs)}", flush=True)
    print(f"checked {checked} graphs in {time.time() - start:.1f} s, {found} with torsion", file=sys.stderr)


if __name__ == "__main__":
    main()
