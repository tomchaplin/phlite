"""Benchmark phlite_grpph against grpphati on random weighted digraphs.

Only dimension 1 is compared, since that is all grpphati computes.

Examples:
    just bench-grpph                                    # default sizes
    just bench-grpph --sizes 50 100 --p 0.2
    just bench-grpph --graph cycle --sizes 100 200
    just bench-grpph --all --only phlite                # every phlite method/algorithm
"""

import numpy as np
from harness import fingerprint, main

# Import everything up front, so that import costs are not counted in the time or peak memory of a run
from phlite_grpph import grpph
import networkx as nx
from grpphati.pipelines import grounded

INF = float("inf")


def phlite(method="cohomology", algorithm="clearing", representatives=False, parallel=False):
    def run(graph, case):
        n, edges = graph
        threads = case["threads"] if parallel else None
        barcode = grpph(
            n,
            edges,
            method=method,
            algorithm=algorithm,
            representatives=representatives,
            threads=threads,
        )
        return fingerprint([list(barcode.pairings[1]) + [(b, INF) for b in barcode.essential[1]]])

    return run


def grpphati(pipeline_name):
    def run(graph, case):
        n, edges = graph
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_weighted_edges_from(edges)
        result = getattr(grounded, pipeline_name)(g)
        return fingerprint([[(float(b), float(d)) for b, d in result.barcode]])

    return run


# Names starting with '*' only run with --all
IMPLEMENTATIONS = {
    "grpphati (GrPPH)": grpphati("GrPPH"),
    "grpphati (GrPPH_par_wedge)": grpphati("GrPPH_par_wedge"),
    "phlite (clearing)": phlite(),
    "phlite (parallel)": phlite(algorithm="parallel", parallel=True),
    "phlite (clearing, reps)": phlite(representatives=True),
    "*phlite (parallel-lock-free)": phlite(algorithm="parallel-lock-free", parallel=True),
    "*phlite (standard)": phlite(algorithm="standard"),
    "*phlite (homology, clearing)": phlite(method="homology"),
}


def make_input(case):
    """A weighted digraph with integer weights in 1..5."""
    rng = np.random.default_rng(case["seed"])
    n = case["n"]
    if case["graph"] == "random":
        mask = rng.uniform(size=(n, n)) < case["p"]
        np.fill_diagonal(mask, False)
        pairs = list(zip(*np.nonzero(mask)))
    elif case["graph"] == "cycle":
        pairs = [(i, (i + 1) % n) for i in range(n)]
    elif case["graph"] == "complete":
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    else:
        raise ValueError(case["graph"])
    weights = rng.integers(1, 6, size=len(pairs))
    return n, [(int(i), int(j), float(w)) for (i, j), w in zip(pairs, weights)]


def case_label(case):
    p = f" p={case['p']}" if case["graph"] == "random" else ""
    return f"{case['graph']} n={case['n']}{p}"


def add_arguments(parser):
    parser.add_argument("--graph", choices=["random", "cycle", "complete"], default="random")
    parser.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100], help="numbers of nodes")
    parser.add_argument("--p", type=float, default=0.1, help="edge probability for random graphs (default 0.1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="also run the slower phlite variants")


def cases_from_args(args):
    for n in args.sizes:
        yield {"graph": args.graph, "n": n, "p": args.p, "seed": args.seed}


if __name__ == "__main__":
    main(__doc__, IMPLEMENTATIONS, make_input, case_label, add_arguments, cases_from_args, reference="grpphati (GrPPH)")
