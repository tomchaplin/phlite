"""Benchmark phlite_rips against ripser and giotto-ph on random point clouds.

Examples:
    just bench-rips                                  # default sizes
    just bench-rips --sizes 200 400 --max-dim 2
    just bench-rips --all --only phlite              # every phlite method/algorithm
"""

import numpy as np
from harness import fingerprint, main

# Import everything up front, so that import costs are not counted in the time or peak memory of a run
from phlite_rips import Algorithm, rips
from ripser import ripser
from gph import ripser_parallel

INF = float("inf")


def phlite(method="cohomology", algorithm=lambda threads: Algorithm.serial(), representatives=False):
    """`algorithm` maps the thread count to an `Algorithm`."""

    def run(distances, case):
        barcode = rips(
            distances,
            case["max_dim"],
            method=method,
            algorithm=algorithm(case["threads"]),
            representatives=representatives,
        )
        return fingerprint(
            [list(p) + [(b, INF) for b in e] for p, e in zip(barcode.pairings, barcode.essential)]
        )

    return run


def ripser_py(distances, case):
    dgms = ripser(distances, distance_matrix=True, maxdim=case["max_dim"])["dgms"]
    return fingerprint([[(float(b), float(d)) for b, d in dgm] for dgm in dgms])


def giotto(parallel):
    def run(distances, case):
        threads = case["threads"] if parallel else 1
        dgms = ripser_parallel(distances, maxdim=case["max_dim"], metric="precomputed", n_threads=threads)["dgms"]
        return fingerprint([[(float(b), float(d)) for b, d in dgm] for dgm in dgms])

    return run


# Names starting with '*' only run with --all
IMPLEMENTATIONS = {
    "ripser": ripser_py,
    "giotto-ph (1 thread)": giotto(parallel=False),
    "giotto-ph (parallel)": giotto(parallel=True),
    "phlite (clearing)": phlite(),
    "phlite (parallel)": phlite(algorithm=Algorithm.parallel),
    "phlite (clearing, reps)": phlite(representatives=True),
    "*phlite (parallel-lock-free)": phlite(algorithm=lambda threads: Algorithm.parallel(threads).with_storage("lock-free")),
    "*phlite (standard)": phlite(algorithm=lambda threads: Algorithm.standard()),
    "*phlite (homology, clearing)": phlite(method="homology"),
}


def make_input(case):
    """Points uniformly in the unit cube, with distances rounded to float32 so every implementation sees the same values."""
    rng = np.random.default_rng(case["seed"])
    points = rng.uniform(size=(case["n"], case["ambient_dim"]))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return distances.astype(np.float32).astype(np.float64)


def case_label(case):
    return f"n={case['n']} dim≤{case['max_dim']}"


def add_arguments(parser):
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 300], help="numbers of points")
    parser.add_argument("--max-dim", type=int, default=2, help="maximum homology dimension (default 2)")
    parser.add_argument("--ambient-dim", type=int, default=3, help="dimension of the cube to sample from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="also run the slower phlite variants")


def cases_from_args(args):
    for n in args.sizes:
        yield {"n": n, "max_dim": args.max_dim, "ambient_dim": args.ambient_dim, "seed": args.seed}


if __name__ == "__main__":
    main(__doc__, IMPLEMENTATIONS, make_input, case_label, add_arguments, cases_from_args, reference="ripser")
