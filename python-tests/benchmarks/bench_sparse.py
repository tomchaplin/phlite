"""Benchmark phlite_py (the `phlite` package) against PHAT and LoPHAT on explicit boundary matrices over Z/2.

Every implementation gets the same columns, as a list of `(dimension, boundary)` pairs in filtration order (and phlite
also gets the filtration values, so that it can leave out zero-length bars). By default the time includes loading the
columns from Python; the `(reduce)` variants load them before the clock starts.

Kinds of complex (`--kind`):
    rips    Rips complex of n random points in the unit square, up to triangles (or `--top-dim`)
    grid    Cubical complex of an n x n grid with a lower-star filtration of smoothed noise
    cells   A random cellular complex (see tests/test_sparse.py): n vertices, and n cells in each dimension 1 to 3
    dense   The same, but with dense boundaries (each basis cycle is in a cell's boundary with probability `--density`)

Inputs are generated once and cached in a temporary directory.

Examples:
    just bench-sparse                                   # Rips complexes, default sizes
    just bench-sparse --kind grid --sizes 200 400 800
    just bench-sparse --kind dense --sizes 100 200 400
    just bench-sparse --all --only phlite               # every phlite variant
"""

import itertools
import math
import os
import pickle
import random
import sys
import tempfile

import numpy as np
from harness import main

# Import everything up front, so that import costs are not counted in the time or peak memory of a run
import lophat
import phat
import phlite

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from test_sparse import random_complex  # noqa: E402

Algorithm = phlite.Algorithm
INF = math.inf


# ====== Inputs ===========================================================


def rips_complex(n, top_dim, seed):
    """The Rips complex of `n` random points in the unit square, up to dimension `top_dim`."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(size=(n, 2))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1).astype(np.float32).astype(np.float64)
    simplices = [list(itertools.combinations(range(n), k + 1)) for k in range(top_dim + 1)]
    values = [np.zeros(n)]
    for k in range(1, top_dim + 1):
        s = np.array(simplices[k], dtype=np.int64)
        pairs = list(itertools.combinations(range(k + 1), 2))
        values.append(np.max([distances[s[:, i], s[:, j]] for i, j in pairs], axis=0))
    return _sorted_columns(simplices, values, _simplicial_faces)


def _simplicial_faces(simplex):
    return [simplex[:i] + simplex[i + 1 :] for i in range(len(simplex))] if len(simplex) > 1 else []


def grid_complex(n, seed):
    """The cubical complex of an n x n grid of vertices, filtered by smoothed noise on the vertices (lower star)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, n))
    # Smooth with a few box blurs, so that there are features at several scales
    for _ in range(3):
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0) + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5
    height = noise.astype(np.float32).astype(np.float64)
    vertex = lambda i, j: i * n + j  # noqa: E731
    cells = [[(vertex(i, j),) for i in range(n) for j in range(n)], [], []]
    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                cells[1].append((vertex(i, j), vertex(i, j + 1)))
            if i + 1 < n:
                cells[1].append((vertex(i, j), vertex(i + 1, j)))
            if i + 1 < n and j + 1 < n:
                cells[2].append((vertex(i, j), vertex(i, j + 1), vertex(i + 1, j), vertex(i + 1, j + 1)))
    flat = height.ravel()
    values = [np.array([flat[list(c)].max() for c in cells_k]) for cells_k in cells]

    def faces(cell):
        if len(cell) == 2:
            return [(cell[0],), (cell[1],)]
        if len(cell) == 4:
            a, b, c, d = cell
            return [(a, b), (a, c), (b, d), (c, d)]
        return []

    return _sorted_columns(cells, values, faces)


def _sorted_columns(cells, values, faces):
    """Columns in filtration order (by value, then dimension), and the values in that order."""
    keys = [(v, k, i) for k, vs in enumerate(values) for i, v in enumerate(vs.tolist())]
    keys.sort()
    position = {}
    for new, (_, k, i) in enumerate(keys):
        position[cells[k][i]] = new
    columns = [(k, sorted(position[f] for f in faces(cells[k][i]))) for _, k, i in keys]
    return columns, np.array([v for v, _, _ in keys])


CACHE = os.path.join(tempfile.gettempdir(), "phlite-bench-sparse")


def make_input(case):
    columns, values = cached_complex(case)
    data = {"columns": columns, "values": values}
    # The `(reduce)` variants get their matrices built up front
    name = case["implementation"]
    if "reduce)" in name:
        if name.startswith("phat"):
            data["matrix"] = phat.boundary_matrix(columns=columns)
        else:
            data["filtered"] = phlite.ChainComplex.from_columns(columns).filtration(values)
    return data


def cached_complex(case):
    """The complex for `case`, generated once and then cached in a temporary directory (generating is slow)."""
    key = "-".join(str(case[k]) for k in ["kind", "n", "top_dim", "density", "seed"])
    path = os.path.join(CACHE, f"{key}.pickle")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        pass
    complex = generate_complex(case)
    os.makedirs(CACHE, exist_ok=True)
    temporary = f"{path}.{os.getpid()}"
    with open(temporary, "wb") as f:
        pickle.dump(complex, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return complex


def generate_complex(case):
    kind, n, seed = case["kind"], case["n"], case["seed"]
    if kind == "rips":
        columns, values = rips_complex(n, case["top_dim"], seed)
    elif kind == "grid":
        columns, values = grid_complex(n, seed)
    else:
        density = case["density"] if kind == "dense" else None
        columns, values = random_complex(random.Random(seed), 3, n, n, 50, density=density)
        values = np.array(values, dtype=np.float64)
    return columns, values


# ====== Implementations ==================================================


def pairs_fingerprint(columns, values, pairs):
    """The fingerprint of a barcode given as pairs of cell indices (every unpaired cell is essential)."""

    def compute():
        dims = [[] for _ in range(1 + max(d for d, _ in columns))]
        paired = set()
        for b, d in pairs:
            b, d = int(b), int(d)
            paired.update((b, d))
            dims[columns[b][0]].append((values[b], values[d]))
        for i, (dim, _) in enumerate(columns):
            if i not in paired:
                dims[dim].append((values[i], INF))
        return fingerprint(dims)

    return compute


def fingerprint(dims):
    """Number of finite and infinite bars, and total finite persistence, per dimension (ignoring zero-length bars)."""
    out = []
    for bars in dims:
        bars = [(float(b), float(d)) for b, d in bars if b < d]
        finite = [d - b for b, d in bars if d != INF]
        out.append([len(finite), len(bars) - len(finite), round(sum(finite), 6)])
    return out


def phat_impl(reduction=phat.reductions.twist_reduction, representation=None, prebuilt=False):
    def run(data, case):
        if prebuilt:
            matrix = data["matrix"]
        else:
            kwargs = {} if representation is None else {"representation": representation}
            matrix = phat.boundary_matrix(columns=data["columns"], **kwargs)
        pairs = matrix.compute_persistence_pairs(reduction=reduction)
        return pairs_fingerprint(data["columns"], data["values"], pairs)

    return run


def lophat_impl(parallel):
    def run(data, case):
        options = lophat.LoPhatOptions()
        options.num_threads = case["threads"] if parallel else 1
        diagram = lophat.compute_pairings(data["columns"], True, options)
        return pairs_fingerprint(data["columns"], data["values"], diagram.paired)

    return run


def phlite_impl(algorithm=lambda threads: Algorithm.serial(), method="cohomology", build="columns", representatives=False):
    """`algorithm` maps the thread count to an `Algorithm`; `build` is 'columns', 'csc' or 'prebuilt'."""

    def run(data, case):
        if build == "prebuilt":
            filtered = data["filtered"]
        elif build == "csc":
            columns = data["columns"]
            indptr = np.cumsum([0] + [len(c) for _, c in columns], dtype=np.int64)
            indices = np.fromiter((f for _, c in columns for f in c), dtype=np.int64, count=int(indptr[-1]))
            dimensions = np.fromiter((d for d, _ in columns), dtype=np.int64, count=len(columns))
            filtered = phlite.ChainComplex.from_csc(indptr, indices, dimensions).filtration(data["values"])
        else:
            filtered = phlite.ChainComplex.from_columns(data["columns"]).filtration(data["values"])
        barcode = filtered.persistence(
            method=method, algorithm=algorithm(case["threads"]), representatives=representatives
        )
        return lambda: fingerprint(barcode.diagrams())

    return run


# Names starting with '*' only run with --all
IMPLEMENTATIONS = {
    "phat (twist)": phat_impl(),
    "phat (twist, reduce)": phat_impl(prebuilt=True),
    "*phat (chunk)": phat_impl(phat.reductions.chunk_reduction),
    "*phat (standard, vector_vector)": phat_impl(
        phat.reductions.standard_reduction, phat.representations.vector_vector
    ),
    "lophat (1 thread)": lophat_impl(parallel=False),
    "lophat (parallel)": lophat_impl(parallel=True),
    "phlite (serial)": phlite_impl(),
    "phlite (serial, reduce)": phlite_impl(build="prebuilt"),
    "phlite (parallel)": phlite_impl(Algorithm.parallel),
    "phlite (parallel, reduce)": phlite_impl(Algorithm.parallel, build="prebuilt"),
    "phlite (homology, reduce)": phlite_impl(method="homology", build="prebuilt"),
    "*phlite (serial, csc)": phlite_impl(build="csc"),
    "*phlite (heap, reduce)": phlite_impl(lambda threads: Algorithm.serial().with_columns("heap"), build="prebuilt"),
    "*phlite (no skipping, reduce)": phlite_impl(
        lambda threads: Algorithm.serial().with_skip_zero_apparent_pairs(False), build="prebuilt"
    ),
    "*phlite (standard, reduce)": phlite_impl(lambda threads: Algorithm.standard(), build="prebuilt"),
    "*phlite (homology, parallel, reduce)": phlite_impl(Algorithm.parallel, method="homology", build="prebuilt"),
    "*phlite (reps, reduce)": phlite_impl(build="prebuilt", representatives=True),
}


def case_label(case):
    kind = case["kind"]
    if kind == "rips":
        return f"rips n={case['n']} dim≤{case['top_dim']}"
    if kind == "grid":
        return f"grid {case['n']}x{case['n']}"
    if kind == "dense":
        return f"dense n={case['n']} p={case['density']}"
    return f"cells n={case['n']}"


def add_arguments(parser):
    parser.add_argument("--kind", choices=["rips", "grid", "cells", "dense"], default="rips")
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="sizes (default depends on --kind)")
    parser.add_argument("--top-dim", type=int, default=2, help="for rips: the top simplex dimension (default 2)")
    parser.add_argument("--density", type=float, default=0.3, help="for dense: probability of each basis cycle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="also run the other PHAT and phlite variants")


DEFAULT_SIZES = {"rips": [50, 100, 150], "grid": [100, 300, 500], "cells": [500, 1000, 3000], "dense": [100, 200, 400]}


def cases_from_args(args):
    for n in args.sizes or DEFAULT_SIZES[args.kind]:
        yield {"kind": args.kind, "n": n, "top_dim": args.top_dim, "density": args.density, "seed": args.seed}


if __name__ == "__main__":
    main(__doc__, IMPLEMENTATIONS, make_input, case_label, add_arguments, cases_from_args, reference="phat (twist)")
