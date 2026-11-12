"""Benchmark phlite_py (the `phlite` package) against PHAT and LoPHAT on explicit boundary matrices over Z/2.

Every implementation gets the same columns, as a list of `(dimension, boundary)` pairs in filtration order (and phlite
also gets the filtration values, so that it can leave out zero-length bars). Times include loading the columns from
Python, except for `(reduce)` variants, whose matrix is built before the clock starts.

Kinds of complex, random (whose bars are mostly short):
    rips    Rips complex of n random points in the unit square, up to triangles
    grid    Cubical complex of an n x n grid with a lower-star filtration of smoothed noise
    cells   A random cellular complex (see tests/test_sparse.py): n vertices, and n cells in each dimension 1 to 3
    dense   The same, but with dense boundaries (each basis cycle is in a cell's boundary with probability 0.3)
and structured, with long bars:
    rips    with `shape`: Rips complex of n noisy points on a torus or a wedge of circles (see datasets.py)
    rings   Cubical complex of an n x n image of a few noisy rings (lower star of the distance to them): a long H1 bar
            for each ring
    torus   A triangulated n x n torus, filtered by the lower star of its height (a Morse function) plus noise:
            essential H0, H1 (two) and H2 bars, and long bars from the critical points

Inputs are generated once and cached in a temporary directory. Cases marked `quick` are the ones `just bench-check` uses.
"""

import functools
import itertools
import math
import os
import pickle
import random
import sys
import tempfile

import numpy as np
import pytest
from common import THREADS, fingerprint, run

# Import everything up front, so that import costs are not counted
import lophat
import phat
import phlite

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from test_sparse import random_complex  # noqa: E402

import datasets  # noqa: E402

Algorithm = phlite.Algorithm
INF = math.inf
REFERENCE = "phat (twist)"
CACHE = os.path.join(tempfile.gettempdir(), "phlite-bench-sparse")


# ====== Inputs ===========================================================


def rips_complex(n, top_dim, seed, shape=None):
    """The Rips complex of `n` points, up to dimension `top_dim`: random in the unit square, or sampled from `shape`."""
    if shape is None:
        points = np.random.default_rng(seed).uniform(size=(n, 2))
    else:
        points = datasets.points(shape, n, seed)
    distances = datasets.distance_matrix(points)
    simplices = [list(itertools.combinations(range(n), k + 1)) for k in range(top_dim + 1)]
    values = [np.zeros(n)]
    for k in range(1, top_dim + 1):
        s = np.array(simplices[k], dtype=np.int64)
        pairs = list(itertools.combinations(range(k + 1), 2))
        values.append(np.max([distances[s[:, i], s[:, j]] for i, j in pairs], axis=0))
    return _sorted_columns(simplices, values, _simplicial_faces)


def _simplicial_faces(simplex):
    return [simplex[:i] + simplex[i + 1 :] for i in range(len(simplex))] if len(simplex) > 1 else []


def smoothed_noise(n, seed):
    """An n x n image of noise, smoothed with a few box blurs, so that there are features at several scales."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, n))
    for _ in range(3):
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0) + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5
    return noise


def rings_image(n, seed, rings=5):
    """An n x n image of the distance to a few rings (random centres and radii) in the unit square, plus noise:
    its lower star has a long H1 bar for each ring, born when the ring closes and dying when its inside fills."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    distance = np.full((n, n), np.inf)
    for _ in range(rings):
        cx, cy = rng.uniform(0.2, 0.8, size=2)
        r = rng.uniform(0.08, 0.2)
        distance = np.minimum(distance, np.abs(np.hypot(x - cx, y - cy) - r))
    return distance + 0.003 * rng.normal(size=(n, n))


def grid_complex(n, seed, image=smoothed_noise):
    """The cubical complex of an n x n grid of vertices, filtered by `image(n, seed)` on the vertices (lower star)."""
    height = image(n, seed).astype(np.float32).astype(np.float64)
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


def torus_complex(n, seed):
    """A triangulated torus (an n x n grid of vertices, with wraparound), filtered by the lower star of the height of
    a torus standing on its side (a Morse function: a minimum, two saddles and a maximum), plus a little noise."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.arange(n) * 2 * np.pi / n, np.arange(n) * 2 * np.pi / n, indexing="ij")
    height = ((2 + np.cos(v)) * np.cos(u) + 0.05 * rng.normal(size=(n, n))).astype(np.float32).astype(np.float64)
    vertex = lambda i, j: (i % n) * n + (j % n)  # noqa: E731
    triangles = set()
    for i in range(n):
        for j in range(n):
            a, b, c, d = vertex(i, j), vertex(i + 1, j), vertex(i, j + 1), vertex(i + 1, j + 1)
            triangles.add(tuple(sorted((a, b, d))))
            triangles.add(tuple(sorted((a, c, d))))
    triangles = sorted(triangles)
    edges = sorted({face for t in triangles for face in _simplicial_faces(t)})
    cells = [[(i,) for i in range(n * n)], edges, triangles]
    flat = height.ravel()
    values = [np.array([flat[list(c)].max() for c in cells_k]) for cells_k in cells]
    return _sorted_columns(cells, values, _simplicial_faces)


def _sorted_columns(cells, values, faces):
    """Columns in filtration order (by value, then dimension), and the values in that order."""
    keys = [(v, k, i) for k, vs in enumerate(values) for i, v in enumerate(vs.tolist())]
    keys.sort()
    position = {}
    for new, (_, k, i) in enumerate(keys):
        position[cells[k][i]] = new
    columns = [(k, sorted(position[f] for f in faces(cells[k][i]))) for _, k, i in keys]
    return columns, np.array([v for v, _, _ in keys])


@functools.cache
def cached_complex(case_key):
    """The complex for a case (given as a sorted tuple of items), generated once and then cached in memory and in a
    temporary directory, since generating is slow. Stored as plain lists, so the cache doesn't depend on numpy."""
    case = dict(case_key)
    key = "-".join(str(case[k]) for k in ["kind", "n", "top_dim", "density", "seed"])
    if case.get("shape"):
        key += f"-{case['shape']}"
    path = os.path.join(CACHE, f"{key}.lists.pickle")
    try:
        with open(path, "rb") as f:
            columns, values = pickle.load(f)
    except FileNotFoundError:
        columns, values = generate_complex(case)
        values = np.asarray(values, dtype=np.float64).tolist()
        os.makedirs(CACHE, exist_ok=True)
        temporary = f"{path}.{os.getpid()}"
        with open(temporary, "wb") as f:
            pickle.dump((columns, values), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    return columns, np.array(values, dtype=np.float64)


def generate_complex(case):
    kind, n, seed = case["kind"], case["n"], case["seed"]
    if kind == "rips":
        columns, values = rips_complex(n, case["top_dim"], seed, case.get("shape"))
    elif kind == "grid":
        columns, values = grid_complex(n, seed)
    elif kind == "rings":
        columns, values = grid_complex(n, seed, rings_image)
    elif kind == "torus":
        columns, values = torus_complex(n, seed)
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


def phat_impl(reduction=phat.reductions.twist_reduction, representation=None):
    def compute(data):
        # The `(reduce)` variants get a prebuilt matrix, see `make_input`
        if "matrix" in data:
            matrix = data["matrix"]
        else:
            kwargs = {} if representation is None else {"representation": representation}
            matrix = phat.boundary_matrix(columns=data["columns"], **kwargs)
        pairs = matrix.compute_persistence_pairs(reduction=reduction)
        return pairs_fingerprint(data["columns"], data["values"], pairs)

    return compute


def lophat_impl(threads):
    def compute(data):
        diagram = lophat.compute_pairings(data["columns"], True, lophat.LoPhatOptions(num_threads=threads))
        return pairs_fingerprint(data["columns"], data["values"], ((b, d) for b, d in diagram.items() if d is not None))

    return compute


def phlite_impl(algorithm=Algorithm.serial, method="cohomology", csc=False, representatives=False):
    """`algorithm` builds the `Algorithm`. The `(reduce)` variants get a prebuilt complex, see `make_input`."""

    def compute(data):
        if "filtered" in data:
            filtered = data["filtered"]
        elif csc:
            columns = data["columns"]
            indptr = np.cumsum([0] + [len(c) for _, c in columns], dtype=np.int64)
            indices = np.fromiter((f for _, c in columns for f in c), dtype=np.int64, count=int(indptr[-1]))
            dimensions = np.fromiter((d for d, _ in columns), dtype=np.int64, count=len(columns))
            filtered = phlite.ChainComplex.from_csc(indptr, indices, dimensions).filtration(data["values"])
        else:
            filtered = phlite.ChainComplex.from_pairs(data["columns"]).filtration(data["values"])
        barcode = filtered.persistence(method=method, algorithm=algorithm(), representatives=representatives)
        return lambda: fingerprint(barcode.diagrams())

    return compute


def representation_variants():
    """Variations on the default (which stores R, with pivot columns, uncached), in both directions, named by how they
    differ, like `phlite (homology, heap, cached, reduce)`. Storing R: every other column representation (uncached),
    and pivot columns cached. `V only` doesn't store R (recomputing each R_j = D V_j instead): every representation
    uncached, and merged columns cached (the default before pivot columns)."""
    for method in ["cohomology", "homology"]:
        for store_r in [True, False]:
            for columns in ["pivot", "merged", "heap", "sorted"]:
                for cache in [False, True]:
                    if store_r and (columns == "pivot") != cache:
                        # The default, or cached non-default representations (caching barely matters when storing R)
                        continue
                    if not store_r and cache and columns != "merged":
                        continue
                    parts = ["homology"] if method == "homology" else []
                    parts += [] if store_r else ["V only"]
                    parts += [columns] if columns != "pivot" else []
                    parts += ["cached"] if cache else []
                    algorithm = lambda columns=columns, cache=cache, store_r=store_r: (
                        Algorithm.serial().with_columns(columns).with_cache(cache).with_store_r(store_r)
                    )
                    yield f"phlite ({', '.join(parts + ['reduce'])})", phlite_impl(algorithm, method=method)


IMPLEMENTATIONS = {
    "phat (twist)": phat_impl(),
    "phat (twist, reduce)": phat_impl(),
    "phat (chunk)": phat_impl(phat.reductions.chunk_reduction),
    "phat (standard, vector_vector)": phat_impl(phat.reductions.standard_reduction, phat.representations.vector_vector),
    "lophat (1 thread)": lophat_impl(1),
    "lophat (parallel)": lophat_impl(THREADS),
    "phlite (serial)": phlite_impl(),
    "phlite (serial, csc)": phlite_impl(csc=True),
    "phlite (serial, reduce)": phlite_impl(),
    "phlite (parallel)": phlite_impl(lambda: Algorithm.parallel(THREADS)),
    "phlite (parallel, reduce)": phlite_impl(lambda: Algorithm.parallel(THREADS)),
    "phlite (homology, reduce)": phlite_impl(method="homology"),
    "phlite (homology, parallel, reduce)": phlite_impl(lambda: Algorithm.parallel(THREADS), method="homology"),
    **{name: impl for name, impl in representation_variants()},
    "phlite (no skipping, reduce)": phlite_impl(lambda: Algorithm.serial().with_skip_zero_apparent_pairs(False)),
    "phlite (standard, reduce)": phlite_impl(Algorithm.standard),
    "phlite (reps, reduce)": phlite_impl(representatives=True),
}


# ====== Cases ============================================================


def case(kind, n, quick=False, large=False, shape=None):
    values = {"kind": kind, "n": n, "top_dim": 2, "density": 0.3, "seed": 0, "large": large, "shape": shape}
    marks = [pytest.mark.quick] if quick else [pytest.mark.large] if large else []
    return pytest.param(values, id=label(values), marks=marks)


def label(case):
    kind, n = case["kind"], case["n"]
    if kind == "rips" and case["shape"]:
        return f"rips {case['shape']} n={n} dim≤{case['top_dim']}"
    if kind == "rips":
        return f"rips n={n} dim≤{case['top_dim']}"
    if kind == "rings":
        return f"rings {n}x{n}"
    if kind == "torus":
        return f"torus mesh {n}x{n}"
    if kind == "grid":
        return f"grid {n}x{n}"
    if kind == "dense":
        return f"dense n={n} p={case['density']}"
    return f"cells n={n}"


CASES = [
    case("rips", 100, quick=True),
    case("rips", 150),
    case("grid", 300, quick=True),
    case("grid", 500),
    case("cells", 1000),
    case("cells", 3000, quick=True),
    case("dense", 100, quick=True),
    case("dense", 200),
    # Structured
    case("rips", 150, shape="torus"),
    case("rips", 150, shape="circles"),
    case("rings", 500),
    case("torus", 300),
    # Timed once each
    case("rips", 200, large=True),
    case("grid", 1000, large=True),
    case("cells", 10000, large=True),
    case("dense", 400, large=True),
    case("rips", 200, large=True, shape="torus"),
    case("rings", 1000, large=True),
    case("torus", 700, large=True),
]


def make_input(case, implementation):
    columns, values = cached_complex(case_key(case))
    data = {"columns": columns, "values": values}
    if "reduce)" in implementation:
        if implementation.startswith("phat"):
            data["matrix"] = phat.boundary_matrix(columns=columns)
        else:
            data["filtered"] = phlite.ChainComplex.from_pairs(columns).filtration(values)
    return data


def case_key(case):
    return tuple(sorted(case.items()))


@functools.cache
def reference(key):
    return IMPLEMENTATIONS[REFERENCE](make_input(dict(key), REFERENCE))()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("case", CASES)
def test_sparse(benchmark, case, implementation):
    run(
        benchmark,
        group=label(case),
        implementation=IMPLEMENTATIONS[implementation],
        make_input=lambda: make_input(case, implementation),
        reference=None if implementation == REFERENCE else functools.partial(reference, case_key(case)),
        rounds=1 if case["large"] else None,
    )
