"""What does describing a complex with Python cell objects cost, and does key= help?

Builds the Vietoris-Rips complex (up to triangles) of random points as
* a ChainComplex from integer columns (the baseline),
* a ChainComplex of frozen dataclasses, matched by hashing,
* the same with key= (an integer encoding of the vertices), and with from_key= too,
* the same with face_keys=True, where boundaries list the keys of faces, so no face objects are built,
then attaches the filtration (each simplex's entrance_time) and computes H_0 and H_1 by cohomology.
"""

import dataclasses
import gc
import itertools
import sys
import time
import tracemalloc

import numpy as np

import phlite

n_points = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(0)
points = rng.random((n_points, 3))
dist = np.linalg.norm(points[:, None] - points[None], axis=-1)
threshold = np.quantile(dist[np.triu_indices(n_points, 1)], 0.25)


@dataclasses.dataclass(frozen=True)
class Simplex:
    vertices: tuple
    entrance_time: float = dataclasses.field(default=None, compare=False)

    @property
    def dimension(self):
        return len(self.vertices) - 1

    def boundary(self):
        vs = self.vertices
        return (
            [Simplex(vs[:i] + vs[i + 1 :]) for i in range(len(vs))]
            if len(vs) > 1
            else []
        )


def diameter(vs):
    return max((dist[a, b] for a, b in itertools.combinations(vs, 2)), default=0.0)


edges = [
    (i, j)
    for i, j in itertools.combinations(range(n_points), 2)
    if dist[i, j] <= threshold
]
nbrs = {i: set() for i in range(n_points)}
for i, j in edges:
    nbrs[i].add(j)
    nbrs[j].add(i)
triangles = [(i, j, k) for i, j in edges for k in nbrs[i] & nbrs[j] if k > j]
simplices = [(v,) for v in range(n_points)] + edges + triangles
cells = [Simplex(s, diameter(s)) for s in simplices]
print(f"{n_points} points: {len(cells)} cells")


def encode(vs):
    k = 0
    for v in vs:
        k = k * (n_points + 1) + v + 1
    return k


def decode(k):
    vs = []
    while k:
        k, d = divmod(k, n_points + 1)
        vs.append(d - 1)
    return Simplex(tuple(reversed(vs)))


times = np.array([c.entrance_time for c in cells])


def timed(name, build):
    # Time without tracemalloc (which slows down every allocation), then measure memory separately
    gc.collect()
    start = time.perf_counter()
    chain = build()
    built = time.perf_counter() - start
    del chain
    gc.collect()
    tracemalloc.start()
    chain = build()
    retained = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    # The same filtration for every variant: entrance times, as an array in cell order
    start = time.perf_counter()
    filtered = chain.filtration(times)
    filter_time = time.perf_counter() - start
    start = time.perf_counter()
    barcode = filtered.persistence(1)
    reduce = time.perf_counter() - start
    print(
        f"{name:<38} build {built * 1e3:7.1f} ms   filtration {filter_time * 1e3:5.1f} ms   "
        f"reduce {reduce * 1e3:5.1f} ms   Python memory kept {retained / 1e6:5.1f} MB   {len(barcode)} bars"
    )


def integer_columns():
    index = {s: i for i, s in enumerate(simplices)}
    columns = [
        [index[s[:i] + s[i + 1 :]] for i in range(len(s))] if len(s) > 1 else []
        for s in simplices
    ]
    return phlite.ChainComplex.from_columns(columns, [len(s) - 1 for s in simplices])


timed("from_columns (integer columns)", integer_columns)
timed("from_cells, hashing", lambda: phlite.ChainComplex.from_cells(cells))
timed(
    "from_cells, key=",
    lambda: phlite.ChainComplex.from_cells(cells, key=lambda c: encode(c.vertices)),
)
timed(
    "from_cells, key= and from_key=",
    lambda: phlite.ChainComplex.from_cells(
        cells, key=lambda c: encode(c.vertices), from_key=decode
    ),
)


def face_keys(c):
    vs = c.vertices
    return [encode(vs[:i] + vs[i + 1 :]) for i in range(len(vs))] if len(vs) > 1 else []


timed(
    "from_cells, key= and face_keys=True",
    lambda: phlite.ChainComplex.from_cells(
        cells, key=lambda c: encode(c.vertices), boundary=face_keys, face_keys=True
    ),
)
