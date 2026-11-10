"""How expensive would matrix oracles written in Python be?

We build a Vietoris-Rips complex (up to triangles) of random points as an explicit filtered complex, then compare
* the full phlite reduction (Rust, explicit matrix), against
* merely calling a Python oracle once per column of the boundary and the coboundary (no reduction at all).

The second is a generous lower bound on a reduction driven by Python oracles: phlite recomputes columns on demand
rather than storing them, so a real reduction calls the oracle for each column at least once and often many times,
and every call would also need the GIL (serialising the parallel algorithms).
"""

import itertools
import sys
import time

import numpy as np

import phlite

n_points = int(sys.argv[1]) if len(sys.argv) > 1 else 120
rng = np.random.default_rng(0)
points = rng.random((n_points, 3))
dist = np.linalg.norm(points[:, None] - points[None], axis=-1)
threshold = np.quantile(dist[np.triu_indices(n_points, 1)], 0.25)

# Cells: vertices, edges, triangles with their diameters, in filtration order
edges = [
    (i, j)
    for i, j in itertools.combinations(range(n_points), 2)
    if dist[i, j] <= threshold
]
edge_set = set(edges)
nbrs = {i: set() for i in range(n_points)}
for i, j in edges:
    nbrs[i].add(j)
    nbrs[j].add(i)
triangles = [(i, j, k) for i, j in edges for k in nbrs[i] & nbrs[j] if k > j]
cells = [((v,), 0.0) for v in range(n_points)]
cells += [(e, dist[e]) for e in edges]
cells += [
    (t, max(dist[t[0], t[1]], dist[t[0], t[2]], dist[t[1], t[2]])) for t in triangles
]
cells.sort(key=lambda c: (c[1], len(c[0])))
index = {s: i for i, (s, _) in enumerate(cells)}
columns = [
    [index[s[:i] + s[i + 1 :]] for i in range(len(s))] if len(s) > 1 else []
    for s, _ in cells
]
values = [v for _, v in cells]
dims = [len(s) - 1 for s, _ in cells]
print(
    f"{n_points} points: {len(cells)} cells ({len(edges)} edges, {len(triangles)} triangles)"
)

start = time.perf_counter()
complex = phlite.FilteredComplex(columns, dims, values)
build = time.perf_counter() - start

for method in ["cohomology", "homology"]:
    start = time.perf_counter()
    barcode = complex.persistence(1, method=method)
    reduce = time.perf_counter() - start
    print(
        f"phlite {method:>10}: build {build * 1e3:7.1f} ms + reduce {reduce * 1e3:7.1f} ms ({len(barcode)} bars in H_0, H_1)"
    )

try:
    import lophat

    start = time.perf_counter()
    lophat.compute_pairings(list(zip(dims, columns)))
    print(
        f"lophat (for scale, all dimensions): {(time.perf_counter() - start) * 1e3:7.1f} ms"
    )
except ImportError:
    pass

# The cheapest conceivable Python oracle: look the column up in a precomputed list
coboundary = [[] for _ in cells]
for j, col in enumerate(columns):
    for i in col:
        coboundary[i].append(j)


class Oracle:
    def __init__(self, cols):
        self.cols = cols

    def column(self, j):
        return self.cols[j]


boundary_oracle, coboundary_oracle = Oracle(columns), Oracle(coboundary)
start = time.perf_counter()
for j in range(len(cells)):
    for _ in boundary_oracle.column(j):
        pass
    for _ in coboundary_oracle.column(j):
        pass
calls = time.perf_counter() - start
print(f"one pass of Python oracle calls (no reduction, no FFI): {calls * 1e3:7.1f} ms")
