"""The training workload for profile-guided optimisation (PGO) of the phlite wheels: every pipeline on mid-sized random
inputs, by the serial and parallel algorithms, cohomology and homology, with and without representatives.

Run it with wheels built with `-Cprofile-generate` installed (see `just pgo` and .github/workflows/release.yml); the
profiles it leaves are merged and used to build the released wheels. It needs only phlite and numpy, and takes about a
minute (instrumented). What it runs is what the optimiser favours, so it should look like real use: change it with the pipelines.
"""

import itertools
import time

import numpy as np

from phlite.clouds import alpha, alpha_complex, delaunay_cech, rips
from phlite.complexes import ChainComplex
from phlite.core import Algorithm
from phlite.images import cubical
from phlite.networks import directed_flag, dowker, flag, grpph, ordered_tuples, pph

rng = np.random.default_rng(0)
ALGORITHMS = [Algorithm.serial, lambda: Algorithm.parallel(2)]


def step(name, function):
    start = time.perf_counter()
    function()
    print(f"{name:40} {time.perf_counter() - start:7.2f}s", flush=True)


def every(name, run, methods=("cohomology", "homology"), representatives=True):
    """`run(method=, algorithm=, representatives=)` by each algorithm and method, and once with representatives."""
    for method in methods:
        for algorithm in ALGORITHMS:
            step(f"{name} {method} {algorithm().name}", lambda: run(method=method, algorithm=algorithm()))
    if representatives:
        step(f"{name} representatives", lambda: run(method=methods[0], algorithm=Algorithm.serial(), representatives=True))


def distances(points):
    return np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))


def rips_complex(n):
    """The Rips complex of `n` random points in the plane up to triangles, as a filtered `ChainComplex`."""
    d = distances(rng.random((n, 2)))
    simplices = [[(i,) for i in range(n)], list(itertools.combinations(range(n), 2)), list(itertools.combinations(range(n), 3))]
    edges, triangles = np.array(simplices[1]), np.array(simplices[2])
    values = [np.zeros(n), d[edges[:, 0], edges[:, 1]],
              np.max([d[triangles[:, i], triangles[:, j]] for i, j in [(0, 1), (0, 2), (1, 2)]], axis=0)]
    keys = sorted((v, k, i) for k, vs in enumerate(values) for i, v in enumerate(vs.tolist()))
    position = {simplices[k][i]: new for new, (_, k, i) in enumerate(keys)}
    pairs = []
    for _, k, i in keys:
        simplex = simplices[k][i]
        faces = [simplex[:j] + simplex[j + 1:] for j in range(len(simplex))] if k else []
        pairs.append((k, sorted(position[f] for f in faces)))
    return ChainComplex.from_pairs(pairs).filtration([v for v, _, _ in keys])


def digraph(n, p, integer=False):
    mask = (rng.random((n, n)) < p) & ~np.eye(n, dtype=bool)
    u, v = np.nonzero(mask)
    w = rng.random(len(u))
    if integer:
        w = 1 + np.floor(w * 20)
    return [(int(a), int(b), float(c)) for a, b, c in zip(u, v, w)]


def main():
    start = time.perf_counter()

    # Point clouds
    cube = distances(rng.random((300, 3)))
    angles = rng.random(400) * 2 * np.pi
    circle = np.c_[np.cos(angles), np.sin(angles)] + rng.normal(scale=0.05, size=(400, 2))
    every("rips cube dim<=2", lambda **kw: rips(cube, 2, **kw), methods=("cohomology",))
    # (Vietoris–Rips by homology is far slower than by cohomology, its default: only on a small cloud, below)
    every("rips circle dim<=1", lambda **kw: rips(distances(circle), 1, **kw), methods=("cohomology",))
    step("rips homology dim<=2", lambda: rips(distances(rng.random((60, 3))), 2, method="homology"))
    step("rips threshold", lambda: rips(cube, 2, threshold=0.3))
    step("rips weighted", lambda: rips(distances(circle), 1, weights=rng.random(400) * 0.1))
    plane = rng.random((20_000, 2))
    every("alpha plane", lambda **kw: alpha(plane, **kw))
    every("delaunay_cech space", lambda **kw: delaunay_cech(rng.random((5_000, 3)), **kw), representatives=False)

    # Explicit complexes: a full Rips complex up to triangles, as integer columns in filtration order (as in
    # bench_complexes.py), and an alpha complex given as simplices with values
    explicit_rips = rips_complex(120)
    every("complexes rips 2d", lambda **kw: explicit_rips.persistence(**kw), representatives=False)
    cells = alpha_complex(rng.random((20_000, 3)))
    simplices = [simplex for simplex, _ in cells]
    values = [value for _, value in cells]
    chain = ChainComplex.from_cells(
        simplices,
        dimension=lambda s: len(s) - 1,
        boundary=lambda s: {s[:i] + s[i + 1:]: 1 for i in range(len(s))} if len(s) > 1 else {},
    )
    filtered = chain.filtration(values)
    every("complexes alpha 3d", lambda **kw: filtered.persistence(**kw))
    step("complexes extended", lambda: filtered.extended_persistence())

    # Images
    noise = rng.random((512, 512))
    every("cubical noise 512^2", lambda **kw: cubical(noise, **kw))
    step("cubical V", lambda: cubical(noise, construction="V"))
    step("cubical 3d", lambda: cubical(rng.random((40, 40, 40))))

    # Networks
    # (GrPPH by homology is far slower than by cohomology, its default, so isn't worth training on)
    every("grpph G(60, 0.1)", lambda **kw: grpph(60, digraph(60, 0.1, integer=True), **kw), methods=("cohomology",))
    every("pph G(40, 0.15)", lambda **kw: pph(40, digraph(40, 0.15), 2, **kw))
    every("directed_flag G(150, 0.1)", lambda **kw: directed_flag(150, digraph(150, 0.1), 2, **kw))
    every("ordered_tuples G(40, 0.2)", lambda **kw: ordered_tuples(40, digraph(40, 0.2), 2, **kw))
    every("dowker G(40, 0.2)", lambda **kw: dowker(40, digraph(40, 0.2), 2, **kw))
    undirected = [(a, b, w) for a, b, w in digraph(300, 0.05) if a < b]
    every("flag G(300, 0.05)", lambda **kw: flag(300, undirected, 2, **kw))
    step("flag implicit", lambda: flag(300, undirected, 2, explicit=False))
    sparse = [(a, b, w) for a, b, w in digraph(150, 0.03) if a < b]
    step("flag shortest paths", lambda: flag(150, sparse, 1, filtration="shortest_path"))

    print(f"{'total':40} {time.perf_counter() - start:7.2f}s")


if __name__ == "__main__":
    main()
