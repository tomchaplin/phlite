"""Time phlite's witness and periodic cubical complexes against GUDHI's, checking the barcodes agree.

Run from python-tests (the 3.11 environment, with gudhi): `uv run --no-sync python benchmarks/gudhi_filtrations_compare.py
[--out FILE]`. Times are best of three, each the whole computation from the input arrays (building the complex
included), on one thread; GUDHI computes over Z/2.
"""

import argparse
import math
import time

import gudhi
import numpy as np

from phlite.clouds import witness
from phlite.core import Algorithm
from phlite.images import cubical


def best(fn, repeats=3):
    times, out = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return out, min(times)


def fmt(t):
    return f"{t * 1e3:.1f} ms" if t < 1 else f"{t:.2f} s"


def bars_from_gudhi(intervals_by_dim, top):
    return [sorted((float(b), float(d)) for b, d in intervals_by_dim(k) if d > b) for k in range(top + 1)]


def bars_from_phlite(barcode, top):
    return [
        sorted([tuple(map(float, p)) for p in barcode.pairings[k]] + [(float(b), math.inf) for b in barcode.essential[k]])
        for k in range(top + 1)
    ]


def agree(a, b):
    return all(len(x) == len(y) and np.allclose(np.array(x).reshape(-1, 2), np.array(y).reshape(-1, 2)) for x, y in zip(a, b))


def witness_rows():
    rows = []
    serial = Algorithm.serial()
    for n_witnesses, n_landmarks, dim, bound in [(2000, 50, 3, math.inf), (10000, 100, 3, 0.02), (20000, 200, 3, 0.01)]:
        rng = np.random.default_rng(0)
        points = rng.random((n_witnesses, dim))
        landmarks = points[rng.choice(n_witnesses, n_landmarks, replace=False)]
        for strong in (False, True):
            cls = gudhi.EuclideanStrongWitnessComplex if strong else gudhi.EuclideanWitnessComplex

            def theirs():
                tree = cls(landmarks=landmarks, witnesses=points).create_simplex_tree(
                    max_alpha_square=bound, limit_dimension=2
                )
                tree.compute_persistence(homology_coeff_field=2, min_persistence=0)
                return tree

            tree, t_gudhi = best(theirs)
            limit = None if bound == math.inf else bound
            ours, t_ours = best(lambda: witness(landmarks, points, 1, strong=strong, max_alpha_square=limit, algorithm=serial))
            ok = agree(bars_from_phlite(ours, 1), bars_from_gudhi(tree.persistence_intervals_in_dimension, 1))
            name = f"{'strong ' if strong else ''}witness, {n_landmarks} of {n_witnesses} points in R^{dim}, α² ≤ {bound}"
            rows.append(f"| {name} | {tree.num_simplices()} | {fmt(t_ours)} | {fmt(t_gudhi)} | {'✓' if ok else '✗'} |")
            print(rows[-1], flush=True)
    return rows


def periodic_rows():
    rows = []
    serial = Algorithm.serial()
    for shape in [(256, 256), (1024, 1024), (32, 32, 32), (64, 64, 64)]:
        rng = np.random.default_rng(1)
        image = rng.random(shape)
        top = len(shape)

        def theirs():
            c = gudhi.PeriodicCubicalComplex(top_dimensional_cells=image, periodic_dimensions=[True] * len(shape))
            c.compute_persistence(homology_coeff_field=2, min_persistence=0)
            return c

        complex_, t_gudhi = best(theirs)
        ours, t_ours = best(lambda: cubical(image, top, periodic=True, algorithm=serial))
        ok = agree(bars_from_phlite(ours, top), bars_from_gudhi(complex_.persistence_intervals_in_dimension, top))
        rows.append(f"| periodic {'×'.join(map(str, shape))}, uniform noise, T | {fmt(t_ours)} | {fmt(t_gudhi)} | {'✓' if ok else '✗'} |")
        print(rows[-1], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    w = witness_rows()
    p = periodic_rows()
    table = "\n".join(
        [
            "## Witness complexes: phlite against GUDHI (dimensions 0 and 1, Z/2; ✓: identical barcodes)",
            "",
            "| complex | simplices | phlite | GUDHI | agree |",
            "|---|---:|---:|---:|:-:|",
            *w,
            "",
            "## Periodic cubical complexes: phlite against GUDHI's PeriodicCubicalComplex (every dimension)",
            "",
            "| image | phlite | GUDHI | agree |",
            "|---|---:|---:|:-:|",
            *p,
        ]
    )
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
