"""Compare phlite's Vietoris-Rips persistence with PerMaViss (https://pypi.org/project/permaviss), which computes it
through the persistence Mayer-Vietoris spectral sequence of a cubical cover (PH of each cover piece and intersection,
the spectral sequence's pages, then the extension problem), over Z/2: the barcodes, up to `max_r`, and time.

PerMaViss needs its own environment (it pins old numpy behaviour); give its Python as `PERMAVISS_PYTHON`:

    PERMAVISS_PYTHON=.../bin/python uv run --no-sync python benchmarks/permaviss_compare.py [--out FILE]

Run from python-tests (phlite.clouds).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np

from phlite.clouds import rips

RUNNER = r"""
import json, sys, time
import numpy as np
from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss
points = np.array(json.loads(sys.stdin.read()))
max_r, max_dim, max_div = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
start = time.perf_counter()
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    ss = create_MV_ss(points, max_r, max_dim, max_div, max_r * 1.01, 2)
elapsed = time.perf_counter() - start
bars = [ph.barcode.tolist() if ph.dim else [] for ph in ss.persistent_homology]
print(json.dumps({"time": elapsed, "bars": bars}))
"""


def point_clouds():
    rng = np.random.default_rng(0)
    cube = lambda n: rng.random((n, 3))  # noqa: E731

    def torus(n, r=1.0, big=3.0):
        a, b = rng.random(n) * 2 * np.pi, rng.random(n) * 2 * np.pi
        return np.c_[(big + r * np.cos(a)) * np.cos(b), (big + r * np.cos(a)) * np.sin(b), r * np.sin(a)]

    return [
        ("cube, 130 points (PerMaViss's example)", cube(130), 0.36, 3, 2),
        ("cube, 300 points", cube(300), 0.3, 3, 2),
        ("torus, 300 points", torus(300), 1.2, 3, 2),
        ("torus, 600 points", torus(600), 1.0, 3, 3),
    ]


def multiset(bars, max_r, digits=6):
    return Counter((round(b, digits), round(min(d, max_r), digits)) for b, d in bars if min(d, max_r) > b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    python = os.environ["PERMAVISS_PYTHON"]
    rows = []
    for name, points, max_r, max_dim, max_div in point_clouds():
        result = subprocess.run(
            [python, "-I", "-c", RUNNER, str(max_r), str(max_dim), str(max_div)],
            input=json.dumps(points.tolist()),
            capture_output=True,
            text=True,
            check=True,
        )
        theirs = json.loads(result.stdout.strip().splitlines()[-1])
        start = time.perf_counter()
        barcode = rips(points=points, max_dim=max_dim - 1, threshold=max_r)
        ours_time = time.perf_counter() - start
        ours = [
            list(barcode.pairings[k]) + [(b, float("inf")) for b in barcode.essential[k]] for k in range(max_dim)
        ]
        agree = all(
            multiset(theirs["bars"][k] if k < len(theirs["bars"]) else [], max_r) == multiset(ours[k], max_r)
            for k in range(max_dim)
        )
        counts = [len(multiset(d, max_r)) for d in ours]
        rows.append(
            f"| {name} | r < {max_r}, dims 0–{max_dim - 1} | {counts} | {theirs['time']:.2f} s | "
            f"{ours_time * 1e3:.1f} ms | {'✓' if agree else '✗'} |"
        )
        print(rows[-1], flush=True)
    table = "\n".join(
        [
            "Vietoris-Rips barcodes over Z/2 up to `max_r`: PerMaViss (through the persistence Mayer-Vietoris spectral "
            "sequence of a cubical cover) against phlite (`clouds.rips`, serial). ✓: the barcodes agree.",
            "",
            "| point cloud | filtration | phlite bars per dimension | PerMaViss | phlite | agree |",
            "|---|---|---|---:|---:|:-:|",
        ]
        + rows
    )
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    sys.exit(main())
