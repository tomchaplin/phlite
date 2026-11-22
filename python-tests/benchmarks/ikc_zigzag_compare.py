"""Compare phlite's image/kernel/cokernel persistence with phimaker, and its zigzag persistence with Dionysus 2:
correctness (identical barcodes) and time.

phimaker and Dionysus run in their own environments; give their Pythons, and `fzz` (Dey and Hou's FastZigzag, built
with a driver that prints the time of its `compute` to stderr):

    PHIMAKER_PYTHON=... DIONYSUS_PYTHON=... FZZ_BIN=... uv run --no-sync python ../benchmarks/ikc_zigzag_compare.py [--out FILE]

Run from python-tests/latest (phlite.complexes and phlite.clouds). Times exclude building inputs and passing them
between processes: phimaker's is `compute_ensemble` (all its threads), Dionysus's `zigzag_homology_persistence`,
phlite's the call (`image_kernel_cokernel` on a prebuilt filtered complex, `rips_image_kernel` from the points, or
`zigzag` from the operations), serial.
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import time

import numpy as np

from phlite.clouds import rips_complex, rips_image_kernel
from phlite.complexes import ChainComplex, zigzag

PHIMAKER_RUNNER = r"""
import json, sys, time
from phimaker import compute_ensemble
matrix = [tuple(x) for x in json.loads(sys.stdin.read())]
start = time.perf_counter()
d = compute_ensemble(matrix)
elapsed = time.perf_counter() - start
out = {"time": elapsed}
for name in ("im", "ker", "cok"):
    dgm = getattr(d, name)
    out[name] = [list(map(int, p)) for p in dgm.paired] + [[int(u), -1] for u in dgm.unpaired]
print(json.dumps(out))
"""

DIONYSUS_RUNNER = r"""
import json, sys, time
import dionysus as d
ops = json.loads(sys.stdin.read())
times = {}
for i, (s, ins) in enumerate(ops):
    times.setdefault(tuple(s), []).append(i + 1)
f = d.Filtration([d.Simplex(list(s), 0) for s in times])
t = [times[tuple(s)] for s in f]
start = time.perf_counter()
zz, dgms, cells = d.zigzag_homology_persistence(f, t)
elapsed = time.perf_counter() - start
bars = [[k, p.birth, p.death] for k, dg in enumerate(dgms) for p in dg]
print(json.dumps({"time": elapsed, "bars": [[k, b, -1 if e == float("inf") else e] for k, b, e in bars]}))
"""


def fmt(t):
    return f"{t * 1e3:.1f} ms" if t < 1 else f"{t:.2f} s"


# ====== Images, kernels, cokernels ======


def torus_mesh(n, seed):
    rng = random.Random(seed)
    height = [rng.randint(0, 9) for _ in range(n * n)]
    v = lambda i, j: (i % n) * n + (j % n)  # noqa: E731
    simplices = {(x,) for x in range(n * n)}
    for i in range(n):
        for j in range(n):
            a, b, c, d = v(i, j), v(i + 1, j), v(i, j + 1), v(i + 1, j + 1)
            for t in (tuple(sorted((a, b, d))), tuple(sorted((a, c, d)))):
                for k in (2, 3):
                    simplices.update(itertools.combinations(t, k))
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    values = [float(max(height[x] for x in s)) for s in ordered]
    subset = {x for x in range(n * n) if (x // n) < n // 2}
    return ordered, values, subset


def rips_case(n, seed, threshold):
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2))
    simplices = rips_complex(points=points, dimension=2, threshold=threshold)
    subset = set(rng.choice(n, n // 2, replace=False).tolist())
    return points, [s for s, _ in simplices], [v for _, v in simplices], subset


def phimaker_run(ordered, values, chosen, python):
    order = sorted(range(len(ordered)), key=lambda i: (values[i], len(ordered[i]), i))
    position = {ordered[i]: p for p, i in enumerate(order)}
    matrix = []
    for i in order:
        s = ordered[i]
        faces = sorted(position[s[:j] + s[j + 1:]] for j in range(len(s))) if len(s) > 1 else []
        matrix.append([s in chosen, len(s) - 1, faces])
    out = subprocess.run([python, "-I", "-c", PHIMAKER_RUNNER], input=json.dumps(matrix), capture_output=True, text=True, check=True)
    theirs = json.loads(out.stdout)
    vals = [values[i] for i in order]
    dims = [len(ordered[i]) - 1 for i in order]
    bars = {}
    for name, key, shift in (("image", "im", 0), ("kernel", "ker", 1), ("cokernel", "cok", 0)):
        bars[name] = sorted(
            (dims[b] - shift, vals[b], vals[e] if e >= 0 else float("inf"))
            for b, e in theirs[key]
            if (e < 0 or vals[e] > vals[b]) and dims[b] - shift <= 1
        )
    return bars, theirs["time"]


def best_of(fn, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return out, min(times)


def phimaker_best(ordered, values, chosen, python, repeats=3):
    if not python:
        return None, None
    runs = [phimaker_run(ordered, values, chosen, python) for _ in range(repeats)]
    return runs[0][0], min(t for _, t in runs)


def ours_explicit(ordered, values, chosen):
    filtered = ChainComplex.from_simplices(ordered).filtration(values)
    mask = [s in chosen for s in ordered]
    result, elapsed = best_of(lambda: filtered.image_kernel_cokernel(mask, max_dim=1))
    bars = {
        name: sorted((k, b, e) for k, d in enumerate(result[name].diagrams()) for b, e in d.tolist())
        for name in ("image", "kernel", "cokernel")
    }
    _, six = best_of(lambda: filtered.six_pack(mask, max_dim=1))
    return bars, (elapsed, six)


def same(a, b):
    return all(len(a[k]) == len(b[k]) and np.allclose(np.array(a[k], dtype=float).reshape(-1, 3), np.array(b[k], dtype=float).reshape(-1, 3)) for k in a)


def ikc_rows(python, quick):
    rows = []
    cases = [("Rips, 100 points", 100, 0.25), ("Rips, 200 points", 200, 0.18)]
    if not quick:
        cases += [("Rips, 400 points", 400, 0.13), ("Rips, 800 points", 800, 0.09), ("Rips, 1600 points", 1600, 0.065)]
    for name, n, threshold in cases:
        points, ordered, values, subset = rips_case(n, 0, threshold)
        chosen = {s for s in ordered if set(s) <= subset}
        ours, (t_explicit, t_six) = ours_explicit(ordered, values, chosen)
        _, t_lazy = best_of(lambda: rips_image_kernel(points=points, subset=sorted(subset), max_dim=1, threshold=threshold))
        theirs, t_phimaker = phimaker_best(ordered, values, chosen, python)
        agree = "—" if theirs is None else ("✓" if same(ours, theirs) else "✗")
        rows.append(f"| {name}, r ≤ {threshold} | {len(ordered)} | {fmt(t_explicit)} | {fmt(t_six)} | {fmt(t_lazy)} | {fmt(t_phimaker) if t_phimaker else '—'} | {agree} |")
        print(rows[-1], flush=True)
    meshes = [("torus mesh 30×30, lower star", 30)] + ([] if quick else [("torus mesh 60×60, lower star", 60)])
    for name, n in meshes:
        ordered, values, subset = torus_mesh(n, 0)
        chosen = {s for s in ordered if set(s) <= subset}
        ours, (t_explicit, t_six) = ours_explicit(ordered, values, chosen)
        theirs, t_phimaker = phimaker_best(ordered, values, chosen, python)
        agree = "—" if theirs is None else ("✓" if same(ours, theirs) else "✗")
        rows.append(f"| {name} | {len(ordered)} | {fmt(t_explicit)} | {fmt(t_six)} | — | {fmt(t_phimaker) if t_phimaker else '—'} | {agree} |")
        print(rows[-1], flush=True)
    return rows


# ====== Zigzag ======


def sliding_rips(n_points, window, threshold, seed):
    """A zigzag of Rips complexes of a sliding window over points arriving in order: each new point brings its
    simplices with the points in the window (up to triangles), and the oldest point leaves with its simplices (cofaces
    first)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * np.pi, n_points)
    points = np.c_[np.cos(t), np.sin(t)] + 0.15 * rng.standard_normal((n_points, 2))
    close = lambda a, b: np.linalg.norm(points[a] - points[b]) <= threshold  # noqa: E731
    ops, current = [], set()
    for new in range(n_points):
        inside = [v for v in range(max(0, new - window), new)]
        ops.append(((new,), True))
        current.add((new,))
        neighbours = [v for v in inside if close(v, new)]
        for v in neighbours:
            ops.append(((v, new), True))
            current.add((v, new))
        for a, b in itertools.combinations(neighbours, 2):
            if (a, b) in current:
                ops.append(((a, b, new), True))
                current.add((a, b, new))
        old = new - window
        if old >= 0:
            leaving = sorted((s for s in current if old in s), key=len, reverse=True)
            for s in leaving:
                ops.append((s, False))
                current.discard(s)
    return ops


def fzz_run(ops, binary):
    """Dey and Hou's FastZigzag (the C++ `fzz` on PHAT, through a driver timing its `compute` only): its bars as
    `(dimension, birth, death)` closed intervals, as phlite's, and its time."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "filt")
        with open(path, "w") as f:
            for s, insert in ops:
                f.write(("i " if insert else "d ") + " ".join(map(str, sorted(s))) + "\n")
        out = subprocess.run([binary, path], capture_output=True, text=True, check=True, cwd=tmp)
        elapsed = float(out.stderr.strip().splitlines()[-1])
        with open(os.path.join(tmp, "filt_pers")) as f:
            bars = sorted(tuple(map(int, line.split())) for line in f if line.strip())
    return bars, elapsed


def zigzag_rows(python, quick):
    rows = []
    cases = [("sliding Rips, 300 points, window 20", 300, 20, 0.35), ("sliding Rips, 1000 points, window 30", 1000, 30, 0.3)]
    if not quick:
        cases += [("sliding Rips, 5000 points, window 40", 5000, 40, 0.25), ("sliding Rips, 20000 points, window 40", 20000, 40, 0.25)]
    if os.environ.get("ZIGZAG_ONLY"):
        pass
    for name, n, window, threshold in cases:
        ops = sliding_rips(n, window, threshold, 0)
        m = len(ops)
        ours, t_ours = best_of(lambda: zigzag(ops))
        ours_bars = sorted((k, b, -1 if e == m else e + 1) for k, b, e, _, _ in ours)
        fzz = os.environ.get("FZZ_BIN")
        if fzz:
            runs = [fzz_run(ops, fzz) for _ in range(3)]
            fzz_bars, t_fzz = runs[0][0], min(t for _, t in runs)
            fzz_agree = "✓" if fzz_bars == sorted((k, b, e) for k, b, e, _, _ in ours) else "✗"
        else:
            t_fzz, fzz_agree = None, "—"
        if python and m <= 1_000_000:
            out = subprocess.run(
                [python, "-I", "-c", DIONYSUS_RUNNER], input=json.dumps([[list(s), i] for s, i in ops]),
                capture_output=True, text=True, check=True,
            )
            theirs = json.loads(out.stdout)
            theirs_bars = sorted((k, int(b), int(e)) for k, b, e in theirs["bars"] if e < 0 or e > b)
            agree = "✓" if theirs_bars == ours_bars else "✗"
            t_theirs = theirs["time"]
        else:
            agree, t_theirs = "—", None
        rows.append(
            f"| {name} | {m} | {len(ours)} | {fmt(t_ours)} | {fmt(t_fzz) if t_fzz else '—'} | {fzz_agree} | "
            f"{fmt(t_theirs) if t_theirs else '—'} | {agree} |"
        )
        print(rows[-1], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    ikc = [] if os.environ.get("ZIGZAG_ONLY") else ikc_rows(os.environ.get("PHIMAKER_PYTHON"), args.quick)
    zz = zigzag_rows(os.environ.get("DIONYSUS_PYTHON"), args.quick)
    table = "\n".join(
        [
            "## Images, kernels, cokernels: phlite against phimaker (dimensions 0 and 1; ✓: identical barcodes)",
            "",
            "phlite: `image_kernel_cokernel` on the explicit complex; the six-pack (`six_pack`, all six barcodes with",
            "representatives); `rips_image_kernel` from the points, lazily. Best of three.",
            "",
            "| complex | simplices | phlite | phlite six-pack + reps | phlite (lazy Rips) | phimaker | agree |",
            "|---|---:|---:|---:|---:|---:|:-:|",
            *ikc,
            "",
            "## Zigzag: phlite (FastZigzag) against Dey and Hou's fzz (C++, on PHAT) and Dionysus 2 (✓: identical barcodes)",
            "",
            "| zigzag | operations | intervals | phlite | fzz | agree | Dionysus | agree |",
            "|---|---:|---:|---:|---:|:-:|---:|:-:|",
            *zz,
        ]
    )
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
