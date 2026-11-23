"""Compare phlite's spectral sequences with other software, on the same filtered complexes: correctness (every entry of
pages E^1, ..., E^R) and time.

* Kenzo (Common Lisp; Ana Romero's spectral sequences, `eff-spsq-basis-dvs`), over Z. Needs SBCL and a checkout of
  https://github.com/miguelmarco/kenzo, given as `KENZO_DIR`.
* spectral-sequence-engine (pure Python, https://github.com/towre676-cloud/spectral-sequence-engine), over F_p.
  Needs a checkout, given as `SSE_DIR`.

Run from python-tests/latest (phlite.complexes needs Python 3.13):

    KENZO_DIR=... SSE_DIR=... uv run --no-sync python ../benchmarks/spectral_compare.py [--quick] [--out FILE]

Prints (and with --out writes) a markdown table. Each tool computes every entry E^r_{p,q} for r = 1..R, every step p
and every degree; times are wall-clock for that, excluding process start-up and loading (Kenzo's load is excluded by
timing inside Lisp; phlite's includes building the complex from Python lists).
"""

import argparse
import itertools
import os
import random
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_spectral_sequences import elementary, lens_space, projective_space, random_filtered, tensor  # noqa: E402

from phlite.complexes import ChainComplex  # noqa: E402

R = 4


# ====== Inputs ======


def hopf(n):
    cells = [(k, {k - 1: 1} if k > 0 and k % 2 == 0 else {}) for k in range(2 * n + 2)]
    return cells, [2 * (k // 2) for k in range(2 * n + 2)]


def skeletal(cells):
    return cells, [d for d, _ in cells]


def product(c, d):
    cells, pairs = tensor(c, d)
    return cells, [c[i][0] for i, _ in pairs]


def torus_mesh(n, seed=0):
    """A triangulated n×n torus, lower-star filtered by random integer heights in 0..9 (simplicial, signed)."""
    rng = random.Random(seed)
    height = {(i, j): rng.randint(0, 9) for i in range(n) for j in range(n)}
    v = lambda i, j: (i % n) * n + (j % n)  # noqa: E731
    tris = set()
    for i in range(n):
        for j in range(n):
            a, b, c, d = v(i, j), v(i + 1, j), v(i, j + 1), v(i + 1, j + 1)
            tris.add(tuple(sorted((a, b, d))))
            tris.add(tuple(sorted((a, c, d))))
    simplices = {(x,) for x in range(n * n)}
    for t in tris:
        for k in (2, 3):
            simplices.update(itertools.combinations(t, k))
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    index = {s: i for i, s in enumerate(ordered)}
    h = [height[divmod(x, n)] for x in range(n * n)]
    cells = [(len(s) - 1, {index[s[:i] + s[i + 1:]]: (-1) ** i for i in range(len(s))} if len(s) > 1 else {}) for s in ordered]
    return cells, [max(h[x] for x in s) for s in ordered]


def cases(quick):
    out = [
        ("lens L(5), 7-dim, by skeleta", skeletal(lens_space(5, 3))),
        ("Hopf S^1 → S^9 → CP^4", hopf(4)),
        ("RP^3 × RP^3 (Künneth)", product(projective_space(3), projective_space(3))),
        ("L(3) × RP^4 (Künneth)", product(lens_space(3, 2), projective_space(4))),
        ("random over Z, 40 pieces", random_filtered(1, n_pieces=40, max_value=8, exact=True, sparse=True)),
        ("random over Z, 150 pieces", random_filtered(2, n_pieces=150, max_value=10, exact=True, sparse=True)),
        ("torus mesh 6×6, lower star", torus_mesh(6)),
        ("torus mesh 10×10, lower star", torus_mesh(10)),
    ]
    if not quick:
        out += [
            ("random over Z, 500 pieces", random_filtered(3, n_pieces=500, max_value=12, exact=True, sparse=True)),
            ("random over Z, 2000 pieces", random_filtered(4, n_pieces=2000, max_value=16, exact=True, sparse=True)),
            ("torus mesh 16×16, lower star", torus_mesh(16)),
            ("torus mesh 24×24, lower star", torus_mesh(24)),
        ]
    return out


# ====== phlite ======


def phlite_pages(cells, values, modulus):
    """{(r, p, n): (rank, elementary divisors)} for r = 1..R, and the time."""
    start = time.perf_counter()
    pairs = [(d, list(b)) for d, b in cells]
    coefficients = [list(b.values()) for _, b in cells]
    ss = (
        ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=modulus)
        .filtration([float(v) for v in values])
        .spectral_sequence(integer_steps=True)
    )
    out = {}
    for r in range(1, R + 1):
        for p, row in enumerate(ss.page(r).rows):
            for n, e in enumerate(row):
                rank, torsion = e if isinstance(e, tuple) else (e, [])
                if rank or torsion:
                    out[r, p, n] = (rank, elementary(torsion))
    return out, time.perf_counter() - start


# ====== Kenzo ======

KENZO_HEADER = """
(require :asdf)
(setf asdf:*compile-file-warnings-behaviour* :ignore asdf:*compile-file-failure-behaviour* :ignore)
(push (truename "{dir}/") asdf:*central-registry*)
(let ((*standard-output* (make-broadcast-stream)) (*error-output* (make-broadcast-stream)))
  (asdf:load-system :kenzo))
(in-package :cat)
;; The finite-spaces module, loaded after Romero's spectral sequences, redefines SUBMATRIX with another arity: reload
;; Romero's file so that its definitions win (SBCL compiles each form as it loads)
(let ((*standard-output* (make-broadcast-stream)) (*error-output* (make-broadcast-stream)))
  (load "{dir}/src/anromero/spectral-sequences.lisp"))
(defvar *basis*)
(defvar *dffr*)
(defvar *fltr*)
"""

KENZO_CASE = """
(setf *basis* (make-hash-table) *dffr* (make-hash-table) *fltr* (make-hash-table))
{tables}
(handler-case
(let* ((fc (build-flcc :cmpr #'f-cmpr
                       :basis #'(lambda (degr) (gethash degr *basis* nil))
                       :intr-dffr #'(lambda (degr gnrt) (apply #'cmbn (1- degr) (gethash gnrt *dffr* nil)))
                       :dffr-strt :gnrt
                       :flin #'(lambda (degr gnrt) (declare (ignore degr)) (gethash gnrt *fltr*))
                       :orgn '(phlite-bench {case})))
       (results nil)
       (start (get-internal-real-time)))
  (loop for r from 1 to {r_max} do
    (loop for p from 0 to {p_max} do
      (loop for n from 0 to {n_max} do
        (let ((divs (second (let ((*standard-output* (make-broadcast-stream))) (eff-spsq-basis-dvs fc r p (- n p))))))
          (push (list r p n divs) results)))))
  (format t "~&TIME {case} ~F~%" (/ (- (get-internal-real-time) start) internal-time-units-per-second))
  (dolist (item results)
    (format t "~&ENTRY {case} ~D ~D ~D~{{ ~D~}}~%" (first item) (second item) (third item) (fourth item))))
  (error (e) (format t "~&FAIL {case} ~A~%" (remove (code-char 10) (princ-to-string e)))))
"""


def kenzo_tables(cells, values):
    lines = []
    by_degree = {}
    for i, (d, _) in enumerate(cells):
        by_degree.setdefault(d, []).append(i)
    for d, gens in by_degree.items():
        lines.append(f"(setf (gethash {d} *basis*) '({' '.join(map(str, gens))}))")
    for i, (_, b) in enumerate(cells):
        terms = " ".join(f"{x} {f}" for f, x in sorted(b.items()) if x)
        if terms:
            lines.append(f"(setf (gethash {i} *dffr*) '({terms}))")
        lines.append(f"(setf (gethash {i} *fltr*) {values[i]})")
    return "\n".join(lines)


def kenzo_pages(all_cases, kenzo_dir, timeout):
    """Run every case in one SBCL session: {case index: (pages, seconds)} (a case that fails or times out is
    missing)."""
    script = KENZO_HEADER.format(dir=kenzo_dir)
    for i, (_, (cells, values)) in enumerate(all_cases):
        script += KENZO_CASE.format(
            tables=kenzo_tables(cells, values),
            case=i,
            r_max=R,
            p_max=max(values),
            n_max=max(d for d, _ in cells),
        )
    with tempfile.NamedTemporaryFile("w", suffix=".lisp", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        output = subprocess.run(
            ["sbcl", "--non-interactive", "--load", path], capture_output=True, text=True, timeout=timeout
        ).stdout
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    finally:
        os.unlink(path)
    out = {}
    failures = {}
    for line in output.splitlines():
        if m := re.match(r"FAIL (\d+) (.*)", line):
            failures[int(m[1])] = m[2][:80]
        elif m := re.match(r"TIME (\d+) ([\d.eE+-]+)", line):
            out.setdefault(int(m[1]), [{}, None])[1] = float(m[2])
        elif m := re.match(r"ENTRY (\d+) (\d+) (\d+) (\d+)((?: -?\d+)*)", line):
            case, r, p, n = map(int, m.groups()[:4])
            divs = [int(x) for x in m[5].split()]
            rank = sum(1 for d in divs if d == 0)
            torsion = elementary([d for d in divs if d > 1])
            if rank or torsion:
                out.setdefault(case, [{}, None])[0][r, p, n] = (rank, torsion)
    return {i: (pages, t) for i, (pages, t) in out.items() if t is not None}, failures


# ====== spectral-sequence-engine ======


def engine_pages(cells, values, prime, sse_dir):
    sys.path.insert(0, sse_dir)
    from spectral_sequence import FilteredChainComplex

    start = time.perf_counter()
    generators = [
        (f"c{i}", d, int(values[i]), {f"c{f}": x for f, x in b.items()}) for i, (d, b) in enumerate(cells)
    ]
    complex_ = FilteredChainComplex(generators, prime=prime)
    out = {}
    for r in range(1, R + 1):
        for (p, q), dim in complex_.page(r).items():
            out[r, p, p + q] = (dim, [])
    return out, time.perf_counter() - start


# ====== Main ======


def fmt(t):
    if t is None:
        return "—"
    return f"{t * 1e3:.1f} ms" if t < 1 else f"{t:.2f} s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--prime", type=int, default=3)
    parser.add_argument("--engine-limit", type=int, default=2500, help="largest complex given to the engine")
    parser.add_argument("--kenzo-timeout", type=int, default=3600)
    args = parser.parse_args()
    kenzo_dir, sse_dir = os.environ.get("KENZO_DIR"), os.environ.get("SSE_DIR")
    all_cases = cases(args.quick)
    kenzo, kenzo_failures = kenzo_pages(all_cases, kenzo_dir, args.kenzo_timeout) if kenzo_dir else ({}, {})
    rows = []
    for i, (name, (cells, values)) in enumerate(all_cases):
        z, tz = phlite_pages(cells, values, 0)
        f, tf = phlite_pages(cells, values, args.prime)
        torsion = sum(1 for e in z.values() if e[1])
        kz = kenzo.get(i)
        k_agree = "—" if kz is None else ("✓" if kz[0] == z else "✗")
        if i in kenzo_failures:
            k_agree = "fails¹" if "FIXNUM" in kenzo_failures[i] else "fails"
        if sse_dir and len(cells) <= args.engine_limit:
            e, te = engine_pages(cells, values, args.prime, sse_dir)
            e_agree = "✓" if e == f else "✗"
        else:
            te, e_agree = None, "—"
        rows.append(
            f"| {name} | {len(cells)} | {torsion} | {fmt(tz)} | {fmt(kz[1] if kz else None)} | {k_agree} | "
            f"{fmt(tf)} | {fmt(te)} | {e_agree} |"
        )
        print(rows[-1], flush=True)
    header = [
        f"Pages E^1..E^{R}, every entry. Over Z: phlite (`modulus=0`) against Kenzo (`eff-spsq-basis-dvs`); over "
        f"F_{args.prime}: phlite against spectral-sequence-engine. ✓: every entry agrees.",
        "",
        "| complex | cells | torsion entries | phlite Z | Kenzo Z | agree | phlite F_p | engine F_p | agree |",
        "|---|---:|---:|---:|---:|:-:|---:|---:|:-:|",
    ]
    notes = ["", "¹ Kenzo's Smith normal form keeps entries in fixnums (63 bits), which overflow on these inputs."]
    table = "\n".join(header + rows + (notes if any("FIXNUM" in f for f in kenzo_failures.values()) else []))
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
