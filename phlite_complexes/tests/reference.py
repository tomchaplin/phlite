"""Slow but obviously-correct reference implementations, for testing."""

import itertools
import random


def random_simplicial_complex(rng, n_vertices, n_top, top_dim, max_value=4):
    """A random filtered simplicial complex, closed under faces, with many ties in filtration value.

    Returns (simplices, values, dimensions, columns, coefficients) where the cells are listed in a valid filtration
    order (by value, then dimension, then at random) and columns/coefficients give the oriented boundary.
    """
    tops = {
        tuple(sorted(rng.sample(range(n_vertices), top_dim + 1))) for _ in range(n_top)
    }
    simplices = set()
    for top in tops:
        for k in range(1, len(top) + 1):
            simplices.update(itertools.combinations(top, k))
    simplices.update((v,) for v in range(n_vertices))
    # A monotone filtration: each simplex appears no earlier than its faces
    value = {}
    for simplex in sorted(simplices, key=len):
        faces = (
            [simplex[:i] + simplex[i + 1 :] for i in range(len(simplex))]
            if len(simplex) > 1
            else []
        )
        value[simplex] = max([rng.randint(0, max_value)] + [value[f] for f in faces])
    ordered = sorted(simplices, key=lambda s: (value[s], len(s), rng.random()))
    index = {s: i for i, s in enumerate(ordered)}
    columns, coefficients = [], []
    for s in ordered:
        if len(s) == 1:
            columns.append([])
            coefficients.append([])
            continue
        faces = [s[:i] + s[i + 1 :] for i in range(len(s))]
        columns.append([index[f] for f in faces])
        coefficients.append([(-1) ** i for i in range(len(s))])
    values = [float(value[s]) for s in ordered]
    dimensions = [len(s) - 1 for s in ordered]
    return ordered, values, dimensions, columns, coefficients


def reduce(columns, coefficients, values, p):
    """The standard algorithm over Z/p, on cells ordered by (value, index).

    Returns (pairs, essential) as cell indices, including zero-length pairs.
    """
    n = len(columns)
    key = lambda i: (values[i], i)
    order = sorted(range(n), key=key)
    reduced = {}
    for j in range(n):
        col = {}
        for row, c in zip(columns[j], coefficients[j]):
            col[row] = (col.get(row, 0) + c) % p
        reduced[j] = {r: c for r, c in col.items() if c}
    pivot_owner = {}
    pairs = []
    for j in order:
        col = reduced[j]
        while col:
            pivot = max(col, key=key)
            if pivot not in pivot_owner:
                break
            other = reduced[pivot_owner[pivot]]
            factor = col[pivot] * pow(other[pivot], p - 2, p) % p
            for r, c in other.items():
                col[r] = (col.get(r, 0) - factor * c) % p
                if col[r] == 0:
                    del col[r]
        if col:
            pivot = max(col, key=key)
            pivot_owner[pivot] = j
            pairs.append((pivot, j))
    births = {b for b, _ in pairs}
    deaths = {d for _, d in pairs}
    essential = sorted(i for i in range(n) if i not in births and i not in deaths)
    return sorted(pairs), essential


def boundary_of_chain(chain, columns, coefficients, p):
    """The boundary of a chain {cell: coefficient} over Z/p."""
    out = {}
    for cell, c in chain.items():
        for row, d in zip(columns[cell], coefficients[cell]):
            out[row] = (out.get(row, 0) + c * d) % p
    return {r: c for r, c in out.items() if c}


def in_span(target, vectors, p):
    """Is the vector `target` ({index: coefficient}) in the span of `vectors` over Z/p?"""
    pivots = {}  # pivot index -> reduced vector

    def reduce_vec(v):
        v = dict(v)
        while v:
            top = max(v)
            if top not in pivots:
                return v
            other = pivots[top]
            factor = v[top] * pow(other[top], p - 2, p) % p
            for r, c in other.items():
                v[r] = (v.get(r, 0) - factor * c) % p
                if v[r] == 0:
                    del v[r]
        return v

    for vec in vectors:
        reduced = reduce_vec(vec)
        if reduced:
            pivots[max(reduced)] = reduced
    return not reduce_vec(target)


def rp2():
    """The 6-vertex triangulation of the real projective plane, in an order valid for the index filtration."""
    # 123, 134, 145, 156, 162, 235, 346, 452, 563, 624 on vertices 1..6
    triangles = ["123", "134", "145", "156", "162", "235", "346", "452", "563", "624"]
    triangles = [tuple(sorted(int(v) - 1 for v in t)) for t in triangles]
    simplices = set()
    for t in triangles:
        for k in range(1, 4):
            simplices.update(itertools.combinations(t, k))
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    index = {s: i for i, s in enumerate(ordered)}
    columns = []
    coefficients = []
    for s in ordered:
        faces = [s[:i] + s[i + 1 :] for i in range(len(s))] if len(s) > 1 else []
        columns.append([index[f] for f in faces])
        coefficients.append([(-1) ** i for i in range(len(faces))])
    dimensions = [len(s) - 1 for s in ordered]
    return ordered, dimensions, columns, coefficients


def all_rng_seeds(count):
    return [random.Random(seed) for seed in range(count)]
