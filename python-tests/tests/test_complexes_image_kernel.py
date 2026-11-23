"""FilteredComplex.image_kernel_cokernel: persistence of the image, kernel and cokernel of H(L) -> H(K) for a
subcomplex L, and of (K, L).

* Against linear algebra over Z/2: at every filtration value t, the bars alive (birth <= t < death) number the
  dimensions of the image, kernel and cokernel of H_k(L_t) -> H_k(K_t), and of H_k(K_t, L_t).
* Against phimaker (when `PHIMAKER_PYTHON` names a Python with phimaker installed): the same barcodes.
"""

import itertools
import json
import os
import random
import subprocess

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st
from examples import examples

from phlite.complexes import ChainComplex


def random_case(seed):
    """A random filtered simplicial complex (simplices as vertex tuples, faces first, values no smaller than faces')
    and a random subcomplex (a closed set of its simplices)."""
    rng = random.Random(seed)
    n = rng.randint(2, 7)
    simplices = {(v,) for v in range(n)}
    for _ in range(rng.randint(1, 10)):
        s = tuple(sorted(set(rng.randrange(n) for _ in range(rng.randint(2, 4)))))
        for k in range(1, len(s) + 1):
            simplices.update(itertools.combinations(s, k))
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    values = {}
    for s in ordered:
        faces = [s[:i] + s[i + 1:] for i in range(len(s))] if len(s) > 1 else []
        values[s] = max([values[f] for f in faces] + [rng.randint(0, 5)])
    # The subcomplex: a random set of simplices and all their faces
    chosen = set()
    for s in ordered:
        if rng.random() < 0.4:
            for k in range(1, len(s) + 1):
                chosen.update(itertools.combinations(s, k))
    return ordered, [float(values[s]) for s in ordered], chosen


def rank(vectors):
    """The rank over Z/2 of integers as bit vectors."""
    pivots = {}
    for v in vectors:
        while v:
            top = v.bit_length() - 1
            if top in pivots:
                v ^= pivots[top]
            else:
                pivots[top] = v
                break
    return len(pivots)


def boundary_bits(s, index):
    return sum(1 << index[s[:i] + s[i + 1:]] for i in range(len(s))) if len(s) > 1 else 0


def kernel_vectors(cells, index):
    """A basis of the cycles (bit vectors over all simplices) among the chains on `cells`."""
    rows = [(boundary_bits(s, index), 1 << index[s]) for s in cells]
    pivots = {}
    out = []
    for b, c in rows:
        while b:
            top = b.bit_length() - 1
            if top in pivots:
                pb, pc = pivots[top]
                b, c = b ^ pb, c ^ pc
            else:
                pivots[top] = (b, c)
                break
        if not b:
            out.append(c)
    return out


def pointwise(ordered, values, chosen, t, k):
    """(dim image, dim kernel, dim cokernel, dim relative) of H_k(L_t) -> H_k(K_t), over Z/2."""
    index = {s: i for i, s in enumerate(ordered)}
    K = [s for s, v in zip(ordered, values) if v <= t]
    L = [s for s in K if s in chosen]
    zk = kernel_vectors([s for s in K if len(s) == k + 1], index)
    zl = kernel_vectors([s for s in L if len(s) == k + 1], index)
    bk = [boundary_bits(s, index) for s in K if len(s) == k + 2]
    bl = [boundary_bits(s, index) for s in L if len(s) == k + 2]
    r_bk, r_bl, r_zl, r_zk = rank(bk), rank(bl), rank(zl), rank(zk)
    r_zl_bk = rank(zl + bk)
    image = r_zl_bk - r_bk
    kernel = (r_zl + r_bk - r_zl_bk) - r_bl
    cokernel = (r_zk - r_bk) - image
    # H_k(K, L): cycles relative to L (chains of K whose boundary is in C(L)), modulo boundaries and C_k(L)
    l_mask = sum(1 << index[s] for s in L)
    k_cells = [s for s in K if len(s) == k + 1]
    rel_rows = [(boundary_bits(s, index) & ~l_mask, 1 << index[s]) for s in k_cells]
    pivots, rel_cycles = {}, []
    for b, c in rel_rows:
        while b:
            top = b.bit_length() - 1
            if top in pivots:
                pb, pc = pivots[top]
                b, c = b ^ pb, c ^ pc
            else:
                pivots[top] = (b, c)
                break
        if not b:
            rel_cycles.append(c)
    in_l_k = [1 << index[s] for s in L if len(s) == k + 1]
    relative = rank(rel_cycles + bk + in_l_k) - rank(bk + in_l_k)
    return image, kernel, cokernel, relative


def alive(barcode, k, t):
    if k >= len(barcode.diagrams()):
        return 0
    return sum(1 for b, d in barcode.diagrams()[k].tolist() if b <= t < d)


@given(st.integers(0, 10**9))
@settings(deadline=None, max_examples=examples(200))
def test_pointwise_dimensions(seed):
    ordered, values, chosen = random_case(seed)
    note(f"{ordered} {values} {sorted(chosen)}")
    filtered = ChainComplex.from_simplices(ordered).filtration(values)
    result = filtered.image_kernel_cokernel([s in chosen for s in ordered])
    top = max(len(s) for s in ordered) - 1
    for t in sorted(set(values)):
        for k in range(top + 1):
            expected = pointwise(ordered, values, chosen, t, k)
            got = tuple(alive(result[name], k, t) for name in ("image", "kernel", "cokernel", "relative"))
            assert got == expected, (t, k)


def test_cells_and_errors():
    # A circle K and its vertices L: the image of H_0(L) has three classes until edges identify them in K; each
    # identification is a class of the kernel, born then and never dying (L doesn't change); H_1(K) isn't hit
    simplices = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2)]
    filtered = ChainComplex.from_simplices(simplices).filtration([0, 0, 0, 1, 2, 3])
    result = filtered.image_kernel_cokernel([(0,), (1,), (2,)])
    inf = float("inf")
    assert sorted(result["image"].diagrams()[0].tolist()) == [[0.0, 1.0], [0.0, 2.0], [0.0, inf]]
    assert sorted(result["kernel"].diagrams()[0].tolist()) == [[1.0, inf], [2.0, inf]]
    assert result["cokernel"].diagrams()[1].tolist() == [[3.0, float("inf")]]
    with pytest.raises(ValueError, match="not a subcomplex"):
        filtered.image_kernel_cokernel([(0, 1)])


PHIMAKER = os.environ.get("PHIMAKER_PYTHON")

PHIMAKER_RUNNER = r"""
import json, sys
from phimaker import compute_ensemble
matrix = [tuple(x) for x in json.loads(sys.stdin.read())]
d = compute_ensemble(matrix)
out = {}
for name in ("im", "ker", "cok"):
    dgm = getattr(d, name)
    out[name] = [list(map(int, p)) for p in dgm.paired] + [[int(u), -1] for u in dgm.unpaired]
print(json.dumps(out))
"""


@pytest.mark.skipif(not PHIMAKER, reason="PHIMAKER_PYTHON not set")
@given(st.integers(0, 10**9))
@settings(deadline=None, max_examples=examples(60))
def test_against_phimaker(seed):
    """Image, kernel and cokernel against phimaker. (phimaker assumes a cell whose boundary is in L is in L, so L is
    the full subcomplex on some vertices; and its relative diagram is of the quotient space K / L, not of the pair, so
    isn't compared: the pointwise test checks ours.)"""
    ordered, values, _ = random_case(seed)
    rng = random.Random(seed)
    vertices = {v for v in range(max(s[0] for s in ordered) + 1) if rng.random() < 0.6} or {0}
    chosen = {s for s in ordered if set(s) <= vertices}
    # Filtration order (value, dimension, index), as phimaker's matrix columns
    order = sorted(range(len(ordered)), key=lambda i: (values[i], len(ordered[i]), i))
    position = {ordered[i]: p for p, i in enumerate(order)}
    matrix = []
    for i in order:
        s = ordered[i]
        faces = sorted(position[s[:j] + s[j + 1:]] for j in range(len(s))) if len(s) > 1 else []
        matrix.append([s in chosen, len(s) - 1, faces])
    result = subprocess.run([PHIMAKER, "-I", "-c", PHIMAKER_RUNNER], input=json.dumps(matrix), capture_output=True, text=True, check=True)
    theirs = json.loads(result.stdout)
    vals = [values[i] for i in order]
    dims = [len(ordered[i]) - 1 for i in order]
    filtered = ChainComplex.from_simplices(ordered).filtration(values)
    ours = filtered.image_kernel_cokernel([s in chosen for s in ordered])
    for name, key, shift in (("image", "im", 0), ("kernel", "ker", 1), ("cokernel", "cok", 0)):
        expected = sorted(
            (dims[b] - shift, vals[b], vals[d] if d >= 0 else float("inf"))
            for b, d in theirs[key]
            if d < 0 or vals[d] > vals[b]
        )
        got = sorted((k, b, e) for k, diagram in enumerate(ours[name].diagrams()) for b, e in diagram.tolist())
        assert got == expected, name


def test_rips_lazy_matches_explicit():
    """clouds.rips_image_kernel (on the lazy Rips boundary) against the explicit route: rips_complex, from_simplices
    and image_kernel_cokernel with the subset's full subcomplex."""
    clouds = pytest.importorskip("phlite.clouds")
    import numpy as np

    rng = np.random.default_rng(3)
    for trial in range(6):
        points = rng.random((14, 2))
        subset = sorted(rng.choice(14, 7, replace=False).tolist())
        lazy = clouds.rips_image_kernel(points=points, subset=subset, max_dim=1, threshold=0.7)
        simplices = clouds.rips_complex(points=points, dimension=2, threshold=0.7)
        filtered = ChainComplex.from_simplices([s for s, _ in simplices]).filtration([v for _, v in simplices])
        explicit = filtered.image_kernel_cokernel([set(s) <= set(subset) for s, _ in simplices], max_dim=1)
        for name in ("image", "kernel", "cokernel", "relative"):
            for k in range(2):
                ours = sorted([tuple(p) for p in lazy[name].pairings[k]] + [(b, float("inf")) for b in lazy[name].essential[k]])
                theirs = sorted(map(tuple, explicit[name].diagrams()[k].tolist()))
                assert len(ours) == len(theirs), (trial, name, k)
                assert np.allclose(ours, theirs), (trial, name, k)
