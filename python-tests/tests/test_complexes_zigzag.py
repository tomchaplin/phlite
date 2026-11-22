"""phlite.complexes.zigzag (FastZigzag) on random zigzags of simplicial complexes, with repeated insertions:

* with insertions only, it is ordinary persistence;
* at every K_i, the intervals containing i count dim H_p(K_i) (over Z/2);
* over every stretch of insertions only (or deletions only) from i to j, the intervals containing [i, j] count the rank
  of the map H(K_i) -> H(K_j) (or H(K_j) -> H(K_i));
* each end is closed or open as the operation that makes it says.
"""

import itertools
import json
import os
import random
import subprocess

import pytest

from hypothesis import given, note, settings
from hypothesis import strategies as st

from phlite.complexes import ChainComplex, Zigzag, ZigzagBar, ZigzagBarcode, zigzag


def random_zigzag(seed, steps=40, n=6):
    rng = random.Random(seed)
    current = set()
    ops = []
    for _ in range(steps):
        if current and rng.random() < 0.4:
            maximal = [s for s in current if not any(set(s) < set(t) for t in current)]
            s = rng.choice(sorted(maximal))
            current.remove(s)
            ops.append((s, False))
        else:
            k = rng.choice([1, 1, 2, 2, 3])
            s = tuple(sorted(rng.sample(range(n), k)))
            if s in current:
                continue
            faces = [s[:i] + s[i + 1:] for i in range(len(s))] if len(s) > 1 else []
            missing = [f for f in faces if f not in current]
            # Insert missing faces first (each a valid operation)
            for f in sorted({g for f in missing for r in range(1, len(f) + 1) for g in itertools.combinations(f, r)}, key=len):
                if f not in current:
                    current.add(f)
                    ops.append((f, True))
            current.add(s)
            ops.append((s, True))
    return ops


def complexes(ops):
    out = [set()]
    for s, insert in ops:
        nxt = set(out[-1])
        (nxt.add if insert else nxt.remove)(s)
        out.append(nxt)
    return out


def rank(vectors):
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


def cycles_and_boundaries(cells, p, index):
    """Bit vectors of a basis of the p-cycles of `cells`, and the boundaries of their (p+1)-cells."""
    def bd(s):
        return sum(1 << index[s[:i] + s[i + 1:]] for i in range(len(s))) if len(s) > 1 else 0
    rows = [(bd(s), 1 << index[s]) for s in cells if len(s) == p + 1]
    pivots, cycles = {}, []
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
            cycles.append(c)
    return cycles, [bd(s) for s in cells if len(s) == p + 2]


def map_rank(small, big, p, index):
    """The rank of H_p(small) -> H_p(big), small ⊆ big."""
    z_small, _ = cycles_and_boundaries(small, p, index)
    _, b_big = cycles_and_boundaries(big, p, index)
    return rank(z_small + b_big) - rank(b_big)


@given(st.integers(0, 10**9))
@settings(deadline=None, max_examples=150)
def test_zigzag(seed):
    ops = random_zigzag(seed)
    note(ops)
    bars = zigzag(ops)
    ks = complexes(ops)
    m = len(ops)
    everything = sorted({s for k in ks for s in k}, key=lambda s: (len(s), s))
    index = {s: i for i, s in enumerate(everything)}
    top = max((len(s) for s in everything), default=1) - 1
    for p in range(top + 1):
        mine = [(b, d) for dim, b, d, _, _ in bars if dim == p]
        for i, k in enumerate(ks):
            expected = map_rank(k, k, p, index)
            assert sum(1 for b, d in mine if b <= i <= d) == expected, (p, i)
        # Monotone stretches
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                kinds = {insert for _, insert in ops[i:j]}
                if len(kinds) != 1:
                    break
                small, big = (ks[i], ks[j]) if kinds == {True} else (ks[j], ks[i])
                assert sum(1 for b, d in mine if b <= i and j <= d) == map_rank(small, big, p, index), (p, i, j)
    for dim, b, d, birth_closed, death_closed in bars:
        assert 0 < b <= d <= m
        assert birth_closed == ops[b - 1][1]
        assert death_closed == (d == m or not ops[d][1])


@given(st.integers(0, 10**9))
@settings(deadline=None, max_examples=100)
def test_insertions_only_is_ordinary_persistence(seed):
    ops = [op for op in random_zigzag(seed) if op[1]]
    # (deletions dropped: keep only simplices whose faces were inserted earlier and that aren't repeated)
    seen, clean = set(), []
    for s, _ in ops:
        faces = [s[:i] + s[i + 1:] for i in range(len(s))] if len(s) > 1 else []
        if s not in seen and all(f in seen for f in faces):
            seen.add(s)
            clean.append((s, True))
    bars = zigzag(clean)
    m = len(clean)
    simplices = [s for s, _ in clean]
    filtered = ChainComplex.from_simplices(simplices).filtration([float(i + 1) for i in range(m)])
    expected = []
    for k, diagram in enumerate(filtered.persistence().diagrams()):
        for b, d in diagram.tolist():
            expected.append((k, int(b), m if d == float("inf") else int(d) - 1, True, d == float("inf")))
    assert sorted(bars) == sorted(expected)


DIONYSUS = os.environ.get("DIONYSUS_PYTHON")

DIONYSUS_RUNNER = r"""
import json, sys
import dionysus as d
ops = json.loads(sys.stdin.read())
times = {}
for i, (s, ins) in enumerate(ops):
    times.setdefault(tuple(s), []).append(i + 1)
f = d.Filtration([d.Simplex(list(s), 0) for s in times])
t = [times[tuple(s)] for s in f]
zz, dgms, cells = d.zigzag_homology_persistence(f, t)
print(json.dumps([[[p.birth, -1 if p.death == float("inf") else p.death] for p in dg] for dg in dgms]))
"""


@pytest.mark.skipif(not DIONYSUS, reason="DIONYSUS_PYTHON not set")
@given(st.integers(0, 10**9))
@settings(deadline=None, max_examples=150)
def test_against_dionysus(seed):
    """The same barcode as Dionysus 2's zigzag_homology_persistence (operation i at time i + 1; its (b, d) is our
    [b, d - 1], its inf our alive at the end), random zigzags with simplices inserted again after deletion."""
    ops = random_zigzag(seed, steps=60)
    note(ops)
    out = subprocess.run([DIONYSUS, "-I", "-c", DIONYSUS_RUNNER], input=json.dumps([[list(s), i] for s, i in ops]),
                         capture_output=True, text=True, check=True)
    theirs = sorted((k, b, float("inf") if e < 0 else e) for k, dg in enumerate(json.loads(out.stdout)) for b, e in dg
                    if e < 0 or e > b)
    ours = sorted((k, b, e) for k, dg in enumerate(zigzag(ops).to_dionysus()) for b, e in dg)
    assert ours == theirs


def test_builder_and_barcode():
    triangle = [(0,), (1,), (2,), (0, 1), (1, 2), (0, 2)]
    barcode = Zigzag().insert(*triangle).delete((0, 2)).barcode()
    assert isinstance(barcode, ZigzagBarcode) and barcode.n_positions == 8
    assert barcode.in_dimension(1) == [ZigzagBar(1, 6, 6, True, True)]
    assert str(barcode.in_dimension(1)[0]) == "H1 [6, 6]"
    assert barcode.rank(6, 1) == 1 and barcode.betti_curve(0).tolist() == [0, 1, 2, 3, 2, 1, 1, 1]
    # The same as the operations; tuples still compare
    ops = Zigzag().insert(*triangle).delete((0, 2)).operations
    assert zigzag(ops) == barcode and list(zigzag(ops)) == [tuple(b) for b in barcode]
    # Faces and cofaces filled in
    zz = Zigzag().insert((0, 1, 2), with_faces=True)
    assert len(zz) == 7 and (0, 1, 2) in zz.current
    zz.delete((0,), with_cofaces=True)
    assert zz.current == {(1,), (2,), (1, 2)}
    with pytest.raises(ValueError, match="already there"):
        Zigzag().insert((0,)).insert((0,))
    with pytest.raises(ValueError):
        Zigzag().insert((0, 1)).barcode()


def closure(simplices):
    return {f for s in simplices for k in range(1, len(s) + 1) for f in itertools.combinations(sorted(s), k)}


@given(st.integers(0, 10**9))
@settings(deadline=None, max_examples=100)
def test_from_complexes_levels(seed):
    """Zigzag.from_complexes: at each level, the Betti numbers are those of K_i (even levels) or K_i ∪ K_{i+1} (odd),
    and over each arrow the intervals spanning it number the rank of the map."""
    rng = random.Random(seed)
    ks = [closure([tuple(rng.sample(range(5), rng.randint(1, 3))) for _ in range(rng.randint(1, 5))])
          for _ in range(rng.randint(1, 5))]
    note(ks)
    levels = [ks[0]]
    for a, b in zip(ks, ks[1:]):
        levels += [a | b, b]
    barcode = Zigzag.from_complexes(ks).barcode(levels=True)
    assert barcode.n_positions == len(levels)
    everything = sorted(set().union(*levels), key=lambda s: (len(s), s))
    index = {s: i for i, s in enumerate(everything)}
    for p in range(3):
        for i, k in enumerate(levels):
            assert barcode.rank(i, p) == map_rank(k, k, p, index), (p, i)
        for i in range(len(levels) - 1):
            small, big = (levels[i], levels[i + 1]) if i % 2 == 0 else (levels[i + 1], levels[i])
            spanning = sum(1 for b in barcode.in_dimension(p) if b.birth <= i and i + 1 <= b.death)
            assert spanning == map_rank(small, big, p, index), (p, i)


def test_rips_windows():
    """Zigzag.rips_windows: the zigzag of Rips complexes of sliding windows has, at each window's level, the Betti
    numbers of that window's Rips complex."""
    pytest.importorskip("phlite.clouds")
    import numpy as np

    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 40)
    points = np.c_[np.cos(t), np.sin(t)] + 0.05 * rng.standard_normal((40, 2))
    zz = Zigzag.rips_windows(points, 12, threshold=0.6, dimension=2)
    barcode = zz.barcode(levels=True)
    assert barcode.n_positions == 2 * (40 - 12) + 1
    for i in range(0, 40 - 12 + 1, 7):
        window = points[i : i + 12]
        close = [[np.linalg.norm(a - b) <= 0.6 for b in window] for a in window]
        simplices = {(v,) for v in range(12)} | {(a, b) for a in range(12) for b in range(a + 1, 12) if close[a][b]}
        simplices |= {(a, b, c) for a, b in [s for s in simplices if len(s) == 2] for c in range(b + 1, 12) if close[a][c] and close[b][c]}
        everything = sorted(simplices, key=lambda s: (len(s), s))
        index = {s: j for j, s in enumerate(everything)}
        for p in range(2):
            assert barcode.rank(2 * i, p) == map_rank(simplices, simplices, p, index), (i, p)
