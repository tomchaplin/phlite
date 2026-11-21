"""phlite.complexes.zigzag (FastZigzag) on random zigzags of simplicial complexes, with repeated insertions:

* with insertions only, it is ordinary persistence;
* at every K_i, the intervals containing i count dim H_p(K_i) (over Z/2);
* over every stretch of insertions only (or deletions only) from i to j, the intervals containing [i, j] count the rank
  of the map H(K_i) -> H(K_j) (or H(K_j) -> H(K_i));
* each end is closed or open as the operation that makes it says.
"""

import itertools
import random

from hypothesis import given, note, settings
from hypothesis import strategies as st

from phlite.complexes import ChainComplex, zigzag


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
