"""FilteredComplex.spectral_sequence: E^1 is the homology of the associated graded F_p / F_{p-1} (computed here
directly, as the homology of each step's cells with the boundary restricted to them), and the last page sums to the
homology of the whole complex."""

import itertools
import random

from hypothesis import given, settings
from hypothesis import strategies as st

from phlite.complexes import ChainComplex


def random_filtered_simplicial(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 7)
    simplices = {(v,) for v in range(n)}
    for _ in range(rng.randint(0, 12)):
        s = tuple(sorted(set(rng.randrange(n) for _ in range(rng.randint(2, 4)))))
        for k in range(1, len(s) + 1):
            simplices.update(itertools.combinations(s, k))
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    index = {s: i for i, s in enumerate(ordered)}
    values = {}
    for s in ordered:
        faces = [s[:i] + s[i + 1:] for i in range(len(s))] if len(s) > 1 else []
        values[s] = max([values[f] for f in faces] + [rng.randint(0, 3)])
    pairs = [(len(s) - 1, [index[s[:i] + s[i + 1:]] for i in range(len(s))] if len(s) > 1 else []) for s in ordered]
    return pairs, [float(values[s]) for s in ordered]


def homology_ranks(pairs, keep):
    """Betti numbers over Z/2 of the cells `keep` with the boundary restricted to them."""
    keep = sorted(keep)
    position = {c: i for i, c in enumerate(keep)}
    sub = [(pairs[c][0], [position[f] for f in pairs[c][1] if f in position]) for c in keep]
    if not sub:
        return {}
    barcode = ChainComplex.from_pairs(sub).filtration([0.0] * len(sub)).persistence()
    ranks = {}
    for k, bars in enumerate(barcode.diagrams()):
        essential = sum(1 for b, d in bars if d == float("inf"))
        if essential:
            ranks[k] = essential
    return ranks


@given(st.integers(0, 10**6))
@settings(deadline=None)
def test_first_page_and_convergence(seed):
    pairs, values = random_filtered_simplicial(seed)
    filtered = ChainComplex.from_pairs(pairs).filtration(values)
    top = max(d for d, _ in pairs)
    steps, pages = filtered.spectral_sequence(r_max=len(set(values)) + 1, max_dim=top)
    assert steps == sorted(set(values))
    for p, value in enumerate(steps):
        graded = homology_ranks(pairs, [c for c, v in enumerate(values) if v == value])
        assert {k: r for k, r in enumerate(pages[0][p]) if r} == graded
    # The last page (r beyond every bar) is the homology of the whole complex
    whole = homology_ranks(pairs, range(len(pairs)))
    last = {}
    for row in pages[-1]:
        for k, r in enumerate(row):
            if r:
                last[k] = last.get(k, 0) + r
    assert last == whole
