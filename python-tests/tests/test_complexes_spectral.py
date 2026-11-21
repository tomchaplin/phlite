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
    ss = filtered.spectral_sequence(max_dim=top)
    steps = ss.steps
    pages = [ss.page(r).rows for r in range(1, len(set(values)) + 2)]
    assert steps == sorted(set(values))
    assert [page.r for page in ss][-1] == ss.degenerates_at
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


# ====== Over the integers (modulus=0) ======

import pytest

RP2 = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 1, 5), (1, 2, 4), (2, 3, 5), (1, 3, 4), (2, 4, 5), (1, 3, 5)]


def signed_simplicial(tops, value):
    """The closure of `tops` as `(pairs, coefficients, values)`, with the alternating signs and `value(simplex)`."""
    simplices = set()
    for top in tops:
        for k in range(1, len(top) + 1):
            simplices.update(itertools.combinations(sorted(top), k))
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    index = {s: i for i, s in enumerate(ordered)}
    faces = lambda s: [s[:i] + s[i + 1:] for i in range(len(s))] if len(s) > 1 else []  # noqa: E731
    pairs = [(len(s) - 1, [index[f] for f in faces(s)]) for s in ordered]
    coefficients = [[(-1) ** i for i in range(len(faces(s)))] for s in ordered]
    return pairs, coefficients, [float(value(s)) for s in ordered]


def test_projective_plane_over_the_integers():
    pairs, coefficients, values = signed_simplicial(RP2, lambda s: 0)
    filtered = ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=0).filtration(values)
    assert filtered.persistent_group(1, 0.0, 0.0) == (0, [2])
    assert filtered.persistent_group(0, 0.0, 0.0) == (1, [])
    assert filtered.persistent_group(2, 0.0, 0.0) == (0, [])
    # Filtered by dimension: E^1 is the chain groups, E^2 = E^infinity the homology, Z/2 in degree 1 at step 1
    pairs, coefficients, values = signed_simplicial(RP2, lambda s: len(s) - 1)
    filtered = ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=0).filtration(values)
    ss = filtered.spectral_sequence()
    assert ss.steps == [0.0, 1.0, 2.0] and ss.integral
    assert ss.page(1).rows == [[(6, []), (0, []), (0, [])], [(0, []), (15, []), (0, [])], [(0, []), (0, []), (10, [])]]
    # d^1 is the boundary: from the triangles injectively (H_2 = 0) onto Z^10 of the edges' Z^15
    assert {(d.source, d.target): d.image for d in ss.page(1).differentials} == {
        ((2, 2), (1, 1)): (10, []),
        ((1, 1), (0, 0)): (5, []),
    }
    for r in (2, 3):
        assert ss.page(r).rows == [[(1, []), (0, []), (0, [])], [(0, []), (0, [2]), (0, [])], [(0, []), (0, []), (0, [])]]
        assert ss.page(r).differentials == []
    assert ss.infinity.torsion(1, 1) == [2]
    assert "Z/2" in ss.page(2).table()
    # Over Z/2 the same filtration has H_1 = H_2 = Z/2 instead
    field = ChainComplex.from_pairs(pairs).filtration(values).spectral_sequence()
    assert field.page(2).rows == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_integers_need_signs_and_have_no_barcode():
    pairs, _, values = signed_simplicial(RP2, lambda s: 0)
    with pytest.raises(ValueError, match="signs"):
        ChainComplex.from_pairs(pairs, modulus=0)
    pairs, coefficients, values = signed_simplicial(RP2, lambda s: 0)
    filtered = ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=0).filtration(values)
    assert filtered.chain.modulus == 0
    with pytest.raises(ValueError, match="no barcode"):
        filtered.persistence()
    with pytest.raises(ValueError, match="start <= end"):
        filtered.persistent_group(1, 1.0, 0.0)


@given(st.integers(0, 10**6))
@settings(deadline=None)
def test_integral_free_ranks(seed):
    """Free ranks over Z are the dimensions over Z/13 where there is no torsion (on these small complexes, torsion
    of any order is rare), for the pages and the persistent groups."""
    pairs, values = random_filtered_simplicial(seed)
    # Signs: the faces of each simplex are listed in order of the vertex left out, so alternate
    coefficients = [[(-1) ** i for i in range(len(faces))] for _, faces in pairs]
    integral = ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=0).filtration(values)
    field = ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=13).filtration(values)
    top = max(d for d, _ in pairs)
    r_max = len(set(values)) + 1
    ss = integral.spectral_sequence(max_dim=top)
    steps = ss.steps
    pages = [ss.page(r) for r in range(1, r_max + 1)]
    if any(page.torsion(p, k) for page in pages for (p, k), _ in page.entries()):
        return
    field_ss = field.spectral_sequence(max_dim=top)
    assert [[[e[0] for e in row] for row in page.rows] for page in pages] == [field_ss.page(r).rows for r in range(1, r_max + 1)]
    for dim in range(top + 1):
        for i, start in enumerate(steps):
            for end in steps[i:]:
                group = integral.persistent_group(dim, start, end)
                if not group[1]:
                    assert group[0] == field.persistent_group(dim, start, end)[0], (dim, start, end)


def boundary_over_z(pairs, coefficients, chain):
    """The boundary of `chain` ({cell: coefficient}) over Z, without zero terms."""
    out = {}
    for cell, x in chain.items():
        for face, c in zip(pairs[cell][1], coefficients[cell]):
            out[face] = out.get(face, 0) + x * c
    return {f: x for f, x in out.items() if x}


def test_groups_and_generators_over_the_integers():
    pairs, coefficients, values = signed_simplicial(RP2, lambda s: 0)
    ss = ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=0).filtration(values).spectral_sequence()
    h1 = ss.page(1).group(0, 1)
    assert h1 == (0, [2]) and str(h1) == "Z/2" and h1.elementary_divisors == [2] and h1.order == 2
    assert str(ss.page(1).group(0, 0)) == "Z" and ss.page(1).group(0, 0) == (1, [])
    [gen] = h1.generators
    assert gen.order == 2 and all(pairs[c][0] == 1 for c in gen.chain)
    assert boundary_over_z(pairs, coefficients, gen.chain) == {}, "a generator of H_1 is a cycle"
    assert h1.classify(gen.chain) == [1]
    assert h1.classify({c: 2 * x for c, x in gen.chain.items()}) == [0]
    assert h1.classify(list(gen.chain.items())) == [1]
    edge = next(i for i, (d, _) in enumerate(pairs) if d == 1)
    assert not h1.contains({edge: 1})
    with pytest.raises(ValueError, match="doesn't represent"):
        h1.classify({edge: 1})
    with pytest.raises(ValueError, match="not a cell"):
        h1.classify({10**6: 1})
    # By the vertices: E^infinity has Z/2 at step 3 in degree 1, generated by a cycle (with boundary in F_{3-r} for
    # every r)
    pairs, coefficients, values = signed_simplicial(RP2, lambda s: max(s))
    ss = ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=0).filtration(values).spectral_sequence()
    g = ss.infinity.group(3, 1)
    assert g == (0, [2])
    [gen] = g.generators
    assert boundary_over_z(pairs, coefficients, gen.chain) == {}
    assert all(values[c] <= 3 for c in gen.chain) and any(values[c] == 3 for c in gen.chain)


def test_generators_over_a_field_and_with_cells():
    # Over Z/2 the projective plane has H_1 = H_2 = Z/2, each with one generator
    simplices = sorted({f for t in RP2 for k in range(1, 4) for f in itertools.combinations(t, k)}, key=lambda s: (len(s), s))
    chain = ChainComplex.from_cells(
        simplices,
        dimension=lambda s: len(s) - 1,
        boundary=lambda s: [s[:i] + s[i + 1:] for i in range(len(s))] if len(s) > 1 else [],
    )
    ss = chain.filtration([0.0] * len(simplices)).spectral_sequence()
    h2 = ss.page(1).group(0, 2)
    assert h2 == 1 and str(h2) == "F_2" and ss.modulus == 2
    [gen] = h2.generators
    assert gen.order is None and len(gen.chain) == 10 and all(len(c) == 3 for c in gen.chain)
    assert h2.classify(gen.chain) == [1]
    # Over Z/3: H_1 = H_2 = 0
    signed = ChainComplex.from_cells(
        simplices,
        dimension=lambda s: len(s) - 1,
        boundary=lambda s: {s[:i] + s[i + 1:]: (-1) ** i for i in range(len(s))} if len(s) > 1 else {},
        modulus=3,
    )
    ss3 = signed.filtration([0.0] * len(simplices)).spectral_sequence()
    assert ss3.page(1).group(0, 1).is_zero and ss3.page(1).group(0, 1).generators == []
