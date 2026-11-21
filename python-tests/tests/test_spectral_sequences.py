"""Spectral sequences of filtered complexes (phlite.complexes, through phlite.core.SpectralSequence): known results
computed from scratch, and properties every spectral sequence must have, fuzzed over random filtered complexes over
the integers (with torsion, and differentials of every length) and over fields.

Ground truth (the integral homology of a complex) comes from a small, independent Smith normal form written here.
"""

import itertools
import math
import random

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from phlite.complexes import ChainComplex

# ====== Ground truth: integral homology by Smith normal form ======


def smith_invariants(rows):
    """The non-zero invariant factors of an integer matrix (a list of rows), by plain elimination."""
    a = [list(r) for r in rows if any(r)]
    out = []
    while a and any(any(r) for r in a):
        # Least non-zero entry to the corner
        _, i, j = min((abs(x), i, j) for i, r in enumerate(a) for j, x in enumerate(r) if x)
        a[0], a[i] = a[i], a[0]
        for r in a:
            r[0], r[j] = r[j], r[0]
        while True:
            p = a[0][0]
            done = True
            for r in a[1:]:
                q = r[0] // p
                if q:
                    for c in range(len(r)):
                        r[c] -= q * a[0][c]
                done &= r[0] == 0
            for c in range(1, len(a[0])):
                q = a[0][c] // p
                if q:
                    for r in a:
                        r[c] -= q * r[0]
                done &= a[0][c] == 0
            if not done:
                _, i, j = min(
                    [(abs(a[i][0]), i, 0) for i in range(len(a)) if a[i][0]]
                    + [(abs(a[0][j]), 0, j) for j in range(len(a[0])) if a[0][j]]
                )
                a[0], a[i] = a[i], a[0]
                for r in a:
                    r[0], r[j] = r[j], r[0]
                continue
            bad = next((r for r in a[1:] if any(x % p for x in r[1:])), None)
            if bad is None:
                break
            for c in range(len(a[0])):
                a[0][c] += bad[c]
        out.append(abs(a[0][0]))
        a = [r[1:] for r in a[1:]]
        a = [r for r in a if any(r)]
    return sorted(out)


def elementary(torsion):
    """Prime powers of a list of orders, sorted (to compare torsion however it is decomposed)."""
    out = []
    for t in torsion:
        n, p = t, 2
        while n > 1:
            q = 1
            while n % p == 0:
                n //= p
                q *= p
            if q > 1:
                out.append(q)
            p += 1
    return sorted(out)


def homology(cells, degree):
    """`H_degree` over Z of the complex `cells` (a list of `(dimension, {face: coefficient})`), as `(rank, torsion)`."""
    k_cells = [i for i, (d, _) in enumerate(cells) if d == degree]
    below = [i for i, (d, _) in enumerate(cells) if d == degree - 1]
    above = [i for i, (d, _) in enumerate(cells) if d == degree + 1]

    def matrix(cols, rows):
        return [[cells[c][1].get(r, 0) for r in rows] for c in cols]  # rows of the transpose: one per column

    low = smith_invariants(matrix(k_cells, below)) if below else []
    high = smith_invariants(matrix(above, k_cells)) if above else []
    return len(k_cells) - len(low) - len(high), elementary([t for t in high if t > 1])


# ====== Building filtered complexes ======


def filtered(cells, values, modulus=0):
    """A FilteredComplex from `(dimension, {face: coefficient})` cells (faces earlier) and values."""
    pairs = [(d, list(b)) for d, b in cells]
    coefficients = [list(b.values()) for _, b in cells]
    return ChainComplex.from_pairs(pairs, coefficients=coefficients, modulus=modulus).filtration([float(v) for v in values])


def cellular(boundaries):
    """A cellular complex with one cell per entry of `boundaries` (`(dimension, {face: coefficient})`)."""
    return list(boundaries)


def projective_space(n):
    """RP^n with one cell in each dimension: the boundary of e_k is (1 + (-1)^k) e_{k-1}."""
    return [(k, {k - 1: 1 + (-1) ** k} if k > 0 and 1 + (-1) ** k else {}) for k in range(n + 1)]


def lens_space(m, n):
    """The (2n+1)-dimensional lens space L(m) (one cell per dimension: d e_{2i} = m e_{2i-1})."""
    return [(k, {k - 1: m} if k > 0 and k % 2 == 0 else {}) for k in range(2 * n + 2)]


def tensor(c, d):
    """The tensor product of two cellular complexes, `(a, b)` in the order of `itertools.product`, with the Koszul sign:
    d(a ⊗ b) = da ⊗ b + (-1)^|a| a ⊗ db."""
    index = {(i, j): n for n, (i, j) in enumerate(itertools.product(range(len(c)), range(len(d))))}
    cells = []
    for i, j in itertools.product(range(len(c)), range(len(d))):
        (da, ba), (db, bb) = c[i], d[j]
        boundary = {}
        for f, x in ba.items():
            boundary[index[f, j]] = boundary.get(index[f, j], 0) + x
        for f, x in bb.items():
            boundary[index[i, f]] = boundary.get(index[i, f], 0) + (-1) ** da * x
        cells.append((da + db, {f: x for f, x in boundary.items() if x}))
    # Faces must come first: order by dimension
    order = sorted(range(len(cells)), key=lambda n: cells[n][0])
    position = {old: new for new, old in enumerate(order)}
    pairs = [(i, j) for i, j in itertools.product(range(len(c)), range(len(d)))]
    return [(cells[o][0], {position[f]: x for f, x in cells[o][1].items()}) for o in order], [pairs[o] for o in order]


def random_filtered(seed, n_pieces=7, max_degree=3, max_value=4, exact=False, sparse=False):
    """A random filtered complex over Z: a sum of elementary pieces (a free cell, or a pair with d e = m e' for m in
    1, 2, 3, 4 or 6), each cell at a random value (no earlier than its face), then a random unitriangular change of
    basis in each degree (each cell plus multiples of cells no later than it), which keeps faces no later than their
    cofaces and mixes everything up (`sparse`: at most two other cells each, keeping coefficients small on big
    complexes). Returns `(cells, values)`, cells in order of dimension."""
    rng = random.Random(seed)
    raw = []  # (dimension, value, partner index or None, multiplier)
    for _ in range(n_pieces if exact else rng.randint(1, n_pieces)):
        k = rng.randint(0, max_degree)
        v = rng.randint(0, max_value)
        if k < max_degree and rng.random() < 0.6:
            m = rng.choice([1, 2, 3, 4, 6])
            raw.append((k, v, None, 0))
            raw.append((k + 1, rng.randint(v, max_value), len(raw) - 1, m))
        else:
            raw.append((k, v, None, 0))
    # Boundaries in the elementary basis
    boundary = [({p: m} if p is not None else {}) for (_, _, p, m) in raw]
    # Change of basis: b_i = e_i + sum a_ij e_j over e_j of the same dimension, no later (value, then index)
    n = len(raw)
    key = lambda i: (raw[i][1], i)  # noqa: E731
    change = []  # change[i] = {j: a_ij} including i: 1
    for i in range(n):
        row = {i: 1}
        earlier = [j for j in range(n) if j != i and raw[j][0] == raw[i][0] and key(j) < key(i)]
        if sparse:
            chosen = rng.sample(earlier, min(len(earlier), rng.randint(0, 2)))
        else:
            chosen = [j for j in earlier if rng.random() < 0.35]
        for j in chosen:
            row[j] = rng.choice([-2, -1, 1, 2])
        change.append(row)
    # The inverse change, by forward substitution in the order of `key`: e_i = b_i - sum a_ij e_j
    inverse = {}
    for i in sorted(range(n), key=key):
        row = {i: 1}
        for j, a in change[i].items():
            if j != i:
                for l, c in inverse[j].items():
                    row[l] = row.get(l, 0) - a * c
        inverse[i] = {l: c for l, c in row.items() if c}
    # d b_i = sum_j a_ij d e_j, in the e basis, then in the b basis
    cells = []
    for i in range(n):
        in_e = {}
        for j, a in change[i].items():
            for f, x in boundary[j].items():
                in_e[f] = in_e.get(f, 0) + a * x
        in_b = {}
        for f, x in in_e.items():
            for l, c in inverse[f].items():
                in_b[l] = in_b.get(l, 0) + x * c
        cells.append((raw[i][0], {f: x for f, x in in_b.items() if x}))
    values = [raw[i][1] for i in range(n)]
    # Order by dimension (faces first), renumbering
    order = sorted(range(n), key=lambda i: (cells[i][0], i))
    position = {old: new for new, old in enumerate(order)}
    out = [(cells[o][0], {position[f]: x for f, x in cells[o][1].items()}) for o in order]
    return out, [values[o] for o in order]


def entry(page, p, k):
    """`(rank, elementary divisors)` of a page entry, over Z or a field."""
    e = page[p, k]
    if isinstance(e, tuple):
        return e[0], elementary(e[1])
    return e, []


# ====== Known results ======


@pytest.mark.parametrize("n", range(1, 7))
def test_projective_spaces_by_skeleta(n):
    """Filtered by skeleta, E^1 is the cellular chain complex, d^1 the cellular boundary, and E^2 = E^infinity is
    the homology: H_k(RP^n) = Z (k = 0, and k = n odd), Z/2 (k odd < n), 0 otherwise."""
    cells = projective_space(n)
    ss = filtered(cells, [d for d, _ in cells], modulus=0).spectral_sequence(integer_steps=True)
    e1 = ss.page(1)
    for k in range(n + 1):
        assert entry(e1, k, k) == (1, [])
        expected = (1, []) if k == 0 or (k == n and n % 2) else ((0, [2]) if k % 2 else (0, []))
        assert entry(ss.page(2), k, k) == expected == entry(ss.infinity, k, k)
        assert homology(cells, k) == expected
    # d^1 from e_k is multiplication by 2 for even k > 0
    assert {d.source for d in e1.differentials} == {(k, k) for k in range(2, n + 1, 2)}
    assert ss.degenerates_at is None or ss.page(2).is_degenerate


@pytest.mark.parametrize("m, n", [(3, 1), (5, 2), (4, 2)])
def test_lens_spaces(m, n):
    cells = lens_space(m, n)
    ss = filtered(cells, [d for d, _ in cells], modulus=0).spectral_sequence(integer_steps=True)
    for k in range(2 * n + 2):
        expected = (1, []) if k in (0, 2 * n + 1) else ((0, elementary([m])) if k % 2 else (0, []))
        assert entry(ss.infinity, k, k) == expected


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_hopf_fibrations(n):
    """S^1 -> S^{2n+1} -> CP^n, filtered by the cells of the base (a cellular model: e_{2i}, e_{2i+1} over the 2i-cell
    of CP^n, with d e_{2i} = e_{2i-1}): E^2_{p,q} = H_p(CP^n) ⊗ H_q(S^1), Z at p = 0, 2, ..., 2n and q = 0, 1; d^2 is an
    isomorphism from (2i, q = 0) to (2i - 2, q = 1) for i = 1, ..., n; E^3 = E^infinity is H(S^{2n+1}): Z at (0, 0) and
    (2n, 2n + 1)."""
    cells = [(k, {k - 1: 1} if k > 0 and k % 2 == 0 else {}) for k in range(2 * n + 2)]
    values = [2 * (k // 2) for k in range(2 * n + 2)]
    ss = filtered(cells, values, modulus=0).spectral_sequence(integer_steps=True)
    e2 = ss.page(2)
    for i in range(n + 1):
        assert entry(e2, 2 * i, 2 * i) == (1, []) and entry(e2, 2 * i, 2 * i + 1) == (1, [])
    assert {(d.source, d.target) for d in e2.differentials} == {((2 * i, 2 * i), (2 * i - 2, 2 * i - 1)) for i in range(1, n + 1)}
    # d^2 on the generators: an isomorphism Z -> Z (coordinate ±1)
    for i in range(1, n + 1):
        g = e2.group(2 * i, 2 * i)
        assert [abs(c) for c in g.d(g.generators[0].chain)] == [1]
    infinity = ss.infinity
    nonzero = {(p, k): e for (p, k), e in infinity.entries()}
    assert nonzero == {(0, 0): (1, []), (2 * n, 2 * n + 1): (1, [])}
    # Over a field, the same
    field = filtered(cells, values, modulus=3).spectral_sequence(integer_steps=True)
    assert dict(field.infinity.entries()) == {(0, 0): 1, (2 * n, 2 * n + 1): 1}


@pytest.mark.parametrize("twisted", [False, True])
def test_circle_bundles_over_the_circle(twisted):
    """The torus and the Klein bottle as S^1-bundles over S^1, filtered by the base: cells v, b (the fibre) at p = 0,
    a, c over the base's 1-cell at p = 1, with d c = 0 (torus) or 2 b (Klein bottle, the monodromy reversing the fibre).
    E^2_{p,q} = H_p(S^1; H_q(S^1)), with twisted coefficients for the Klein bottle: H_0(S^1; Z^-) = Z/2,
    H_1(S^1; Z^-) = 0. So H_1(K) = Z ⊕ Z/2 and H_2(K) = 0."""
    cells = [(0, {}), (1, {}), (1, {}), (2, {1: 2} if twisted else {})]  # v, b, a, c
    ss = filtered(cells, [0, 0, 1, 1], modulus=0).spectral_sequence(integer_steps=True)
    e2 = ss.page(2)
    if twisted:
        assert entry(e2, 0, 1) == (0, [2]) and entry(e2, 1, 2) == (0, []) and entry(e2, 1, 1) == (1, [])
        [gen] = e2.group(0, 1).generators
        assert gen.order == 2 and gen.chain in ({1: 1}, {1: -1})
    else:
        assert entry(e2, 0, 1) == (1, []) and entry(e2, 1, 2) == (1, []) and entry(e2, 1, 1) == (1, [])
    for k in range(3):
        rank = sum(entry(ss.infinity, p, k)[0] for p in range(2))
        assert rank == homology(cells, k)[0]


def test_kunneth_spectral_sequence():
    """RP^2 × RP^2 (the tensor product of cellular chains), filtered by the dimension of the first factor:
    E^2_{p,q} = H_p(RP^2; H_q(RP^2)), which over Z is Z at (0, 0), Z/2 at (1, 0), (0, 1), (1, 1), (2, 1), and 0
    elsewhere; it degenerates there, and converges to H(RP^2 × RP^2) = Z, (Z/2)^2, Z/2, Z/2 (a Tor term), 0."""
    rp2 = projective_space(2)
    cells, pairs = tensor(rp2, rp2)
    values = [rp2[i][0] for i, _ in pairs]
    ss = filtered(cells, values, modulus=0).spectral_sequence(integer_steps=True)
    e2 = ss.page(2)
    expected = {(0, 0): (1, []), (1, 1): (0, [2]), (0, 1): (0, [2]), (1, 2): (0, [2]), (2, 3): (0, [2])}
    for p in range(3):
        for q in range(3):
            assert entry(e2, p, p + q) == expected.get((p, p + q), (0, [])), (p, q)
    assert e2.is_degenerate and ss.page(3).rows == e2.rows
    for k, (rank, torsion) in enumerate([(1, []), (0, [2, 2]), (0, [2]), (0, [2]), (0, [])]):
        assert homology(cells, k) == (rank, torsion)
        assert sorted(q for p in range(3) for q in entry(ss.infinity, p, k)[1]) == torsion


def test_double_complex_filtrations_converge_to_the_same_homology():
    """Both filtrations of a double complex (here a tensor product, by either factor) converge to its homology."""
    c, d = lens_space(3, 1), projective_space(3)
    cells, pairs = tensor(c, d)
    for which in (0, 1):
        values = [(c[i][0] if which == 0 else d[j][0]) for i, j in pairs]
        ss = filtered(cells, values, modulus=0).spectral_sequence(integer_steps=True)
        for k in range(max(dim for dim, _ in cells) + 1):
            rank, torsion = homology(cells, k)
            assert sum(entry(ss.infinity, p, k)[0] for p in range(ss.n_steps)) == rank
            if rank == 0:
                assert math.prod(math.prod(entry(ss.infinity, p, k)[1]) for p in range(ss.n_steps)) == math.prod(torsion)


# ====== Properties of every spectral sequence, fuzzed ======


def random_case(seed, kind):
    if kind == "torsion":
        return random_filtered(seed)
    # A random simplicial complex with the alternating signs
    rng = random.Random(seed)
    n = rng.randint(1, 6)
    simplices = {(v,) for v in range(n)}
    for _ in range(rng.randint(0, 8)):
        s = tuple(sorted(set(rng.randrange(n) for _ in range(rng.randint(2, 4)))))
        for k in range(1, len(s) + 1):
            simplices.update(itertools.combinations(s, k))
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    index = {s: i for i, s in enumerate(ordered)}
    values = {}
    for s in ordered:
        faces = [s[:i] + s[i + 1:] for i in range(len(s))] if len(s) > 1 else []
        values[s] = max([values[f] for f in faces] + [rng.randint(0, 3)])
    cells = [(len(s) - 1, {index[s[:i] + s[i + 1:]]: (-1) ** i for i in range(len(s))} if len(s) > 1 else {}) for s in ordered]
    return cells, [values[s] for s in ordered]


cases = st.tuples(st.integers(0, 10**9), st.sampled_from(["torsion", "simplicial"]))
rings = st.sampled_from([0, 2, 3])


def rank_of(e):
    return e[0] if isinstance(e, tuple) else e


@given(cases, rings)
@settings(deadline=None, max_examples=150)
def test_each_page_is_the_homology_of_the_last(case, modulus):
    """rank E^{r+1} = rank E^r - rank(d^r out) - rank(d^r in), and pages only shrink."""
    cells, values = random_case(*case)
    note(f"{cells} {values}")
    ss = filtered(cells, values, modulus).spectral_sequence(integer_steps=True)
    for r in range(1, ss.n_steps + 1):
        page, following = ss.page(r), ss.page(r + 1)
        out = {d.source: rank_of(d.image) for d in page.differentials}
        into = {d.target: rank_of(d.image) for d in page.differentials}
        for p, row in enumerate(page.rows):
            for k, e in enumerate(row):
                assert rank_of(following[p, k]) == rank_of(e) - out.get((p, k), 0) - into.get((p, k), 0)


@given(cases)
@settings(deadline=None, max_examples=150)
def test_convergence_to_the_homology(case):
    """E^infinity in degree k sums (in rank) to H_k of the whole complex, and when it is finite its order is |H_k|
    (whatever the extensions); E^1 is the homology of the associated graded F_p / F_{p-1}."""
    cells, values = random_case(*case)
    note(f"{cells} {values}")
    ss = filtered(cells, values, 0).spectral_sequence(integer_steps=True)
    top = max(d for d, _ in cells)
    for k in range(top + 1):
        rank, torsion = homology(cells, k)
        column = [entry(ss.infinity, p, k) for p in range(ss.n_steps)]
        assert sum(r for r, _ in column) == rank
        if rank == 0:
            assert math.prod(math.prod(t) for _, t in column) == math.prod(torsion)
    for p in range(ss.n_steps):
        keep = [i for i, v in enumerate(values) if v == p]
        position = {c: n for n, c in enumerate(keep)}
        graded = [(cells[c][0], {position[f]: x for f, x in cells[c][1].items() if f in position}) for c in keep]
        for k in range(top + 1):
            assert entry(ss.page(1), p, k) == (homology(graded, k) if graded else (0, []))


@given(cases, st.sampled_from([2, 3]))
@settings(deadline=None, max_examples=150)
def test_universal_coefficients_on_the_first_page(case, prime):
    """E^1 is the homology of free complexes, so over Z/p its dimension is the free rank over Z plus the torsion
    summands of order divisible by p in the same and the previous degree."""
    cells, values = random_case(*case)
    integral = filtered(cells, values, 0).spectral_sequence(integer_steps=True).page(1)
    field = filtered(cells, values, prime).spectral_sequence(integer_steps=True).page(1)
    for p, row in enumerate(integral.rows):
        for k, (rank, torsion) in enumerate(row):
            divisible = sum(1 for t in torsion if t % prime == 0)
            previous = sum(1 for t in integral[p, k - 1][1] if t % prime == 0) if k else 0
            assert field[p, k] == rank + divisible + previous


@given(cases, rings)
@settings(deadline=None, max_examples=100)
def test_generators_and_classes(case, modulus):
    """Each entry's generators: as many as its invariants (which checks the barcode's ranks against the local
    computation, over a field), each classifies to its unit vector, order × generator to zero, and a combination
    to its coefficients (modulo the orders)."""
    cells, values = random_case(*case)
    note(f"{cells} {values}")
    ss = filtered(cells, values, modulus).spectral_sequence(integer_steps=True)
    rng = random.Random(0)
    for r in (1, 2, 3):
        page = ss.page(r)
        for p, row in enumerate(page.rows):
            for k in range(len(row)):
                g = page.group(p, k)
                gens = g.generators
                assert len(gens) == g.rank + len(g.torsion)
                assert [x.order for x in gens if x.order] == g.torsion
                for i, x in enumerate(gens):
                    assert g.classify(x.chain) == [int(i == j) for j in range(len(gens))]
                    if x.order:
                        assert g.classify({c: x.order * a for c, a in x.chain.items()}) == [0] * len(gens)
                if gens:
                    coefficients = [rng.randint(-3, 3) for _ in gens]
                    combination = {}
                    for a, x in zip(coefficients, gens):
                        for c, b in x.chain.items():
                            combination[c] = combination.get(c, 0) + a * b
                    expected = [a % x.order if x.order else a for a, x in zip(coefficients, gens)]
                    if modulus:
                        expected = [a % modulus for a in coefficients]
                        got = [a % modulus for a in g.classify(combination)]
                        assert got == expected
                    else:
                        assert g.classify(combination) == expected


@given(cases, rings)
@settings(deadline=None, max_examples=100)
def test_d_squared_is_zero_on_elements(case, modulus):
    """d^r ∘ d^r = 0, on generators: d^r [x] = [dx] in E^r_{p-r}, and d^r of (a representative of) that is zero. And
    the images of the generators span the image of d^r (in rank)."""
    cells, values = random_case(*case)
    note(f"{cells} {values}")
    ss = filtered(cells, values, modulus).spectral_sequence(integer_steps=True)
    for r in (1, 2, 3):
        page = ss.page(r)
        images = {d.source: rank_of(d.image) for d in page.differentials}
        for p, row in enumerate(page.rows):
            for k in range(len(row)):
                g = page.group(p, k)
                target = g.d_target
                if target is None:
                    continue
                vectors = []
                for x in g.generators:
                    y = g.d(x.chain)
                    vectors.append(y)
                    # A representative of d^r [x], and d^r of it
                    chain = {}
                    for a, t in zip(y, target.generators):
                        for c, b in t.chain.items():
                            chain[c] = chain.get(c, 0) + a * b
                    twice = target.d(chain)
                    for a, t in zip(twice, target.d_target.generators if target.d_target else []):
                        assert a % t.order == 0 if t.order else a == 0
                # The rank of the images (over Q, or over the field) is the rank of the image of d^r
                assert matrix_rank(vectors, modulus, target) == images.get((p, k), 0)


def matrix_rank(vectors, modulus, target):
    """The rank of the images of the generators: over the field, or over Q on the free coordinates."""
    free = [i for i, t in enumerate(target.generators) if t.order is None]
    rows = [[v[i] for i in free] for v in vectors]
    rows = [r for r in rows if any(r)]
    if not rows:
        return 0
    p = modulus or None
    rank, columns = 0, len(rows[0])
    rows = [list(r) for r in rows]
    for c in range(columns):
        pivot = next((i for i in range(rank, len(rows)) if (rows[i][c] % p if p else rows[i][c])), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i][c]:
                if p:
                    f = rows[i][c] * pow(rows[rank][c], -1, p)
                    rows[i] = [(a - f * b) % p for a, b in zip(rows[i], rows[rank])]
                else:
                    a, b = rows[rank][c], rows[i][c]
                    rows[i] = [a * x - b * y for x, y in zip(rows[i], rows[rank])]
        rank += 1
    return rank


@given(cases, rings)
@settings(deadline=None, max_examples=80)
def test_reparametrising_the_filtration(case, modulus):
    """The spectral sequence depends on the filtration only through the order of its levels: any increasing change of
    values (here v -> 3v + 1/2, and v -> 2^v) gives the same pages over the distinct steps."""
    cells, values = random_case(*case)
    pages = []
    for f in (lambda v: v, lambda v: 3 * v + 0.5, lambda v: 2.0**v):
        ss = filtered(cells, [f(v) for v in values], modulus).spectral_sequence()
        pages.append([ss.page(r).rows for r in (1, 2, 3)])
    assert pages[0] == pages[1] == pages[2]


@given(cases, rings)
@settings(deadline=None, max_examples=80)
def test_cell_order_does_not_matter(case, modulus):
    """Listing the cells in another order (faces still first) gives isomorphic groups."""
    cells, values = random_case(*case)
    rng = random.Random(case[0])
    order = sorted(range(len(cells)), key=lambda i: (cells[i][0], rng.random()))
    position = {old: new for new, old in enumerate(order)}
    shuffled = [(cells[o][0], {position[f]: x for f, x in cells[o][1].items()}) for o in order]
    a = filtered(cells, values, modulus).spectral_sequence(integer_steps=True)
    b = filtered(shuffled, [values[o] for o in order], modulus).spectral_sequence(integer_steps=True)
    for r in (1, 2, 3):
        for p, row in enumerate(a.page(r).rows):
            for k in range(len(row)):
                assert a.page(r).group(p, k) == b.page(r).group(p, k)
