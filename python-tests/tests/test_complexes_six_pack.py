"""Six-packs: FilteredComplex.six_pack (inclusions) and FilteredMap.six_pack (any chain map, by the mapping cylinder).

Over Z/2, at every filtration value t and in every dimension k, against linear algebra on bit vectors:

* the bars of each barcode alive at t number the dimensions of H_k(L_t), H_k(K_t), the image, kernel and cokernel of
  f_*: H_k(L_t) -> H_k(K_t), and the relative homology (of the pair for an inclusion; of the mapping cone for a map,
  coker_k + ker_{k-1} by the long exact sequence);
* their representatives are a basis there: cycles of the right complex, present at t, independent modulo the right
  subspace (and, for the kernel, mapping to boundaries).
"""

import itertools
import random

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from phlite.core import Cone, SixPack
from phlite.complexes import ChainComplex, FilteredMap


def closure(simplices):
    out = set()
    for s in simplices:
        for k in range(1, len(s) + 1):
            out.update(itertools.combinations(s, k))
    return out


def random_complex(rng, n, extra):
    simplices = {(v,) for v in range(n)}
    for _ in range(extra):
        simplices |= closure([tuple(sorted(set(rng.randrange(n) for _ in range(rng.randint(2, 4)))))])
    return simplices


def values_for(ordered, rng, floor=lambda s: 0):
    values = {}
    for s in ordered:
        faces = [s[:i] + s[i + 1 :] for i in range(len(s))] if len(s) > 1 else []
        values[s] = max([values[f] for f in faces] + [rng.randint(0, 5), floor(s)])
    return values


def faces(s):
    return [s[:i] + s[i + 1 :] for i in range(len(s))] if len(s) > 1 else []


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


def independent_mod(vectors, subspace):
    return rank(subspace + vectors) - rank(subspace) == len(vectors)


class Space:
    """A filtered simplicial complex with its simplices as bits."""

    def __init__(self, simplices, values):
        self.simplices = sorted(simplices, key=lambda s: (len(s), s))
        self.index = {s: i for i, s in enumerate(self.simplices)}
        self.values = values

    def bits(self, chain):
        out = 0
        for cell, coefficient in chain:
            if coefficient % 2:
                out ^= 1 << self.index[cell]
        return out

    def boundary(self, bits):
        out = 0
        for i in range(bits.bit_length()):
            if bits >> i & 1:
                for f in faces(self.simplices[i]):
                    out ^= 1 << self.index[f]
        return out

    def present(self, t, k=None):
        return [s for s in self.simplices if self.values[s] <= t and (k is None or len(s) == k + 1)]

    def cycles(self, t, k):
        """A basis of the k-cycles at t."""
        pivots, out = {}, []
        for s in self.present(t, k):
            b, c = self.boundary(1 << self.index[s]), 1 << self.index[s]
            while b:
                top = b.bit_length() - 1
                if top not in pivots:
                    pivots[top] = (b, c)
                    break
                b, c = b ^ pivots[top][0], c ^ pivots[top][1]
            if not b:
                out.append(c)
        return out

    def boundaries(self, t, k):
        return [self.boundary(1 << self.index[s]) for s in self.present(t, k + 1)]

    def at(self, bits, t):
        return all(self.values[self.simplices[i]] <= t for i in range(bits.bit_length()) if bits >> i & 1)


def alive(barcode, t, k):
    return [bar for bar in barcode.in_dimension(k) if bar.alive_at(t)]


def check(pack: SixPack, L: Space, K: Space, f, inclusion: bool, max_dim=None):
    """`f` maps bits of L to bits of K. In dimensions up to `max_dim` (default: all)."""
    times = sorted(set(L.values.values()) | set(K.values.values()))
    top = max(len(s) for s in K.simplices + L.simplices)
    for t in times:
        for k in range(top if max_dim is None else max_dim + 1):
            zl, zk, bl, bk = L.cycles(t, k), K.cycles(t, k), L.boundaries(t, k), K.boundaries(t, k)
            fzl = [f(z) for z in zl]
            h_l, h_k = rank(zl) - rank(bl), rank(zk) - rank(bk)
            image = rank(fzl + bk) - rank(bk)
            dims = {"domain": h_l, "codomain": h_k, "image": image, "kernel": h_l - image, "cokernel": h_k - image}
            reps = {name: [L.bits(b.representative) if name in ("domain", "image", "kernel") else K.bits(b.representative)
                           for b in alive(pack[name], t, k)] for name in dims}
            for name, n in dims.items():
                assert len(reps[name]) == n, (name, t, k)
            space = {"domain": L, "image": L, "kernel": L, "codomain": K, "cokernel": K}
            for name, vectors in reps.items():
                for z in vectors:
                    assert space[name].boundary(z) == 0 and space[name].at(z, t), (name, t, k)
            assert independent_mod(reps["domain"], bl)
            assert independent_mod(reps["codomain"], bk)
            assert independent_mod([f(z) for z in reps["image"]], bk)
            assert all(rank(bk + [f(z)]) == rank(bk) for z in reps["kernel"])
            assert independent_mod(reps["kernel"], bl)
            assert independent_mod(reps["cokernel"], bk + fzl)
            # Relative
            rel = alive(pack.relative, t, k)
            if inclusion:
                l_chains = [1 << K.index[s] for s in L.present(t, k)]
                l_mask = sum(1 << K.index[s] for s in L.simplices)
                vectors = [K.bits(b.representative) for b in rel]
                for z in vectors:
                    assert z & l_mask == 0 and K.at(z, t) and K.boundary(z) & ~l_mask == 0, (t, k)
                assert independent_mod(vectors, bk + l_chains)
                # dim H_k(K_t, L_t), from the relative cycles
                rel_cycles = Space.__new__(Space)
                cyc = [1 << K.index[s] for s in K.present(t, k) if s not in L.index]
                pivots, rc = {}, []
                for c in cyc:
                    b, cc = K.boundary(c) & ~l_mask, c
                    while b:
                        top_bit = b.bit_length() - 1
                        if top_bit not in pivots:
                            pivots[top_bit] = (b, cc)
                            break
                        b, cc = b ^ pivots[top_bit][0], cc ^ pivots[top_bit][1]
                    if not b:
                        rc.append(cc)
                assert len(vectors) == rank(rc + bk + l_chains) - rank(bk + l_chains), (t, k)
            else:
                # The mapping cone: (a, b) with a a k-chain of K, b a (k-1)-chain of L, as a | b << n_K;
                # d(a, b) = (da + f b, db)
                n_k = len(K.simplices)
                low = (1 << n_k) - 1

                def cone_bits(chain):
                    out = 0
                    for cell, coefficient in chain:
                        if coefficient % 2:
                            out ^= (1 << (n_k + L.index[cell.cell])) if isinstance(cell, Cone) else (1 << K.index[cell])
                    return out

                def cone_boundary(v):
                    a, b = v & low, v >> n_k
                    return (K.boundary(a) ^ f(b)) | (L.boundary(b) << n_k)

                vectors = [cone_bits(b.representative) for b in rel]
                for v in vectors:
                    assert cone_boundary(v) == 0 and K.at(v & low, t) and L.at(v >> n_k, t), (t, k)
                cone_boundaries = [K.boundary(1 << K.index[s]) for s in K.present(t, k + 1)] + [
                    f(1 << L.index[s]) | (L.boundary(1 << L.index[s]) << n_k) for s in L.present(t, k)
                ]
                assert independent_mod(vectors, cone_boundaries)
                # coker_k + ker_{k-1}
                if k == 0:
                    ker_below = 0
                else:
                    zl1, bl1, bk1 = L.cycles(t, k - 1), L.boundaries(t, k - 1), K.boundaries(t, k - 1)
                    image1 = rank([f(z) for z in zl1] + bk1) - rank(bk1)
                    ker_below = rank(zl1) - rank(bl1) - image1
                assert len(vectors) == dims["cokernel"] + ker_below, (t, k)


@given(st.integers(0, 10**9))
@settings(deadline=None, max_examples=150)
def test_inclusion(seed):
    rng = random.Random(seed)
    simplices = random_complex(rng, rng.randint(2, 7), rng.randint(1, 10))
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    values = values_for(ordered, rng)
    chosen = closure([s for s in ordered if rng.random() < 0.4])
    note(f"{ordered} {values} {sorted(chosen)}")
    filtered = ChainComplex.from_simplices(ordered).filtration([values[s] for s in ordered])
    pack = filtered.six_pack([s in chosen for s in ordered])
    K = Space(ordered, values)
    L = Space(chosen, {s: values[s] for s in chosen})
    to_k = lambda bits: sum(1 << K.index[L.simplices[i]] for i in range(bits.bit_length()) if bits >> i & 1)  # noqa: E731
    check(pack, L, K, to_k, inclusion=True)


def random_map(seed):
    rng = random.Random(seed)
    n_l, n_k = rng.randint(1, 6), rng.randint(1, 5)
    l_simplices = random_complex(rng, n_l, rng.randint(0, 8))
    g = [rng.randrange(n_k) for _ in range(n_l)]
    image = lambda s: tuple(sorted({g[v] for v in s}))  # noqa: E731
    k_simplices = random_complex(rng, n_k, rng.randint(0, 6)) | closure([image(s) for s in l_simplices])
    k_ordered = sorted(k_simplices, key=lambda s: (len(s), s))
    k_values = values_for(k_ordered, rng)
    l_ordered = sorted(l_simplices, key=lambda s: (len(s), s))
    l_values = values_for(l_ordered, rng, floor=lambda s: k_values[image(s)])
    return l_ordered, l_values, k_ordered, k_values, g


@given(st.integers(0, 10**9))
@settings(deadline=None, max_examples=150)
def test_simplicial_map(seed):
    l_ordered, l_values, k_ordered, k_values, g = random_map(seed)
    note(f"L={l_ordered} {l_values}\nK={k_ordered} {k_values}\ng={g}")
    Lf = ChainComplex.from_simplices(l_ordered).filtration([l_values[s] for s in l_ordered])
    Kf = ChainComplex.from_simplices(k_ordered).filtration([k_values[s] for s in k_ordered])
    pack = FilteredMap.simplicial(Lf, Kf, g).six_pack()
    L, K = Space(l_ordered, l_values), Space(k_ordered, k_values)

    def f(bits):
        out = 0
        for i in range(bits.bit_length()):
            if bits >> i & 1:
                s = L.simplices[i]
                t = tuple(sorted({g[v] for v in s}))
                if len(t) == len(s):
                    out ^= 1 << K.index[t]
        return out

    check(pack, L, K, f, inclusion=False)


@given(st.integers(0, 10**9))
@settings(deadline=None, max_examples=60)
def test_cylinder_matches_inclusion(seed):
    """An inclusion given as a FilteredMap (by the mapping cylinder) has the same six barcodes as
    FilteredComplex.six_pack (no cylinder)."""
    rng = random.Random(seed)
    simplices = random_complex(rng, rng.randint(2, 7), rng.randint(1, 10))
    ordered = sorted(simplices, key=lambda s: (len(s), s))
    values = values_for(ordered, rng)
    chosen = sorted(closure([s for s in ordered if rng.random() < 0.5]), key=lambda s: (len(s), s)) or [ordered[0]]
    Kf = ChainComplex.from_simplices(ordered).filtration([values[s] for s in ordered])
    Lf = ChainComplex.from_simplices(chosen).filtration([values[s] for s in chosen])
    by_map = FilteredMap(Lf, Kf, {s: [s] for s in chosen}).six_pack(representatives=False)
    direct = Kf.six_pack(chosen, representatives=False)
    for name in by_map:
        assert by_map[name].to_gudhi() == direct[name].to_gudhi(), name


def test_api_and_errors():
    K = ChainComplex.from_simplices([(0,), (1,), (2,), (0, 1), (1, 2), (0, 2)]).filtration([0, 0, 0, 1, 2, 3])
    pack = K.six_pack([(0,), (1,), (2,)], max_dim=1)
    assert list(pack) == ["domain", "codomain", "image", "kernel", "cokernel", "relative"]
    assert pack.kernel.diagram(0).tolist() == [[1.0, float("inf")], [2.0, float("inf")]]
    assert pack.ranks(2.5, 0) == {"domain": 3, "codomain": 1, "image": 1, "kernel": 2, "cokernel": 0, "relative": 0}
    assert pack.cokernel.essential(1)[0].cells is not None
    assert "kernel" in repr(pack)
    no_reps = K.six_pack([(0,)], representatives=False)
    assert all(bar.representative is None for barcode in no_reps.values() for bar in barcode)
    # Maps: filtration, dimension, chain map, field
    L = ChainComplex.from_simplices([(0,), (1,), (0, 1)]).filtration([0, 0, 0])
    with pytest.raises(ValueError, match="enters later"):
        FilteredMap.simplicial(L, K, {0: 0, 1: 1})
    with pytest.raises(ValueError, match="another dimension"):
        FilteredMap(L, K, {(0,): [(0, 1)]})
    with pytest.raises(ValueError, match="chain map"):
        FilteredMap(L, L, {(0, 1): [(0, 1)]})
    K3 = ChainComplex.from_simplices([(0,), (1,), (0, 1)], modulus=3).filtration([0, 0, 0])
    with pytest.raises(ValueError, match="same field"):
        FilteredMap(L, K3, {})
    # Over Z/3: the domain and codomain are the complexes' own barcodes
    L3 = ChainComplex.from_simplices([(0,), (1,), (2,), (0, 1), (1, 2), (0, 2)], modulus=3).filtration([0, 0, 0, 1, 2, 3])
    pack3 = FilteredMap.simplicial(L3, L3, [0, 1, 2]).six_pack()
    assert pack3.codomain.to_gudhi() == pack3.domain.to_gudhi()
    assert len(pack3.kernel) == 0 and len(pack3.cokernel) == 0


def rips_space(points, threshold, dim=2):
    clouds = pytest.importorskip("phlite.clouds")
    simplices = clouds.rips_complex(points=points, dimension=dim, threshold=threshold)
    values = {s: v for s, v in simplices}
    ordered = sorted(values, key=lambda s: (len(s), s))
    return ordered, values


@pytest.mark.parametrize("trial", range(8))
def test_rips_lazy(trial):
    """clouds.rips_six_pack (lazy, an inclusion) and clouds.rips_map_six_pack (lazy, through the cylinder) against the
    explicit complexes, with representatives checked."""
    clouds = pytest.importorskip("phlite.clouds")
    import numpy as np

    rng = np.random.default_rng(trial)
    points = rng.random((9, 2))
    threshold = 0.6
    ordered, values = rips_space(points, threshold)
    K = Space(ordered, values)
    # A subset
    subset = sorted(rng.choice(9, 5, replace=False).tolist())
    pack = clouds.rips_six_pack(points=points, subset=subset, max_dim=1, threshold=threshold)
    chosen = [s for s in ordered if set(s) <= set(subset)]
    explicit = ChainComplex.from_simplices(ordered).filtration([values[s] for s in ordered]).six_pack(chosen, max_dim=1)
    for name in pack:
        assert np.allclose(
            np.array([b for _, b in pack[name].to_gudhi()]).reshape(-1, 2),
            np.array([b for _, b in explicit[name].to_gudhi()]).reshape(-1, 2),
        ), name
    L = Space(chosen, {s: values[s] for s in chosen})
    to_k = lambda bits: sum(1 << K.index[L.simplices[i]] for i in range(bits.bit_length()) if bits >> i & 1)  # noqa: E731
    check(pack, L, K, to_k, inclusion=True, max_dim=1)
    # A map that doesn't stretch: squash onto a line, or shrink, with points identified
    if trial % 2:
        # Projection onto the x-axis, with some points sharing x (identified)
        points = np.c_[rng.choice([0.0, 0.2, 0.4, 0.6], 9), rng.random(9)]
        ordered, values = rips_space(points, threshold)
        image_points = points * [1.0, 0.0]
    else:
        image_points = 0.7 * points
    unique, vertex_map = np.unique(image_points, axis=0, return_inverse=True)
    vertex_map = vertex_map.ravel().tolist()
    mapped = clouds.rips_map_six_pack(points, unique, vertex_map, max_dim=1, threshold=threshold, metric="euclidean")
    k_ordered, k_values = rips_space(unique, threshold)
    Lf = ChainComplex.from_simplices(ordered).filtration([values[s] for s in ordered])
    Kf = ChainComplex.from_simplices(k_ordered).filtration([k_values[s] for s in k_ordered])
    explicit = FilteredMap.simplicial(Lf, Kf, vertex_map).six_pack(max_dim=1)
    for name in mapped:
        assert np.allclose(
            np.array([b for _, b in mapped[name].to_gudhi()]).reshape(-1, 2),
            np.array([b for _, b in explicit[name].to_gudhi()]).reshape(-1, 2),
        ), name
    L, K = Space(ordered, values), Space(k_ordered, k_values)

    def f(bits):
        out = 0
        for i in range(bits.bit_length()):
            if bits >> i & 1:
                s = L.simplices[i]
                t = tuple(sorted({vertex_map[v] for v in s}))
                if len(t) == len(s):
                    out ^= 1 << K.index[t]
        return out

    check(mapped, L, K, f, inclusion=False, max_dim=1)
