"""Fuzz phlite.images.cubical: against GUDHI's and Cubical Ripser's cubical complexes (T- and V-constructions, 1 to 3
axes), against every cubical complex built by brute force (any construction, filtration and number of axes, with NaN
and infinite pixels) and reduced by LoPHAT, with representatives checked from first principles; and against invariants:
the Euler characteristic of every sublevel set, superlevel sets as sublevel sets of the negated image, symmetries of
the grid, and monotone maps of the values."""

import itertools
import math

import cripser
import gudhi
import lophat
import numpy as np
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from grid import algorithm_grid
from mod2 import Span, chain_sum
from phlite.core import Algorithm
from phlite.images import cubical

INF = math.inf
METHODS = ["cohomology", "homology"]
ALGORITHMS = [
    Algorithm.standard(),
    Algorithm.serial(),
    Algorithm.parallel(),
    Algorithm.serial().with_columns("merged").with_cache(True),
    Algorithm.clearing().with_store_r(),
]


@st.composite
def images(draw, max_axes=3, max_pixels=60, special=True):
    """Small images: integer values (lots of ties) or float32 ones, sometimes with NaN and infinite pixels."""
    axes = draw(st.integers(1, max_axes))
    side = max(1, int(max_pixels ** (1 / axes)))
    shape = tuple(draw(st.integers(1, side)) for _ in range(axes))
    if draw(st.booleans()):
        value = st.integers(0, 4).map(float)
    else:
        value = st.floats(-10, 10, allow_nan=False, width=32)
    if special and draw(st.booleans()):
        value = st.one_of(value, st.sampled_from([math.nan, INF, -INF]))
    pixels = [draw(value) for _ in range(math.prod(shape))]
    return np.array(pixels, dtype=float).reshape(shape)


def diagram(barcode, dim):
    return sorted(list(barcode.pairings[dim]) + [(b, INF) for b in barcode.essential[dim]])


def diagrams(barcode):
    return [diagram(barcode, dim) for dim in range(len(barcode.pairings))]


# ====== Brute force ========================================================


def cells_of(image, construction, filtration):
    """Every cell of the complex (as doubled-grid coordinates) with the value it enters at in the filtration
    (sublevel; for superlevel sets, of the negated image), leaving out those that never enter."""
    values = np.where(np.isnan(image), INF, image if filtration == "sublevel" else -image)
    shape = image.shape
    if construction == "V":
        extent = [2 * n - 1 for n in shape]
    else:
        extent = [2 * n + 1 for n in shape]
    cells = {}
    for c in itertools.product(*(range(e) for e in extent)):
        if construction == "V":
            # The largest of its vertices: pixels c/2 rounded either way along odd axes
            around = itertools.product(*([x // 2] if x % 2 == 0 else [x // 2, x // 2 + 1] for x in c))
            value = max(values[p] for p in around)
        else:
            # The smallest of the top cells it is a face of: pixels (c-1)/2 and (c+1)/2 along even axes, inside
            choices = [[(x - 1) // 2] if x % 2 else [p for p in (x // 2 - 1, x // 2) if 0 <= p < n] for x, n in zip(c, shape)]
            value = min(values[p] for p in itertools.product(*choices))
        if value != INF:
            cells[c] = float(value)
    return cells


def cube_boundary(cell):
    faces = []
    for a, x in enumerate(cell):
        if x % 2:
            faces.append(cell[:a] + (x - 1,) + cell[a + 1 :])
            faces.append(cell[:a] + (x + 1,) + cell[a + 1 :])
    return faces


def dimension(cell):
    return sum(x % 2 for x in cell)


def lophat_diagrams(cells, max_dim, sign=1.0):
    if not cells:
        # LoPHAT can't anti-transpose an empty matrix
        return [[] for _ in range(max_dim + 1)]
    order = sorted(cells, key=lambda c: (cells[c], dimension(c), c))
    index = {c: i for i, c in enumerate(order)}
    columns = [(dimension(c), sorted(index[f] for f in cube_boundary(c))) for c in order]
    pairing = lophat.compute_pairings(columns, True, lophat.LoPhatOptions())
    bars = [[] for _ in range(max_dim + 1)]
    paired = set()
    for b, d in pairing.paired:
        paired.update((b, d))
        dim = dimension(order[b])
        if dim <= max_dim and cells[order[b]] < cells[order[d]]:
            bars[dim].append((sign * cells[order[b]], sign * cells[order[d]]))
    for i, c in enumerate(order):
        if i not in paired and dimension(c) <= max_dim:
            bars[dimension(c)].append((sign * cells[c], sign * INF))
    # Superlevel bars are (birth, death) with birth > death; essential ones die at -inf
    return [sorted((b, INF if d == -INF else d) for b, d in dim) for dim in bars]


def check_representatives(cells, barcode, sign):
    for dim in range(len(barcode.pairings)):
        cofaces = [(v, c) for c, v in cells.items() if dimension(c) == dim + 1]

        def boundaries_before(time, strict):
            return Span(frozenset(cube_boundary(c)) for v, c in cofaces if (v < time if strict else v <= time))

        bars = list(zip(barcode.pairings[dim], barcode.pairing_representatives[dim]))
        bars += [((b, INF), rep) for b, rep in zip(barcode.essential[dim], barcode.essential_representatives[dim])]
        for (birth, death), rep in bars:
            note(f"dim={dim} bar=({birth}, {death}) rep={rep}")
            rep = [tuple(c) for c in rep]
            assert all(c in cells and dimension(c) == dim for c in rep), "not a cell of the complex"
            if dim > 0:
                assert not chain_sum(frozenset(cube_boundary(c)) for c in rep), "not a cycle"
            assert max(cells[c] for c in rep) == sign * birth, "not born at birth"
            death = INF if death == INF else sign * death
            chain = frozenset(rep)
            assert chain not in boundaries_before(death, strict=True), "already a boundary before death"
            if death != INF:
                assert chain in boundaries_before(death, strict=False), "not a boundary at death"


@given(
    image=images(max_axes=4, max_pixels=40),
    construction=st.sampled_from(["T", "V"]),
    filtration=st.sampled_from(["sublevel", "superlevel"]),
)
def test_matches_brute_force(image, construction, filtration):
    note(f"image={image.tolist()}")
    max_dim = image.ndim - 1
    sign = 1.0 if filtration == "sublevel" else -1.0
    cells = cells_of(image, construction, filtration)
    expected = lophat_diagrams(cells, max_dim, sign)
    for method, algorithm in itertools.product(METHODS, ALGORITHMS):
        note(f"method={method} algorithm={algorithm!r}")
        barcode = cubical(
            image, construction=construction, filtration=filtration, method=method, algorithm=algorithm,
            representatives=True,
        )
        assert diagrams(barcode) == expected
        check_representatives(cells, barcode, sign)


# ====== Against GUDHI and Cubical Ripser ==================================


def reference_diagrams(pairs, n_dims):
    """(dim, birth, death) triples, without zero-length bars, as sorted diagrams."""
    bars = [[] for _ in range(n_dims)]
    for dim, b, d in pairs:
        if dim < n_dims and d != b:
            bars[int(dim)].append((float(b), INF if d >= 1e308 else float(d)))
    return [sorted(dim) for dim in bars]


@given(image=images(special=False, max_pixels=200), construction=st.sampled_from(["T", "V"]))
def test_matches_gudhi(image, construction):
    note(f"image={image.tolist()}")
    complex = gudhi.CubicalComplex(**{"top_dimensional_cells" if construction == "T" else "vertices": image})
    pairs = [(dim, b, d) for dim, (b, d) in complex.persistence(homology_coeff_field=2, min_persistence=-1)]
    assert diagrams(cubical(image, construction=construction)) == reference_diagrams(pairs, image.ndim)


@given(image=images(special=False, max_pixels=200), construction=st.sampled_from(["T", "V"]))
def test_matches_cripser(image, construction):
    note(f"image={image.tolist()}")
    if construction == "T":
        pairs = cripser.computePH_T(image, maxdim=image.ndim - 1)
    else:
        # Cubical Ripser's V-construction leaves out the edges of a 2D image along an axis of length 1 (so [[0, 0]]
        # has two components); dropping axes of length 1 doesn't change the V-construction's complex
        squeezed = image.reshape([n for n in image.shape if n > 1] or [1])
        pairs = cripser.computePH(squeezed, maxdim=squeezed.ndim - 1)
    pairs = [tuple(row[:3]) for row in pairs]
    assert diagrams(cubical(image, construction=construction)) == reference_diagrams(pairs, image.ndim)


# ====== Invariants =========================================================


def euler_characteristic(cells, t):
    return sum((-1) ** dimension(c) for c, v in cells.items() if v <= t)


@given(image=images(max_pixels=100), construction=st.sampled_from(["T", "V"]))
def test_euler_characteristic(image, construction):
    """At every value t, the alternating sum of the numbers of bars alive at t is the Euler characteristic of the
    sublevel set at t."""
    barcode = cubical(image, construction=construction)
    cells = cells_of(image, construction, "sublevel")
    for t in sorted(set(cells.values())):
        alive = sum((-1) ** dim * sum(b <= t < d for b, d in diagram(barcode, dim)) for dim in range(image.ndim))
        assert alive == euler_characteristic(cells, t), f"t={t}"


@given(image=images(), construction=st.sampled_from(["T", "V"]), seed=st.integers(0, 2**32 - 1))
def test_symmetries_and_monotone_maps(image, construction, seed):
    """Flipping or permuting the axes changes nothing; an increasing map of the values maps the bars; superlevel sets
    are the sublevel sets of the negated image, reported in the image's values; and every algorithm agrees."""
    rng = np.random.default_rng(seed)
    expected = diagrams(cubical(image, construction=construction))
    permuted = np.transpose(image, rng.permutation(image.ndim))
    flipped = np.flip(image, axis=int(rng.integers(image.ndim)))
    assert diagrams(cubical(permuted, construction=construction)) == expected
    assert diagrams(cubical(flipped, construction=construction)) == expected

    # Strictly increasing maps, exactly so in floating point: doubling, and each finite value to its rank
    finite = np.unique(image[np.isfinite(image)])
    rank = {float(v): float(i) for i, v in enumerate(finite)}
    for increasing in [lambda x: 2 * x, lambda x: rank.get(x, x)]:
        mapped = diagrams(cubical(np.vectorize(increasing, otypes=[float])(image), construction=construction))
        assert mapped == [sorted((increasing(b), increasing(d)) for b, d in dim) for dim in expected]

    superlevel = diagrams(cubical(-image, construction=construction, filtration="superlevel"))
    assert superlevel == [sorted((-b, -d if d != INF else INF) for b, d in dim) for dim in expected]

    for method, algorithm in itertools.product(METHODS, algorithm_grid(Algorithm, row_ranks=False)):
        barcode = cubical(image, construction=construction, method=method, algorithm=algorithm)
        assert diagrams(barcode) == expected, f"{method} {algorithm!r}"


@given(image=images(special=False, max_axes=2, max_pixels=100))
def test_betti_numbers(image):
    """With every pixel finite the complex is a box: one essential class in dimension 0, none above."""
    for construction in ["T", "V"]:
        barcode = cubical(image, construction=construction)
        assert [len(e) for e in barcode.essential] == [1] + [0] * (image.ndim - 1)
        assert barcode.essential[0] == [float(np.min(image))]


def test_input_handling():
    image = np.arange(12.0).reshape(3, 4)
    expected = diagrams(cubical(image))
    assert diagrams(cubical(image.tolist())) == expected
    assert diagrams(cubical(image.astype(np.int32))) == expected
    assert diagrams(cubical(np.asfortranarray(image))) == expected
    assert diagrams(cubical(image[:, ::-1][:, ::-1])) == expected
    assert diagrams(cubical(np.zeros((0, 3)))) == [[], []]
    with pytest.raises(ValueError, match="axes"):
        cubical(np.zeros((2,) * 5))
    with pytest.raises(ValueError, match="construction"):
        cubical(image, construction="X")
    with pytest.raises(ValueError, match="filtration"):
        cubical(image, filtration="lower")
    assert Algorithm.serial().resolve("images.cubical").columns == "merged"
