import dataclasses
import math
import random

import pytest

from helpers import cell_filtered, filtered
from reference import random_simplicial_complex, rp2


@dataclasses.dataclass(frozen=True)
class Simplex:
    """A simplex with the full protocol: dimension property, boundary() method, entrance_time."""

    vertices: tuple
    entrance_time: float = dataclasses.field(default=None, compare=False, hash=False)

    @property
    def dimension(self):
        return len(self.vertices) - 1

    def boundary(self):
        if len(self.vertices) == 1:
            return []
        # Faces don't know their entrance time; they only need to compare equal to the real cell
        return [
            Simplex(self.vertices[:i] + self.vertices[i + 1 :])
            for i in range(len(self.vertices))
        ]


class GrpphatiStyleColumn:
    """Mimics grpphati's Column: dimension() and boundary() are methods, entrance_time an attribute."""

    def __init__(self, vertices, entrance_time=None):
        self.vertices = tuple(vertices)
        self.entrance_time = entrance_time

    def __eq__(self, other):
        return (
            isinstance(other, GrpphatiStyleColumn) and self.vertices == other.vertices
        )

    def __hash__(self):
        return hash((GrpphatiStyleColumn, self.vertices))

    def __repr__(self):
        return f"Col {self.vertices} :: {self.entrance_time}"

    def dimension(self):
        return len(self.vertices) - 1

    def boundary(self):
        vs = self.vertices
        return (
            [GrpphatiStyleColumn(vs[:i] + vs[i + 1 :]) for i in range(len(vs))]
            if len(vs) > 1
            else []
        )


def random_case(seed):
    rng = random.Random(seed)
    simplices, values, dimensions, columns, _ = random_simplicial_complex(
        rng, n_vertices=rng.randint(4, 8), n_top=rng.randint(1, 6), top_dim=3
    )
    indexed = filtered(columns, dimensions, values)
    return simplices, values, indexed


def as_cells(barcode, simplices):
    """The index barcode, translated to vertex tuples."""
    return (
        [(simplices[b], simplices[d]) for b, d in barcode.pairings],
        [simplices[b] for b in barcode.essential],
    )


def vertex_tuples(cell_barcode, vertices=lambda c: c.vertices):
    return (
        [(vertices(b), vertices(d)) for b, d in cell_barcode.pairings],
        [vertices(b) for b in cell_barcode.essential],
    )


# ====== Agreement with FilteredComplex ====================================


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("method", ["cohomology", "homology"])
def test_protocol_cells(seed, method):
    simplices, values, indexed = random_case(seed)
    expected = as_cells(indexed.persistence(method=method), simplices)
    cells = [Simplex(s, v) for s, v in zip(simplices, values)]
    # Shuffle: cells may be given in any order, entrance times decide (ties broken by position)
    barcode = cell_filtered(cells).persistence(method=method)
    assert vertex_tuples(barcode) == expected


@pytest.mark.parametrize("seed", range(20))
def test_grpphati_style_columns(seed):
    simplices, values, indexed = random_case(seed)
    expected = as_cells(indexed.persistence(), simplices)
    cells = [GrpphatiStyleColumn(s, v) for s, v in zip(simplices, values)]
    assert vertex_tuples(cell_filtered(cells).persistence()) == expected


@pytest.mark.parametrize("seed", range(20))
def test_plain_tuples_with_functions(seed):
    simplices, values, indexed = random_case(seed)
    expected = as_cells(indexed.persistence(), simplices)
    value_of = dict(zip(simplices, values))

    def faces(s):
        return [s[:i] + s[i + 1 :] for i in range(len(s))] if len(s) > 1 else []

    for vals in [value_of, value_of.__getitem__, values]:
        complex = cell_filtered(
            simplices, boundary=faces, dimension=lambda s: len(s) - 1, values=vals
        )
        assert vertex_tuples(complex.persistence(), lambda c: c) == expected
    # A mapping from cells to boundaries
    complex = cell_filtered(
        {s: faces(s) for s in simplices},
        dimension=lambda s: len(s) - 1,
        values=value_of,
    )
    assert vertex_tuples(complex.persistence(), lambda c: c) == expected


@pytest.mark.parametrize("seed", range(20))
def test_cells_in_any_order(seed):
    simplices, _, _ = random_case(seed)
    # The index filtration (no ties, so no zero-length bars)
    index = {s: i for i, s in enumerate(simplices)}
    columns = [[index[f.vertices] for f in Simplex(s).boundary()] for s in simplices]
    indexed = filtered(columns, [len(s) - 1 for s in simplices])
    pairs, essential = as_cells(indexed.persistence(), simplices)
    # The same entrance times, but give the cells backwards (cofaces before faces)
    cells = [Simplex(s, float(i)) for i, s in enumerate(simplices)][::-1]
    got_pairs, got_essential = vertex_tuples(cell_filtered(cells).persistence())
    # Bars are listed in the order the cells were given, so compare as sets
    assert sorted(got_pairs) == sorted(pairs)
    assert sorted(got_essential) == sorted(essential)
    # Representatives are in decreasing filtration order, even though positions now run the other way
    for method in ["cohomology", "homology"]:
        for bar in (
            cell_filtered(cells).persistence(method=method, representatives=True).bars
        ):
            times = [cell.entrance_time for cell, _ in bar.representative]
            assert bar.representative[0][0] == bar.birth
            assert times == sorted(times, reverse=True)


# ====== key / from_key ====================================================


class Unhashable:
    __hash__ = None

    def __init__(self, vertices, value):
        self.vertices = tuple(vertices)
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Unhashable) and self.vertices == other.vertices

    @property
    def dimension(self):
        return len(self.vertices) - 1


@pytest.mark.parametrize("seed", range(15))
def test_key_and_from_key(seed):
    simplices, values, indexed = random_case(seed)
    expected = as_cells(indexed.persistence(), simplices)
    n = max(v for s in simplices for v in s) + 1

    def encode(vertices):
        # Base-n digits, offset so that different lengths can't collide
        k = 0
        for v in vertices:
            k = k * (n + 1) + v + 1
        return k

    def decode(k):
        vs = []
        while k:
            k, d = divmod(k, n + 1)
            vs.append(d - 1)
        return tuple(reversed(vs))

    by_vertices = {s: Unhashable(s, v) for s, v in zip(simplices, values)}
    cells = list(by_vertices.values())

    def boundary(c):
        vs = c.vertices
        return (
            [by_vertices[vs[:i] + vs[i + 1 :]] for i in range(len(vs))]
            if len(vs) > 1
            else []
        )

    key = lambda c: encode(c.vertices)  # noqa: E731
    with pytest.raises(TypeError, match="not hashable"):
        cell_filtered(cells, boundary=boundary, values=lambda c: c.value)
    complex = cell_filtered(cells, boundary=boundary, values=lambda c: c.value, key=key)
    assert vertex_tuples(complex.persistence()) == expected
    # Without the cells stored: results are rebuilt from keys
    # Values functions would be given the rebuilt cells (here tuples), so give values in cell order instead
    complex = cell_filtered(
        cells,
        boundary=boundary,
        values=[c.value for c in cells],
        key=key,
        from_key=decode,
    )
    assert vertex_tuples(complex.persistence(), lambda c: c) == expected


@pytest.mark.parametrize("seed", range(15))
def test_face_keys(seed):
    simplices, values, indexed = random_case(seed)
    expected = as_cells(indexed.persistence(), simplices)
    keys = {s: 1000 + i for i, s in enumerate(simplices)}
    complex = cell_filtered(
        [Simplex(s, v) for s, v in zip(simplices, values)],
        key=lambda c: keys[c.vertices],
        # Faces given by key, not as Simplex objects
        boundary=lambda c: [keys[f.vertices] for f in c.boundary()],
        face_keys=True,
    )
    assert vertex_tuples(complex.persistence()) == expected
    with pytest.raises(ValueError, match="not the key of a cell"):
        cell_filtered(
            [Simplex((0,), 0), Simplex((0, 1), 1)],
            key=lambda c: len(c.vertices),
            boundary=lambda c: [5] if c.dimension else [],
            face_keys=True,
        )
    with pytest.raises(ValueError, match="must contain integer keys"):
        cell_filtered(
            [Simplex((0,), 0), Simplex((0, 1), 1)],
            key=lambda c: len(c.vertices),
            face_keys=True,
        )
    with pytest.raises(ValueError, match="needs key"):
        cell_filtered([], face_keys=True)


def test_duplicate_keys():
    cells = [Simplex((0,), 0), Simplex((1,), 0)]
    with pytest.raises(ValueError, match="same key 7"):
        cell_filtered(cells, key=lambda c: 7)
    with pytest.raises(ValueError, match="from_key can only"):
        cell_filtered(cells, from_key=lambda k: k)


# ====== Representatives and coefficients ==================================


def oriented_boundary(s):
    if len(s) == 1:
        return {}
    return {s[:i] + s[i + 1 :]: (-1) ** i for i in range(len(s))}


@pytest.mark.parametrize("p", [2, 3, 5])
def test_rp2_with_coefficients(p):
    simplices, *_ = rp2()
    complex = cell_filtered(
        simplices, boundary=oriented_boundary, dimension=lambda s: len(s) - 1, modulus=p
    )
    barcode = complex.persistence(representatives=True)
    betti = [sum(len(b) - 1 == k for b in barcode.essential) for k in range(3)]
    assert betti == ([1, 1, 1] if p == 2 else [1, 0, 0])
    for bar in barcode.bars:
        rep = bar.representative
        assert rep[0][0] == bar.birth
        assert all(0 < c < p for _, c in rep)
        # A cycle: its boundary vanishes mod p
        total = {}
        for cell, c in rep:
            for face, d in oriented_boundary(cell).items():
                total[face] = (total.get(face, 0) + c * d) % p
        assert not any(total.values())
        assert barcode.representative(bar.birth) == rep
        assert barcode.bar(bar.birth).representative == rep


def test_bar_lookups():
    cells = [
        Simplex(s, t)
        for s, t in [
            ((0,), 0),
            ((1,), 0),
            ((2,), 1),
            ((0, 1), 2),
            ((1, 2), 3),
            ((0, 2), 4),
        ]
    ]
    complex = cell_filtered(cells)
    assert complex.chain.index_of(Simplex((1, 2))) == 4
    assert complex.chain.cell_at(4) is cells[4]
    assert Simplex((0, 2)) in complex.chain and Simplex((5,)) not in complex.chain
    barcode = complex.persistence()
    assert barcode.pairings == [(cells[1], cells[3]), (cells[2], cells[4])]
    assert barcode.essential == [cells[0], cells[5]]
    bar = barcode.bar(Simplex((0, 2)))
    assert (
        bar.dimension,
        bar.birth_value,
        bar.death_value,
        bar.death,
        bar.is_essential,
    ) == (1, 4.0, math.inf, None, True)
    assert barcode.representative(Simplex((0, 2))) is None
    assert barcode.diagram(1).tolist() == [[4.0, math.inf]]
    assert barcode.complex is complex
    with pytest.raises(ValueError, match="No bar is born"):
        barcode.bar(Simplex((0, 1)))
    with pytest.raises(ValueError, match="not a cell"):
        barcode.bar(Simplex((7,)))


def test_filtered_complex_bars():
    barcode = filtered([[], [], [0, 1]], [0, 0, 1]).persistence(representatives=True)
    bars = barcode.bars
    assert [(b.birth, b.death) for b in bars] == [(0, None), (1, 2)]
    assert bars[1].representative == [(1, 1), (0, 1)]


# ====== Errors name the cells ===============================================


def test_errors_name_cells():
    with pytest.raises(
        ValueError,
        match=r"boundary of Col \(0, 1\) :: 1 contains Col \(2,\) :: None, which is not a cell",
    ):
        cell_filtered(
            [
                GrpphatiStyleColumn((0,), 0),
                GrpphatiStyleColumn((1,), 0),
                _Bad((0, 1), 1),
            ]
        )
    with pytest.raises(ValueError, match="given more than once"):
        cell_filtered([Simplex((0,), 0), Simplex((0,), 1)])
    with pytest.raises(
        ValueError, match=r"Simplex\(vertices=\(0, 1\).*enter the filtration first"
    ):
        cell_filtered([Simplex((0,), 2), Simplex((1,), 0), Simplex((0, 1), 1)])
    with pytest.raises(ValueError, match="Could not get the dimension"):
        cell_filtered([(0,)], boundary=lambda c: [])
    # The first cell has an entrance time, so every cell needs one
    with pytest.raises(
        ValueError,
        match=r"The cell Simplex\(vertices=\(1,\).* has no filtration value \(pass extend=True",
    ):
        cell_filtered([Simplex((0,), 0), Simplex((1,))])
    with pytest.raises(ValueError, match="dimension one lower"):
        cell_filtered(
            [(0,), (0, 1)],
            boundary=lambda c: [(0,)] if len(c) == 2 else [],
            dimension=lambda c: 0,
        )


class _Bad(GrpphatiStyleColumn):
    def boundary(self):
        return [GrpphatiStyleColumn((2,))]


def test_real_grpphati():
    """grpphati's own columns, compared against grpphati's own pipeline."""
    nx = pytest.importorskip("networkx")
    pytest.importorskip("grpphati")
    from grpphati.filtrations import ShortestPathFiltration
    from grpphati.homologies import DirectedFlagComplexHomology, RegularPathHomology
    from grpphati.pipelines.grounded import make_grounded_pipeline

    for homology in [RegularPathHomology, DirectedFlagComplexHomology]:
        pipeline = make_grounded_pipeline(ShortestPathFiltration, homology)
        for seed in range(10):
            G = nx.gnp_random_graph(8, 0.35, seed=seed, directed=True)
            rng = random.Random(seed)
            for u, v in G.edges:
                G[u][v]["weight"] = rng.randint(1, 4)
            expected = sorted(map(tuple, pipeline(G).barcode))
            filtration = ShortestPathFiltration(G)
            cols = homology.get_cells(
                [0, 1], filtration.ground(G)
            ) + homology.get_cells([2], filtration)
            barcode = cell_filtered(cols).persistence(1)
            got = sorted(
                (b.birth_value, b.death_value) for b in barcode.bars if b.dimension == 1
            )
            assert got == expected
