"""One chain complex, many filtrations; and extending partial filtrations."""

import itertools
import math
import random

import numpy as np
import pytest

import phlite
from reference import random_simplicial_complex, reduce


def chain_for(seed):
    rng = random.Random(seed)
    simplices, values, dimensions, columns, coefficients = random_simplicial_complex(
        rng, n_vertices=rng.randint(4, 8), n_top=rng.randint(1, 6), top_dim=3
    )
    return (
        rng,
        simplices,
        phlite.ChainComplex.from_columns(columns, dimensions),
        columns,
    )


def bars(barcode):
    return barcode.pairings, barcode.essential


def expected(columns, values):
    pairs, essential = reduce(columns, [[1] * len(c) for c in columns], values, 2)
    return [(b, d) for b, d in pairs if values[b] != values[d]], essential


@pytest.mark.parametrize("seed", range(20))
def test_many_lower_star_filtrations(seed):
    rng, simplices, chain, columns = chain_for(seed)
    n_vertices = sum(len(s) == 1 for s in simplices)
    for _ in range(5):
        f = [rng.randint(0, 5) for _ in range(max(max(s) for s in simplices) + 1)]
        lower_star = [float(max(f[v] for v in s)) for s in simplices]
        # Values on vertices only, extended; as a mapping, a function, and a NaN-padded array
        vertex_values = {
            i: float(f[s[0]]) for i, s in enumerate(simplices) if len(s) == 1
        }
        assert len(vertex_values) == n_vertices
        padded = np.full(len(simplices), np.nan)
        padded[list(vertex_values)] = list(vertex_values.values())
        for values in [vertex_values, vertex_values.get, padded]:
            filtered = chain.filtration(values, extend=True)
            assert filtered.values.tolist() == lower_star
            assert bars(filtered.persistence()) == expected(columns, lower_star)
        # The same, given in full
        assert bars(chain.filtration(lower_star).persistence()) == expected(
            columns, lower_star
        )
        assert bars(chain.filtration(np.array(lower_star)).persistence()) == expected(
            columns, lower_star
        )


def test_default_is_cell_order():
    chain = phlite.ChainComplex.from_columns([[], [], [0, 1]], [0, 0, 1])
    filtered = chain.filtration()
    assert filtered.values.tolist() == [0.0, 1.0, 2.0]
    assert filtered.value(2) == 2.0
    assert filtered.chain is chain


def test_extend_keeps_given_values():
    chain = phlite.ChainComplex.from_columns([[], [], [0, 1], [0, 1]], [0, 0, 1, 1])
    filtered = chain.filtration({0: 1.0, 1: 2.0, 3: 7.0}, extend=True)
    assert filtered.values.tolist() == [1.0, 2.0, 2.0, 7.0]


def test_missing_values():
    chain = phlite.ChainComplex.from_columns([[], [], [0, 1]], [0, 0, 1])
    with pytest.raises(
        ValueError, match="The cell 2 has no filtration value \\(pass extend=True"
    ):
        chain.filtration({0: 0.0, 1: 0.0})
    with pytest.raises(
        ValueError, match="The cell 1 has no filtration value, and no faces"
    ):
        chain.filtration({0: 0.0}, extend=True)
    with pytest.raises(ValueError, match="needs some values"):
        chain.filtration(extend=True)
    with pytest.raises(ValueError, match="NaN"):
        chain.filtration([0.0, 1.0, math.nan])
    with pytest.raises(ValueError, match="values has length"):
        chain.filtration([0.0])
    with pytest.raises(
        ValueError, match="The boundary of cell 2 .* enter the filtration first"
    ):
        chain.filtration([0.0, 3.0, 1.0])


def test_cells_entrance_time_and_override():
    class Cell:
        def __init__(self, vs, t=None):
            self.vertices, self.entrance_time = vs, t

        def __eq__(self, other):
            return self.vertices == other.vertices

        def __hash__(self):
            return hash(self.vertices)

        def dimension(self):
            return len(self.vertices) - 1

        def boundary(self):
            vs = self.vertices
            return (
                [Cell(vs[:i] + vs[i + 1 :]) for i in range(len(vs))]
                if len(vs) > 1
                else []
            )

    cells = [Cell((0,), 0.0), Cell((1,), 1.0), Cell((0, 1), 2.0)]
    chain = phlite.ChainComplex.from_cells(cells)
    assert chain.filtration().values.tolist() == [0.0, 1.0, 2.0]
    # Override with a mapping keyed by cells, or a function of cells; faces built fresh work as keys
    assert chain.filtration(
        {Cell((0,)): 5.0, Cell((1,)): 0.0}, extend=True
    ).values.tolist() == [5.0, 0.0, 5.0]
    assert chain.filtration(lambda c: 10.0 * len(c.vertices)).values.tolist() == [
        10.0,
        10.0,
        20.0,
    ]
    barcode = chain.filtration(
        {Cell((0,)): 5.0, Cell((1,)): 0.0}, extend=True
    ).persistence()
    assert barcode.essential == [cells[1]]
    assert barcode.pairings == []  # vertex 0 and the edge both enter at 5


def test_reuse_across_dimensions_and_fields():
    simplices = [s for k in range(1, 4) for s in itertools.combinations(range(4), k)]
    chain = phlite.ChainComplex.from_cells(
        simplices,
        dimension=lambda s: len(s) - 1,
        boundary=lambda s: (
            {s[:i] + s[i + 1 :]: (-1) ** i for i in range(len(s))} if len(s) > 1 else {}
        ),
        modulus=3,
    )
    for height in range(4):
        # Sweep a vertex to the top: the 2-sphere is born when the last triangle enters
        f = {(v,): float(v == height) for v in range(4)}
        barcode = chain.filtration(f, extend=True).persistence(representatives=True)
        (sphere,) = [bar for bar in barcode.bars if bar.dimension == 2]
        assert sphere.is_essential and sphere.birth_value == 1.0
        assert {c for c, _ in sphere.representative} == {
            s for s in simplices if len(s) == 3
        }
