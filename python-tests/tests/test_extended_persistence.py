"""Fuzz phlite.complexes' FilteredComplex.extended_persistence: the lower-star filtration of vertex values (on graphs,
their flag complexes, and random 2-complexes) against GUDHI's SimplexTree.extend_filtration and extended_persistence;
any filtration against the cone complex built here and reduced by ordinary persistence; and every algorithm.

(GUDHI isn't in the `latest` environment, so this file runs in the main one, despite testing phlite.complexes.)

GUDHI rescales the values into [-2, -1] going up and [1, 2] going down, so its values are within rounding of the
exact ones, and a pair whose two values are equal can come back a rounding error apart (which GUDHI keeps, and phlite
drops as a bar of length zero): the comparison allows for both."""

import itertools
import math

import gudhi
import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from grid import algorithm_grid
from phlite.complexes import ChainComplex
from phlite.core import Algorithm

KINDS = ["ordinary", "relative", "extended_plus", "extended_minus"]


@st.composite
def complexes(draw, max_vertices=8):
    """Simplicial complexes `(n, simplices, values)`: a random graph, perhaps expanded to its flag complex up to
    triangles, or random triangles with their faces; vertex values small integers (ties) or floats."""
    n = draw(st.integers(1, max_vertices))
    kind = draw(st.sampled_from(["graph", "flag", "triangles"]))
    simplices = {(v,) for v in range(n)}
    if kind == "triangles":
        for t in itertools.combinations(range(n), 3):
            if draw(st.floats(0, 1)) < 0.3:
                simplices |= {t, t[:2], t[1:], (t[0], t[2])}
        for e in itertools.combinations(range(n), 2):
            if draw(st.floats(0, 1)) < 0.2:
                simplices.add(e)
    else:
        density = draw(st.sampled_from([0.3, 0.6, 0.95]))
        edges = [e for e in itertools.combinations(range(n), 2) if draw(st.floats(0, 1)) < density]
        simplices |= set(edges)
        if kind == "flag":
            simplices |= {t for t in itertools.combinations(range(n), 3) if all(f in simplices for f in itertools.combinations(t, 2))}
    if draw(st.booleans()):
        values = [float(draw(st.integers(-3, 3))) for _ in range(n)]
    else:
        values = [draw(st.floats(-10, 10, allow_nan=False, width=32)) for _ in range(n)]
    return n, sorted(simplices, key=lambda s: (len(s), s)), values


def simplicial(simplices, modulus=2):
    """The chain complex of `simplices` (tuples of vertices, faces before cofaces), with signed boundaries."""
    return ChainComplex.from_cells(
        simplices,
        dimension=lambda s: len(s) - 1,
        boundary=lambda s: {s[:i] + s[i + 1 :]: (-1) ** i for i in range(len(s))} if len(s) > 1 else {},
        modulus=modulus,
    )


def lower_star(simplices, values, modulus=2):
    """The chain complex of `simplices`, filtered by the lower-star filtration of the vertex values."""
    chain = simplicial(simplices, modulus)
    return chain.filtration(lambda s: values[s[0]] if len(s) == 1 else None, extend=True)


def gudhi_extended(simplices, values, modulus):
    tree = gudhi.SimplexTree()
    for s in simplices:
        tree.insert(list(s), filtration=max(values[v] for v in s))
    tree.extend_filtration()
    diagrams = tree.extended_persistence(homology_coeff_field=modulus, min_persistence=-1)
    scale = max([abs(x) for x in values] + [1.0])
    top = max(len(s) for s in simplices)
    out = []
    for diagram in diagrams:
        by_dim = [[] for _ in range(top + 1)]
        for d, (b, e) in diagram:
            if abs(b - e) > 1e-9 * scale:
                by_dim[d].append((b, e))
        out.append(by_dim)
    return out, scale


def close(ours, theirs, scale):
    """Equal within rounding, after sorting each by values rounded to it (values a rounding error apart, like 0 and
    1e-38, which GUDHI can't tell apart, could otherwise sort differently)."""
    # (Ties broken by the exact values, so that both sides sort the same way)
    key = lambda bar: (tuple(round(x / (1e-7 * scale)) for x in bar), bar)
    ours, theirs = sorted(ours, key=key), sorted(theirs, key=key)
    return len(ours) == len(theirs) and all(
        math.isclose(a, c, abs_tol=1e-9 * scale) and math.isclose(b, d, abs_tol=1e-9 * scale)
        for (a, b), (c, d) in zip(ours, theirs)
    )


def check(barcode, expected, scale):
    """Ours against GUDHI's, both without pairs shorter than rounding."""
    for kind, theirs in zip(KINDS, expected):
        ours = getattr(barcode, kind)
        for dim in range(max(len(ours), len(theirs))):
            o = [(b, d) for b, d in (ours[dim] if dim < len(ours) else []) if abs(b - d) > 1e-9 * scale]
            t = theirs[dim] if dim < len(theirs) else []
            assert close(o, t, scale), (kind, dim, o, t)


@given(complex_=complexes(), modulus=st.sampled_from([2, 3]))
def test_lower_star_matches_gudhi(complex_, modulus):
    n, simplices, values = complex_
    note(f"simplices={simplices} values={values}")
    expected, scale = gudhi_extended(simplices, values, modulus)
    for method in ["cohomology", "homology"]:
        check(lower_star(simplices, values, modulus).extended_persistence(method=method), expected, scale)


@given(complex_=complexes(max_vertices=6))
def test_every_algorithm_agrees(complex_):
    n, simplices, values = complex_
    expected, scale = gudhi_extended(simplices, values, 2)
    for algorithm in algorithm_grid(Algorithm):
        check(lower_star(simplices, values).extended_persistence(algorithm=algorithm), expected, scale)


def cone_by_hand(simplices, f):
    """Extended persistence of the filtration `f` (a value per simplex) from its definition: the cone vertex, the
    complex by f, then the cones by decreasing g (the least f of a simplex and its faces), reduced as an ordinary
    filtered complex; bars classified by where their cells are."""
    index = {s: i for i, s in enumerate(simplices)}
    g = {}
    for s in simplices:
        g[s] = min([f[index[s]]] + [g[t] for t in itertools.combinations(s, len(s) - 1) if len(s) > 1])
    up = sorted(simplices, key=lambda s: (f[index[s]], len(s), index[s]))
    down = sorted(simplices, key=lambda s: (-g[s], len(s), index[s]))
    cells = [("omega",)] + [("up",) + s for s in up] + [("down",) + s for s in down]
    position = {c: i for i, c in enumerate(cells)}
    columns, dims = [], []
    for c in cells:
        part, s = c[0], c[1:]
        faces = [s[:i] + s[i + 1 :] for i in range(len(s))] if len(s) > 1 else []
        if part == "omega":
            columns.append([])
            dims.append(0)
        elif part == "up":
            columns.append([position[("up",) + t] for t in faces])
            dims.append(len(s) - 1)
        else:
            columns.append([position[("up",) + s]] + ([position[("omega",)]] if len(s) == 1 else [position[("down",) + t] for t in faces]))
            dims.append(len(s))
    chain = ChainComplex.from_columns(columns, dims)
    barcode = chain.filtration().persistence()
    value = lambda i: f[index[cells[i][1:]]] if cells[i][0] == "up" else g[cells[i][1:]]
    out = {kind: [] for kind in KINDS}
    for bar in barcode.bars:
        if bar.death is None:
            continue
        b, d = bar.birth, bar.death
        bv, dv = value(b), value(d)
        if bv == dv:
            continue
        if cells[b][0] == "up" and cells[d][0] == "up":
            out["ordinary"].append((dims[b], bv, dv))
        elif cells[b][0] == "down":
            out["relative"].append((dims[b], bv, dv))
        else:
            out["extended_plus" if bv < dv else "extended_minus"].append((dims[b], bv, dv))
    return {kind: sorted(bars) for kind, bars in out.items()}


@given(complex_=complexes(max_vertices=6), data=st.data())
def test_any_filtration_matches_the_definition(complex_, data):
    """A filtration that isn't lower-star (each simplex no earlier than its faces, but otherwise anything), against
    the cone built here."""
    n, simplices, _ = complex_
    f = []
    for s in simplices:
        least = max([f[simplices.index(t)] for t in itertools.combinations(s, len(s) - 1)] if len(s) > 1 else [-5.0])
        f.append(least + float(data.draw(st.integers(0, 3))))
    barcode = simplicial(simplices).filtration(f).extended_persistence()
    expected = cone_by_hand(simplices, f)
    got = {kind: sorted(item for item in barcode.bars if item[0] == name) for kind, name in zip(KINDS, ["ordinary", "relative", "extended+", "extended-"])}
    for kind in KINDS:
        assert sorted((dim, b, d) for _, dim, b, d, _, _ in got[kind]) == expected[kind], kind


def test_a_cycle_and_the_gudhi_format():
    simplices = [(0,), (1,), (2,), (3,), (0, 1), (1, 2), (2, 3), (0, 3)]
    barcode = lower_star(simplices, [0.0, 2.0, 1.0, 3.0]).extended_persistence()
    assert barcode.ordinary[0] == [(1.0, 2.0)]
    assert barcode.relative[1] == [(2.0, 1.0)]
    assert barcode.extended_plus[0] == [(0.0, 3.0)]
    assert barcode.extended_minus[1] == [(3.0, 0.0)]
    assert barcode.to_gudhi() == [[(0, (1.0, 2.0))], [(1, (2.0, 1.0))], [(0, (0.0, 3.0))], [(1, (3.0, 0.0))]]
    assert len(barcode) == 4 and "extended_minus=[H1: 1]" in repr(barcode)
    # Each bar's cells, as the complex's own: the loop is born with an edge at its highest point (3: (2, 3) or
    # (0, 3)), and dies with the cone on a cell at its lowest (0: vertex 0, or an edge containing it)
    (kind, dim, birth, death, birth_cell, death_cell), = [b for b in barcode.bars if b[0] == "extended-"]
    assert birth_cell in [(2, 3), (0, 3)] and 0 in death_cell
    # It is phlite.core's, with the algorithm that computed it
    from phlite.core import ExtendedBarcode
    assert isinstance(barcode, ExtendedBarcode) and barcode.algorithm.store_r is True
