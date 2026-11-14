"""GrPPH written on phlite (examples/grpph.py), against grpphati's published results and hand computations."""

import pathlib
import sys

import pytest

nx = pytest.importorskip("networkx")
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))
from grpph import grpph  # noqa: E402


def digraph(edges, weight=1):
    G = nx.DiGraph()
    G.add_edges_from(edges, weight=weight)
    return G


@pytest.mark.parametrize("modulus", [2, 3])
def test_long_square(modulus):
    # grpphati's README: [[0, 3]]
    G = digraph([(0, 1), (0, 2), (1, 3), (2, 3)], weight=3)
    barcode = grpph(G, modulus=modulus, representatives=True)
    assert barcode.diagram(1).tolist() == [[0.0, 3.0]]
    (bar,) = [bar for bar in barcode.bars if bar.dimension == 1]
    assert bar.death == ("s", 0, 1, 2, 3)
    assert {cell for cell, _ in bar.representative} == {
        ("e", 0, 1),
        ("e", 1, 3),
        ("e", 0, 2),
        ("e", 2, 3),
    }


@pytest.mark.parametrize("modulus", [2, 3])
def test_directed_triangle(modulus):
    # The 1-cycle on the triangle is filled by the directed triangle once the 2-path 0 -> 1 -> 2 enters
    barcode = grpph(digraph([(0, 1), (1, 2), (0, 2)], weight=2), modulus=modulus)
    assert barcode.diagram(1).tolist() == [[0.0, 2.0]]


@pytest.mark.parametrize("modulus", [2, 3])
def test_double_edge(modulus):
    barcode = grpph(digraph([(0, 1), (1, 0)], weight=1.5), modulus=modulus)
    assert barcode.diagram(1).tolist() == [[0.0, 1.5]]


def test_tree_has_no_h1():
    assert grpph(digraph([(0, 1), (1, 2), (1, 3)])).diagram(1).shape == (0, 2)


def test_wedge_of_complete_digraphs():
    # grpphati's README: 4802 bars
    N = 50
    G_1 = nx.relabel_nodes(
        nx.complete_graph(N, create_using=nx.DiGraph), lambda x: (x, 1) if x > 0 else x
    )
    G_2 = nx.relabel_nodes(
        nx.complete_graph(N, create_using=nx.DiGraph), lambda x: (x, 2) if x > 0 else x
    )
    assert len(grpph(nx.compose(G_1, G_2)).diagram(1)) == 4802
