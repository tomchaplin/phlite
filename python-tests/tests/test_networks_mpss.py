"""Fuzz phlite.networks.magnitude_path_spectral_sequence: its first page is magnitude homology (phlite.networks.
magnitude), the diagonal of its second page is the path homology of the digraph (Asao), computed here as the
essential bars of phlite.networks.pph with every edge at weight 1, and pages only shrink."""

from hypothesis import given, note, settings
from hypothesis import strategies as st

from phlite.networks import magnitude, magnitude_path_spectral_sequence, pph


@st.composite
def digraphs(draw):
    n = draw(st.integers(1, 6))
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    edges = draw(st.lists(st.sampled_from(pairs), unique=True, max_size=len(pairs))) if pairs else []
    return n, edges


@given(digraphs(), st.integers(0, 4), st.sampled_from([2, 3]))
@settings(deadline=None)
def test_first_page_is_magnitude_homology(graph, l_max, p):
    n, edges = graph
    pages = magnitude_path_spectral_sequence(n, edges, l_max=l_max, r_max=1, modulus=p)
    assert pages[0] == magnitude(n, edges, l_max=l_max, modulus=p)


@given(digraphs(), st.integers(1, 4))
@settings(deadline=None)
def test_second_page_diagonal_is_path_homology(graph, max_dim):
    n, edges = graph
    l_max = max_dim + 1
    note(f"{n} {edges}")
    e2 = magnitude_path_spectral_sequence(n, edges, l_max=l_max, k_max=max_dim, r_max=2)[1]
    barcode = pph(n, [(a, b, 1.0) for a, b in edges], max_dim)
    for k in range(max_dim + 1):
        assert e2[k][k] == len(barcode.essential[k]), k


@given(digraphs(), st.integers(1, 5))
@settings(deadline=None)
def test_pages_shrink(graph, l_max):
    n, edges = graph
    pages = magnitude_path_spectral_sequence(n, edges, l_max=l_max, r_max=4)
    for before, after in zip(pages, pages[1:]):
        for row_before, row_after in zip(before, after):
            for b, a in zip(row_before, row_after):
                if a is not None:
                    assert b is not None and a <= b
