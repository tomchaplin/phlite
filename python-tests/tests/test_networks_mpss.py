"""Fuzz phlite.networks.magnitude_path_spectral_sequence: its first page is magnitude homology (phlite.networks.
magnitude), the diagonal of its second page is the path homology of the digraph (Asao), computed here as the
essential bars of phlite.networks.pph with every edge at weight 1, and pages only shrink."""

from hypothesis import given, note, settings
from hypothesis import strategies as st

from phlite.networks import magnitude, magnitude_path_spectral_sequence, pph


def pages(*args, r_max, **kwargs):
    """Pages E^1, ..., E^r_max as lists of rows (`[l][k]`)."""
    ss = magnitude_path_spectral_sequence(*args, **kwargs)
    return [ss.page(r).rows for r in range(1, r_max + 1)]


def as_magnitude(page):
    """An integral page in the form of `magnitude(..., integral=True)`."""
    free = [[None if e is None else e[0] for e in row] for row in page.rows]
    torsion = {}
    for l, row in enumerate(page.rows):
        for k, e in enumerate(row):
            if e is not None and e[1]:
                torsion[(k, l)] = {t: e[1].count(t) for t in e[1]}
    return {"free": free, "torsion": torsion}


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
    assert pages(n, edges, l_max=l_max, r_max=1, modulus=p)[0] == magnitude(n, edges, l_max=l_max, modulus=p)


@given(digraphs(), st.integers(1, 4))
@settings(deadline=None)
def test_second_page_diagonal_is_path_homology(graph, max_dim):
    n, edges = graph
    l_max = max_dim + 1
    note(f"{n} {edges}")
    e2 = pages(n, edges, l_max=l_max, k_max=max_dim, r_max=2)[1]
    barcode = pph(n, [(a, b, 1.0) for a, b in edges], max_dim)
    for k in range(max_dim + 1):
        assert e2[k][k] == len(barcode.essential[k]), k


@given(digraphs(), st.integers(1, 5))
@settings(deadline=None)
def test_pages_shrink(graph, l_max):
    n, edges = graph
    computed = pages(n, edges, l_max=l_max, r_max=4)
    for before, after in zip(computed, computed[1:]):
        for row_before, row_after in zip(before, after):
            for b, a in zip(row_before, row_after):
                if a is not None:
                    assert b is not None and a <= b


@st.composite
def small_digraphs(draw):
    n = draw(st.integers(1, 5))
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    edges = draw(st.lists(st.sampled_from(pairs), unique=True, max_size=min(len(pairs), 8))) if pairs else []
    return n, edges


@given(small_digraphs(), st.integers(0, 3))
@settings(deadline=None, max_examples=60)
def test_integral_pages(graph, l_max):
    """Over the integers: free ranks are the dimensions over Z/251 (where torsion of order 251 is out of reach), and
    the first page is integral magnitude homology."""
    n, edges = graph
    note(f"{n} {edges}")
    ss = magnitude_path_spectral_sequence(n, edges, l_max=l_max, integral=True)
    integral = [as_magnitude(ss.page(r)) for r in range(1, 4)]
    assert [page["free"] for page in integral] == pages(n, edges, l_max=l_max, r_max=3, modulus=251)
    assert integral[0] == magnitude(n, edges, l_max=l_max, integral=True)


@given(small_digraphs(), st.integers(1, 4), st.booleans())
@settings(deadline=None, max_examples=60)
def test_differentials_give_the_next_page(graph, l_max, integral):
    """rank E^{r+1} = rank E^r - rank(d^r out) - rank(d^r in), wherever all three are known."""
    n, edges = graph
    ss = magnitude_path_spectral_sequence(n, edges, l_max=l_max, integral=integral)
    rank = lambda e: e[0] if isinstance(e, tuple) else e  # noqa: E731
    for r in range(1, 4):
        page, following = ss.page(r), ss.page(r + 1)
        out = {d.source: rank(d.image) for d in page.differentials}
        into = {d.target: rank(d.image) for d in page.differentials}
        for l, row in enumerate(page.rows):
            for k, entry in enumerate(row):
                after = following.rows[l][k]
                # Differentials into (l, k) come from l + r, which must be known too
                if entry is None or after is None or l + 2 * r > l_max + 1:
                    continue
                assert rank(after) == rank(entry) - out.get((l, k), 0) - into.get((l, k), 0), (r, l, k)


def test_driver_and_display():
    """Directed 4-cycle (Hepworth and Roff): E^1 has rank 4 at (k, l) = (0, 0), (1, 1), then pairs along the cycle,
    and E^4 = E^infinity is a single generator at (0, 0)."""
    cycle = [(0, 1), (1, 2), (2, 3), (3, 0)]
    ss = magnitude_path_spectral_sequence(4, cycle, l_max=8, k_max=4)
    assert ss.current is None
    e1 = ss.next()
    assert ss.current is e1 and e1.r == 1 and e1[0, 0] == 4 and e1[1, 1] == 4
    assert ss.next().r == 2 and ss.page(2) is ss.current
    e4 = ss.page(4)
    assert e4[0, 0] == 1 and all(e in (0, None) for (l, k), e in e4.entries() if (l, k) != (0, 0))
    assert [p.r for p in ss][:4] == [1, 2, 3, 4]
    text = str(e1)
    assert text.startswith("E^1 of the MPSS") and "│ k=" in text and "│ l=0" in text
    assert "<table" in e1._repr_html_() and "<table" in ss._repr_html_()
    # Over the integers, the same (no torsion), and entries are (rank, torsion)
    zz = magnitude_path_spectral_sequence(4, cycle, l_max=4, integral=True)
    assert zz.page(1)[1, 1] == (4, []) and zz.integral and not ss.integral
    assert "?" in zz.page(4).table()


def test_plot():
    import pytest

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ss = magnitude_path_spectral_sequence(4, [(0, 1), (1, 2), (2, 3), (3, 0)], l_max=5)
    ax = ss.page(2).plot()
    assert ax.get_xlabel() == "k" and ax.get_ylabel() == "l"
