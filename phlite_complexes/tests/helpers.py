"""Shorthands for building filtered complexes in tests."""

from phlite import complexes
def filtered(
    boundaries,
    dimensions,
    values=None,
    *,
    coefficients=None,
    modulus=2,
    extend=False,
):
    chain = complexes.ChainComplex.from_columns(
        boundaries, dimensions, coefficients=coefficients, modulus=modulus
    )
    return chain.filtration(values, extend=extend)


def filtered_pairs(pairs, values=None, *, coefficients=None, modulus=2):
    chain = complexes.ChainComplex.from_pairs(
        pairs, coefficients=coefficients, modulus=modulus
    )
    return chain.filtration(values)


def filtered_csc(indptr, indices, dimensions, values=None, *, data=None, modulus=2):
    return complexes.ChainComplex.from_csc(
        indptr, indices, dimensions, data=data, modulus=modulus
    ).filtration(values)


def cell_filtered(cells, *, values=None, extend=False, **kwargs):
    return complexes.ChainComplex.from_cells(cells, **kwargs).filtration(
        values, extend=extend
    )
