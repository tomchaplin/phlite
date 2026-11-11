"""Shorthands for building filtered complexes in tests."""

import phlite


def filtered(
    boundaries,
    dimensions=None,
    values=None,
    *,
    coefficients=None,
    modulus=2,
    extend=False,
):
    chain = phlite.ChainComplex.from_columns(
        boundaries, dimensions, coefficients=coefficients, modulus=modulus
    )
    return chain.filtration(values, extend=extend)


def filtered_csc(indptr, indices, dimensions, values=None, *, data=None, modulus=2):
    return phlite.ChainComplex.from_csc(
        indptr, indices, dimensions, data=data, modulus=modulus
    ).filtration(values)


def cell_filtered(cells, *, values=None, extend=False, **kwargs):
    return phlite.ChainComplex.from_cells(cells, **kwargs).filtration(
        values, extend=extend
    )
