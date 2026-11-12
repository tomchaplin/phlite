"""Exact linear algebra over Z/p, with sparse vectors as dicts {index: coefficient}, for checking barcodes."""


def _normalise(vector, p):
    return {i: c % p for i, c in vector.items() if c % p}


class Echelon:
    """A basis of the span of some vectors over Z/p, in echelon form, for ranks and membership tests."""

    def __init__(self, p, vectors=()):
        self.p = p
        self._pivots = {}  # leading index -> vector with that leading index, scaled so the leading coefficient is 1
        for vector in vectors:
            self.add(vector)

    def reduce(self, vector):
        """What is left of `vector` after cancelling every leading index it shares with the basis."""
        p = self.p
        vector = _normalise(vector, p)
        while vector:
            top = max(vector)
            basis = self._pivots.get(top)
            if basis is None:
                return vector
            factor = vector[top]
            for i, c in basis.items():
                vector[i] = (vector.get(i, 0) - factor * c) % p
                if vector[i] == 0:
                    del vector[i]
        return vector

    def add(self, vector):
        """Add `vector` to the span, returning whether that made it bigger."""
        vector = self.reduce(vector)
        if not vector:
            return False
        top = max(vector)
        inverse = pow(vector[top], self.p - 2, self.p)
        self._pivots[top] = {i: c * inverse % self.p for i, c in vector.items()}
        return True

    def __contains__(self, vector):
        return not self.reduce(vector)

    def __len__(self):
        return len(self._pivots)


def rank(vectors, p):
    return len(Echelon(p, vectors))


def kernel(cells, boundaries, p):
    """A basis of the cycles among `cells` (their boundaries cancel) over Z/p, as {cell: coefficient} chains."""
    # Gaussian elimination on the boundaries, tracking which combination of cells each reduced column is
    pivots = {}  # leading row -> (reduced boundary, combination), with leading coefficient 1
    basis = []
    for cell in cells:
        column, combination = _normalise(boundaries[cell], p), {cell: 1}
        while column and max(column) in pivots:
            top = max(column)
            other_column, other_combination = pivots[top]
            factor = column[top]
            for i, c in other_column.items():
                column[i] = (column.get(i, 0) - factor * c) % p
                if column[i] == 0:
                    del column[i]
            for i, c in other_combination.items():
                combination[i] = (combination.get(i, 0) - factor * c) % p
                if combination[i] == 0:
                    del combination[i]
        if column:
            inverse = pow(column[max(column)], p - 2, p)
            pivots[max(column)] = (
                {i: c * inverse % p for i, c in column.items()},
                {i: c * inverse % p for i, c in combination.items()},
            )
        else:
            basis.append(combination)
    return basis


def boundary_of(chain, boundaries, p):
    """The boundary of a chain {cell: coefficient}."""
    out = {}
    for cell, c in chain.items():
        for face, d in boundaries[cell].items():
            out[face] = (out.get(face, 0) + c * d) % p
    return _normalise(out, p)
