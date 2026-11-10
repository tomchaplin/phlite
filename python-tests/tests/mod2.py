"""Linear algebra over Z/2, with chains stored as sets of cells, for checking representatives."""

from collections import Counter


def boundary(cell):
    """The boundary of a simplex (a sorted tuple of vertices) over Z/2, as a set of faces."""
    if len(cell) <= 1:
        return set()
    return {cell[:i] + cell[i + 1 :] for i in range(len(cell))}


def chain_sum(chains):
    """Sum a collection of chains (sets of cells) over Z/2."""
    counts = Counter(cell for chain in chains for cell in chain)
    return {cell for cell, count in counts.items() if count % 2}


class Span:
    """The span of some chains over Z/2, for testing membership.

    Chains are encoded as integers, with bit i set when cell i (in order of first appearance) is present.
    """

    def __init__(self, chains=()):
        self._index = {}
        self._pivots = {}  # leading bit -> reduced vector with that leading bit
        for chain in chains:
            self.add(chain)

    def _encode(self, chain, grow):
        vector = 0
        for cell in chain:
            if cell not in self._index:
                if not grow:
                    # A cell we have never seen cannot be cancelled out
                    return None
                self._index[cell] = len(self._index)
            vector ^= 1 << self._index[cell]
        return vector

    def _reduce(self, vector):
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in self._pivots:
                return vector
            vector ^= self._pivots[pivot]
        return 0

    def add(self, chain):
        vector = self._reduce(self._encode(chain, grow=True))
        if vector:
            self._pivots[vector.bit_length() - 1] = vector

    def __contains__(self, chain):
        vector = self._encode(chain, grow=False)
        return vector is not None and self._reduce(vector) == 0
