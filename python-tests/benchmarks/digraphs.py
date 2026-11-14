"""Random weighted digraphs for the network benchmarks."""

import functools

import numpy as np


@functools.cache
def random_digraph(n, p, seed=0):
    """`n` nodes, each edge `a -> b` (`a != b`) present with probability `p`, weights uniform in [0, 1] rounded to
    float32 (so every implementation reads the same values)."""
    rng = np.random.default_rng(seed)
    present = rng.random((n, n)) < p
    np.fill_diagonal(present, False)
    weights = rng.random((n, n)).astype(np.float32).astype(np.float64)
    sources, targets = np.nonzero(present)
    return [(int(a), int(b), float(weights[a, b])) for a, b in zip(sources, targets)]


def adjacency(n, edges):
    """The dense adjacency matrix flagser reads: weights, infinity for no edge, 0 on the diagonal."""
    matrix = np.full((n, n), np.inf)
    np.fill_diagonal(matrix, 0.0)
    for a, b, w in edges:
        matrix[a, b] = w
    return matrix
