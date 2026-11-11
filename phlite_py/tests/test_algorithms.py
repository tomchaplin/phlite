"""Every algorithm and option gives the reference barcode, including skipping zero-length apparent pairs."""

import itertools
import random

import pytest

import phlite
from helpers import filtered
from reference import random_simplicial_complex, reduce

Algorithm = phlite.Algorithm


def case(seed):
    rng = random.Random(seed)
    # Small max_value means many ties, so many zero-length (apparent) pairs
    _, values, dimensions, columns, _ = random_simplicial_complex(
        rng, n_vertices=rng.randint(4, 9), n_top=rng.randint(1, 8), top_dim=3, max_value=2
    )
    pairs, essential = reduce(columns, [[1] * len(c) for c in columns], values, 2)
    pairs = [(b, d) for b, d in pairs if values[b] != values[d]]
    return filtered(columns, dimensions, values), max(dimensions), (sorted(pairs), sorted(essential))


def algorithms(top):
    bases = [Algorithm.standard(), Algorithm.clearing(), Algorithm.parallel_clearing()]
    for base, columns in itertools.product(bases, ["merged", "heap"]):
        algorithm = base.with_columns(columns)
        if base.name == "standard":
            yield algorithm
            continue
        for skip in [True, False]:
            for order in [None, list(range(top + 1)), list(range(top, -1, -1))]:
                configured = algorithm.with_skip_zero_apparent_pairs(skip).with_dimension_order(order)
                yield configured
                if base.name == "parallel-clearing":
                    yield configured.with_storage("lock-free").with_threads(2)


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("method", ["cohomology", "homology"])
def test_every_algorithm(seed, method):
    complex, top, expected = case(seed)
    for algorithm in algorithms(top):
        for reps in [False, True]:
            barcode = complex.persistence(method=method, algorithm=algorithm, representatives=reps)
            assert (barcode.pairings, barcode.essential) == expected, repr(algorithm)


def test_presets_and_names():
    complex, _, expected = case(0)
    for algorithm in ["standard", "clearing", "parallel", "parallel-lock-free", None, Algorithm.serial(), Algorithm.parallel(2)]:
        barcode = complex.persistence(algorithm=algorithm)
        assert (barcode.pairings, barcode.essential) == expected
    assert complex.persistence(algorithm="parallel", threads=2).pairings == expected[0]
    assert Algorithm.serial() == Algorithm.clearing() and Algorithm.serial().skip_zero_apparent_pairs
    assert Algorithm.parallel() == Algorithm.parallel_clearing() and Algorithm.parallel().storage == "locking"
    assert repr(Algorithm.parallel(4)) == "Algorithm.parallel_clearing().with_threads(4)"


def test_bad_options():
    complex, *_ = case(0)
    with pytest.raises(ValueError, match="can't be set for the standard"):
        Algorithm.standard().with_skip_zero_apparent_pairs()
    with pytest.raises(ValueError, match="can't be set for the clearing"):
        Algorithm.serial().with_threads(2)
    with pytest.raises(ValueError, match="can't be set for the clearing"):
        Algorithm.serial().with_storage("lock-free")
    with pytest.raises(ValueError, match="Unknown columns"):
        Algorithm.serial().with_columns("vec")
    with pytest.raises(ValueError, match="threads must be positive"):
        Algorithm.parallel(0)
    with pytest.raises(ValueError, match="threads can only be set for a parallel"):
        complex.persistence(algorithm=Algorithm.serial(), threads=2)
    with pytest.raises(TypeError, match="must be an Algorithm"):
        complex.persistence(algorithm=3)
