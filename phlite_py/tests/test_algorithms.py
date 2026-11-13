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
    for base, columns, cache in itertools.product(bases, ["merged", "heap", "sorted", "pivot"], [True, False]):
        if columns == "pivot" and base.name != "clearing":
            continue  # only serial clearing uses pivot columns (asking the others for them is an error)
        algorithm = base.with_columns(columns).with_cache(cache)
        if base.name == "standard":
            yield algorithm
            continue
        for skip in [True, False]:
            for order in [None, list(range(top + 1)), list(range(top, -1, -1))]:
                configured = algorithm.with_skip_zero_apparent_pairs(skip).with_dimension_order(order)
                yield configured
                if base.name == "clearing":
                    yield configured.with_store_r()
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


def test_presets():
    complex, _, expected = case(0)
    for algorithm in [None, Algorithm.serial(), Algorithm.parallel(), Algorithm.parallel(2)]:
        barcode = complex.persistence(algorithm=algorithm)
        assert (barcode.pairings, barcode.essential) == expected
    assert Algorithm.serial() == Algorithm.clearing() and Algorithm.serial().skip_zero_apparent_pairs
    assert Algorithm.parallel() == Algorithm.parallel_clearing() and Algorithm.parallel().storage == "locking"
    # The repr gives every option, resolved, so it means the same in every package
    assert repr(Algorithm.parallel(4)) == (
        'Algorithm.parallel_clearing().with_columns("merged").with_cache(False)'
        '.with_skip_zero_apparent_pairs(True).with_storage("locking").with_threads(4)'
    )
    assert repr(Algorithm.serial()) == (
        'Algorithm.clearing().with_columns("pivot").with_cache(False).with_store_r(True)'
        ".with_skip_zero_apparent_pairs(True)"
    )
    assert repr(Algorithm.standard()) == 'Algorithm.standard().with_columns("merged").with_cache(False)'
    for algorithm in [Algorithm.serial().with_dimension_order([2, 1, 0]), Algorithm.parallel(), Algorithm.standard()]:
        assert eval(repr(algorithm), {"Algorithm": Algorithm}) == algorithm
    # phlite's defaults: pivot columns, uncached, storing R (serial clearing only)
    serial = Algorithm.serial()
    assert (serial.columns, serial.cache, serial.store_r) == ("pivot", False, True)
    # Only serial clearing uses pivot columns, so the others' presets merge, and say so
    assert not Algorithm.parallel().store_r and Algorithm.parallel().columns == "merged"
    assert Algorithm.standard().columns == "merged"
    for algorithm in [Algorithm.parallel(), Algorithm.standard()]:
        with pytest.raises(ValueError, match="only used by serial clearing"):
            algorithm.with_columns("pivot")
    assert Algorithm.serial().with_cache().cache
    assert repr(Algorithm.serial().with_columns("sorted").with_cache()) == (
        'Algorithm.clearing().with_columns("sorted").with_cache(True).with_store_r(True)'
        ".with_skip_zero_apparent_pairs(True)"
    )


def test_bad_options():
    complex, *_ = case(0)
    with pytest.raises(ValueError, match="can't be set for the standard"):
        Algorithm.standard().with_skip_zero_apparent_pairs()
    with pytest.raises(ValueError, match="can't be set for the clearing"):
        Algorithm.serial().with_threads(2)
    with pytest.raises(ValueError, match="can't be set for the clearing"):
        Algorithm.serial().with_storage("lock-free")
    with pytest.raises(ValueError, match="store_r can't be set for the parallel-clearing"):
        Algorithm.parallel().with_store_r()
    with pytest.raises(ValueError, match="store_r can't be set for the standard"):
        Algorithm.standard().with_store_r()
    # phlite's serial preset stores R; the other algorithms can't
    assert Algorithm.serial().store_r and not Algorithm.parallel().store_r and not Algorithm.standard().store_r
    assert "with_store_r(False)" in repr(Algorithm.serial().with_store_r(False))
    with pytest.raises(ValueError, match="Unknown columns"):
        Algorithm.serial().with_columns("vec")
    with pytest.raises(ValueError, match="threads must be positive"):
        Algorithm.parallel(0)
    for not_an_algorithm in [3, "clearing"]:
        with pytest.raises(TypeError, match="Algorithm"):
            complex.persistence(algorithm=not_an_algorithm)
    with pytest.raises(TypeError, match="threads"):
        complex.persistence(threads=2)
