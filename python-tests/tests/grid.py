"""Every algorithm and option of a `phlite` package's `Algorithm` class, shared by the fuzz tests of all three packages."""

import itertools

COLUMNS = ["merged", "heap", "sorted", "pivot"]


def algorithm_grid(Algorithm, top=None, threads=2):
    """Every combination of algorithm, column representation, caching, skipping zero-length apparent pairs and (if
    `top`, the top dimension, is given) dimension order, plus storing R for serial clearing and the thread count and
    storage of parallel clearing."""
    bases = [Algorithm.standard(), Algorithm.clearing(), Algorithm.parallel_clearing()]
    orders = [None] if top is None else [None, list(range(top + 1)), list(range(top, -1, -1))]
    for base, columns, cache in itertools.product(bases, COLUMNS, [True, False]):
        algorithm = base.with_columns(columns).with_cache(cache)
        if base.name == "standard":
            yield algorithm
            continue
        for skip, order in itertools.product([True, False], orders):
            configured = algorithm.with_skip_zero_apparent_pairs(skip)
            if order is not None:
                configured = configured.with_dimension_order(order)
            yield configured
            if base.name == "clearing":
                yield configured.with_store_r()
            if base.name == "parallel-clearing":
                yield configured.with_threads(threads)
                yield configured.with_storage("lock-free").with_threads(threads)
