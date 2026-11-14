"""Every algorithm and option of `phlite.core.Algorithm`, shared by the fuzz tests of every pipeline."""

import itertools

COLUMNS = ["merged", "heap", "sorted", "pivot"]


def algorithm_grid(Algorithm, top=None, threads=2, row_ranks=True):
    """Every valid combination of algorithm, column representation, caching, skipping zero-length apparent pairs and (if
    `top`, the top dimension, is given) dimension order, plus storing R for serial clearing and the thread count and
    storage of parallel clearing. `row_ranks`: whether the pipeline's matrices know their row ranks, without which
    'pivot' columns are refused."""
    bases = [Algorithm.standard(), Algorithm.clearing(), Algorithm.parallel_clearing()]
    orders = [None] if top is None else [None, list(range(top + 1)), list(range(top, -1, -1))]
    for base, columns, cache in itertools.product(bases, COLUMNS, [True, False]):
        if columns == "pivot" and not row_ranks:
            continue
        try:
            algorithm = base.with_columns(columns).with_cache(cache)
        except ValueError:
            # 'pivot' columns: only for serial clearing
            continue
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
