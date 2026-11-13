# Choosing an algorithm

Each package has an `Algorithm` class (`phlite.Algorithm`, `phlite_rips.Algorithm`, `phlite_grpph.Algorithm`), built
from the same code, whose presets use the package's own defaults.

```python
from phlite import Algorithm

Algorithm.serial()                          # the default
Algorithm.parallel(threads=8)
Algorithm.clearing().with_columns("heap").with_cache(True)
```

Start from a preset or an algorithm, then adjust with `with_*` methods (each returns a new `Algorithm`):

| Start from | Is |
|---|---|
| `Algorithm.serial()` | `Algorithm.clearing()`: [clearing](../algorithms/clearing.md), skipping zero-length apparent pairs |
| `Algorithm.parallel(threads=None)` | `Algorithm.parallel_clearing().with_threads(threads)`: [parallel clearing](../algorithms/parallel.md) on all cores by default |
| `Algorithm.standard()` | the [standard algorithm](../algorithms/standard.md) |

| Option | What | `phlite` | `phlite_rips` | `phlite_grpph` |
|---|---|---|---|---|
| `with_columns` | the [column representation](../columns/overview.md): `'merged'`, `'heap'`, `'sorted'`, `'pivot'` | `'pivot'` (`'merged'` for standard and parallel) | `'heap'` | `'merged'` |
| `with_cache` | [cache columns](../columns/caching.md) of the boundary | `False` | `False` | `True` |
| `with_store_r` | [store R](../algorithms/store-r.md) (serial clearing only) | `True` | `False` | `False` |
| `with_skip_zero_apparent_pairs` | [skip](../algorithms/apparent-pairs.md#zero-length-apparent-pairs) zero-length apparent pairs | `True` (not for standard) | | |
| `with_dimension_order` | reduce dimensions in this order (clearing) | detected | | |
| `with_storage` | `'locking'` or `'lock-free'` (parallel) | `'locking'` | | |
| `with_threads` | thread count (parallel) | all cores | | |

Options an algorithm can't use are errors, not ignored: `with_threads` on `serial()`, or `'pivot'` columns on the
parallel algorithm or in `phlite_rips` and `phlite_grpph` (whose matrices have no row ranks).

**Every option is visible.** The `repr` lists every resolved option, as a chain that rebuilds it:

```python
>>> phlite.Algorithm.serial()
Algorithm.clearing().with_columns("pivot").with_cache(False).with_store_r(True).with_skip_zero_apparent_pairs(True)
>>> phlite_rips.Algorithm.parallel(4)
Algorithm.parallel_clearing().with_columns("heap").with_cache(False).with_skip_zero_apparent_pairs(True).with_storage("locking").with_threads(4)
```

and each option is a property (`.columns`, `.cache`, `.store_r`, ...).

Every algorithm gives the same barcode; only the time and memory differ. The defaults won their package's benchmarks;
see [choosing a representation](../columns/choosing.md) for when to change them.
