# Choosing an algorithm

Every pipeline (`phlite.complexes.FilteredComplex.persistence`, `phlite.clouds.rips`, `phlite.networks.grpph`, ...)
takes an `algorithm`, a `phlite.core.Algorithm`:

```python
from phlite.core import Algorithm

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

| Option | What | Default |
|---|---|---|
| `with_columns` | the [column representation](../columns/overview.md): `'merged'`, `'heap'`, `'sorted'`, `'pivot'` | the pipeline's (below) |
| `with_cache` | [cache columns](../columns/caching.md) of the boundary | the pipeline's |
| `with_store_r` | [store R](../algorithms/store-r.md) (serial clearing only) | the pipeline's |
| `with_skip_zero_apparent_pairs` | [skip](../algorithms/apparent-pairs.md#zero-length-apparent-pairs) zero-length apparent pairs | `True` (not for standard) |
| `with_dimension_order` | reduce dimensions in this order (clearing) | detected |
| `with_storage` | `'locking'` or `'lock-free'` (parallel) | `'locking'` |
| `with_threads` | thread count (parallel) | all cores |

Options an algorithm can't use are errors, not ignored: `with_threads` on `serial()`, or `'pivot'` columns on the
parallel algorithm, as soon as you ask; `'pivot'` columns in a pipeline whose matrices have no row ranks (Vietoris–Rips,
GrPPH), when it runs.

## Each pipeline's defaults

The column representation, caching and storing R are best chosen per kind of input, so the presets leave them unset
(`None`), and each pipeline fills them in with its own defaults when it runs. So does `with_union_find`: whether, by
cohomology, to find dimension 0 by union-find (as Ripser and Cubical Ripser do) rather than by reducing it, which the
pipelines that can (Vietoris–Rips and cubical complexes, `—` below for the others, which refuse it) do by default:

<!-- defaults:start -->
| Pipeline | Function | Columns (serial) | Cache (serial) | Store R (serial) | Columns (parallel) | Cache (parallel) | Union-find |
|---|---|---|---|---|---|---|---|
| `complexes` | `phlite.complexes.FilteredComplex.persistence` | `'pivot'` | `False` | `True` | `'merged'` | `False` | — |
| `clouds.rips` | `phlite.clouds.rips` | `'heap'` | `False` | `False` | `'heap'` | `False` | `True` |
| `clouds.alpha` | `phlite.clouds.alpha` | `'pivot'` | `False` | `True` | `'merged'` | `False` | — |
| `clouds.delaunay_cech` | `phlite.clouds.delaunay_cech` | `'pivot'` | `False` | `True` | `'merged'` | `False` | — |
| `networks.grpph` | `phlite.networks.grpph` | `'merged'` | `True` | `False` | `'merged'` | `True` | — |
| `networks.pph` | `phlite.networks.pph` | `'merged'` | `True` | `False` | `'merged'` | `True` | — |
| `networks.directed_flag` | `phlite.networks.directed_flag` | `'merged'` | `True` | `False` | `'merged'` | `True` | — |
| `networks.ordered_tuples` | `phlite.networks.ordered_tuples` | `'merged'` | `True` | `False` | `'merged'` | `True` | — |
| `networks.dowker` | `phlite.networks.dowker` | `'merged'` | `True` | `False` | `'merged'` | `True` | — |
| `images.cubical` | `phlite.images.cubical` | `'merged'` | `False` | `True` | `'merged'` | `False` | `True` |
<!-- defaults:end -->

(`python-tests/tests/test_core.py` checks this table, and each function's docstring, against the pipelines
themselves; a new pipeline needs a row. Pipelines of packages that aren't installed are left out of `defaults()`.)
The same, from Python:

```python
>>> phlite.core.defaults()["clouds.rips"]
{'serial': Algorithm('clearing', columns='heap', cache=False, store_r=False, skip_zero_apparent_pairs=True, dimension_order=None),
 'parallel': Algorithm('parallel-clearing', columns='heap', cache=False, skip_zero_apparent_pairs=True, dimension_order=None, storage='locking', threads=None)}
```

**Every option is visible.** An `Algorithm` shows the options it leaves to the pipeline, `resolve` fills them in for
one, and every barcode records the algorithm that computed it:

```python
>>> Algorithm.serial()
Algorithm('clearing', columns=<pipeline default>, cache=<pipeline default>, store_r=<pipeline default>, skip_zero_apparent_pairs=True, dimension_order=None)
>>> Algorithm.serial().resolve("complexes")
Algorithm('clearing', columns='pivot', cache=False, store_r=True, skip_zero_apparent_pairs=True, dimension_order=None)
>>> phlite.clouds.rips(distances).algorithm
Algorithm('clearing', columns='heap', cache=False, store_r=False, skip_zero_apparent_pairs=True, dimension_order=None)
```

Each option is also a property (`.columns`, `.cache`, `.store_r`, ...), `None` while it is left to the pipeline.

Every algorithm gives the same barcode; only the time and memory differ. The defaults won their pipeline's
benchmarks; see [choosing a representation](../columns/choosing.md) for when to change them.
