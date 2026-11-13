# phlite_grpph

This library uses [`phlite`](https://github.com/tomchaplin/phlite) to compute grounded persistent path homology (GrPPH) [[1]](#1).

## Usage

The Python module exposes a single function, `grpph`, which accepts either a [networkx](https://networkx.org/) digraph:

```python
import networkx as nx
from phlite_grpph import grpph

G = nx.DiGraph()
G.add_weighted_edges_from([("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0), ("d", "a", 1.0)])

barcode = grpph(G, representatives=True)

barcode.essential                 # essential[k]: births of the infinite bars in dimension k
barcode.pairings                  # pairings[k]: (birth, death) of the finite bars in dimension k
barcode.pairing_representatives   # pairing_representatives[k][i]: a cycle representing pairings[k][i]
barcode.essential_representatives # essential_representatives[k][i]: a cycle representing essential[k][i]
```

or the number of nodes (labelled `0, ..., n - 1`) and a list of `(source, target, weight)` edges:

```python
barcode = grpph(4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)], representatives=True)
```

With networkx, weights are read from the `weight` attribute (change this with `weight="cost"`), defaulting to 1 when missing,
and `MultiDiGraph`s are accepted (the lightest of parallel edges determines distances).
Undirected graphs are rejected: pass `G.to_directed()` to get an edge each way.

Representatives are lists of cells, each a tuple of nodes: `(a,)` for a node and `(a, b)` for an edge.
With a networkx graph these are your node labels, otherwise node indices.
They are `None` unless you pass `representatives=True`.
Bars of length zero are omitted, and within each dimension bars are sorted by birth then death.

Edge weights must be non-negative; an infinite weight means the edge is only used for grounding.
Self-loops are not supported.

The keyword-only arguments choose how the barcode is computed:

| Argument | Values | Default |
|---|---|---|
| `method` | `'cohomology'`: reduce the coboundary matrix, computing representatives by involution. `'homology'`: reduce the boundary matrix (needs every 2-cell up front, so uses far more memory). | `'cohomology'` |
| `algorithm` | An `Algorithm`, see [below](#choosing-an-algorithm). | `Algorithm.serial()` |
| `representatives` | Whether to compute a representative cycle for every bar. | `False` |

The computation releases the GIL. For more examples see the `examples` folder.

### Choosing an algorithm

The `algorithm` argument takes an `Algorithm`. Start from a preset:

* `Algorithm.serial()` (the default): clearing, skipping zero-length apparent pairs;
* `Algorithm.parallel(threads=None)`: the same, with each dimension reduced in parallel on `threads` threads (default: all cores).

or from an algorithm itself, `Algorithm.standard()`, `Algorithm.clearing()` or `Algorithm.parallel_clearing()`,
then adjust it with the `with_*` methods, each of which returns a new `Algorithm`:

| Method | Values | Default |
|---|---|---|
| `with_columns` | How the column being reduced is stored: `'merged'` (sorted columns, merged lazily, so never looking below the pivot), `'heap'` (every entry pushed onto one heap) or `'sorted'` (one sorted vector: fast when the columns added stay near the pivot, slow when they reach far below it). | `'merged'` (`'heap'` in `phlite_rips`; in `phlite`, `'pivot'` for serial clearing, see [its README](../phlite_py/README.md#choosing-an-algorithm)) |
| `with_cache` | Whether to cache the columns of the boundary matrix that get added, rather than recomputing them each time. Pays off when columns are expensive to compute, but holds them in memory until their dimension is done. | `True` (`False` in `phlite_rips` and `phlite`) |
| `with_store_r` | Keep each reduced column of R = D V and add it directly when cancelling a pivot (as PHAT does), rather than recompute it from V (serial clearing only). Pays off when columns need many additions, as in dense or cubical complexes, at the cost of memory. | `False` (`True` for serial clearing in `phlite`) |
| `with_skip_zero_apparent_pairs` | Whether to leave zero-length apparent pairs out of the reduction's diagram (clearing only). They would be dropped from the barcode anyway, and recording them can dominate the time and memory. | `True` |
| `with_dimension_order` | The order to reduce dimensions in (clearing only), or `None` to detect the efficient order from the matrix. | `None` |
| `with_storage` | Concurrent data structures (parallel only): `'locking'` or `'lock-free'`. | `'locking'` |
| `with_threads` | Number of threads (parallel only), or `None` for all cores. | `None` |

```python
from phlite_grpph import Algorithm

Algorithm.parallel(threads=8)
Algorithm.clearing().with_columns("heap").with_skip_zero_apparent_pairs(False)
```

Every algorithm gives the same barcode; only the time and memory differ.
An `Algorithm` shows every option it resolves to, as a chain that rebuilds it: `repr(phlite_grpph.Algorithm.serial())` is
`Algorithm.clearing().with_columns("merged").with_cache(True).with_store_r(False).with_skip_zero_apparent_pairs(True)`,
and each option is also a property (`.columns`, `.cache`, `.store_r`, ...). `serial()` is the same as `clearing()`, and
`parallel(threads)` as `parallel_clearing().with_threads(threads)`. `'pivot'` columns (see `phlite`) are refused here and in `phlite_rips`, whose matrices don't give the row ranks they need.
Options that don't apply to an algorithm (such as `with_threads` on `Algorithm.serial()`) raise a `ValueError`.

### Migrating from 0.1

`grpph` now returns a `Barcode` object rather than a tuple, and `grpph_with_involution` has been removed:

```python
# 0.1
essential, pairings = grpph(n, edges)
essential, pairings, reps = grpph_with_involution(n, edges)
# 0.2
barcode = grpph(n, edges, representatives=True)
essential, pairings = barcode.essential, barcode.pairings
reps = barcode.pairing_representatives[1]
```

Note that in 0.1 `reps` was in no particular order, whereas `pairing_representatives[1]` lines up with `pairings[1]`.
Representatives are now also available for essential bars and in dimension 0, and you can pass a networkx graph directly.
Self-loops, negative or `NaN` weights and nodes outside `0, ..., n - 1` now raise a `ValueError`.

## Development

Building requires [`just`](https://github.com/casey/just) and [`uv`](https://github.com/astral-sh/uv).
Run `just` to list all recipes; the main ones are:

```sh
just develop          # create .venv and install a debug build of the extension
just develop-release  # same, but an optimised build
just wheel            # build a release wheel into dist/
just stubs            # regenerate phlite_grpph.pyi
just examples         # run everything in examples/
```

Then use `uv run --no-sync python ...` (or activate `.venv`) to run Python against the installed build.
Pass a different interpreter with e.g. `just python=3.12 develop`.

## References

<a id="1">[1]</a>
Chaplin, T., Harrington, H.A. and Tillmann, U., 2022.
Grounded persistent path homology: a stable, topological descriptor for weighted digraphs.
arXiv preprint [arXiv:2210.11274](https://arxiv.org/abs/2210.11274).
