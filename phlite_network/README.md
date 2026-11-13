# phlite_network

Persistent homology of networks with [`phlite`](https://github.com/tomchaplin/phlite). So far one filtration:

* `phlite_network::grpph`: grounded persistent path homology (GrPPH) [[1]](#1) of weighted digraphs.

In Python, this crate is the `phlite.network` submodule of the `phlite` package (its bindings live in
`phlite_py/src/python/network`). It was called `phlite_grpph` (crate and Python package) until the packages merged;
`grpph` itself is unchanged.

## In Python

`phlite.network.grpph` accepts either a [networkx](https://networkx.org/) digraph:

```python
import networkx as nx
from phlite.network import grpph

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
| `algorithm` | A `phlite.Algorithm`, see [below](#choosing-an-algorithm). | `Algorithm.serial()` |
| `representatives` | Whether to compute a representative cycle for every bar. | `False` |

It returns a `phlite.ValueBarcode`, which also records the algorithm that computed it (`barcode.algorithm`).
The computation releases the GIL. For more examples see `phlite_py/examples/network_*.py`.

### Choosing an algorithm

The `algorithm` argument takes a `phlite.Algorithm`, shared by every pipeline of the `phlite` package. Start from a
preset:

* `Algorithm.serial()` (the default): clearing, skipping zero-length apparent pairs;
* `Algorithm.parallel(threads=None)`: the same, with each dimension reduced in parallel on `threads` threads (default: all cores).

or from an algorithm itself, `Algorithm.standard()`, `Algorithm.clearing()` or `Algorithm.parallel_clearing()`,
then adjust it with the `with_*` methods, each of which returns a new `Algorithm`:

| Method | Values | Default |
|---|---|---|
| `with_columns` | How the column being reduced is stored: `'merged'` (sorted columns, merged lazily, so never looking below the pivot), `'heap'` (every entry pushed onto one heap) or `'sorted'` (one sorted vector: fast when the columns added stay near the pivot, slow when they reach far below it). | `'merged'` for `grpph` (`'heap'` for `phlite.cloud.rips`; for explicit complexes, `'pivot'` for serial clearing, see [`phlite_py`'s README](../phlite_py/README.md#choosing-an-algorithm)) |
| `with_cache` | Whether to cache the columns of the boundary matrix that get added, rather than recomputing them each time. Pays off when columns are expensive to compute, but holds them in memory until their dimension is done. | `True` for `grpph` (`False` for `phlite.cloud.rips` and explicit complexes) |
| `with_store_r` | Keep each reduced column of R = D V and add it directly when cancelling a pivot (as PHAT does), rather than recompute it from V (serial clearing only). Pays off when columns need many additions, as in dense or cubical complexes, at the cost of memory. | `False` for `grpph` (`True` for serial clearing of explicit complexes) |
| `with_skip_zero_apparent_pairs` | Whether to leave zero-length apparent pairs out of the reduction's diagram (clearing only). They would be dropped from the barcode anyway, and recording them can dominate the time and memory. | `True` |
| `with_dimension_order` | The order to reduce dimensions in (clearing only), or `None` to detect the efficient order from the matrix. | `None` |
| `with_storage` | Concurrent data structures (parallel only): `'locking'` or `'lock-free'`. | `'locking'` |
| `with_threads` | Number of threads (parallel only), or `None` for all cores. | `None` |

```python
from phlite import Algorithm

Algorithm.parallel(threads=8)
Algorithm.clearing().with_columns("heap").with_skip_zero_apparent_pairs(False)
```

Every algorithm gives the same barcode; only the time and memory differ.
The last three options' defaults depend on the pipeline, so the presets leave them unset (`None`) and each pipeline
fills them in (here the pipeline `network.grpph`). An `Algorithm` shows every option, those left to the pipeline as
`<pipeline default>`; `Algorithm.serial().resolve("network.grpph")` fills them in, and `phlite.defaults()` lists
every pipeline's. `serial()` is the same as `clearing()`, and `parallel(threads)` as
`parallel_clearing().with_threads(threads)`. `'pivot'` columns are refused here and in `phlite.cloud.rips`, whose
matrices don't give the row ranks they need.
Options that don't apply to an algorithm (such as `with_threads` on `Algorithm.serial()`) raise a `ValueError`.

### Migrating from `phlite_grpph`

`phlite_grpph` 0.2 is now `phlite.network`: `from phlite_grpph import grpph` becomes
`from phlite.network import grpph`, and `phlite_grpph.Algorithm` is `phlite.Algorithm`. The barcode is a
`phlite.ValueBarcode`, with the same attributes as 0.2's `Barcode`.

From `phlite_grpph` 0.1, `grpph` returns a barcode object rather than a tuple, and `grpph_with_involution` has been
removed:

```python
# phlite_grpph 0.1
essential, pairings = grpph(n, edges)
essential, pairings, reps = grpph_with_involution(n, edges)
# now
barcode = grpph(n, edges, representatives=True)
essential, pairings = barcode.essential, barcode.pairings
reps = barcode.pairing_representatives[1]
```

Note that in 0.1 `reps` was in no particular order, whereas `pairing_representatives[1]` lines up with `pairings[1]`.
Representatives are now also available for essential bars and in dimension 0, and you can pass a networkx graph directly.
Self-loops, negative or `NaN` weights and nodes outside `0, ..., n - 1` now raise a `ValueError`.

## Development

The Rust side is `src/grpph` (with the `phlite_network_grpph` binary, which times a complete digraph); the Python
bindings are in `phlite_py/src/python/network/grpph.rs`, built as part of the `phlite` package (see
[`phlite_py`](../phlite_py/README.md)).

## References

<a id="1">[1]</a>
Chaplin, T., Harrington, H.A. and Tillmann, U., 2022.
Grounded persistent path homology: a stable, topological descriptor for weighted digraphs.
arXiv preprint [arXiv:2210.11274](https://arxiv.org/abs/2210.11274).
