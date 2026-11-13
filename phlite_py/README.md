# phlite (Python)

Python bindings for `phlite`: persistent homology of explicit, sparse, filtered chain complexes.

You describe a chain complex once — as integer columns (like [PHAT](https://github.com/blazs/phat) and
[LoPHAT](https://github.com/tomchaplin/lophat)), as sparse arrays, or with your own Python objects as cells — then attach
any number of filtrations to it and compute persistence. Results come back in terms of your cells, with representative
cycles and coefficients in Z/p if you like.

```python
import phlite

# A hollow tetrahedron, as (dimension, boundary) pairs
boundaries = [
    (0, []), (0, []), (0, []), (0, []),
    (1, [0, 1]), (1, [0, 2]), (1, [1, 2]), (1, [0, 3]), (1, [1, 3]), (1, [2, 3]),
    (2, [4, 7, 8]), (2, [5, 7, 9]), (2, [6, 8, 9]), (2, [4, 5, 6]),
]
chain = phlite.ChainComplex.from_pairs(boundaries)

barcode = chain.filtration().persistence()  # cells enter in index order, as in PHAT
barcode.pairings   # [(1, 4), (2, 5), (3, 7), (6, 12), (8, 10), (9, 11)]
barcode.essential  # [0, 13]

values = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
barcode = chain.filtration(values).persistence()
barcode.diagram(1)  # [[1, 3], [2, 3], [2, 3]]
```

## 1. Build a chain complex

A `ChainComplex` is the cells, their dimensions and their boundaries — no filtration. It is validated and indexed once.

**From integer columns** (cells are `0, ..., n - 1`):

```python
phlite.ChainComplex.from_pairs(pairs, *, coefficients=None, modulus=2)
phlite.ChainComplex.from_columns(boundaries, dimensions, *, coefficients=None, modulus=2)
```

`from_pairs` takes a list of `(dimension, boundary)` pairs, as PHAT and LoPHAT do; `from_columns` takes the
boundaries and the dimensions as two separate lists. Each boundary is a list of cell indices, and repeated entries are
summed. The two are about as fast as each other (around 0.1 µs per cell); for larger complexes, `from_csc` below
avoids reading Python lists at all.

**From CSC arrays** — the fast path when you already have an integer-indexed sparse matrix. numpy arrays are read
straight into Rust, with no per-entry Python objects:

```python
m = m.tocsc()  # e.g. a scipy.sparse boundary matrix
chain = phlite.ChainComplex.from_csc(m.indptr, m.indices, dimensions, data=m.data)
```

**From your own cell objects:**

```python
@dataclasses.dataclass(frozen=True)
class Simplex:
    vertices: tuple
    entrance_time: float = dataclasses.field(default=None, compare=False)  # optional

    @property
    def dimension(self):  # an attribute, property or method
        return len(self.vertices) - 1

    def boundary(self):   # faces, or a mapping {face: coefficient}
        vs = self.vertices
        return [Simplex(vs[:i] + vs[i + 1 :]) for i in range(len(vs))] if len(vs) > 1 else []

chain = phlite.ChainComplex.from_cells(simplices)
```

A cell needs to be hashable, with a dimension and a boundary. This is the same protocol as grpphati's `Column`, so
grpphati's columns work as they are. Faces only need to compare equal to the cell they mean, so `boundary()` can build
fresh objects. Plain types work too, by saying how to get the pieces:

```python
chain = phlite.ChainComplex.from_cells(
    simplices,                                  # e.g. tuples of vertices
    dimension=lambda s: len(s) - 1,
    boundary=lambda s: {s[:i] + s[i + 1 :]: (-1) ** i for i in range(len(s))} if len(s) > 1 else {},
    modulus=3,
)
```

Each cell's index (how `phlite` identifies it) is its position in the input; `chain.index_of(cell)` and
`chain.cell_at(i)` convert. Error messages name cells by their `repr`.

## 2. Attach a filtration

```python
filtered = chain.filtration(values=None, *, extend=False)
```

Cells enter in order of filtration value, breaking ties by index. Every face must enter before its cofaces, otherwise
you get a `ValueError` naming the cells. `values` can be

* a sequence or numpy array in cell order (the fastest);
* a mapping from cells to values;
* a function of a cell;
* omitted: `cell.entrance_time` if the cells have one, otherwise the cell order.

With `extend=True` values may be missing (absent from a mapping, `None`, or NaN in an array), and a missing value is the
largest value of the cell's faces. Values on vertices give the lower-star filtration:

```python
chain = phlite.ChainComplex.from_cells(simplices, dimension=lambda s: len(s) - 1, boundary=faces)
for t in times:
    barcode = chain.filtration({(v,): height(v, t) for v in vertices}, extend=True).persistence(1)
```

`filtered.values` gives the resulting value of every cell.

## 3. Compute persistence

```python
barcode = filtered.persistence(
    max_dim=None,             # default: the top dimension of the complex
    method="cohomology",      # or "homology"
    algorithm=None,           # an Algorithm, default Algorithm.serial(); see below
    representatives=False,
)
```

For example `filtered.persistence(algorithm=phlite.Algorithm.parallel(threads=8))`.

The `Barcode` is in terms of your cells (ints for `from_pairs`, `from_columns` and `from_csc`):

* `bars`: every bar, with `dimension`, `birth`, `death` (cells, `None` if essential), `birth_value`, `death_value`,
  `persistence`, `is_essential` and `representative`;
* `pairings`, `essential`: `(birth, death)` cells of finite bars, and births of infinite ones;
* `bar(birth)`, `representative(birth)`: look up the bar born at a cell;
* `diagram(dim)`, `diagrams()`: `(k, 2)` numpy arrays of `(birth, death)` values, `inf` for essential bars.

Representatives are lists of `(cell, coefficient)` pairs in decreasing filtration order, so the birth cell comes first.
As with Ripser, `method="cohomology"` is usually faster; it computes representatives by involution. It is the default
because it never lost by much in the benchmarks: homology was about 7% faster on a cubical complex of random noise, but
slower on the other cubical complexes and meshes, and about 3× slower on cell complexes. If your complexes are cubical,
time both.

By default the serial algorithm stores the reduced columns of R = DV, as PHAT does, and adds them directly when
cancelling pivots. (`phlite_rips` and `phlite_grpph` store only V and recompute columns of R, like Ripser, which is
better when almost every column is an apparent pair.) Reducing a prebuilt matrix on one thread, against PHAT's twist
reduction:

| input | random | structured (long bars) |
|---|---|---|
| Rips complexes, by cohomology | phlite 7–10× faster (random points in the plane) | phlite 9–13× faster (noisy torus, wedge of circles) |
| cubical complexes | PHAT 1.1–1.2× faster (smoothed noise) | PHAT 1.2–1.3× faster (an image of rings) |
| triangulated surfaces | | level, to PHAT 1.1× faster (a torus filtered by height) |
| sparse cell complexes | phlite 1.1–2× faster | |
| dense boundaries (cells with hundreds of faces) | level | |

See [`python-tests/BENCHMARKS.md`](../python-tests/BENCHMARKS.md) for these and comparisons with LoPHAT.

**Zero-length bars are omitted** (birth and death at the same filtration value), as everywhere in `phlite`. With the
default index filtration no bar has length zero, so you get every pair, as with PHAT.

### Choosing an algorithm

The `algorithm` argument takes an `Algorithm`. Start from a preset:

* `Algorithm.serial()` (the default): clearing, skipping zero-length apparent pairs;
* `Algorithm.parallel(threads=None)`: the same, with each dimension reduced in parallel on `threads` threads (default: all cores).

or from an algorithm itself, `Algorithm.standard()`, `Algorithm.clearing()` or `Algorithm.parallel_clearing()`,
then adjust it with the `with_*` methods, each of which returns a new `Algorithm`:

| Method | Values | Default |
|---|---|---|
| `with_columns` | How the column being reduced is stored: `'pivot'` (a dense array over every cell, with a tree of bitsets marking those present, as in PHAT: adding an entry flips a bit; serial clearing only, so the standard and parallel presets use `'merged'`, and asking them for `'pivot'` is an error), `'merged'` (sorted columns, merged lazily, so never looking below the pivot), `'heap'` (every entry pushed onto one heap) or `'sorted'` (one sorted vector: fast when the columns added stay near the pivot, slow when they reach far below it). | `'pivot'` (`'merged'` for standard and parallel) |
| `with_cache` | Whether to cache the columns of the boundary matrix that get added, rather than recomputing them each time. Columns are stored sorted, so this only saves a copy, which here costs more than it saves. | `False` |
| `with_store_r` | Keep each reduced column of R = D V and add it directly when cancelling a pivot (as PHAT does), rather than recompute it from V (serial clearing only). Pays off when columns need many additions, as in dense or cubical complexes, at the cost of memory. | `True` for serial clearing |
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
An `Algorithm` shows every option it resolves to, as a chain that rebuilds it: `repr(phlite.Algorithm.serial())` is
`Algorithm.clearing().with_columns("pivot").with_cache(False).with_store_r(True).with_skip_zero_apparent_pairs(True)`,
and each option is also a property (`.columns`, `.cache`, `.store_r`, ...). `serial()` is the same as `clearing()`, and
`parallel(threads)` as `parallel_clearing().with_threads(threads)`.
Options that don't apply to an algorithm (such as `with_threads` on `Algorithm.serial()`) raise a `ValueError`.

## Example: GrPPH

`examples/grpph.py` implements grounded persistent path homology of a weighted digraph, as in
[grpphati](https://github.com/tomchaplin/grpphati), in about 60 lines. Cells are tagged tuples (`("e", a, b)`,
`("t", a, m, b)`, ...), the 2-cells are grpphati's filtration-adapted basis of the regular path complex, and the
boundaries are signed so it also works over Z/p:

```python
time = grpph_cells(G)  # {cell: entrance time}
chain = phlite.ChainComplex.from_cells(time.keys(), dimension=lambda c: DIMENSION[c[0]], boundary=boundary)
barcode = chain.filtration(time).persistence(max_dim=1)
```

It reproduces grpphati's README results: `[[0, 3]]` for its long square, and 4802 bars for the wedge of two complete
digraphs on 50 nodes (960k cells: 0.9 s to list the cells in Python, 1.4 s for `from_cells` and `filtration`, and the
reduction takes a few milliseconds). Representatives come back as paths, e.g.
`[(("e", 2, 3), 1), (("e", 1, 3), 1), (("e", 0, 2), 1), (("e", 0, 1), 1)]`.

Here the 2-cell basis depends on the filtration, so a new chain complex is built per digraph. Reusing a chain
complex pays off when one complex has many filtrations, like the lower-star example above.

## Choosing cell identifiers yourself

By default cells are matched up through a Python dict, using your `__hash__` and `__eq__`. You can instead give
`key=`, an injective map from cells to integers (e.g. a combinatorial encoding of a simplex), matched in a Rust hash map.

* `key=` lets cells be unhashable, and avoids your `__hash__`/`__eq__`, which helps when those are expensive.
* `from_key=` (the inverse of `key`) means the chain complex stores only keys, not cells, and rebuilds cells for
  results. (Value functions and mappings passed to `filtration` then see the rebuilt cells.)
* `face_keys=True` means `boundary` returns the keys of the faces rather than face objects. This is the real
  speed-up, since building face objects is usually the slowest part.

`examples/cell_cost.py` builds the Vietoris-Rips complex of 300 random points (193k cells) from frozen dataclass
simplices, attaches their diameters as a numpy array, and computes H_0 and H_1:

| | build | filtration | memory kept |
|---|---|---|---|
| `from_columns` (indexing done by hand, in Python) | 152 ms | 24 ms | – |
| `from_cells`, hashing | 411 ms | 24 ms | 17.4 MB |
| `from_cells`, `key=` | 403 ms | 32 ms | 1.5 MB |
| `from_cells`, `key=`, `from_key=` | 424 ms | 23 ms | 0 MB |
| `from_cells`, `key=`, `face_keys=True` | 279 ms | 23 ms | 1.5 MB |

The reduction takes about 4 ms in every case. For simplices this cheap to hash, `key=` alone saves memory but not time:
most of the build is spent calling `boundary()`, which `face_keys=True` makes cheaper.

## How it works

`phlite` is built around lazy matrix oracles. Here the oracle is an explicit matrix, stored compactly:

* A `ChainComplex` stores the boundary matrix in CSC format with `u32` cell indices (and zero-sized Z/2 coefficients),
  validated and with repeated entries summed.
* `filtration` renumbers the cells by their position in the filtration, so that the reduction, which visits cells in
  that order, reads the matrices in order rather than all over memory, and compares positions rather than looking up
  filtration values (cells are numbered back in the results).
  It sorts every column in decreasing filtration order, so a column's pivot is its first entry and its
  smallest entry its last: `column_pivot` and `column_min` are O(1), and most columns need nothing more.
  It also builds the transpose (the coboundary), sorted the same way, so `first_column_in_row` is O(1) and the clearing
  algorithms recognise apparent pairs for free; the transpose also gives cohomology representatives by involution.
  That is about 8 bytes per non-zero entry and 36 bytes per cell for each filtered complex (which can have up to
  2³² − 2 non-zero entries).

So the same `phlite` reduction runs as for `phlite_rips` and `phlite_grpph`, on the matrix you supplied.

## Why not matrix oracles written in Python?

It is tempting to let users implement `MatrixOracle` in Python, keeping `phlite`'s laziness.
Unfortunately the per-column cost of Python dwarfs everything else:
every column access would need the GIL, a Python call and a Python object per entry
(and the GIL would serialise the parallel algorithms).
`phlite` *recomputes* columns on demand rather than storing them, so this cost is paid many times per column.

`examples/oracle_cost.py` measures this on a Vietoris-Rips complex (vertices, edges, triangles) of random points in
the unit cube, computing H_0 and H_1. Merely calling the cheapest possible Python oracle (looking a column up in a
list, without crossing into Rust) *once* per column of the boundary and coboundary takes longer than `phlite`'s entire
cohomology reduction:

| points | cells | `phlite` cohomology (build + reduce) | one pass of Python oracle calls |
|---|---|---|---|
| 120 | 13k | 5.2 ms + 0.8 ms | 3.1 ms |
| 200 | 59k | 12.9 ms + 1.5 ms | 16.7 ms |
| 300 | 193k | 58.5 ms + 5.7 ms | 64.0 ms |

If your complex is too big to write down, the better route is a small Rust crate implementing the oracle,
as `phlite_rips` and `phlite_grpph` do, with its own Python bindings.

## Development

Requires `just` and `uv`.

```sh
just develop-release     # build and install into .venv
just test                # pytest: against a brute-force reduction, LoPHAT and grpphati's published results
just stubs               # regenerate phlite.pyi
cargo test -p phlite_py  # the Rust side, without Python
```
