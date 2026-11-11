# phlite (Python)

Python bindings for `phlite`: persistent homology of an explicit, sparse, filtered chain complex.

The interface is close to [PHAT](https://github.com/blazs/phat) and [LoPHAT](https://github.com/tomchaplin/lophat):
you give the boundary matrix (and the dimension of every cell), and get back pairs of cell indices.
On top of that you can give real-valued filtration values, coefficients in Z/p, and ask for representative cycles.

```python
import phlite

# A hollow tetrahedron, as (dimension, boundary) pairs; cells enter in index order
boundaries = [
    (0, []), (0, []), (0, []), (0, []),
    (1, [0, 1]), (1, [0, 2]), (1, [1, 2]), (1, [0, 3]), (1, [1, 3]), (1, [2, 3]),
    (2, [4, 7, 8]), (2, [5, 7, 9]), (2, [6, 8, 9]), (2, [4, 5, 6]),
]
complex = phlite.FilteredComplex(boundaries)
barcode = complex.persistence(representatives=True)

barcode.pairings         # [(1, 4), (2, 5), (3, 7), (6, 12), (8, 10), (9, 11)]
barcode.essential        # [0, 13]
barcode.representatives  # {0: [0], 1: [1, 0], ..., 13: [13, 12, 11, 10]}
```

## Building a complex

Cells are numbered `0, ..., n - 1`.

```python
phlite.FilteredComplex(boundaries, dimensions=None, values=None, *, coefficients=None, modulus=2)
```

* `boundaries`: the boundary of every cell, as a list of columns (lists or arrays of cell indices),
  or as `(dimension, column)` pairs. Repeated entries are summed. Any iterable works.
* `dimensions`: the dimension of every cell. Required unless you gave `(dimension, column)` pairs.
  (We don't guess: a cell with zero boundary could have any dimension.)
* `values`: the filtration value of every cell. Default: the cell's index, as in PHAT.
  Cells enter in order of value, breaking ties by index. `inf` means the cell never appears.
* `coefficients`: the coefficient of every boundary entry, matching `boundaries` (default all 1).
* `modulus`: the coefficient field, Z/p for p in 2, 3, 5, 7, 11, 13.

For large complexes, build from CSC arrays, which reads numpy arrays without going through Python objects:

```python
phlite.FilteredComplex.from_csc(indptr, indices, dimensions, values=None, *, data=None, modulus=2)

m = m.tocsc()  # a scipy.sparse boundary matrix
phlite.FilteredComplex.from_csc(m.indptr, m.indices, dimensions, values, data=m.data)
```

The complex is validated on construction: every face must have dimension one lower than its coface and enter the
filtration before it, otherwise you get a `ValueError` naming the offending cells.
We do *not* check that the boundary squares to zero.

## Computing persistence

```python
barcode = complex.persistence(
    max_dim=None,             # default: the top dimension of the complex
    method="cohomology",      # or "homology"
    algorithm="clearing",     # or "standard", "parallel", "parallel-lock-free"
    representatives=False,
    threads=None,             # for the parallel algorithms
)
```

`FilteredComplex` is immutable and stores everything it needs, so computing several barcodes from one complex is cheap.

The `Barcode` has:

* `pairings`: `(birth, death)` cell indices of the finite bars, sorted by birth;
* `essential`: birth cells of the infinite bars;
* `representatives`: `{birth: cycle}` (if requested). Over Z/2 a cycle is a list of cells; over Z/p it is a list of
  `(cell, coefficient)` pairs. Cells are in decreasing filtration order, so the first is the birth cell;
* `diagram(dim)`, `diagrams()`: `(k, 2)` numpy arrays of `(birth, death)` filtration values, `inf` for essential bars;
* `bars`: every bar as a `Bar`, with `dimension`, `birth`, `death`, `birth_value`, `death_value` and
  `representative` (`(cell, coefficient)` pairs);
* `dimension_of(birth)`.

**Zero-length bars are omitted** (birth and death with the same filtration value), as everywhere in `phlite`:
there are usually far more of them than anything else, and skipping them lets the reduction skip apparent pairs.
With the default index filtration no bar has length zero, so you get every pair, as with PHAT.

## Using your own cell objects

`CellComplex` lets you describe the complex with your own Python objects, and gives the results back in terms of
them. A cell needs to be hashable, with a dimension and a boundary; this is the same protocol as grpphati's
`Column`, so grpphati columns work as they are:

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

barcode = phlite.CellComplex(cells).persistence(representatives=True)
for bar in barcode.bars:
    bar.dimension, bar.birth, bar.death, bar.birth_value, bar.death_value
    bar.representative  # [(Simplex(...), 1), ...], birth cell first
barcode.pairings        # [(Simplex(...), Simplex(...)), ...]
barcode.bar(some_simplex), barcode.representative(some_simplex)
```

* Faces only need to compare equal to the cell they mean, so `boundary()` can build fresh objects
  (without entrance times).
* Cells can be given in any order: they enter by filtration value, and their order only breaks ties.
  Values come from `values=` (a mapping, a function or a sequence), else `cell.entrance_time` if the cells have one,
  else the order given.
* Plain types work too, by saying how to get the pieces:
  `phlite.CellComplex(simplices, dimension=lambda s: len(s) - 1, boundary=faces, values=times)`,
  or pass a mapping `{cell: boundary}`.
* Internally each cell is given its position in `cells` as its index, and the complex is reduced exactly like a
  `FilteredComplex`; error messages name your cells by `repr`.

### Choosing the identifiers yourself

By default cells are matched up through a Python dict, using your `__hash__` and `__eq__`. You can instead give
`key=`, an injective map from cells to integers (e.g. a combinatorial encoding of a simplex); keys are matched in
a Rust hash map.

* `key=` lets cells be unhashable, and avoids your `__hash__`/`__eq__`, which helps when those are expensive.
* `from_key=` (the inverse of `key`) means the complex stores only the keys, not the cells, and rebuilds cells
  when you ask for results.
* `face_keys=True` means `boundary` returns the keys of the faces rather than face objects. This is the real
  speed-up, since building face objects is usually the slowest part.

`examples/cell_cost.py` builds the Vietoris-Rips complex of 300 random points (193k cells) from frozen dataclass
simplices, then computes H_0 and H_1:

| | build | memory kept by the complex |
|---|---|---|
| `FilteredComplex` from integer columns (indexing done by hand, in Python) | 175 ms | – |
| `CellComplex`, hashing | 448 ms | 17.4 MB |
| `CellComplex`, `key=` | 455 ms | 1.5 MB |
| `CellComplex`, `key=`, `from_key=` | 483 ms | 0 MB |
| `CellComplex`, `key=`, `face_keys=True` | 276 ms | 1.5 MB |

The reduction itself takes about 4 ms in every case. For simplices this cheap to hash, `key=` alone saves memory
but not time: most of the build is spent calling `boundary()`, which `face_keys=True` makes cheaper.

## How it works

`phlite` is built around lazy matrix oracles. Here the oracle is just an explicit matrix, stored compactly:

* the boundary matrix in CSC format, with `u32` cell indices (and zero-sized Z/2 coefficients), and its transpose
  (the coboundary), so about 8 bytes per non-zero entry plus 16 per cell;
* every column sorted in decreasing filtration order. A column's pivot is then its first entry and its smallest entry
  its last, so `column_pivot` and `column_min` are O(1) — and most columns need nothing more;
* the transpose sorted the same way, so `first_column_in_row` is O(1) and the clearing algorithms recognise
  apparent pairs for free. The transpose also lets cohomology compute representatives by involution.

So the same `phlite` reduction runs as for `phlite_rips` and `phlite_grpph`, on the matrix you supplied.
As with Ripser, `method="cohomology"` is usually much faster.

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
just develop-release   # build and install into .venv
just test              # pytest, against a brute-force reduction and LoPHAT
just stubs             # regenerate phlite.pyi
cargo test -p phlite_py  # the Rust side, without Python
```
