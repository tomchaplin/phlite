//! `ChainComplex`: a chain complex (no filtration yet), and the translation between Python cells and indices.
//!
//! Cells are identified by dense `u32` indices, in the order they are given.
//! * Built from columns or CSC arrays, the cells *are* those indices (Python ints).
//! * Built from Python cell objects, cells are matched up through a Python dict (using their `__hash__` and
//!   `__eq__`), or by `key(cell)`, an injective function to integers, through a Rust hash map. That avoids
//!   Python-level hashing, allows unhashable cells, and, together with `from_key`, avoids keeping the cells alive.

use std::collections::HashMap;

use pyo3::{
    exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyMapping},
};
use pyo3_stub_gen::derive::*;

use super::{read_columns, read_floats, read_ints, AnyChain, Dimensions, FilteredComplex};
use crate::complex::RawChain;

/// How to get from a cell to its index.
enum CellIndex {
    /// Cells are their indices.
    Positions,
    /// cell -> index, using the cells' own `__hash__` and `__eq__`.
    Hash(Py<PyDict>),
    /// `key(cell)` -> index.
    Key {
        key: Py<PyAny>,
        ids: HashMap<i64, u32>,
    },
}

/// How to get from an index to its cell.
enum CellStore {
    /// Cells are their indices.
    Positions,
    /// The cells themselves.
    Objects(Py<PyList>),
    /// Their keys, and the inverse of `key`.
    Keys { keys: Vec<i64>, from_key: Py<PyAny> },
}

fn repr(obj: &Bound<'_, PyAny>) -> String {
    obj.repr()
        .map_or_else(|_| "<unprintable cell>".to_string(), |r| r.to_string())
}

/// Read `cell.<name>`, calling it if it is a method.
fn attribute<'py>(cell: &Bound<'py, PyAny>, name: &str, hint: &str) -> PyResult<Bound<'py, PyAny>> {
    let attr = cell.getattr(name).map_err(|e| {
        let err = PyTypeError::new_err(format!(
            "Cell {} has no `{name}` attribute; {hint}",
            repr(cell)
        ));
        err.set_cause(cell.py(), Some(e));
        err
    })?;
    if attr.is_callable() {
        attr.call0()
    } else {
        Ok(attr)
    }
}

/// Wrap an error from a user callback with some context.
fn context(py: Python<'_>, err: PyErr, message: String) -> PyErr {
    let wrapped = PyValueError::new_err(message);
    wrapped.set_cause(py, Some(err));
    wrapped
}

impl CellIndex {
    fn key_of(py: Python<'_>, key: &Py<PyAny>, cell: &Bound<'_, PyAny>) -> PyResult<i64> {
        key.bind(py)
            .call1((cell,))
            .and_then(|k| k.extract::<i64>())
            .map_err(|e| {
                context(
                    py,
                    e,
                    format!(
                        "key({}) must return an integer that fits in 64 bits",
                        repr(cell)
                    ),
                )
            })
    }

    /// The index of `cell`, or `None` if it is not a cell of a complex with `n` cells.
    fn lookup(&self, cell: &Bound<'_, PyAny>, n: usize) -> PyResult<Option<u32>> {
        let py = cell.py();
        match self {
            Self::Positions => Ok(cell
                .extract::<i64>()
                .ok()
                .filter(|&i| 0 <= i && (i as usize) < n)
                .map(|i| i as u32)),
            Self::Hash(dict) => dict
                .bind(py)
                .get_item(cell)?
                .map(|i| i.extract::<u32>())
                .transpose(),
            Self::Key { key, ids } => Ok(ids.get(&Self::key_of(py, key, cell)?).copied()),
        }
    }
}

/// Read an optional float: `None` is missing.
fn optional_float(value: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if value.is_none() {
        Ok(None)
    } else {
        value.extract().map(Some)
    }
}

/// A chain complex: cells, their dimensions and boundaries, with no filtration yet.
///
/// Build one with `from_pairs` (`(dimension, boundary)` pairs, as in PHAT and LoPHAT), `from_columns` (boundaries
/// and dimensions as separate sequences), `from_csc` (CSC arrays, e.g. a scipy sparse matrix, read without going
/// through Python objects), or `from_cells` (your own Python objects as cells). Then attach a filtration with `filtration(values)`; one chain complex can be
/// given any number of filtrations, and is validated and indexed only once.
///
/// For a complex built from columns or arrays, cells are their indices (ints); for one built from cells, they are
/// your objects, and every result is given in terms of them.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite")]
pub struct ChainComplex {
    chain: AnyChain,
    index: CellIndex,
    store: CellStore,
}

impl ChainComplex {
    /// The cell with index `i`, as a Python object.
    pub(crate) fn cell<'py>(&self, py: Python<'py>, i: u32) -> PyResult<Bound<'py, PyAny>> {
        match &self.store {
            CellStore::Positions => Ok(i.into_pyobject(py)?.into_any()),
            CellStore::Objects(cells) => cells.bind(py).get_item(i as usize),
            CellStore::Keys { keys, from_key } => from_key.bind(py).call1((keys[i as usize],)),
        }
    }

    /// The index of `cell`, or `None` if it is not a cell of the complex.
    pub(crate) fn lookup(&self, cell: &Bound<'_, PyAny>) -> PyResult<Option<u32>> {
        self.index.lookup(cell, self.chain.n_cells())
    }

    /// The index of `cell`, or an error if it is not a cell of the complex.
    pub(crate) fn index_or_err(&self, cell: &Bound<'_, PyAny>) -> PyResult<u32> {
        self.lookup(cell)?.ok_or_else(|| {
            PyValueError::new_err(format!("{} is not a cell of the complex", repr(cell)))
        })
    }

    /// A description of cell `i` for error messages.
    pub(crate) fn name(&self, py: Python<'_>, i: u32) -> String {
        match &self.store {
            CellStore::Positions => format!("cell {i}"),
            _ => format!(
                "cell {}",
                self.cell(py, i)
                    .map_or_else(|_| format!("at position {i}"), |c| repr(&c))
            ),
        }
    }

    fn from_raw(py: Python<'_>, raw: RawChain, modulus: u64) -> PyResult<Self> {
        let chain = py
            .detach(|| AnyChain::build(raw, modulus))?
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            chain,
            index: CellIndex::Positions,
            store: CellStore::Positions,
        })
    }

    /// The filtration values, or `None` for the order of the cells.
    fn read_values(
        &self,
        py: Python<'_>,
        values: Option<&Bound<'_, PyAny>>,
        extend: bool,
    ) -> PyResult<Option<Vec<f64>>> {
        let n = self.chain.n_cells();
        let per_cell = |get: &dyn Fn(&Bound<'_, PyAny>) -> PyResult<Option<f64>>| {
            (0..n as u32)
                .map(|i| {
                    let cell = self.cell(py, i)?;
                    get(&cell).map_err(|e| {
                        context(
                            py,
                            e,
                            format!("Could not get the filtration value of {}", repr(&cell)),
                        )
                    })
                })
                .collect::<PyResult<Vec<Option<f64>>>>()
        };
        let partial: Vec<Option<f64>> = match values {
            None => {
                let has_objects = !matches!(self.store, CellStore::Positions);
                if has_objects && n > 0 && self.cell(py, 0)?.hasattr("entrance_time")? {
                    // Like grpphati's columns: if cells know their entrance time, use it
                    per_cell(&|cell| {
                        optional_float(&attribute(
                            cell,
                            "entrance_time",
                            "the first cell has an entrance_time, so every cell needs one (or pass values)",
                        )?)
                    })?
                } else if extend {
                    return Err(PyValueError::new_err(
                        "extend=True needs some values to extend",
                    ));
                } else {
                    return Ok(None);
                }
            }
            Some(f) if f.is_callable() => per_cell(&|cell| optional_float(&f.call1((cell,))?))?,
            Some(m) if m.cast::<PyMapping>().is_ok() => {
                let dict = m.cast::<PyDict>().ok();
                per_cell(&|cell| {
                    let value = match &dict {
                        Some(dict) => dict.get_item(cell)?,
                        None => match m.get_item(cell) {
                            Ok(v) => Some(v),
                            Err(e) if e.is_instance_of::<PyKeyError>(py) => None,
                            Err(e) => return Err(e),
                        },
                    };
                    value.map_or(Ok(None), |v| optional_float(&v))
                })?
            }
            Some(sequence) => {
                let values = read_floats(sequence, "values")?;
                if values.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "values has length {} but there are {n} cells",
                        values.len()
                    )));
                }
                // NaN marks a missing value when extending (otherwise it is rejected later)
                values
                    .into_iter()
                    .map(|v| (!(extend && v.is_nan())).then_some(v))
                    .collect()
            }
        };
        let values = if extend {
            py.detach(|| self.chain.extend(partial))
                .map_err(|e| PyValueError::new_err(e.describe(|i| self.name(py, i))))?
        } else {
            partial
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    v.ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "The {} has no filtration value (pass extend=True to take it from its faces)",
                            self.name(py, i as u32)
                        ))
                    })
                })
                .collect::<PyResult<_>>()?
        };
        Ok(Some(values))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl ChainComplex {
    /// A chain complex from its boundary columns and the dimension of every cell. Cells are the integers
    /// `0, ..., n - 1`.
    ///
    /// Args:
    ///     boundaries: The boundary of every cell, as a sequence of cell indices. Repeated entries are summed.
    ///     dimensions: The dimension of every cell.
    ///     coefficients: The coefficient of every boundary entry, as a sequence of sequences matching `boundaries`
    ///         (default: all 1). Reduced modulo `modulus`, and entries that become 0 are dropped.
    ///     modulus: The characteristic of the coefficient field: 2, 3, 5, 7, 11 or 13.
    #[staticmethod]
    #[pyo3(signature = (boundaries, dimensions, *, coefficients = None, modulus = 2))]
    fn from_columns(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Iterable[typing.Sequence[builtins.int]]", imports = ("builtins", "typing")))]
        boundaries: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.integer], typing.Sequence[builtins.int]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        dimensions: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Iterable[typing.Sequence[builtins.int]]]", imports = ("builtins", "typing")))]
        coefficients: Option<&Bound<'_, PyAny>>,
        modulus: u64,
    ) -> PyResult<Self> {
        let raw = read_columns(boundaries, Dimensions::Given(dimensions), coefficients)?;
        Self::from_raw(py, raw, modulus)
    }

    /// A chain complex from `(dimension, boundary)` pairs, one per cell, as in PHAT and LoPHAT.
    /// Cells are the integers `0, ..., n - 1`.
    ///
    /// Args:
    ///     pairs: The dimension and boundary of every cell, the boundary as a sequence of cell indices.
    ///         Repeated entries in a boundary are summed.
    ///     coefficients: The coefficient of every boundary entry, as a sequence of sequences matching the
    ///         boundaries (default: all 1). Reduced modulo `modulus`, and entries that become 0 are dropped.
    ///     modulus: The characteristic of the coefficient field: 2, 3, 5, 7, 11 or 13.
    #[staticmethod]
    #[pyo3(signature = (pairs, *, coefficients = None, modulus = 2))]
    fn from_pairs(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Iterable[tuple[builtins.int, typing.Sequence[builtins.int]]]", imports = ("builtins", "typing")))]
        pairs: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Iterable[typing.Sequence[builtins.int]]]", imports = ("builtins", "typing")))]
        coefficients: Option<&Bound<'_, PyAny>>,
        modulus: u64,
    ) -> PyResult<Self> {
        let raw = read_columns(pairs, Dimensions::Paired, coefficients)?;
        Self::from_raw(py, raw, modulus)
    }

    /// A chain complex from a boundary matrix in compressed sparse column (CSC) format.
    /// Cells are the integers `0, ..., n - 1`.
    ///
    /// The boundary of cell `j` is `indices[indptr[j]:indptr[j + 1]]`, with coefficients from `data` if given.
    /// numpy arrays are read directly, so this is the fastest way to build a large complex.
    /// For a scipy sparse matrix `m`, use `ChainComplex.from_csc(m.indptr, m.indices, dimensions, data=m.data)`
    /// after converting with `m = m.tocsc()`.
    ///
    /// Args:
    ///     indptr: Column pointers, of length `n + 1` for `n` cells.
    ///     indices: The row (face) of every non-zero entry.
    ///     dimensions: The dimension of every cell.
    ///     data: The coefficient of every non-zero entry (default: all 1). Must be whole numbers.
    ///     modulus: The characteristic of the coefficient field: 2, 3, 5, 7, 11 or 13.
    #[staticmethod]
    #[pyo3(signature = (indptr, indices, dimensions, *, data = None, modulus = 2))]
    fn from_csc(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.integer], typing.Sequence[builtins.int]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        indptr: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.integer], typing.Sequence[builtins.int]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        indices: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.integer], typing.Sequence[builtins.int]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        dimensions: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Union[numpy.typing.NDArray[numpy.number], typing.Sequence[builtins.int]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        data: Option<&Bound<'_, PyAny>>,
        modulus: u64,
    ) -> PyResult<Self> {
        let raw = RawChain {
            dimensions: read_ints(dimensions, "dimensions")?,
            indptr: read_ints(indptr, "indptr")?,
            indices: read_ints(indices, "indices")?,
            coefficients: data.map(|d| read_ints(d, "data")).transpose()?,
        };
        Self::from_raw(py, raw, modulus)
    }

    /// A chain complex whose cells are your own Python objects.
    ///
    /// Every cell needs a dimension and a boundary, and must be hashable (unless you pass `key`).
    /// By default these come from the cells themselves, so a cell type only needs to implement
    ///
    /// ```python
    /// class MyCell(Protocol):
    ///     def __hash__(self) -> int: ...
    ///     def __eq__(self, other) -> bool: ...
    ///     def dimension(self) -> int: ...      # or an attribute or property
    ///     def boundary(self) -> Iterable[MyCell] | Mapping[MyCell, int]: ...  # or an attribute or property
    /// ```
    ///
    /// This is the same as grpphati's `Column` protocol, so its columns work as cells
    /// (and `filtration()` uses their `entrance_time`).
    /// A boundary is an iterable of faces (each with coefficient 1) or a mapping from faces to coefficients.
    /// Faces only need to compare equal to (and hash the same as) the cell they refer to; they needn't be the same
    /// object. Each cell's index is its position in `cells`.
    ///
    /// Args:
    ///     cells: The cells, or a mapping from each cell to its boundary.
    ///     boundary: A function giving the boundary of a cell (default: the mapping's values, else `cell.boundary()`).
    ///     dimension: A function giving the dimension of a cell (default: `cell.dimension`).
    ///         Handy for plain types, e.g. `dimension=lambda simplex: len(simplex) - 1` for tuples of vertices.
    ///     key: An injective function from cells to integers (fitting in 64 bits). When given, cells are matched up by
    ///         key instead of by hashing, so need not be hashable.
    ///     from_key: The inverse of `key`. When given, the complex stores the keys rather than the cells and
    ///         reconstructs cells for results, saving memory when cells are large.
    ///     face_keys: If true, boundaries list the *keys* of faces rather than faces (needs `key`). This is the fastest
    ///         way to describe a large complex, since no face objects need to be built.
    ///     modulus: The characteristic of the coefficient field: 2, 3, 5, 7, 11 or 13.
    #[staticmethod]
    #[pyo3(signature = (cells, *, boundary = None, dimension = None, key = None, from_key = None, face_keys = false, modulus = 2))]
    #[allow(clippy::too_many_arguments)]
    fn from_cells(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Union[typing.Iterable[typing.Any], typing.Mapping[typing.Any, typing.Union[typing.Iterable[typing.Any], typing.Mapping[typing.Any, builtins.int]]]]", imports = ("builtins", "typing")))]
        cells: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[typing.Any], typing.Union[typing.Iterable[typing.Any], typing.Mapping[typing.Any, builtins.int]]]]", imports = ("builtins", "typing")))]
        boundary: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[typing.Any], builtins.int]]", imports = ("builtins", "typing")))]
        dimension: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[typing.Any], builtins.int]]", imports = ("builtins", "typing")))]
        key: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[builtins.int], typing.Any]]", imports = ("builtins", "typing")))]
        from_key: Option<&Bound<'_, PyAny>>,
        face_keys: bool,
        modulus: u64,
    ) -> PyResult<Self> {
        if from_key.is_some() && key.is_none() {
            return Err(PyValueError::new_err(
                "from_key can only be given together with key",
            ));
        }
        if face_keys && key.is_none() {
            return Err(PyValueError::new_err("face_keys=True needs key"));
        }

        // ---- Pass 1: the cells, their indices and dimensions ----
        let as_mapping = cells.cast::<PyMapping>().ok();
        let items = match &as_mapping {
            Some(mapping) => mapping.call_method0("items")?.try_iter()?,
            None => cells.try_iter()?,
        };
        let mut index = match key {
            Some(key) => CellIndex::Key {
                key: key.clone().unbind(),
                ids: HashMap::new(),
            },
            None => CellIndex::Hash(PyDict::new(py).unbind()),
        };
        let mut all_cells: Vec<Bound<'_, PyAny>> = vec![];
        let mut keys: Vec<i64> = vec![];
        let mut mapped_boundaries: Vec<Bound<'_, PyAny>> = vec![];
        let mut dimensions: Vec<i64> = vec![];
        for item in items {
            let item = item?;
            let cell = if as_mapping.is_some() {
                let (cell, bdry): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item.extract()?;
                mapped_boundaries.push(bdry);
                cell
            } else {
                item
            };
            let i = all_cells.len();
            if i >= u32::MAX as usize {
                return Err(PyValueError::new_err("Too many cells"));
            }
            match &mut index {
                CellIndex::Positions => unreachable!("cells are indexed by hash or key"),
                CellIndex::Hash(dict) => {
                    let dict = dict.bind(py);
                    if dict.contains(&cell).map_err(|e| {
                        let err = PyTypeError::new_err(format!(
                            "Cell {} is not hashable; make it hashable, or pass key=",
                            repr(&cell)
                        ));
                        err.set_cause(py, Some(e));
                        err
                    })? {
                        return Err(PyValueError::new_err(format!(
                            "Cell {} is given more than once",
                            repr(&cell)
                        )));
                    }
                    dict.set_item(&cell, i)?;
                }
                CellIndex::Key { key, ids } => {
                    let k = CellIndex::key_of(py, key, &cell)?;
                    if let Some(j) = ids.insert(k, i as u32) {
                        return Err(PyValueError::new_err(format!(
                            "Cells {} and {} have the same key {k}, but keys must be distinct",
                            repr(&all_cells[j as usize]),
                            repr(&cell)
                        )));
                    }
                    keys.push(k);
                }
            }
            let dim = match dimension {
                Some(f) => f.call1((&cell,)),
                None => attribute(
                    &cell,
                    "dimension",
                    "give cells a dimension, or pass dimension=",
                ),
            }
            .and_then(|d| d.extract::<i64>())
            .map_err(|e| {
                context(
                    py,
                    e,
                    format!("Could not get the dimension of {}", repr(&cell)),
                )
            })?;
            dimensions.push(dim);
            all_cells.push(cell);
        }
        let n = all_cells.len();
        // ---- Pass 2: boundaries, resolved to indices ----
        let mut indptr: Vec<i64> = Vec::with_capacity(n + 1);
        indptr.push(0);
        let mut indices: Vec<i64> = vec![];
        let mut coefficients: Option<Vec<i64>> = None;
        for (i, cell) in all_cells.iter().enumerate() {
            let bdry = if as_mapping.is_some() {
                Ok(mapped_boundaries[i].clone())
            } else if let Some(f) = boundary {
                f.call1((cell,))
            } else {
                attribute(
                    cell,
                    "boundary",
                    "give cells a boundary() method, pass a mapping from cells to boundaries, or pass boundary=",
                )
            }
            .map_err(|e| context(py, e, format!("Could not get the boundary of {}", repr(cell))))?;

            let mut push = |face: &Bound<'_, PyAny>, coeff: i64| -> PyResult<()> {
                let j = match &index {
                    // The boundary gives keys directly, so no key() call (or face object) is needed
                    CellIndex::Key { ids, .. } if face_keys => {
                        let k: i64 = face.extract().map_err(|e| {
                            context(
                                py,
                                e,
                                format!(
                                    "With face_keys=True, the boundary of {} must contain integer keys",
                                    repr(cell)
                                ),
                            )
                        })?;
                        ids.get(&k).copied()
                    }
                    _ => index.lookup(face, n)?,
                };
                let Some(j) = j else {
                    return Err(PyValueError::new_err(format!(
                        "The boundary of {} contains {}, which is not {} of the complex",
                        repr(cell),
                        repr(face),
                        if face_keys {
                            "the key of a cell"
                        } else {
                            "a cell"
                        }
                    )));
                };
                indices.push(j as i64);
                if let Some(coefficients) = coefficients.as_mut() {
                    coefficients.push(coeff);
                } else if coeff != 1 {
                    // The first coefficient other than 1: from now on, record them all
                    let mut all = vec![1; indices.len() - 1];
                    all.push(coeff);
                    coefficients = Some(all);
                }
                Ok(())
            };
            if let Ok(chain) = bdry.cast::<PyMapping>() {
                for item in chain.call_method0("items")?.try_iter()? {
                    let (face, coeff): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
                    let coeff = coeff.extract::<i64>().map_err(|e| {
                        context(
                            py,
                            e,
                            format!(
                                "Coefficients in the boundary of {} must be integers",
                                repr(cell)
                            ),
                        )
                    })?;
                    push(&face, coeff)?;
                }
            } else {
                for face in bdry.try_iter().map_err(|e| {
                    context(
                        py,
                        e,
                        format!(
                            "The boundary of {} must be an iterable of cells or a mapping from cells to coefficients",
                            repr(cell)
                        ),
                    )
                })? {
                    push(&face?, 1)?;
                }
            }
            indptr.push(indices.len() as i64);
        }

        let raw = RawChain {
            dimensions,
            indptr,
            indices,
            coefficients,
        };
        let chain = py.detach(|| AnyChain::build(raw, modulus))?.map_err(|e| {
            PyValueError::new_err(e.describe(|i| format!("cell {}", repr(&all_cells[i as usize]))))
        })?;
        let store = match from_key {
            Some(from_key) => CellStore::Keys {
                keys,
                from_key: from_key.clone().unbind(),
            },
            None => CellStore::Objects(PyList::new(py, &all_cells)?.unbind()),
        };
        Ok(Self {
            chain,
            index,
            store,
        })
    }

    /// Attach a filtration, giving a complex ready to compute persistence.
    ///
    /// Cells enter in order of their filtration value, breaking ties by index (their position in the input).
    /// Every face must enter before its cofaces, otherwise a `ValueError` is raised.
    ///
    /// Args:
    ///     values: The filtration value of every cell, as a sequence or array in cell order (fastest),
    ///         a mapping from cells, or a function of a cell. By default: `cell.entrance_time` if the cells
    ///         have one, and otherwise the order of the cells (so cell `j` has value `j`).
    ///         Infinite values mean the cell never appears.
    ///         (For a complex built with `from_key`, a function or mapping is given the cells rebuilt from their keys.)
    ///     extend: If true, values may be missing (absent from the mapping, `None`, or NaN in a sequence),
    ///         and a missing value is the largest value of the cell's faces. For example, values on the vertices of
    ///         a simplicial complex extend to its lower-star filtration.
    #[pyo3(signature = (values = None, *, extend = false))]
    fn filtration(
        slf: &Bound<'_, Self>,
        #[gen_stub(override_type(type_repr = "typing.Union[typing.Sequence[builtins.float], numpy.typing.NDArray[numpy.floating], typing.Mapping[typing.Any, typing.Optional[builtins.float]], typing.Callable[[typing.Any], typing.Optional[builtins.float]], None]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        values: Option<&Bound<'_, PyAny>>,
        extend: bool,
    ) -> PyResult<FilteredComplex> {
        let py = slf.py();
        let this = slf.get();
        let values = this.read_values(py, values, extend)?;
        let complex = py
            .detach(|| this.chain.filter(values))
            .map_err(|e| PyValueError::new_err(e.describe(|i| this.name(py, i))))?;
        Ok(FilteredComplex {
            complex,
            chain: slf.clone().unbind(),
        })
    }

    /// The cell with index `index` (its position in the input).
    fn cell_at<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        if index >= self.chain.n_cells() {
            return Err(PyIndexError::new_err(format!(
                "There are only {} cells",
                self.chain.n_cells()
            )));
        }
        self.cell(py, index as u32)
    }

    /// The index of `cell` (its position in the input), which is how `phlite` identifies it.
    fn index_of(&self, cell: &Bound<'_, PyAny>) -> PyResult<u32> {
        self.index_or_err(cell)
    }

    fn __contains__(&self, cell: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.lookup(cell)?.is_some())
    }

    /// The number of cells.
    #[getter]
    fn n_cells(&self) -> usize {
        self.chain.n_cells()
    }

    /// The number of non-zero entries in the boundary matrix (after summing repeated entries).
    #[getter]
    fn n_entries(&self) -> usize {
        self.chain.n_entries()
    }

    /// The largest dimension of a cell, or `None` for an empty complex.
    #[getter]
    fn dimension(&self) -> Option<usize> {
        self.chain.top_dimension()
    }

    /// The characteristic of the coefficient field.
    #[getter]
    fn modulus(&self) -> u64 {
        self.chain.modulus()
    }

    fn __len__(&self) -> usize {
        self.n_cells()
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "ChainComplex(n_cells={}, n_entries={}, dimension={}, modulus={})",
            self.n_cells(),
            self.n_entries(),
            self.dimension()
                .map_or("None".to_string(), |d| d.to_string()),
            self.modulus()
        )
    }

    #[gen_stub(skip)]
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        match &self.index {
            CellIndex::Positions => {}
            CellIndex::Hash(dict) => visit.call(dict)?,
            CellIndex::Key { key, .. } => visit.call(key)?,
        }
        match &self.store {
            CellStore::Positions => {}
            CellStore::Objects(cells) => visit.call(cells)?,
            CellStore::Keys { from_key, .. } => visit.call(from_key)?,
        }
        Ok(())
    }
}
