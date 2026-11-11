//! `CellComplex`: a filtered complex whose cells are arbitrary Python objects.
//!
//! Cells are resolved to dense `u32` indices (in the order they are given) and handed to the same
//! [`Complex`](crate::complex::Complex) as `FilteredComplex` uses; results are translated back to the cell objects.
//!
//! Resolving a cell to its index normally goes through a Python dict, so cells must be hashable.
//! Alternatively the user can supply `key`, an injective function from cells to integers, which is resolved through
//! a Rust hash map instead. That avoids Python-level `__hash__`/`__eq__` (often the bottleneck for rich cell types),
//! allows unhashable cells, and, together with `from_key`, avoids keeping the cell objects alive at all.

use std::collections::HashMap;

use numpy::PyArray2;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyMapping},
};
use pyo3_stub_gen::derive::*;

use super::{chain, read_floats, AnyComplex, Bar, Barcode};
use crate::complex::RawComplex;

/// How to get from a cell to its index.
enum CellIndex {
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
    /// The cells themselves.
    Objects(Py<PyList>),
    /// Their keys, and the inverse of `key`.
    Keys { keys: Vec<i64>, from_key: Py<PyAny> },
}

/// Where the filtration values come from.
enum Values<'py> {
    /// Use the order in which cells are given.
    Order,
    /// `cell.entrance_time`, as in grpphati.
    EntranceTime,
    /// A sequence in the same order as the cells.
    Sequence(Vec<f64>),
    /// `values[cell]`.
    Mapping(Bound<'py, PyMapping>),
    /// `values(cell)`.
    Function(Bound<'py, PyAny>),
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

    /// The index of `cell`, or `None` if it is not a cell of the complex.
    fn lookup(&self, cell: &Bound<'_, PyAny>) -> PyResult<Option<u32>> {
        let py = cell.py();
        match self {
            Self::Hash(dict) => dict
                .bind(py)
                .get_item(cell)?
                .map(|i| i.extract::<u32>())
                .transpose(),
            Self::Key { key, ids } => Ok(ids.get(&Self::key_of(py, key, cell)?).copied()),
        }
    }
}

impl CellStore {
    fn cell<'py>(&self, py: Python<'py>, index: u32) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Objects(cells) => cells.bind(py).get_item(index as usize),
            Self::Keys { keys, from_key } => from_key.bind(py).call1((keys[index as usize],)),
        }
    }
}

/// A filtered chain complex whose cells are your own Python objects.
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
///     entrance_time: float                 # optional: the filtration value (or a method)
/// ```
///
/// This is the same protocol as grpphati's `Column`, so its columns work as cells.
/// Faces in a boundary only need to compare equal to (and hash the same as) the cell they refer to;
/// they needn't be the same object, nor carry an entrance time.
///
/// A boundary is an iterable of faces (each with coefficient 1) or a mapping from faces to coefficients.
/// Every face must itself be a cell of the complex. Cells enter the filtration in order of their filtration value,
/// breaking ties by the order in which they are given (by default, they enter in the order given).
///
/// Internally each cell is assigned an index, its position in `cells`, and the complex is reduced exactly as for a
/// `FilteredComplex`; the results are translated back to your cells.
///
/// Args:
///     cells: The cells, or a mapping from each cell to its boundary.
///     boundary: A function giving the boundary of a cell (default: the mapping's values, else `cell.boundary()`).
///     dimension: A function giving the dimension of a cell (default: `cell.dimension`).
///         Handy for plain types, e.g. `dimension=lambda simplex: len(simplex) - 1` for tuples of vertices.
///     values: Filtration values: a mapping from cells, a function of a cell, or a sequence in the same order as
///         `cells`. By default, `cell.entrance_time` if the first cell has one, and otherwise the order of `cells`.
///         Infinite values mean the cell never appears.
///     key: An injective function from cells to integers (fitting in 64 bits). When given, cells are matched up by
///         key instead of by hashing, so need not be hashable. This is faster when hashing your cells is expensive.
///     from_key: The inverse of `key`. When given, the complex stores the keys rather than the cells and
///         reconstructs cells for the results, saving memory when cells are large.
///     face_keys: If true, boundaries list the *keys* of faces rather than faces (needs `key`). This is the fastest
///         way to describe a large complex, since no face objects need to be built.
///     modulus: The characteristic of the coefficient field: 2, 3, 5, 7, 11 or 13.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite")]
pub struct CellComplex {
    complex: AnyComplex,
    index: CellIndex,
    store: CellStore,
}

impl CellComplex {
    fn cell<'py>(&self, py: Python<'py>, index: u32) -> PyResult<Bound<'py, PyAny>> {
        self.store.cell(py, index)
    }

    fn index_or_err(&self, cell: &Bound<'_, PyAny>) -> PyResult<u32> {
        self.index.lookup(cell)?.ok_or_else(|| {
            PyValueError::new_err(format!("{} is not a cell of the complex", repr(cell)))
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CellComplex {
    #[new]
    #[pyo3(signature = (cells, *, boundary = None, dimension = None, values = None, key = None, from_key = None, face_keys = false, modulus = 2))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Union[typing.Iterable[typing.Any], typing.Mapping[typing.Any, typing.Union[typing.Iterable[typing.Any], typing.Mapping[typing.Any, builtins.int]]]]", imports = ("builtins", "typing")))]
        cells: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[typing.Any], typing.Union[typing.Iterable[typing.Any], typing.Mapping[typing.Any, builtins.int]]]]", imports = ("builtins", "typing")))]
        boundary: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[typing.Any], builtins.int]]", imports = ("builtins", "typing")))]
        dimension: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Union[typing.Mapping[typing.Any, builtins.float], typing.Callable[[typing.Any], builtins.float], typing.Sequence[builtins.float], numpy.typing.NDArray[numpy.floating], None]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        values: Option<&Bound<'_, PyAny>>,
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
        let values = match values {
            None => Values::Order,
            Some(v) if v.is_callable() => Values::Function(v.clone()),
            Some(v) => match v.cast::<PyMapping>() {
                Ok(mapping) => Values::Mapping(mapping.clone()),
                Err(_) => Values::Sequence(read_floats(v, "values")?),
            },
        };

        // ---- Pass 1: the cells, their indices, dimensions and values ----
        let as_mapping = cells.cast::<PyMapping>().ok();
        let mut values = values;
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
        let mut cell_values: Vec<f64> = vec![];
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
            if i == 0 && matches!(values, Values::Order) && cell.hasattr("entrance_time")? {
                // Like grpphati's columns: if cells know their entrance time, use it
                values = Values::EntranceTime;
            }
            let value = match &values {
                Values::Order | Values::Sequence(_) => None,
                Values::EntranceTime => Some(attribute(
                    &cell,
                    "entrance_time",
                    "the first cell has an entrance_time, so every cell needs one (or pass values=)",
                )),
                Values::Mapping(m) => Some(m.get_item(&cell)),
                Values::Function(f) => Some(f.call1((&cell,))),
            };
            if let Some(value) = value {
                let value = value.and_then(|v| v.extract::<f64>()).map_err(|e| {
                    context(
                        py,
                        e,
                        format!("Could not get the filtration value of {}", repr(&cell)),
                    )
                })?;
                cell_values.push(value);
            }
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
                    _ => index.lookup(face)?,
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

        let raw = RawComplex {
            dimensions,
            values: match values {
                Values::Order => None,
                Values::Sequence(v) => Some(v),
                Values::Mapping(_) | Values::Function(_) | Values::EntranceTime => {
                    Some(cell_values)
                }
            },
            indptr,
            indices,
            coefficients,
        };
        let complex = py
            .detach(|| AnyComplex::build(raw, modulus))?
            .map_err(|e| {
                PyValueError::new_err(
                    e.describe(|i| format!("cell {}", repr(&all_cells[i as usize]))),
                )
            })?;
        let store = match from_key {
            Some(from_key) => CellStore::Keys {
                keys,
                from_key: from_key.clone().unbind(),
            },
            None => CellStore::Objects(PyList::new(py, &all_cells)?.unbind()),
        };
        Ok(Self {
            complex,
            index,
            store,
        })
    }

    /// Compute persistent homology in dimensions `0, ..., max_dim`; see `FilteredComplex.persistence`.
    ///
    /// Returns:
    ///     The barcode, in terms of your cells. Bars of length zero are omitted.
    #[pyo3(signature = (max_dim = None, *, method = "cohomology", algorithm = "clearing", representatives = false, threads = None))]
    fn persistence(
        slf: &Bound<'_, Self>,
        max_dim: Option<usize>,
        #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
        method: &str,
        #[gen_stub(override_type(type_repr = "typing.Literal['standard', 'clearing', 'parallel', 'parallel-lock-free']", imports = ("typing")))]
        algorithm: &str,
        representatives: bool,
        threads: Option<usize>,
    ) -> PyResult<CellBarcode> {
        let barcode = slf.get().complex.persistence(
            slf.py(),
            max_dim,
            method,
            algorithm,
            representatives,
            threads,
        )?;
        Ok(CellBarcode {
            barcode,
            complex: slf.clone().unbind(),
        })
    }

    /// The cell at position `index` (its position in `cells`).
    fn cell_at<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        if index >= self.complex.n_cells() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "There are only {} cells",
                self.complex.n_cells()
            )));
        }
        self.cell(py, index as u32)
    }

    /// The position of `cell` in `cells`, which is how `phlite` identifies it.
    fn index_of(&self, cell: &Bound<'_, PyAny>) -> PyResult<u32> {
        self.index_or_err(cell)
    }

    fn __contains__(&self, cell: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.index.lookup(cell)?.is_some())
    }

    /// The number of cells.
    #[getter]
    fn n_cells(&self) -> usize {
        self.complex.n_cells()
    }

    /// The number of non-zero entries in the boundary matrix (after summing repeated entries).
    #[getter]
    fn n_entries(&self) -> usize {
        self.complex.n_entries()
    }

    /// The largest dimension of a cell, or `None` for an empty complex.
    #[getter]
    fn dimension(&self) -> Option<usize> {
        self.complex.top_dimension()
    }

    /// The characteristic of the coefficient field.
    #[getter]
    fn modulus(&self) -> u64 {
        self.complex.modulus()
    }

    fn __len__(&self) -> usize {
        self.n_cells()
    }

    fn __repr__(&self) -> String {
        format!(
            "CellComplex(n_cells={}, n_entries={}, dimension={}, modulus={})",
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
            CellIndex::Hash(dict) => visit.call(dict)?,
            CellIndex::Key { key, .. } => visit.call(key)?,
        }
        match &self.store {
            CellStore::Objects(cells) => visit.call(cells)?,
            CellStore::Keys { from_key, .. } => visit.call(from_key)?,
        }
        Ok(())
    }
}

/// A persistence barcode of a `CellComplex`, in terms of its cells.
///
/// Bars are identified by their birth cell (no cell gives birth to two bars).
/// Bars of length zero are omitted. Bars are sorted by the position of their birth cell in the complex.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite")]
pub struct CellBarcode {
    barcode: Barcode,
    complex: Py<CellComplex>,
}

impl CellBarcode {
    fn cell<'py>(&self, py: Python<'py>, index: u32) -> PyResult<Bound<'py, PyAny>> {
        self.complex.get().cell(py, index)
    }

    fn find(&self, birth: &Bound<'_, PyAny>) -> PyResult<&super::Record> {
        let complex = self.complex.get();
        let index = complex.index_or_err(birth)?;
        self.barcode
            .find(index)
            .ok_or_else(|| PyValueError::new_err(format!("No bar is born at {}", repr(birth))))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CellBarcode {
    /// Every bar, sorted by birth.
    #[getter]
    fn bars(&self, py: Python<'_>) -> PyResult<Vec<Bar>> {
        self.barcode.make_bars(py, |i| self.cell(py, i))
    }

    /// The finite bars, as `(birth, death)` pairs of cells, sorted by birth.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[tuple[typing.Any, typing.Any]]", imports = ("builtins", "typing")))]
    fn pairings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for bar in &self.barcode.bars {
            if let Some(death) = bar.death {
                list.append((self.cell(py, bar.birth)?, self.cell(py, death)?))?;
            }
        }
        Ok(list)
    }

    /// The birth cells of the essential (infinite) bars, sorted.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[typing.Any]", imports = ("builtins", "typing")))]
    fn essential<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for bar in &self.barcode.bars {
            if bar.death.is_none() {
                list.append(self.cell(py, bar.birth)?)?;
            }
        }
        Ok(list)
    }

    /// The bar born at the cell `birth`.
    fn bar(&self, py: Python<'_>, birth: &Bound<'_, PyAny>) -> PyResult<Bar> {
        Bar::from_record(py, self.find(birth)?, |i| self.cell(py, i))
    }

    /// A representative cycle for the bar born at `birth`, as a list of `(cell, coefficient)` pairs in decreasing
    /// filtration order, or `None` if representatives were not requested.
    #[gen_stub(override_return_type(type_repr = "typing.Optional[builtins.list[tuple[typing.Any, builtins.int]]]", imports = ("builtins", "typing")))]
    fn representative<'py>(
        &self,
        py: Python<'py>,
        birth: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let record = self.find(birth)?;
        record
            .representative
            .as_ref()
            .map(|rep| chain(py, rep, |c| self.cell(py, c)))
            .transpose()
    }

    /// The persistence diagram in dimension `dim`; see `Barcode.diagram`.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float64]", imports = ("numpy", "numpy.typing")))]
    fn diagram<'py>(&self, py: Python<'py>, dim: usize) -> Bound<'py, PyArray2<f64>> {
        self.barcode.diagram_array(py, dim)
    }

    /// The persistence diagrams in dimensions `0, ..., max_dim`; see `Barcode.diagram`.
    #[gen_stub(override_return_type(type_repr = "builtins.list[numpy.typing.NDArray[numpy.float64]]", imports = ("builtins", "numpy", "numpy.typing")))]
    fn diagrams<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        (0..self.barcode.n_dims)
            .map(|dim| self.barcode.diagram_array(py, dim))
            .collect()
    }

    /// The complex this barcode was computed from.
    #[getter]
    fn complex(&self, py: Python<'_>) -> Py<CellComplex> {
        self.complex.clone_ref(py)
    }

    fn __len__(&self) -> usize {
        self.barcode.bars.len()
    }

    fn __repr__(&self) -> String {
        self.barcode.repr("CellBarcode")
    }

    #[gen_stub(skip)]
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.complex)
    }
}
