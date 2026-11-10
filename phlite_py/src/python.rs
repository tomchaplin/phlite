//! The `phlite` Python module (feature `python`).

use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use ordered_float::NotNan;
use phlite::{
    barcode::Bar,
    fields::{Z11, Z13, Z2, Z3, Z5, Z7},
};
use phlite_pyutils::{parse_method, AlgorithmArg};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use pyo3_stub_gen::derive::*;

use crate::{
    complex::{Cell, Complex, RawComplex},
    field::IntField,
};

// ====== Coefficient fields ===============================================

/// The supported moduli, for error messages.
const MODULI: &str = "2, 3, 5, 7, 11 or 13";

/// A complex over one of the supported fields.
/// Each field is a separate monomorphisation of the whole reduction, so we keep the list short.
enum AnyComplex {
    Z2(Complex<Z2>),
    Z3(Complex<Z3>),
    Z5(Complex<Z5>),
    Z7(Complex<Z7>),
    Z11(Complex<Z11>),
    Z13(Complex<Z13>),
}

/// Run `$body` with `$c` bound to the complex inside `$any`, whatever its field.
macro_rules! dispatch {
    ($any:expr, $c:ident => $body:expr) => {
        match $any {
            AnyComplex::Z2($c) => $body,
            AnyComplex::Z3($c) => $body,
            AnyComplex::Z5($c) => $body,
            AnyComplex::Z7($c) => $body,
            AnyComplex::Z11($c) => $body,
            AnyComplex::Z13($c) => $body,
        }
    };
}

impl AnyComplex {
    fn build(raw: RawComplex, modulus: u64) -> PyResult<Self> {
        let complex = match modulus {
            2 => Complex::from_raw(raw).map(Self::Z2),
            3 => Complex::from_raw(raw).map(Self::Z3),
            5 => Complex::from_raw(raw).map(Self::Z5),
            7 => Complex::from_raw(raw).map(Self::Z7),
            11 => Complex::from_raw(raw).map(Self::Z11),
            13 => Complex::from_raw(raw).map(Self::Z13),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported modulus {other}, expected one of {MODULI}"
                )))
            }
        };
        complex.map_err(PyValueError::new_err)
    }

    fn modulus(&self) -> u64 {
        fn of<CF: IntField>(_: &Complex<CF>) -> u64 {
            CF::MODULUS
        }
        dispatch!(self, c => of(c))
    }
}

// ====== Reading arrays ===================================================

fn is_numpy(obj: &Bound<'_, PyAny>) -> bool {
    obj.get_type()
        .module()
        .is_ok_and(|module| module.to_cow().is_ok_and(|m| m == "numpy"))
}

/// Read a 1D numpy array of dtype `T` (in any layout), if `obj` is one.
fn try_numpy<T: numpy::Element + Copy>(obj: &Bound<'_, PyAny>) -> Option<Vec<T>> {
    let array = obj.cast::<PyArray1<T>>().ok()?;
    let array = array.readonly();
    Some(array.as_array().iter().copied().collect())
}

fn check_1d(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if let Ok(array) = obj.cast::<PyUntypedArray>() {
        if array.ndim() != 1 {
            return Err(PyValueError::new_err(format!(
                "{name} must be 1-dimensional, but has {} dimensions",
                array.ndim()
            )));
        }
    }
    Ok(())
}

fn integral(x: f64, name: &str) -> PyResult<i64> {
    if x.fract() == 0.0 && x.abs() < 2f64.powi(62) {
        Ok(x as i64)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must contain integers, but contains {x}"
        )))
    }
}

/// Read a sequence of integers. Integer numpy arrays (and float arrays of whole numbers, as scipy uses for `data`)
/// are read directly, without converting each entry through Python.
fn read_ints(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    if is_numpy(obj) {
        if let Some(v) = try_numpy::<i64>(obj) {
            return Ok(v);
        }
        if let Some(v) = try_numpy::<i32>(obj) {
            return Ok(v.into_iter().map(i64::from).collect());
        }
        if let Some(v) = try_numpy::<u32>(obj) {
            return Ok(v.into_iter().map(i64::from).collect());
        }
        if let Some(v) = try_numpy::<u64>(obj) {
            return v
                .into_iter()
                .map(|x| {
                    i64::try_from(x).map_err(|_| {
                        PyValueError::new_err(format!("{name} contains {x}, which is too large"))
                    })
                })
                .collect();
        }
        if let Some(v) = try_numpy::<f64>(obj) {
            return v.into_iter().map(|x| integral(x, name)).collect();
        }
        if let Some(v) = try_numpy::<f32>(obj) {
            return v.into_iter().map(|x| integral(x as f64, name)).collect();
        }
        check_1d(obj, name)?;
    }
    obj.extract().map_err(|e| {
        let err = PyTypeError::new_err(format!("{name} must be a sequence of integers"));
        err.set_cause(obj.py(), Some(e));
        err
    })
}

/// Read a sequence of floats, reading float numpy arrays directly.
fn read_floats(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if is_numpy(obj) {
        if let Some(v) = try_numpy::<f64>(obj) {
            return Ok(v);
        }
        if let Some(v) = try_numpy::<f32>(obj) {
            return Ok(v.into_iter().map(f64::from).collect());
        }
        check_1d(obj, name)?;
    }
    obj.extract().map_err(|e| {
        let err = PyTypeError::new_err(format!("{name} must be a sequence of floats"));
        err.set_cause(obj.py(), Some(e));
        err
    })
}

/// If `item` is a `(dimension, column)` pair, split it.
/// A column is never mistaken for a pair, since the entries of a column are integers.
fn split_pair<'py>(item: &Bound<'py, PyAny>) -> Option<(i64, Bound<'py, PyAny>)> {
    let tuple = item.cast::<PyTuple>().ok()?;
    if tuple.len() != 2 {
        return None;
    }
    let first = tuple.get_item(0).ok()?;
    let second = tuple.get_item(1).ok()?;
    match (first.extract::<i64>(), second.extract::<i64>()) {
        (Ok(dim), Err(_)) => Some((dim, second)),
        _ => None,
    }
}

/// Parse the columns of the boundary matrix, given as a sequence of columns or of `(dimension, column)` pairs.
fn read_columns(
    boundaries: &Bound<'_, PyAny>,
    dimensions: Option<&Bound<'_, PyAny>>,
    coefficients: Option<&Bound<'_, PyAny>>,
) -> PyResult<RawComplex> {
    let mut raw = RawComplex {
        indptr: vec![0],
        ..Default::default()
    };
    let mut pair_dims: Vec<i64> = vec![];
    let mut coeffs: Option<Vec<i64>> = coefficients.map(|_| vec![]);
    let mut coeff_iter = coefficients.map(|c| c.try_iter()).transpose()?;
    let mut saw_pair = None;

    for (j, item) in boundaries.try_iter()?.enumerate() {
        let item = item?;
        let (column, is_pair) = match split_pair(&item) {
            Some((dim, column)) => {
                pair_dims.push(dim);
                (column, true)
            }
            None => (item, false),
        };
        if *saw_pair.get_or_insert(is_pair) != is_pair {
            return Err(PyValueError::new_err(format!(
                "boundaries[{j}] is {} but earlier entries are not; \
                 give either every column or every (dimension, column) pair",
                if is_pair {
                    "a (dimension, column) pair"
                } else {
                    "a column"
                }
            )));
        }
        let column = read_ints(&column, &format!("boundaries[{j}]"))?;

        if let (Some(iter), Some(coeffs)) = (coeff_iter.as_mut(), coeffs.as_mut()) {
            let Some(c) = iter.next() else {
                return Err(PyValueError::new_err(format!(
                    "coefficients has fewer entries than boundaries (no entry for column {j})"
                )));
            };
            let c = read_ints(&c?, &format!("coefficients[{j}]"))?;
            if c.len() != column.len() {
                return Err(PyValueError::new_err(format!(
                    "coefficients[{j}] has length {} but boundaries[{j}] has length {}",
                    c.len(),
                    column.len()
                )));
            }
            coeffs.extend(c);
        }
        raw.indices.extend(column);
        raw.indptr.push(raw.indices.len() as i64);
    }
    if let Some(iter) = coeff_iter.as_mut() {
        if iter.next().is_some() {
            return Err(PyValueError::new_err(
                "coefficients has more entries than boundaries",
            ));
        }
    }
    raw.coefficients = coeffs;

    raw.dimensions = match (saw_pair == Some(true), dimensions) {
        (true, Some(_)) => {
            return Err(PyValueError::new_err(
                "Dimensions were given both as (dimension, column) pairs and as the dimensions argument",
            ))
        }
        (true, None) => pair_dims,
        (false, Some(dims)) => read_ints(dims, "dimensions")?,
        // No columns at all: an empty complex
        (false, None) if saw_pair.is_none() => vec![],
        (false, None) => {
            return Err(PyValueError::new_err(
                "The dimension of every cell is needed: pass dimensions=..., \
                 or give boundaries as (dimension, column) pairs",
            ))
        }
    };
    Ok(raw)
}

// ====== FilteredComplex ==================================================

/// A filtered chain complex, given by its sparse boundary matrix, ready to compute persistent homology.
///
/// Cells are numbered `0, ..., n - 1` and the boundary of cell `j` is a linear combination of cells.
/// Cells enter the filtration in order of their filtration value, breaking ties by index
/// (by default the value of cell `j` is `j`, as in PHAT).
/// Every face must have dimension one lower than its coface and enter the filtration before it,
/// otherwise a `ValueError` is raised.
///
/// The complex is validated and stored on construction (as the boundary matrix and its transpose, with
/// each column sorted), so you can compute several barcodes from one complex cheaply.
///
/// Args:
///     boundaries: The boundary of every cell, as a sequence of columns (each a sequence of cell indices),
///         or as a sequence of `(dimension, column)` pairs, as in PHAT and LoPHAT.
///         Repeated entries in a column are summed.
///     dimensions: The dimension of every cell. Required unless `boundaries` gives `(dimension, column)` pairs.
///     values: The filtration value of every cell (default: the cell's index).
///         Infinite values are allowed, and mean the cell never appears.
///     coefficients: The coefficient of every boundary entry, as a sequence of sequences matching `boundaries`
///         (default: all 1). Reduced modulo `modulus`, and entries that become 0 are dropped.
///     modulus: The characteristic of the coefficient field: 2, 3, 5, 7, 11 or 13.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite")]
pub struct FilteredComplex {
    complex: AnyComplex,
}

#[gen_stub_pymethods]
#[pymethods]
impl FilteredComplex {
    #[new]
    #[pyo3(signature = (boundaries, dimensions = None, values = None, *, coefficients = None, modulus = 2))]
    fn new(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Iterable[typing.Union[typing.Sequence[builtins.int], tuple[builtins.int, typing.Sequence[builtins.int]]]]", imports = ("builtins", "typing")))]
        boundaries: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Union[numpy.typing.NDArray[numpy.integer], typing.Sequence[builtins.int]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        dimensions: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        values: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Iterable[typing.Sequence[builtins.int]]]", imports = ("builtins", "typing")))]
        coefficients: Option<&Bound<'_, PyAny>>,
        modulus: u64,
    ) -> PyResult<Self> {
        let mut raw = read_columns(boundaries, dimensions, coefficients)?;
        raw.values = values.map(|v| read_floats(v, "values")).transpose()?;
        let complex = py.detach(|| AnyComplex::build(raw, modulus))?;
        Ok(Self { complex })
    }

    /// Build a complex from a boundary matrix in compressed sparse column (CSC) format.
    ///
    /// The boundary of cell `j` is `indices[indptr[j]:indptr[j + 1]]`, with coefficients from `data` if given.
    /// numpy arrays are read directly, so this is the fastest way to build a large complex.
    /// For a scipy sparse matrix `m`, use
    /// `FilteredComplex.from_csc(m.indptr, m.indices, dimensions, data=m.data)`
    /// after converting with `m = m.tocsc()`.
    ///
    /// Args:
    ///     indptr: Column pointers, of length `n + 1` for `n` cells.
    ///     indices: The row (face) of every non-zero entry.
    ///     dimensions: The dimension of every cell.
    ///     values: The filtration value of every cell (default: the cell's index).
    ///     data: The coefficient of every non-zero entry (default: all 1). Must be whole numbers.
    ///     modulus: The characteristic of the coefficient field: 2, 3, 5, 7, 11 or 13.
    #[staticmethod]
    #[pyo3(signature = (indptr, indices, dimensions, values = None, *, data = None, modulus = 2))]
    fn from_csc(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.integer], typing.Sequence[builtins.int]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        indptr: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.integer], typing.Sequence[builtins.int]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        indices: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.integer], typing.Sequence[builtins.int]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        dimensions: &Bound<'_, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        values: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Union[numpy.typing.NDArray[numpy.number], typing.Sequence[builtins.int]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
        data: Option<&Bound<'_, PyAny>>,
        modulus: u64,
    ) -> PyResult<Self> {
        let raw = RawComplex {
            dimensions: read_ints(dimensions, "dimensions")?,
            values: values.map(|v| read_floats(v, "values")).transpose()?,
            indptr: read_ints(indptr, "indptr")?,
            indices: read_ints(indices, "indices")?,
            coefficients: data.map(|d| read_ints(d, "data")).transpose()?,
        };
        let complex = py.detach(|| AnyComplex::build(raw, modulus))?;
        Ok(Self { complex })
    }

    /// Compute persistent homology in dimensions `0, ..., max_dim`.
    ///
    /// Args:
    ///     max_dim: The maximum homology dimension to compute (default: the largest dimension of a cell).
    ///         Lower is faster: with `method='cohomology'` cells above dimension `max_dim + 1` are never looked at.
    ///     method: `'cohomology'` (the default) reduces the coboundary matrix, which is usually much faster.
    ///         `'homology'` reduces the boundary matrix instead.
    ///     algorithm: The reduction algorithm: `'standard'`, `'clearing'` (the default), `'parallel'`
    ///         (clearing, with each dimension reduced in parallel) or `'parallel-lock-free'`
    ///         (the same, using lock-free data structures).
    ///     representatives: Whether to also compute a representative cycle for every bar.
    ///         With `method='cohomology'` these are computed via involution.
    ///     threads: The number of threads for the parallel algorithms (default: all cores).
    ///
    /// Returns:
    ///     The barcode. Bars of length zero (birth and death at the same filtration value) are omitted.
    #[pyo3(signature = (max_dim = None, *, method = "cohomology", algorithm = "clearing", representatives = false, threads = None))]
    fn persistence(
        &self,
        py: Python<'_>,
        max_dim: Option<usize>,
        #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
        method: &str,
        #[gen_stub(override_type(type_repr = "typing.Literal['standard', 'clearing', 'parallel', 'parallel-lock-free']", imports = ("typing")))]
        algorithm: &str,
        representatives: bool,
        threads: Option<usize>,
    ) -> PyResult<Barcode> {
        let method = parse_method(method)?;
        let algorithm = AlgorithmArg::parse(algorithm)?;
        dispatch!(&self.complex, c => py.detach(|| {
            algorithm.run(threads, |algorithm| {
                let bars = c.barcode(max_dim, method, algorithm, representatives);
                Barcode::new(c, bars, max_dim, representatives)
            })
        }))
    }

    /// The number of cells.
    #[getter]
    fn n_cells(&self) -> usize {
        dispatch!(&self.complex, c => c.n_cells())
    }

    /// The number of non-zero entries in the boundary matrix (after summing repeated entries).
    #[getter]
    fn n_entries(&self) -> usize {
        dispatch!(&self.complex, c => c.n_entries())
    }

    /// The largest dimension of a cell, or `None` for an empty complex.
    #[getter]
    fn dimension(&self) -> Option<usize> {
        dispatch!(&self.complex, c => c.top_dimension())
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
            "FilteredComplex(n_cells={}, n_entries={}, dimension={}, modulus={})",
            self.n_cells(),
            self.n_entries(),
            self.dimension()
                .map_or("None".to_string(), |d| d.to_string()),
            self.modulus()
        )
    }
}

// ====== Barcode ==========================================================

/// A bar, with the complex's data copied in so that the barcode is self-contained.
struct Record {
    birth: u32,
    death: Option<u32>,
    dimension: usize,
    birth_value: f64,
    /// Infinite for essential bars.
    death_value: f64,
    /// `(cell, coefficient)` pairs, in decreasing filtration order.
    representative: Option<Vec<(u32, u64)>>,
}

/// A persistence barcode, as computed by `FilteredComplex.persistence`.
///
/// Bars are identified by their birth cell (no cell gives birth to two bars) and, for finite bars,
/// their death cell, as indices into the complex.
/// Bars of length zero are omitted.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite")]
pub struct Barcode {
    /// Sorted by birth cell.
    bars: Vec<Record>,
    n_dims: usize,
    modulus: u64,
    representatives: bool,
}

impl Barcode {
    fn new<CF: IntField>(
        complex: &Complex<CF>,
        bars: Vec<Bar<Cell, NotNan<f64>, CF>>,
        max_dim: Option<usize>,
        representatives: bool,
    ) -> Self {
        let n_dims = max_dim.or(complex.top_dimension()).map_or(0, |d| d + 1);
        let mut bars: Vec<Record> = bars
            .into_iter()
            .map(|bar| Record {
                birth: bar.birth.0,
                death: bar.death.as_ref().map(|(cell, _)| cell.0),
                dimension: complex.dimension(bar.birth),
                birth_value: bar.birth_value.into_inner(),
                death_value: bar
                    .death
                    .map_or(f64::INFINITY, |(_, value)| value.into_inner()),
                representative: bar.representative.map(|rep| {
                    rep.into_iter()
                        .map(|(coeff, cell)| (cell.0, coeff.to_int()))
                        .collect()
                }),
            })
            .collect();
        bars.sort_unstable_by_key(|bar| bar.birth);
        Self {
            bars,
            n_dims,
            modulus: CF::MODULUS,
            representatives,
        }
    }

    fn diagram_array<'py>(&self, py: Python<'py>, dim: usize) -> Bound<'py, PyArray2<f64>> {
        let mut points: Vec<(f64, f64)> = self
            .bars
            .iter()
            // Nothing appears at time infinity, so classes born then never exist
            .filter(|bar| bar.dimension == dim && bar.birth_value != f64::INFINITY)
            .map(|bar| (bar.birth_value, bar.death_value))
            .collect();
        points.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        let flat: Vec<f64> = points.iter().flat_map(|&(b, d)| [b, d]).collect();
        Array2::from_shape_vec((points.len(), 2), flat)
            .expect("two values per point")
            .into_pyarray(py)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Barcode {
    /// The finite bars, as `(birth, death)` pairs of cell indices, sorted by birth.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[tuple[builtins.int, builtins.int]]", imports = ("builtins")))]
    fn pairings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.bars
                .iter()
                .filter_map(|bar| bar.death.map(|death| (bar.birth, death))),
        )
    }

    /// The birth cells of the essential (infinite) bars, sorted.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[builtins.int]", imports = ("builtins")))]
    fn essential<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.bars
                .iter()
                .filter(|bar| bar.death.is_none())
                .map(|bar| bar.birth),
        )
    }

    /// A representative cycle for every bar, keyed by birth cell, or `None` if they were not requested.
    ///
    /// Over Z/2 a cycle is a list of cells; over Z/p it is a list of `(cell, coefficient)` pairs
    /// with coefficients in `1, ..., p - 1`. Cells are listed in decreasing filtration order.
    /// A finite bar's representative is a cycle that becomes a boundary at its death.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[typing.Union[builtins.dict[builtins.int, builtins.list[builtins.int]], builtins.dict[builtins.int, builtins.list[tuple[builtins.int, builtins.int]]]]]", imports = ("builtins", "typing")))]
    fn representatives<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if !self.representatives {
            return Ok(None);
        }
        let dict = PyDict::new(py);
        for bar in &self.bars {
            let rep = bar
                .representative
                .as_ref()
                .expect("representatives were requested");
            let rep = if self.modulus == 2 {
                PyList::new(py, rep.iter().map(|&(cell, _)| cell))?
            } else {
                PyList::new(py, rep.iter().copied())?
            };
            dict.set_item(bar.birth, rep)?;
        }
        Ok(Some(dict))
    }

    /// The persistence diagram in dimension `dim`, as an array of shape `(k, 2)` of `(birth, death)`
    /// filtration values, sorted. Essential bars have death `inf`.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float64]", imports = ("numpy", "numpy.typing")))]
    fn diagram<'py>(&self, py: Python<'py>, dim: usize) -> Bound<'py, PyArray2<f64>> {
        self.diagram_array(py, dim)
    }

    /// The persistence diagrams in dimensions `0, ..., max_dim`, see `diagram`.
    #[gen_stub(override_return_type(type_repr = "builtins.list[numpy.typing.NDArray[numpy.float64]]", imports = ("builtins", "numpy", "numpy.typing")))]
    fn diagrams<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        (0..self.n_dims)
            .map(|dim| self.diagram_array(py, dim))
            .collect()
    }

    /// The dimension of the bar born at `birth` (the dimension of the cell `birth`).
    fn dimension_of(&self, birth: u32) -> PyResult<usize> {
        self.bars
            .binary_search_by_key(&birth, |bar| bar.birth)
            .map(|i| self.bars[i].dimension)
            .map_err(|_| PyValueError::new_err(format!("No bar is born at cell {birth}")))
    }

    fn __len__(&self) -> usize {
        self.bars.len()
    }

    fn __repr__(&self) -> String {
        let mut finite = vec![0; self.n_dims];
        let mut essential = vec![0; self.n_dims];
        for bar in &self.bars {
            let counts = if bar.death.is_some() {
                &mut finite
            } else {
                &mut essential
            };
            if let Some(count) = counts.get_mut(bar.dimension) {
                *count += 1;
            }
        }
        format!(
            "Barcode(pairings={finite:?}, essential={essential:?}, representatives={})",
            if self.representatives {
                "True"
            } else {
                "False"
            }
        )
    }
}

// ====== The module =======================================================

/// Persistent homology of sparse filtered chain complexes, computed with `phlite`.
#[pymodule(name = "phlite", gil_used = false)]
fn phlite_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<FilteredComplex>()?;
    m.add_class::<Barcode>()?;
    Ok(())
}
