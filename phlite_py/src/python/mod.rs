//! The `phlite` Python module (feature `python`).
//!
//! * [`ChainComplex`] holds a validated chain complex (no filtration), and how to translate between Python cells and
//!   cell indices.
//! * [`FilteredComplex`] is a chain complex with a filtration attached, ready to reduce.
//! * [`Barcode`] holds the result, and translates cell indices back to cells.

use numpy::{
    ndarray::Array2, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use ordered_float::NotNan;
use phlite::columns::ColumnRepresentation;
use phlite::{
    barcode::Bar as PhliteBar,
    fields::{Z11, Z13, Z2, Z3, Z5, Z7},
};
use phlite_pyutils::{
    add_algorithm_class, parse_method, AlgorithmConfig, AlgorithmDefaults, AlgorithmVisitor,
    Columns,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyList, PyTuple},
};
use pyo3_stub_gen::derive::*;

use crate::{
    complex::{Cell, Chain, Complex, ComplexError, RawChain},
    field::IntField,
};

/// Run `$body` with `$c` bound to the inner value of `$any` (an `AnyChain` or `AnyComplex`), whatever its field.
macro_rules! dispatch {
    ($enum:ident, $any:expr, $c:ident => $body:expr) => {
        match $any {
            $enum::Z2($c) => $body,
            $enum::Z3($c) => $body,
            $enum::Z5($c) => $body,
            $enum::Z7($c) => $body,
            $enum::Z11($c) => $body,
            $enum::Z13($c) => $body,
        }
    };
}

mod chain;
mod filtered;
pub use chain::ChainComplex;
pub use filtered::{Barcode, FilteredComplex};

// ====== Coefficient fields ===============================================

/// The supported moduli, for error messages.
const MODULI: &str = "2, 3, 5, 7, 11 or 13";

/// A chain complex over one of the supported fields.
/// Each field is a separate monomorphisation of the whole reduction, so we keep the list short.
pub(crate) enum AnyChain {
    Z2(Chain<Z2>),
    Z3(Chain<Z3>),
    Z5(Chain<Z5>),
    Z7(Chain<Z7>),
    Z11(Chain<Z11>),
    Z13(Chain<Z13>),
}

/// A filtered complex over one of the supported fields.
pub(crate) enum AnyComplex {
    Z2(Complex<Z2>),
    Z3(Complex<Z3>),
    Z5(Complex<Z5>),
    Z7(Complex<Z7>),
    Z11(Complex<Z11>),
    Z13(Complex<Z13>),
}

impl AnyChain {
    /// Build a chain complex over Z/`modulus`.
    /// The outer error is for an unsupported modulus, the inner for an invalid complex.
    fn build(raw: RawChain, modulus: u64) -> PyResult<Result<Self, ComplexError>> {
        Ok(match modulus {
            2 => Chain::from_raw(raw).map(Self::Z2),
            3 => Chain::from_raw(raw).map(Self::Z3),
            5 => Chain::from_raw(raw).map(Self::Z5),
            7 => Chain::from_raw(raw).map(Self::Z7),
            11 => Chain::from_raw(raw).map(Self::Z11),
            13 => Chain::from_raw(raw).map(Self::Z13),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported modulus {other}, expected one of {MODULI}"
                )))
            }
        })
    }

    fn filter(&self, values: Option<Vec<f64>>) -> Result<AnyComplex, ComplexError> {
        Ok(match self {
            Self::Z2(c) => AnyComplex::Z2(c.filter(values)?),
            Self::Z3(c) => AnyComplex::Z3(c.filter(values)?),
            Self::Z5(c) => AnyComplex::Z5(c.filter(values)?),
            Self::Z7(c) => AnyComplex::Z7(c.filter(values)?),
            Self::Z11(c) => AnyComplex::Z11(c.filter(values)?),
            Self::Z13(c) => AnyComplex::Z13(c.filter(values)?),
        })
    }

    fn extend(&self, partial: Vec<Option<f64>>) -> Result<Vec<f64>, ComplexError> {
        dispatch!(Self, self, c => c.extend(partial))
    }

    fn n_cells(&self) -> usize {
        dispatch!(Self, self, c => c.n_cells())
    }

    fn n_entries(&self) -> usize {
        dispatch!(Self, self, c => c.n_entries())
    }

    fn top_dimension(&self) -> Option<usize> {
        dispatch!(Self, self, c => c.top_dimension())
    }

    fn modulus(&self) -> u64 {
        fn of<CF: IntField>(_: &Chain<CF>) -> u64 {
            CF::MODULUS
        }
        dispatch!(Self, self, c => of(c))
    }
}

impl AnyComplex {
    /// Compute the barcode (without the GIL).
    fn persistence(
        &self,
        py: Python<'_>,
        max_dim: Option<usize>,
        method: &str,
        algorithm: &AlgorithmConfig,
        representatives: bool,
    ) -> PyResult<BarData> {
        let method = parse_method(method)?;
        dispatch!(Self, self, complex => py.detach(|| {
            algorithm.run(Run {
                complex,
                max_dim,
                method,
                representatives,
            })
        }))
    }

    fn n_cells(&self) -> usize {
        dispatch!(Self, self, c => c.n_cells())
    }

    fn value(&self, cell: u32) -> f64 {
        dispatch!(Self, self, c => c.value(Cell(cell)))
    }
}

/// Compute a barcode with whichever algorithm is configured.
struct Run<'a, CF> {
    complex: &'a Complex<CF>,
    max_dim: Option<usize>,
    method: phlite::barcode::Method,
    representatives: bool,
}

impl<CF: IntField> AlgorithmVisitor for Run<'_, CF> {
    type Output = BarData;

    fn visit<R: ColumnRepresentation>(
        self,
        algorithm: phlite::barcode::Algorithm<'_, R>,
    ) -> BarData {
        let bars = self
            .complex
            .barcode(self.max_dim, self.method, algorithm, self.representatives);
        BarData::new(self.complex, bars, self.max_dim, self.representatives)
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
pub(crate) fn read_ints(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    // Lists and tuples first: they're how columns usually come, one per cell, so this is the hot path of
    // `from_columns`, and the checks below cost more than reading a short column
    let fast = if let Ok(list) = obj.cast::<PyList>() {
        Some(collect_ints(list.iter(), list.len()))
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        Some(collect_ints(tuple.iter(), tuple.len()))
    } else {
        None
    };
    if let Some(ints) = fast {
        return ints.map_err(|e| not_ints(obj, name, e));
    }
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
    obj.extract().map_err(|e| not_ints(obj, name, e))
}

fn collect_ints<'py>(
    items: impl Iterator<Item = Bound<'py, PyAny>>,
    len: usize,
) -> PyResult<Vec<i64>> {
    let mut ints = Vec::with_capacity(len);
    for item in items {
        ints.push(item.extract::<i64>()?);
    }
    Ok(ints)
}

fn not_ints(obj: &Bound<'_, PyAny>, name: &str, cause: PyErr) -> PyErr {
    let err = PyTypeError::new_err(format!("{name} must be a sequence of integers"));
    err.set_cause(obj.py(), Some(cause));
    err
}

/// Read a sequence of floats, reading float numpy arrays directly.
pub(crate) fn read_floats(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
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

/// Split a `(dimension, column)` pair, given as a tuple or list.
fn split_pair<'py>(item: Bound<'py, PyAny>, j: usize) -> PyResult<(i64, Bound<'py, PyAny>)> {
    let pair = if let Ok(tuple) = item.cast::<PyTuple>() {
        (tuple.len() == 2).then(|| (tuple.get_item(0), tuple.get_item(1)))
    } else if let Ok(list) = item.cast::<PyList>() {
        (list.len() == 2).then(|| (list.get_item(0), list.get_item(1)))
    } else {
        None
    };
    let Some((dim, column)) = pair else {
        return Err(PyTypeError::new_err(format!(
            "pairs[{j}] must be a (dimension, column) pair"
        )));
    };
    let dim = dim?.extract::<i64>().map_err(|e| {
        let err = PyTypeError::new_err(format!("pairs[{j}][0] must be an integer (a dimension)"));
        err.set_cause(item.py(), Some(e));
        err
    })?;
    Ok((dim, column?))
}

/// Where `read_columns` finds the dimension of each cell.
pub(crate) enum Dimensions<'a, 'py> {
    /// A separate sequence of dimensions.
    Given(&'a Bound<'py, PyAny>),
    /// Each entry is a `(dimension, column)` pair.
    Paired,
}

/// Parse the columns of the boundary matrix, given as a sequence of columns or of `(dimension, column)` pairs,
/// according to `dimensions`.
pub(crate) fn read_columns(
    columns: &Bound<'_, PyAny>,
    dimensions: Dimensions<'_, '_>,
    coefficients: Option<&Bound<'_, PyAny>>,
) -> PyResult<RawChain> {
    let (paired, name) = match dimensions {
        Dimensions::Given(_) => (false, "boundaries"),
        Dimensions::Paired => (true, "pairs"),
    };
    let mut raw = RawChain {
        indptr: vec![0],
        ..Default::default()
    };
    let mut coeffs: Option<Vec<i64>> = coefficients.map(|_| vec![]);
    let mut coeff_iter = coefficients.map(|c| c.try_iter()).transpose()?;

    for (j, item) in columns.try_iter()?.enumerate() {
        let item = item?;
        let column = if paired {
            let (dim, column) = split_pair(item, j)?;
            raw.dimensions.push(dim);
            column
        } else {
            item
        };
        let label = if paired {
            format!("pairs[{j}][1]")
        } else {
            format!("boundaries[{j}]")
        };
        let column = read_ints(&column, &label)?;

        if let (Some(iter), Some(coeffs)) = (coeff_iter.as_mut(), coeffs.as_mut()) {
            let Some(c) = iter.next() else {
                return Err(PyValueError::new_err(format!(
                    "coefficients has fewer entries than {name} (no entry for column {j})"
                )));
            };
            let c = read_ints(&c?, &format!("coefficients[{j}]"))?;
            if c.len() != column.len() {
                return Err(PyValueError::new_err(format!(
                    "coefficients[{j}] has length {} but {label} has length {}",
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
            return Err(PyValueError::new_err(format!(
                "coefficients has more entries than {name}"
            )));
        }
    }
    raw.coefficients = coeffs;
    if let Dimensions::Given(dims) = dimensions {
        raw.dimensions = read_ints(dims, "dimensions")?;
    }
    Ok(raw)
}

// ====== Bars =============================================================

/// A bar, with the complex's data copied in so that the barcode is self-contained.
pub(crate) struct Record {
    birth: u32,
    death: Option<u32>,
    dimension: usize,
    birth_value: f64,
    /// Infinite for essential bars.
    death_value: f64,
    /// `(cell, coefficient)` pairs, in decreasing filtration order.
    representative: Option<Vec<(u32, u64)>>,
}

/// The bars of a barcode, in terms of cell indices (the Python `Barcode` translates them to cells).
pub(crate) struct BarData {
    /// Sorted by birth cell.
    pub(crate) bars: Vec<Record>,
    pub(crate) n_dims: usize,
    pub(crate) representatives: bool,
}

impl BarData {
    pub(crate) fn new<CF: IntField>(
        complex: &Complex<CF>,
        bars: Vec<PhliteBar<Cell, NotNan<f64>, CF>>,
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
                    let mut rep: Vec<(u32, u64)> = rep
                        .into_iter()
                        .map(|(coeff, cell)| (cell.0, coeff.to_int()))
                        .collect();
                    // phlite gives no particular order; list in decreasing filtration order, so the birth cell is first
                    rep.sort_by(|(a, _), (b, _)| {
                        let value = |c: u32| complex.value(Cell(c));
                        value(*b).total_cmp(&value(*a)).then(b.cmp(a))
                    });
                    rep
                }),
            })
            .collect();
        bars.sort_unstable_by_key(|bar| bar.birth);
        Self {
            bars,
            n_dims,
            representatives,
        }
    }

    pub(crate) fn diagram_array<'py>(
        &self,
        py: Python<'py>,
        dim: usize,
    ) -> Bound<'py, PyArray2<f64>> {
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

    /// The bar born at cell `birth`, if any.
    pub(crate) fn find(&self, birth: u32) -> Option<&Record> {
        self.bars
            .binary_search_by_key(&birth, |bar| bar.birth)
            .ok()
            .map(|i| &self.bars[i])
    }

    /// Every bar as a [`Bar`], with cells converted to Python by `cell`.
    pub(crate) fn make_bars<'py>(
        &self,
        py: Python<'py>,
        cell: impl Fn(u32) -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Vec<Bar>> {
        self.bars
            .iter()
            .map(|bar| Bar::from_record(py, bar, &cell))
            .collect()
    }

    pub(crate) fn repr(&self, name: &str) -> String {
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
            "{name}(pairings={finite:?}, essential={essential:?}, representatives={})",
            if self.representatives {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// A chain as a list of `(cell, coefficient)` pairs.
pub(crate) fn chain<'py>(
    py: Python<'py>,
    rep: &[(u32, u64)],
    cell: impl Fn(u32) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for &(c, coeff) in rep {
        list.append((cell(c)?, coeff))?;
    }
    Ok(list)
}

// ====== Bar ==============================================================

/// A bar of a barcode: a homology class with its birth and death.
///
/// Cells are ints for a complex built from columns or arrays, and your own objects for one built from cells.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite")]
pub struct Bar {
    dimension: usize,
    birth: Py<PyAny>,
    death: Option<Py<PyAny>>,
    birth_value: f64,
    death_value: f64,
    representative: Option<Py<PyList>>,
}

impl Bar {
    /// Convert a [`Record`], with cells converted to Python by `cell`.
    pub(crate) fn from_record<'py>(
        py: Python<'py>,
        bar: &Record,
        cell: impl Fn(u32) -> PyResult<Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Bar {
            dimension: bar.dimension,
            birth: cell(bar.birth)?.unbind(),
            death: bar.death.map(&cell).transpose()?.map(Bound::unbind),
            birth_value: bar.birth_value,
            death_value: bar.death_value,
            representative: bar
                .representative
                .as_ref()
                .map(|rep| chain(py, rep, &cell))
                .transpose()?
                .map(Bound::unbind),
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Bar {
    /// The homology dimension (the dimension of the birth cell).
    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    /// The cell whose entry creates the class.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing")))]
    fn birth(&self, py: Python<'_>) -> Py<PyAny> {
        self.birth.clone_ref(py)
    }

    /// The cell whose entry kills the class, or `None` for an essential bar.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[typing.Any]", imports = ("typing")))]
    fn death(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.death.as_ref().map(|d| d.clone_ref(py))
    }

    /// The filtration value of the birth cell.
    #[getter]
    fn birth_value(&self) -> f64 {
        self.birth_value
    }

    /// The filtration value of the death cell, or `inf` for an essential bar.
    #[getter]
    fn death_value(&self) -> f64 {
        self.death_value
    }

    /// `death_value - birth_value`.
    #[getter]
    fn persistence(&self) -> f64 {
        self.death_value - self.birth_value
    }

    /// Whether the bar is infinite.
    #[getter]
    fn is_essential(&self) -> bool {
        self.death.is_none()
    }

    /// A representative cycle as a list of `(cell, coefficient)` pairs in decreasing filtration order
    /// (so the first cell is the birth cell), or `None` if representatives were not requested.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[builtins.list[tuple[typing.Any, builtins.int]]]", imports = ("builtins", "typing")))]
    fn representative<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyList>> {
        self.representative.as_ref().map(|r| r.bind(py).clone())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let death = match &self.death {
            Some(d) => d.bind(py).repr()?.to_string(),
            None => "None".to_string(),
        };
        Ok(format!(
            "Bar(dimension={}, birth={}, death={death}, birth_value={}, death_value={})",
            self.dimension,
            self.birth.bind(py).repr()?,
            self.birth_value,
            self.death_value
        ))
    }
}

// ====== The module =======================================================

/// Persistent homology of sparse filtered chain complexes, computed with `phlite`.
#[pymodule(name = "phlite", gil_used = false)]
fn phlite_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<ChainComplex>()?;
    m.add_class::<FilteredComplex>()?;
    m.add_class::<Barcode>()?;
    m.add_class::<Bar>()?;
    add_algorithm_class(
        m,
        "phlite",
        AlgorithmDefaults {
            // Explicit complexes often need many additions per column (dense or cubical complexes, homology), where
            // adding stored columns of R is much faster than recomputing them from V, and a dense pivot column (the
            // complex knows the rank of every cell) is much faster than merging. Columns are stored sorted, so
            // caching them only saves a copy, which costs more than it saves.
            columns: Columns::Pivot,
            cache: false,
            store_r: true,
        },
    )?;
    Ok(())
}
