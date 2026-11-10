//! Shared helpers for `phlite`'s Python bindings (`phlite_grpph` and `phlite_rips`).
//!
//! * [`AlgorithmArg`] and [`parse_method`] turn the `algorithm`, `threads` and `method` arguments into a [`phlite::barcode::Algorithm`] and [`Method`].
//! * [`Barcode`] is the Python result type, built from a list of [`Bar`]s via [`SortedBars`].
//!   Add it to your module with [`add_barcode_class`].

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Bar, Method},
    reduction::{Clearing, Standard, Storage},
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyTuple},
};
use pyo3_stub_gen::derive::*;

/// Python type of the `algorithm` argument, for `#[gen_stub(override_type(type_repr = ...))]`.
/// Attribute arguments must be literals, so copy this rather than referencing it.
pub const ALGORITHM_TYPE: &str =
    "typing.Literal['standard', 'clearing', 'parallel', 'parallel-lock-free']";
/// Python type of the `method` argument, for `#[gen_stub(override_type(type_repr = ...))]`.
/// Attribute arguments must be literals, so copy this rather than referencing it.
pub const METHOD_TYPE: &str = "typing.Literal['cohomology', 'homology']";

// ====== Choosing an algorithm ============================================

/// A reduction algorithm, as named from Python.
///
/// Use [`parse`](Self::parse) on the `algorithm` argument, then [`run`](Self::run) to get an [`Algorithm`] (and a thread pool, if needed).
/// In the stubs, type the argument as [`ALGORITHM_TYPE`].
#[derive(Debug, Clone, Copy)]
pub enum AlgorithmArg {
    Standard,
    Clearing,
    Parallel(Storage),
}

impl AlgorithmArg {
    /// Parse the name of an algorithm.
    pub fn parse(name: &str) -> PyResult<Self> {
        match name {
            "standard" => Ok(Self::Standard),
            "clearing" => Ok(Self::Clearing),
            "parallel" => Ok(Self::Parallel(Storage::Locking)),
            #[cfg(feature = "lock-free")]
            "parallel-lock-free" => Ok(Self::Parallel(Storage::LockFree)),
            #[cfg(not(feature = "lock-free"))]
            "parallel-lock-free" => Err(PyValueError::new_err(
                "phlite was built without the 'lock-free' feature",
            )),
            other => Err(PyValueError::new_err(format!(
                "Unknown algorithm {other:?}, expected one of 'standard', 'clearing', 'parallel' or 'parallel-lock-free'"
            ))),
        }
    }

    /// Run `f` with the corresponding [`Algorithm`], on a pool of `threads` threads if given.
    /// Errors if `threads` is given for a serial algorithm.
    /// With `threads`, `f` itself runs on the pool, so any other rayon work it does (such as building a basis) uses it too.
    pub fn run<T: Send>(
        self,
        threads: Option<usize>,
        f: impl FnOnce(Algorithm<'_>) -> T + Send,
    ) -> PyResult<T> {
        let storage = match (self, threads) {
            (Self::Standard, None) => return Ok(f(Algorithm::Standard(Standard::new()))),
            (Self::Clearing, None) => return Ok(f(Algorithm::Clearing(Clearing::new()))),
            (Self::Standard | Self::Clearing, Some(_)) => return Err(PyValueError::new_err(
                "threads can only be set for the 'parallel' and 'parallel-lock-free' algorithms",
            )),
            (Self::Parallel(storage), _) => storage,
        };
        let algorithm = Clearing::new().parallel().storage(storage);
        match threads {
            None => Ok(f(Algorithm::ParallelClearing(algorithm))),
            Some(0) => Err(PyValueError::new_err("threads must be positive")),
            Some(n) => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| {
                        PyValueError::new_err(format!("Could not build thread pool: {e}"))
                    })?;
                let algorithm = algorithm.in_pool(&pool);
                Ok(pool.install(|| f(Algorithm::ParallelClearing(algorithm))))
            }
        }
    }
}

/// Parse the `method` argument. In the stubs, type it as [`METHOD_TYPE`].
pub fn parse_method(name: &str) -> PyResult<Method> {
    match name {
        "cohomology" => Ok(Method::Cohomology),
        "homology" => Ok(Method::Homology),
        other => Err(PyValueError::new_err(format!(
            "Unknown method {other:?}, expected 'cohomology' or 'homology'"
        ))),
    }
}

// ====== The result type ==================================================

/// A persistence barcode, split by dimension.
///
/// Bars of length zero are omitted.
/// Within each dimension, bars are sorted by birth (and then death).
#[gen_stub_pyclass]
// The module is set when the class is added to a module, see `add_barcode_class`
#[pyclass(frozen)]
pub struct Barcode {
    essential: Py<PyList>,
    pairings: Py<PyList>,
    essential_representatives: Option<Py<PyList>>,
    pairing_representatives: Option<Py<PyList>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Barcode {
    /// `essential[k]` lists the birth times of the infinite bars in dimension `k`.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[builtins.list[builtins.float]]", imports = ("builtins")))]
    fn essential(&self, py: Python<'_>) -> Py<PyList> {
        self.essential.clone_ref(py)
    }

    /// `pairings[k]` lists the `(birth, death)` times of the finite bars in dimension `k`.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[builtins.list[tuple[builtins.float, builtins.float]]]", imports = ("builtins")))]
    fn pairings(&self, py: Python<'_>) -> Py<PyList> {
        self.pairings.clone_ref(py)
    }

    /// `essential_representatives[k][i]` is a cycle representing the bar `essential[k][i]`,
    /// or `None` if representatives were not requested.
    ///
    /// A cycle is a list of cells, each given as a tuple of vertices (indices, or node labels for a networkx graph).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[builtins.list[builtins.list[builtins.list[tuple[builtins.int, ...]]]]]", imports = ("builtins", "typing")))]
    fn essential_representatives(&self, py: Python<'_>) -> Option<Py<PyList>> {
        self.essential_representatives
            .as_ref()
            .map(|reps| reps.clone_ref(py))
    }

    /// `pairing_representatives[k][i]` is a cycle representing the bar `pairings[k][i]`,
    /// or `None` if representatives were not requested.
    ///
    /// A cycle is a list of cells, each given as a tuple of vertices (indices, or node labels for a networkx graph).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[builtins.list[builtins.list[builtins.list[tuple[builtins.int, ...]]]]]", imports = ("builtins", "typing")))]
    fn pairing_representatives(&self, py: Python<'_>) -> Option<Py<PyList>> {
        self.pairing_representatives
            .as_ref()
            .map(|reps| reps.clone_ref(py))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let counts = |list: &Py<PyList>| -> PyResult<Vec<usize>> {
            list.bind(py)
                .iter()
                .map(|dim| dim.len())
                .collect::<PyResult<_>>()
        };
        Ok(format!(
            "Barcode(essential={:?}, pairings={:?}, representatives={})",
            counts(&self.essential)?,
            counts(&self.pairings)?,
            if self.essential_representatives.is_some() {
                "True"
            } else {
                "False"
            },
        ))
    }
}

/// A representative cycle, as a list of cells each given by its vertices, if requested
type Rep = Option<Vec<Vec<u32>>>;

/// The bars of a barcode, sorted and split by dimension, still on the Rust side.
/// Build with [`new`](Self::new) (which can be done without the GIL), then convert with [`into_python`](Self::into_python).
pub struct SortedBars {
    essential: Vec<Vec<(f64, Rep)>>,
    pairings: Vec<Vec<((f64, f64), Rep)>>,
}

impl SortedBars {
    /// Split `bars` into dimensions `0..n_dims`, dropping zero-length bars, then sort.
    /// Cells with infinite filtration value are treated as never appearing.
    /// `cell_info` gives the dimension of a cell and the vertices it contains.
    pub fn new<C, CF>(
        bars: Vec<Bar<C, NotNan<f64>, CF>>,
        n_dims: usize,
        cell_info: impl Fn(&C) -> (usize, Vec<u32>),
    ) -> Self {
        let mut essential = vec![vec![]; n_dims];
        let mut pairings = vec![vec![]; n_dims];
        for bar in bars {
            if bar.is_zero_length() {
                continue;
            }
            let (dim, _) = cell_info(&bar.birth);
            let rep = bar.representative.map(|rep| {
                rep.iter()
                    .map(|(_coeff, cell)| cell_info(cell).1)
                    .collect::<Vec<_>>()
            });
            let birth = bar.birth_value.into_inner();
            // Nothing appears at time infinity, so classes born then never exist and those dying then are essential
            if birth == f64::INFINITY {
                continue;
            }
            match bar.death.map(|(_, death)| death.into_inner()) {
                None | Some(f64::INFINITY) => essential[dim].push((birth, rep)),
                Some(death) => pairings[dim].push(((birth, death), rep)),
            }
        }
        for dim in essential.iter_mut() {
            dim.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        }
        for dim in pairings.iter_mut() {
            dim.sort_by(|a, b| {
                (a.0 .0.total_cmp(&b.0 .0))
                    .then_with(|| a.0 .1.total_cmp(&b.0 .1))
                    .then_with(|| a.1.cmp(&b.1))
            });
        }
        Self {
            essential,
            pairings,
        }
    }

    /// Convert to a Python [`Barcode`]; `representatives` must match whether the bars have representatives.
    ///
    /// If `labels` is given then vertex `i` appears in representatives as `labels[i]`, otherwise as the integer `i`.
    pub fn into_python(
        self,
        py: Python<'_>,
        representatives: bool,
        labels: Option<&[Py<PyAny>]>,
    ) -> PyResult<Barcode> {
        let cell_to_python = |cell: &Vec<u32>| -> PyResult<Bound<'_, PyTuple>> {
            match labels {
                None => PyTuple::new(py, cell),
                Some(labels) => PyTuple::new(py, cell.iter().map(|&v| labels[v as usize].bind(py))),
            }
        };
        let reps_to_python = |dims: Vec<Vec<&Rep>>| -> PyResult<Py<PyList>> {
            let dims = dims
                .into_iter()
                .map(|bars| {
                    let bars = bars
                        .into_iter()
                        .map(|rep| {
                            let cells = rep
                                .as_ref()
                                .expect("representative was requested")
                                .iter()
                                .map(cell_to_python)
                                .collect::<PyResult<Vec<_>>>()?;
                            PyList::new(py, cells)
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    PyList::new(py, bars)
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, dims)?.unbind())
        };

        let essential: Vec<Vec<f64>> = self
            .essential
            .iter()
            .map(|dim| dim.iter().map(|(birth, _)| *birth).collect())
            .collect();
        let pairings: Vec<Vec<(f64, f64)>> = self
            .pairings
            .iter()
            .map(|dim| dim.iter().map(|(bar, _)| *bar).collect())
            .collect();

        Ok(Barcode {
            essential: PyList::new(py, essential)?.unbind(),
            pairings: PyList::new(py, pairings)?.unbind(),
            essential_representatives: representatives
                .then(|| reps_to_python(only_reps(&self.essential)))
                .transpose()?,
            pairing_representatives: representatives
                .then(|| reps_to_python(only_reps(&self.pairings)))
                .transpose()?,
        })
    }
}

fn only_reps<T>(dims: &[Vec<(T, Rep)>]) -> Vec<Vec<&Rep>> {
    dims.iter()
        .map(|bars| bars.iter().map(|(_, rep)| rep).collect())
        .collect()
}

// ====== Registering ======================================================

/// Add [`Barcode`] to the module `m`, reporting `public_module` (the name users import it from) as its `__module__`.
pub fn add_barcode_class(m: &Bound<'_, PyModule>, public_module: &str) -> PyResult<()> {
    m.add_class::<Barcode>()?;
    m.getattr("Barcode")?.setattr("__module__", public_module)?;
    Ok(())
}
