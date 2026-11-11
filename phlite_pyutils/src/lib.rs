//! Shared helpers for `phlite`'s Python bindings (`phlite_grpph`, `phlite_rips` and `phlite_py`).
//!
//! * [`AlgorithmConfig`] is the Python `Algorithm` class, a builder for every algorithm and option, with presets.
//!   Take the `algorithm` argument as `Option<Bound<'_, AlgorithmConfig>>`, defaulting to
//!   [`AlgorithmConfig::default`] (`Algorithm.serial()`), and call [`AlgorithmConfig::run`] with an
//!   [`AlgorithmVisitor`] to run the configured [`phlite::barcode::Algorithm`]. Add the class to your module with
//!   [`add_algorithm_class`].
//! * [`parse_method`] reads the `method` argument.
//! * [`Barcode`] is the Python result type, built from a list of [`Bar`]s via [`SortedBars`].
//!   Add it to your module with [`add_barcode_class`].

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Bar, Method},
    columns::{ColumnRepresentation, HeapColumns, MergedColumns},
    reduction::{Clearing, Standard, Storage},
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyTuple},
};
use pyo3_stub_gen::derive::*;

/// Python type of the `method` argument, for `#[gen_stub(override_type(type_repr = ...))]`.
/// Attribute arguments must be literals, so copy this rather than referencing it.
pub const METHOD_TYPE: &str = "typing.Literal['cohomology', 'homology']";

// ====== Choosing an algorithm ============================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Standard,
    Clearing,
    ParallelClearing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Columns {
    Merged,
    Heap,
}

impl Columns {
    fn name(self) -> &'static str {
        match self {
            Self::Merged => "merged",
            Self::Heap => "heap",
        }
    }
}

fn storage_name(storage: Storage) -> &'static str {
    match storage {
        #[cfg(feature = "lock-free")]
        Storage::LockFree => "lock-free",
        _ => "locking",
    }
}

/// A reduction algorithm and its options.
///
/// Start from a preset, `Algorithm.serial()` or `Algorithm.parallel()`, which are the recommended configurations,
/// or from one of the algorithms themselves, `Algorithm.standard()`, `Algorithm.clearing()` or
/// `Algorithm.parallel_clearing()`, then adjust it with the `with_*` methods (each returns a new `Algorithm`):
///
/// ```python
/// Algorithm.parallel(threads=8)
/// Algorithm.clearing().with_columns("heap").with_skip_zero_apparent_pairs(False)
/// ```
///
/// Pass the result as the `algorithm` argument. Every algorithm gives the same barcode; only the time and memory differ.
#[gen_stub_pyclass]
// The module is set when the class is added to a module, see `add_algorithm_class`
#[pyclass(frozen, eq, name = "Algorithm")]
#[derive(Debug, Clone, PartialEq)]
pub struct AlgorithmConfig {
    kind: Kind,
    columns: Columns,
    skip_zero_apparent_pairs: bool,
    dimension_order: Option<Vec<usize>>,
    storage: Storage,
    threads: Option<usize>,
}

/// What to do with an [`Algorithm`]: implement this to call your (generic) barcode function,
/// then pass it to [`AlgorithmConfig::run`].
///
/// This is a trait rather than a closure since the column representation is a type parameter.
pub trait AlgorithmVisitor: Send {
    type Output: Send;
    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output;
}

impl AlgorithmConfig {
    fn base(kind: Kind) -> Self {
        Self {
            kind,
            columns: Columns::Merged,
            // The barcode omits zero-length bars anyway, so there's no point recording them
            skip_zero_apparent_pairs: kind != Kind::Standard,
            dimension_order: None,
            storage: Storage::Locking,
            threads: None,
        }
    }

    fn require(&self, kinds: &[Kind], option: &str) -> PyResult<()> {
        if kinds.contains(&self.kind) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "{option} can't be set for the {} algorithm",
                self.name()
            )))
        }
    }

    /// Run `visitor` with the configured [`Algorithm`], on a pool of `threads` threads if set.
    /// With threads set, the visitor itself runs on the pool, so any other rayon work it does (such as building a basis)
    /// uses it too.
    pub fn run<V: AlgorithmVisitor>(&self, visitor: V) -> PyResult<V::Output> {
        match self.columns {
            Columns::Merged => self.run_with(MergedColumns, visitor),
            Columns::Heap => self.run_with(HeapColumns, visitor),
        }
    }

    fn build_clearing<R: ColumnRepresentation>(&self, columns: R) -> Clearing<R> {
        let mut clearing = Clearing::new().with_columns(columns);
        // (`mut` for the dimension order)
        if let Some(order) = &self.dimension_order {
            clearing = clearing.with_dimension_order(order.iter().copied());
        }
        clearing.skip_zero_apparent_pairs(self.skip_zero_apparent_pairs)
    }

    fn run_with<R: ColumnRepresentation, V: AlgorithmVisitor>(
        &self,
        columns: R,
        visitor: V,
    ) -> PyResult<V::Output> {
        match self.kind {
            Kind::Standard => {
                Ok(visitor.visit(Algorithm::Standard(Standard::new().with_columns(columns))))
            }
            Kind::Clearing => Ok(visitor.visit(Algorithm::Clearing(self.build_clearing(columns)))),
            Kind::ParallelClearing => {
                let algorithm = self
                    .build_clearing(columns)
                    .parallel()
                    .storage(self.storage);
                match self.threads {
                    None => Ok(visitor.visit(Algorithm::ParallelClearing(algorithm))),
                    Some(n) => {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(n)
                            .build()
                            .map_err(|e| {
                                PyValueError::new_err(format!("Could not build thread pool: {e}"))
                            })?;
                        let algorithm = algorithm.in_pool(&pool);
                        Ok(pool.install(|| visitor.visit(Algorithm::ParallelClearing(algorithm))))
                    }
                }
            }
        }
    }
}

/// `Algorithm.serial()`, the default for an omitted `algorithm` argument.
impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self::serial()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl AlgorithmConfig {
    /// The recommended serial algorithm: clearing, skipping zero-length apparent pairs.
    #[staticmethod]
    fn serial() -> Self {
        Self::base(Kind::Clearing)
    }

    /// The recommended parallel algorithm: clearing with each dimension reduced in parallel, skipping zero-length
    /// apparent pairs, on `threads` threads (default: all cores).
    #[staticmethod]
    #[pyo3(signature = (threads = None))]
    fn parallel(threads: Option<usize>) -> PyResult<Self> {
        Self::base(Kind::ParallelClearing).with_threads(threads)
    }

    /// The standard algorithm, with no optimisations.
    #[staticmethod]
    fn standard() -> Self {
        Self::base(Kind::Standard)
    }

    /// The standard algorithm with the clearing optimisation. Skips zero-length apparent pairs by default.
    #[staticmethod]
    fn clearing() -> Self {
        Self::base(Kind::Clearing)
    }

    /// Clearing, with the columns of each dimension reduced in parallel.
    /// Skips zero-length apparent pairs by default, and uses all cores and locking storage.
    #[staticmethod]
    fn parallel_clearing() -> Self {
        Self::base(Kind::ParallelClearing)
    }

    /// How the column being reduced is represented: `'merged'` (the default) or `'heap'`.
    fn with_columns(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Literal['merged', 'heap']", imports = ("typing")))]
        columns: &str,
    ) -> PyResult<Self> {
        let columns = match columns {
            "merged" => Columns::Merged,
            "heap" => Columns::Heap,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown columns {other:?}, expected 'merged' or 'heap'"
                )))
            }
        };
        Ok(Self {
            columns,
            ..self.clone()
        })
    }

    /// Whether to leave zero-length apparent pairs out of the diagram (clearing algorithms only).
    ///
    /// Such pairs give zero-length bars, which the barcode omits anyway, but in many filtrations nearly every pair is
    /// one, so recording them dominates the time and memory of the reduction.
    #[pyo3(signature = (skip = true))]
    fn with_skip_zero_apparent_pairs(&self, skip: bool) -> PyResult<Self> {
        if skip {
            self.require(
                &[Kind::Clearing, Kind::ParallelClearing],
                "skip_zero_apparent_pairs",
            )?;
        }
        Ok(Self {
            skip_zero_apparent_pairs: skip,
            ..self.clone()
        })
    }

    /// The order in which to reduce dimensions (clearing algorithms only), or `None` to detect it from the matrix.
    ///
    /// Clearing saves work when dimensions are reduced in the direction the matrix maps them (ascending for
    /// cohomology, descending for homology), which is what is detected. Only the dimensions listed are reduced, so
    /// list every dimension the barcode needs.
    #[pyo3(signature = (dimension_order))]
    fn with_dimension_order(&self, dimension_order: Option<Vec<usize>>) -> PyResult<Self> {
        self.require(&[Kind::Clearing, Kind::ParallelClearing], "dimension_order")?;
        Ok(Self {
            dimension_order,
            ..self.clone()
        })
    }

    /// The concurrent data structures for the parallel algorithm: `'locking'` (the default) or `'lock-free'`.
    fn with_storage(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Literal['locking', 'lock-free']", imports = ("typing")))]
        storage: &str,
    ) -> PyResult<Self> {
        self.require(&[Kind::ParallelClearing], "storage")?;
        let storage = match storage {
            "locking" => Storage::Locking,
            #[cfg(feature = "lock-free")]
            "lock-free" => Storage::LockFree,
            #[cfg(not(feature = "lock-free"))]
            "lock-free" => {
                return Err(PyValueError::new_err(
                    "phlite was built without the 'lock-free' feature",
                ))
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown storage {other:?}, expected 'locking' or 'lock-free'"
                )))
            }
        };
        Ok(Self {
            storage,
            ..self.clone()
        })
    }

    /// The number of threads for the parallel algorithm, or `None` for all cores.
    #[pyo3(signature = (threads))]
    fn with_threads(&self, threads: Option<usize>) -> PyResult<Self> {
        if threads.is_some() {
            self.require(&[Kind::ParallelClearing], "threads")?;
        }
        if threads == Some(0) {
            return Err(PyValueError::new_err("threads must be positive"));
        }
        Ok(Self {
            threads,
            ..self.clone()
        })
    }

    /// `'standard'`, `'clearing'` or `'parallel-clearing'`.
    #[getter]
    fn name(&self) -> &'static str {
        match self.kind {
            Kind::Standard => "standard",
            Kind::Clearing => "clearing",
            Kind::ParallelClearing => "parallel-clearing",
        }
    }

    /// `'merged'` or `'heap'`.
    #[getter]
    fn columns(&self) -> &'static str {
        self.columns.name()
    }

    /// Whether zero-length apparent pairs are skipped.
    #[getter]
    fn skip_zero_apparent_pairs(&self) -> bool {
        self.skip_zero_apparent_pairs
    }

    /// The order dimensions are reduced in, or `None` if detected.
    #[getter]
    fn dimension_order(&self) -> Option<Vec<usize>> {
        self.dimension_order.clone()
    }

    /// `'locking'` or `'lock-free'` for the parallel algorithm, else `None`.
    #[getter]
    fn storage(&self) -> Option<&'static str> {
        (self.kind == Kind::ParallelClearing).then(|| storage_name(self.storage))
    }

    /// The number of threads for the parallel algorithm (`None`: all cores).
    #[getter]
    fn threads(&self) -> Option<usize> {
        self.threads
    }

    /// The Python expression that builds this algorithm.
    fn __repr__(&self) -> String {
        let mut out = format!(
            "Algorithm.{}()",
            match self.kind {
                Kind::Standard => "standard",
                Kind::Clearing => "clearing",
                Kind::ParallelClearing => "parallel_clearing",
            }
        );
        let default = Self::base(self.kind);
        if self.columns != default.columns {
            out += &format!(".with_columns({:?})", self.columns.name());
        }
        if self.skip_zero_apparent_pairs != default.skip_zero_apparent_pairs {
            out += &format!(
                ".with_skip_zero_apparent_pairs({})",
                if self.skip_zero_apparent_pairs {
                    "True"
                } else {
                    "False"
                }
            );
        }
        if let Some(order) = &self.dimension_order {
            out += &format!(".with_dimension_order({order:?})");
        }
        if self.storage != default.storage {
            out += &format!(".with_storage({:?})", storage_name(self.storage));
        }
        if let Some(threads) = self.threads {
            out += &format!(".with_threads({threads})");
        }
        out
    }
}

/// Add the `Algorithm` class to your module, as `public_module.Algorithm`.
pub fn add_algorithm_class(m: &Bound<'_, PyModule>, public_module: &str) -> PyResult<()> {
    m.add_class::<AlgorithmConfig>()?;
    m.getattr("Algorithm")?
        .setattr("__module__", public_module)?;
    Ok(())
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
