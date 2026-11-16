//! `Algorithm`: the reduction algorithm and its options, shared by every pipeline (see [`crate::algorithm`]).

use pyo3::{prelude::*, types::PyDict};
use pyo3_stub_gen::derive::*;

use crate::spec::{spec_from_dict, spec_to_dict};

use crate::algorithm::{
    storage_from_name, storage_name, AlgorithmSpec, Columns, Kind, ResolvedAlgorithm,
};

/// A reduction algorithm and its options, for every pipeline of every phlite package.
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
///
/// The column representation, caching and storing R are best chosen per kind of input, so each pipeline (an
/// explicit complex, `phlite.clouds.rips`, `phlite.networks.grpph`, ...) has its own defaults, and the presets leave
/// these options unset (`None`, shown as `<pipeline default>`) until the algorithm is used. `phlite.core.defaults()`
/// lists every pipeline's defaults, `algorithm.resolve(pipeline)` fills them in, and every barcode records the
/// algorithm that computed it (`barcode.algorithm`).
#[gen_stub_pyclass]
#[pyclass(frozen, eq, name = "Algorithm", module = "phlite.core")]
#[derive(Debug, Clone, PartialEq)]
pub struct AlgorithmConfig {
    pub(crate) spec: AlgorithmSpec,
}

impl AlgorithmConfig {
    fn with(&self, change: impl FnOnce(&mut AlgorithmSpec)) -> PyResult<Self> {
        let mut spec = self.spec.clone();
        change(&mut spec);
        spec.validate()?;
        Ok(Self { spec })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl AlgorithmConfig {
    /// The recommended serial algorithm: clearing, skipping zero-length apparent pairs. The same as `clearing()`
    /// (and shown as that), and the default for an omitted `algorithm` argument.
    #[staticmethod]
    fn serial() -> Self {
        Self {
            spec: AlgorithmSpec::new(Kind::Clearing),
        }
    }

    /// The recommended parallel algorithm: clearing with each dimension reduced in parallel, skipping zero-length
    /// apparent pairs, on `threads` threads (default: all cores). The same as
    /// `parallel_clearing().with_threads(threads)` (and shown as that).
    #[staticmethod]
    #[pyo3(signature = (threads = None))]
    fn parallel(threads: Option<usize>) -> PyResult<Self> {
        Self::parallel_clearing().with_threads(threads)
    }

    /// The standard algorithm, with no optimisations.
    #[staticmethod]
    fn standard() -> Self {
        Self {
            spec: AlgorithmSpec::new(Kind::Standard),
        }
    }

    /// The standard algorithm with the clearing optimisation. Skips zero-length apparent pairs by default.
    #[staticmethod]
    fn clearing() -> Self {
        Self::serial()
    }

    /// Clearing, with the columns of each dimension reduced in parallel.
    /// Skips zero-length apparent pairs by default, and uses all cores and locking storage.
    #[staticmethod]
    fn parallel_clearing() -> Self {
        Self {
            spec: AlgorithmSpec::new(Kind::ParallelClearing),
        }
    }

    /// How the column being reduced is represented: `'merged'`, `'heap'`, `'sorted'` or `'pivot'`, or `None` for the
    /// pipeline's default: `'pivot'` for serial clearing in `phlite.complexes` (otherwise `'merged'`), `'heap'` in
    /// `phlite.clouds.rips` and `'merged'` in `phlite.networks.grpph`. Every choice gives the same barcode; only time
    /// and memory differ.
    ///
    /// Reducing a column means repeatedly finding its pivot (largest entry) and adding other columns to cancel it.
    ///
    /// * `'merged'` never builds the column: it merges the columns added lazily, with one heap slot per column added,
    ///   so it never reads anything below the pivot. The safe choice, fast on every kind of matrix tried.
    /// * `'heap'` pushes every entry of every column added onto one heap. No sorting, so it is good when the columns
    ///   added are short and few (e.g. Vietoris–Rips cohomology), but slows down when many entries are added.
    /// * `'sorted'` keeps the column as one sorted vector and merges each batch of added columns into it. The fastest
    ///   when the columns added stay near the pivot (typically homology of simplicial or cell complexes, about 1.5×
    ///   faster than `'merged'`), but very slow when they reach far below it (15–35× slower on GrPPH or the cohomology
    ///   of cubical complexes). Measure before using it.
    /// * `'pivot'` keeps the column as a dense array over all rows, with a tree of bitsets marking which are present
    ///   (as PHAT does): adding an entry flips a bit, and the pivot is found with a few bit-scans. The fastest when
    ///   many columns are added to each, as for dense or cubical complexes, especially with `with_store_r()`. Needs a
    ///   matrix that knows the rank of every row in filtration order (`phlite`'s explicit complexes do, Vietoris–Rips
    ///   and GrPPH don't), and only serial clearing uses it: otherwise using it is an error (for an algorithm other
    ///   than serial clearing, as soon as it is asked for; for a pipeline without row ranks, when it runs).
    /// * `'dynamic'` is `'heap'` until a column's heap holds at least `threshold` terms (`None`: 4096) and the rows
    ///   popped from it so far averaged at least two terms each, after which each row is kept in it at most once, a
    ///   hash map holding the sum of its terms (flagser's working column). Opt-in: only worth it when columns pile up
    ///   the same rows over and over, as in very large directed flag complexes (24% faster than `'merged'` on one);
    ///   elsewhere it is slightly slower than `'heap'` and slower than `'merged'` where that wins.
    #[pyo3(signature = (columns, threshold = None))]
    fn with_columns(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Literal['merged', 'heap', 'sorted', 'pivot', 'dynamic']]", imports = ("typing")))]
        columns: Option<&str>,
        threshold: Option<usize>,
    ) -> PyResult<Self> {
        let columns = columns.map(Columns::from_name).transpose()?;
        if threshold.is_some() && columns != Some(Columns::Dynamic) {
            return Err(crate::AlgorithmError("A threshold is only used by 'dynamic' columns".into()).into());
        }
        self.with(|spec| {
            spec.columns = columns;
            spec.threshold = threshold;
        })
    }

    /// Whether to cache the columns of the boundary matrix as they are added during the reduction, or `None` for the
    /// pipeline's default: caching in `phlite.networks.grpph`, but not in `phlite.clouds.rips`, whose columns are
    /// cheap to recompute, nor in `phlite.complexes`, whose columns are stored sorted.
    ///
    /// Caching saves recomputing (and re-sorting) a column every time it is added, which pays off when columns are
    /// expensive to compute (about 3× faster for GrPPH), but keeps every column needed in a dimension in memory
    /// until that dimension is done (most of the memory for Vietoris–Rips). For `phlite.complexes.ChainComplex`, whose columns
    /// are stored sorted, it only saves copying them, and with the default pivot columns and stored R it is slower:
    /// about 20% on cell complexes and nearly 2× on cubical complexes.
    #[pyo3(signature = (cache = Some(true)))]
    fn with_cache(&self, cache: Option<bool>) -> PyResult<Self> {
        self.with(|spec| spec.cache = cache)
    }

    /// Whether to keep each reduced column of R = D V and add it directly when cancelling a pivot, rather than
    /// recompute it from V (serial clearing only), or `None` for the pipeline's default: storing R for serial
    /// clearing in `phlite.complexes`, but not in `phlite.clouds.rips` or `phlite.networks.grpph`, where almost every column is
    /// an apparent pair and needs no reduction.
    ///
    /// Only V is stored by default, so cancelling a pivot with an earlier column j adds one column of the boundary
    /// matrix for every term of V_j. Storing R (as PHAT does) adds R_j as one column instead, which pays off when V_j
    /// gets long, as in dense or cubical complexes, at the cost of memory for the stored columns. Vietoris–Rips
    /// cohomology, where almost every pair needs no reduction, gains little.
    #[pyo3(signature = (store = Some(true)))]
    fn with_store_r(&self, store: Option<bool>) -> PyResult<Self> {
        self.with(|spec| spec.store_r = store)
    }

    /// Whether to leave zero-length apparent pairs out of the diagram (clearing algorithms only).
    ///
    /// Such pairs give zero-length bars, which the barcode omits anyway, but in many filtrations nearly every pair is
    /// one, so recording them dominates the time and memory of the reduction.
    #[pyo3(signature = (skip = true))]
    fn with_skip_zero_apparent_pairs(&self, skip: bool) -> PyResult<Self> {
        self.with(|spec| spec.skip_zero_apparent_pairs = skip)
    }

    /// The order in which to reduce dimensions (clearing algorithms only), or `None` to detect it from the matrix.
    ///
    /// Clearing saves work when dimensions are reduced in the direction the matrix maps them (ascending for
    /// cohomology, descending for homology), which is what is detected. Only the dimensions listed are reduced, so
    /// list every dimension the barcode needs.
    #[pyo3(signature = (dimension_order))]
    fn with_dimension_order(&self, dimension_order: Option<Vec<usize>>) -> PyResult<Self> {
        // Unlike the other options, even `None` only makes sense for clearing
        self.spec
            .require(&[Kind::Clearing, Kind::ParallelClearing], "dimension_order")?;
        self.with(|spec| spec.dimension_order = dimension_order)
    }

    /// The concurrent data structures for the parallel algorithm: `'locking'` (the default) or `'lock-free'`.
    fn with_storage(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Literal['locking', 'lock-free']", imports = ("typing")))]
        storage: &str,
    ) -> PyResult<Self> {
        self.spec.require(&[Kind::ParallelClearing], "storage")?;
        let storage = storage_from_name(storage)?;
        self.with(|spec| spec.storage = storage)
    }

    /// The number of threads for the parallel algorithm (and for a parallel setup, see `with_parallel_setup`), or
    /// `None` for all cores.
    #[pyo3(signature = (threads))]
    fn with_threads(&self, threads: Option<usize>) -> PyResult<Self> {
        self.with(|spec| spec.threads = threads)
    }

    /// Whether to find dimension 0 by union-find rather than by reducing it, when computing by cohomology, or `None`
    /// for the pipeline's default: union-find in `phlite.clouds.rips` (as Ripser does) and in `phlite.images.cubical`
    /// (which also finds the top dimension by union-find on the dual, as Cubical Ripser does); other pipelines can't,
    /// and refuse it.
    ///
    /// The edges that kill a component then never enter the reduction (as clearing would leave them out), so only
    /// the dimensions above are reduced. Reducing dimension 0 is most of the time of a Vietoris–Rips barcode of points
    /// in the plane (2–3× slower than union-find), and of a cubical one. Representatives come from the union-find too
    /// (a component's: its birth vertex plus the oldest vertex of the component it joins).
    #[pyo3(signature = (union_find = Some(true)))]
    fn with_union_find(&self, union_find: Option<bool>) -> PyResult<Self> {
        self.with(|spec| spec.union_find = union_find)
    }

    /// Whether to do the work before the reduction in parallel (building the bases of the matrices, sorting cells for
    /// union-find, enumerating paths), or `None` (the default) to follow the algorithm: in parallel for parallel
    /// clearing, serially otherwise. So `Algorithm.serial().with_parallel_setup()` builds in parallel and reduces
    /// serially, and `Algorithm.parallel().with_parallel_setup(False)` the other way round. With `with_threads`, a
    /// parallel setup runs on that many threads (for a serial algorithm too), else on all cores. Pipelines with
    /// nothing to set up in parallel (explicit complexes, alpha, GrPPH) ignore it.
    #[pyo3(signature = (parallel = Some(true)))]
    fn with_parallel_setup(&self, parallel: Option<bool>) -> PyResult<Self> {
        self.with(|spec| spec.parallel_setup = parallel)
    }

    /// Whether the setup runs in parallel, or `None` to follow the algorithm.
    #[getter]
    fn parallel_setup(&self) -> Option<bool> {
        self.spec.parallel_setup
    }

    /// Whether dimension 0 is found by union-find, or `None` for the pipeline's default.
    #[getter]
    fn union_find(&self) -> Option<bool> {
        self.spec.union_find
    }

    /// `'standard'`, `'clearing'` or `'parallel-clearing'`.
    #[getter]
    fn name(&self) -> &'static str {
        self.spec.kind.name()
    }

    /// `'merged'`, `'heap'`, `'sorted'`, `'pivot'` or `'dynamic'`, or `None` for the pipeline's default.
    #[getter]
    fn columns(&self) -> Option<&'static str> {
        self.spec.columns.map(Columns::name)
    }

    /// The threshold of `'dynamic'` columns, or `None` for the default.
    #[getter]
    fn threshold(&self) -> Option<usize> {
        self.spec.threshold
    }

    /// Whether columns of the boundary matrix are cached, or `None` for the pipeline's default.
    #[getter]
    fn cache(&self) -> Option<bool> {
        self.spec.cache
    }

    /// Whether reduced columns of R are stored (serial clearing only), or `None` for the pipeline's default.
    #[getter]
    fn store_r(&self) -> Option<bool> {
        self.spec.store_r
    }

    /// Whether zero-length apparent pairs are skipped.
    #[getter]
    fn skip_zero_apparent_pairs(&self) -> bool {
        self.spec.skip_zero_apparent_pairs
    }

    /// The order dimensions are reduced in, or `None` if detected.
    #[getter]
    fn dimension_order(&self) -> Option<Vec<usize>> {
        self.spec.dimension_order.clone()
    }

    /// `'locking'` or `'lock-free'` for the parallel algorithm, else `None`.
    #[getter]
    fn storage(&self) -> Option<&'static str> {
        (self.spec.kind == Kind::ParallelClearing).then(|| storage_name(self.spec.storage))
    }

    /// The number of threads for the parallel algorithm (`None`: all cores).
    #[getter]
    fn threads(&self) -> Option<usize> {
        self.spec.threads
    }

    /// A copy with every option the pipeline would fill in set to its value there, so that it shows exactly what
    /// would run. `pipeline` is a name from `phlite.core.defaults()`: `'complexes'`
    /// (`phlite.complexes.FilteredComplex.persistence`), `'clouds.rips'`, `'networks.grpph'`, ... Raises a
    /// `ValueError` for an unknown pipeline (or one whose package isn't installed), or an option the pipeline can't
    /// take (such as `'pivot'` columns where the matrices don't know their row ranks).
    fn resolve(&self, py: Python<'_>, pipeline: &str) -> PyResult<Self> {
        let resolved = self.spec.resolve(&super::registry::find(py, pipeline)?)?;
        Ok(Self::from(&resolved))
    }

    /// The algorithm as plain Python data, for the other phlite packages to read (see `phlite_bindings`). A
    /// versioned format: the `"protocol"` key says which.
    #[gen_stub(skip)]
    fn _spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        spec_to_dict(py, &self.spec)
    }

    /// The algorithm described by `spec`, as `_spec()` gives it.
    #[staticmethod]
    #[gen_stub(skip)]
    fn _from_spec(spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            spec: spec_from_dict(spec)?,
        })
    }

    /// Every option, with those left to the pipeline shown as `<pipeline default>` (see `resolve`). Only the options
    /// that apply to the algorithm are shown.
    pub(crate) fn __repr__(&self) -> String {
        let spec = &self.spec;
        let python_bool = |b: bool| if b { "True" } else { "False" };
        let unset = "<pipeline default>";
        let mut fields = vec![format!("'{}'", spec.kind.name())];
        fields.push(format!(
            "columns={}",
            spec.columns
                .map_or(unset.to_string(), |c| format!("'{}'", c.name()))
        ));
        if let Some(threshold) = spec.threshold {
            fields.push(format!("threshold={threshold}"));
        }
        fields.push(format!("cache={}", spec.cache.map_or(unset, python_bool)));
        if spec.kind == Kind::Clearing {
            fields.push(format!(
                "store_r={}",
                spec.store_r.map_or(unset, python_bool)
            ));
        }
        if spec.kind != Kind::Standard {
            fields.push(format!(
                "skip_zero_apparent_pairs={}",
                python_bool(spec.skip_zero_apparent_pairs)
            ));
            fields.push(format!(
                "dimension_order={}",
                spec.dimension_order
                    .as_ref()
                    .map_or("None".to_string(), |order| format!("{order:?}"))
            ));
        }
        if spec.kind == Kind::ParallelClearing {
            fields.push(format!("storage='{}'", storage_name(spec.storage)));
        }
        if spec.kind == Kind::ParallelClearing || spec.threads.is_some() {
            fields.push(format!(
                "threads={}",
                spec.threads.map_or("None".to_string(), |t| t.to_string())
            ));
        }
        // Only shown when set
        if let Some(parallel) = spec.parallel_setup {
            fields.push(format!("parallel_setup={}", python_bool(parallel)));
        }
        // Only shown when set (or resolved for a pipeline that can use it)
        if let Some(union_find) = spec.union_find {
            fields.push(format!("union_find={}", python_bool(union_find)));
        }
        format!("Algorithm({})", fields.join(", "))
    }
}

impl From<&ResolvedAlgorithm> for AlgorithmConfig {
    /// The algorithm with every option set as resolved.
    fn from(resolved: &ResolvedAlgorithm) -> Self {
        Self {
            spec: resolved.to_spec(),
        }
    }
}
