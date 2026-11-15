//! The shared Python bindings of phlite.
//!
//! phlite's Python packages are separate wheels sharing the namespace package `phlite`: `phlite-core`
//! (`phlite.core`, built from this crate with the feature `extension`), and one per kind of input, each built from
//! its own crate: `phlite.complexes` (`phlite_complexes`), `phlite.clouds` (`phlite_clouds`), `phlite.networks`
//! (`phlite_networks`) and `phlite.images` (`phlite_images`). Each is its own extension module, with its own copy of
//! this crate, so they can't share Rust classes (each module would have its own type objects). Instead `phlite.core`
//! owns the classes every package shares, `Algorithm` and `Barcode`, and the others reach them through Python, with
//! the helpers here:
//!
//! * **Versions.** Every package is released in lockstep on the workspace's version, and pins `phlite-core` to it.
//!   [`init_package`], called first thing in a package's `#[pymodule]`, imports `phlite.core`, refuses to load against
//!   any other version (with an error saying how to fix it), and registers the package's pipelines.
//! * **Algorithms.** `phlite.core.Algorithm._spec()` returns the algorithm as plain Python data (see
//!   [`spec::spec_to_dict`]), which [`algorithm_from_py`] reads back and resolves against the pipeline's
//!   [`AlgorithmDefaults`]. Take the `algorithm` argument as `Option<Bound<'_, PyAny>>` and type it in the stubs as
//!   `typing.Optional[phlite.core.Algorithm]`.
//! * **Barcodes.** Build the bars with [`SortedBars::new`] (without the GIL), then [`SortedBars::into_python`]
//!   converts them to Python lists and hands them to the private constructor `phlite.core.Barcode._from_parts`.
//!   Type the result in the stubs as `phlite.core.Barcode`. A package with its own barcode class records the
//!   algorithm with [`algorithm_to_py`].
//! * **Registry.** [`init_package`] registers each pipeline's defaults with `phlite.core`, which is how
//!   `phlite.core.defaults()` and `Algorithm.resolve(pipeline)` know them.
//!
//! [`parse_method`] reads the `method` argument.
//!
//! # Adding a pipeline
//!
//! In its area's crate (e.g. `phlite_clouds`), next to the Rust implementation: a file `src/python/<pipeline>.rs`
//! with the function and its `DEFAULTS` (an [`AlgorithmDefaults`] named `"<area>.<pipeline>"`, whose values its
//! docstring states in a line `Pipeline defaults (`'<area>.<pipeline>'`): columns=..., cache=..., store_r=...`), added
//! to the `PIPELINES` and `register` of `src/python/mod.rs`. A typical function:
//! 1. resolves `algorithm` with [`algorithm_from_py`] against its `DEFAULTS`, and reads `method` with
//!    [`parse_method`];
//! 2. computes its bars without the GIL (`py.detach`), running an [`AlgorithmVisitor`] with
//!    [`ResolvedAlgorithm::run`] and sorting the bars into [`SortedBars`];
//! 3. returns [`SortedBars::into_python`].

pub mod algorithm;
mod bars;
#[cfg(feature = "extension")]
mod core;
pub mod spec;

use phlite::barcode::Method;
use pyo3::{
    exceptions::{PyImportError, PyTypeError, PyValueError},
    prelude::*,
    sync::PyOnceLock,
    types::PyDict,
};

pub use algorithm::{
    AlgorithmDefaults, AlgorithmError, AlgorithmSpec, AlgorithmVisitor, Columns, Kind,
    ResolvedAlgorithm,
};
pub use bars::SortedBars;
use spec::{spec_from_dict, spec_to_dict};

/// The version every phlite package is released on (the workspace's).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

impl From<AlgorithmError> for PyErr {
    fn from(error: AlgorithmError) -> Self {
        PyValueError::new_err(error.0)
    }
}

// ====== Reaching phlite.core =============================================

/// The attributes of `phlite.core` that the other packages call, each looked up once per extension module.
#[derive(Clone, Copy)]
pub(crate) enum CoreAttribute {
    BarcodeFromParts,
    AlgorithmFromSpec,
    RegisterPipeline,
}

pub(crate) fn core_attribute(py: Python<'_>, attribute: CoreAttribute) -> PyResult<&Bound<'_, PyAny>> {
    static CELLS: [PyOnceLock<Py<PyAny>>; 3] = [PyOnceLock::new(), PyOnceLock::new(), PyOnceLock::new()];
    let path: &[&str] = match attribute {
        CoreAttribute::BarcodeFromParts => &["Barcode", "_from_parts"],
        CoreAttribute::AlgorithmFromSpec => &["Algorithm", "_from_spec"],
        CoreAttribute::RegisterPipeline => &["_register_pipeline"],
    };
    CELLS[attribute as usize]
        .get_or_try_init(py, || {
            let mut object = py.import("phlite.core")?.into_any();
            for name in path {
                object = object.getattr(*name)?;
            }
            Ok::<_, PyErr>(object.unbind())
        })
        .map(|object| object.bind(py))
}

/// Set up the package `package` (its pip name, e.g. `"phlite-clouds"`) at `version` (its `CARGO_PKG_VERSION`): check
/// that the installed `phlite-core` has the same version, raising an `ImportError` that says how to fix it otherwise,
/// then register `pipelines` with it. Call this first thing in the package's `#[pymodule]`.
pub fn init_package(
    py: Python<'_>,
    package: &str,
    version: &str,
    pipelines: &[AlgorithmDefaults],
) -> PyResult<()> {
    let fix = format!("install matching versions, e.g. `pip install phlite-core=={version} {package}=={version}`");
    let core = py.import("phlite.core").map_err(|e| {
        let err = PyImportError::new_err(format!("{package} {version} needs phlite-core {version}: {fix}"));
        err.set_cause(py, Some(e));
        err
    })?;
    let found: Option<String> = core.getattr("__version__").and_then(|v| v.extract()).ok();
    if found.as_deref() != Some(version) {
        return Err(PyImportError::new_err(format!(
            "{package} {version} needs phlite-core {version}, but phlite-core {} is installed: {fix}",
            found.as_deref().unwrap_or("(of unknown version)")
        )));
    }
    let register = core_attribute(py, CoreAttribute::RegisterPipeline)?;
    for defaults in pipelines {
        register.call1((
            defaults.pipeline,
            defaults.columns.name(),
            defaults.cache,
            defaults.store_r,
            defaults.row_ranks,
            defaults.union_find,
        ))?;
    }
    Ok(())
}

/// Add a package's functions to its extension module `m` with `register`, so that they report the public module they
/// are re-exported from, `public` (e.g. `"phlite.clouds"`), as their `__module__`, rather than the private extension
/// module (`phlite.clouds._clouds`): pyo3 takes it from the module's `__name__` when a function is added.
pub fn add_functions(
    m: &Bound<'_, PyModule>,
    public: &str,
    register: impl FnOnce(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let name = m.getattr("__name__")?;
    m.setattr("__name__", public)?;
    let result = register(m);
    m.setattr("__name__", name)?;
    result
}

// ====== Arguments ========================================================

/// Parse the `method` argument. In the stubs, type it as `typing.Literal['cohomology', 'homology']`.
pub fn parse_method(name: &str) -> PyResult<Method> {
    match name {
        "cohomology" => Ok(Method::Cohomology),
        "homology" => Ok(Method::Homology),
        other => Err(PyValueError::new_err(format!(
            "Unknown method {other:?}, expected 'cohomology' or 'homology'"
        ))),
    }
}

/// Read the `algorithm` argument: a `phlite.core.Algorithm`, or `None` for `Algorithm.serial()`, resolved against the
/// pipeline's `defaults`. Anything else is a `TypeError`.
pub fn algorithm_from_py(
    algorithm: Option<&Bound<'_, PyAny>>,
    defaults: &AlgorithmDefaults,
) -> PyResult<ResolvedAlgorithm> {
    let spec = match algorithm {
        None => AlgorithmSpec::default(),
        Some(obj) if obj.is_none() => AlgorithmSpec::default(),
        Some(obj) => {
            let not_an_algorithm = || {
                let type_name = obj
                    .get_type()
                    .name()
                    .map_or_else(|_| "?".to_string(), |n| n.to_string());
                PyTypeError::new_err(format!(
                    "algorithm must be a phlite.core.Algorithm (e.g. Algorithm.parallel()) or None, not {type_name}"
                ))
            };
            let spec = obj.getattr("_spec").map_err(|_| not_an_algorithm())?.call0()?;
            spec_from_dict(spec.cast::<PyDict>().map_err(|_| not_an_algorithm())?)?
        }
    };
    Ok(spec.resolve(defaults)?)
}

/// `algorithm` as a `phlite.core.Algorithm`, with every option set as resolved, for a barcode to record.
pub fn algorithm_to_py<'py>(
    py: Python<'py>,
    algorithm: &ResolvedAlgorithm,
) -> PyResult<Bound<'py, PyAny>> {
    core_attribute(py, CoreAttribute::AlgorithmFromSpec)?
        .call1((spec_to_dict(py, &algorithm.to_spec())?,))
}

// ====== Stubs ============================================================

/// What a package's stub generator writes: the stubs of its module `module` (e.g. `"phlite.clouds"`), as
/// `python/phlite/<area>/__init__.pyi` under `manifest_dir` (the crate's `CARGO_MANIFEST_DIR`). The extension module
/// itself is private (`phlite.clouds._clouds`); the module's `__init__.py` re-exports it, so the stubs describe that.
///
/// Call it from a function in the package's own crate, which is where its classes and functions are gathered.
pub fn stub_info_for(module: &str, manifest_dir: &str) -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let manifest_dir = std::path::Path::new(manifest_dir);
    let config = pyo3_stub_gen::StubInfo::from_pyproject_toml(manifest_dir.join("pyproject.toml"))?.config;
    pyo3_stub_gen::StubInfo::from_project_root(module.to_string(), manifest_dir.join("python"), true, config)
}

#[cfg(feature = "extension")]
pub use crate::core::stub_info;

#[cfg(test)]
mod tests {
    use std::path::Path;

    /// Every Python package pins `phlite-core` to exactly the workspace's version, and the meta-package `phlite`
    /// (whose version can't come from Cargo) is on it, so that bumping the workspace's version is all a release needs
    /// besides these lines.
    #[test]
    fn packages_are_pinned_to_the_workspace_version() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let pin = format!("\"phlite-core=={}\"", super::VERSION);
        for package in ["phlite_complexes", "phlite_clouds", "phlite_networks", "phlite_images"] {
            let pyproject = std::fs::read_to_string(root.join(package).join("pyproject.toml")).unwrap();
            assert!(pyproject.contains(&pin), "{package}/pyproject.toml should depend on {pin}");
        }
        let meta = std::fs::read_to_string(root.join("phlite_meta").join("pyproject.toml")).unwrap();
        assert!(meta.contains(&format!("version = \"{}\"", super::VERSION)));
        for package in ["core", "complexes", "clouds", "networks", "images"] {
            let pin = format!("\"phlite-{package}=={}\"", super::VERSION);
            assert!(meta.contains(&pin), "phlite_meta/pyproject.toml should depend on {pin}");
        }
    }
}
