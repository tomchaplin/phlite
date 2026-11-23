//! The registry of pipelines: every function of every installed phlite package that computes a barcode, with the
//! [`AlgorithmDefaults`] it fills in unset `Algorithm` options with.
//!
//! Each package registers its pipelines when it is imported (`phlite_bindings::init_package`, which calls
//! `_register_pipeline`). A pipeline is named after its package and function, `'<area>.<function>'` (e.g.
//! `'clouds.rips'`), or just `'<area>'`; [`find`] and `defaults()` import the packages they need, so the registry
//! knows every installed package's pipelines even before you import them.

use std::sync::Mutex;

use pyo3::{
    exceptions::{PyModuleNotFoundError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pyo3_stub_gen::derive::*;

use super::AlgorithmConfig;
use crate::algorithm::{AlgorithmDefaults, AlgorithmSpec, Columns, Kind};

/// The pipelines registered so far, in the order they were.
static REGISTRY: Mutex<Vec<AlgorithmDefaults>> = Mutex::new(Vec::new());

fn registered(name: &str) -> Option<AlgorithmDefaults> {
    REGISTRY
        .lock()
        .unwrap()
        .iter()
        .find(|defaults| defaults.pipeline == name)
        .copied()
}

/// Register a pipeline's defaults (for phlite's packages, when they are imported; not a public API, so not in the
/// stubs).
#[pyfunction]
pub(crate) fn _register_pipeline(
    name: &str,
    columns: &str,
    cache: bool,
    store_r: bool,
    row_ranks: bool,
    union_find: Option<bool>,
    collapse: Option<bool>,
) -> PyResult<()> {
    let mut registry = REGISTRY.lock().unwrap();
    let columns = Columns::from_name(columns)?;
    let entry = |pipeline| AlgorithmDefaults {
        pipeline,
        columns,
        cache,
        store_r,
        row_ranks,
        union_find,
        collapse,
    };
    match registry.iter_mut().find(|defaults| defaults.pipeline == name) {
        // Registered again (a module re-imported, say): keep the name, take the new defaults
        Some(defaults) => *defaults = entry(defaults.pipeline),
        // One small leak per pipeline per process, for `AlgorithmDefaults`' `&'static str`
        None => registry.push(entry(Box::leak(name.to_owned().into_boxed_str()))),
    }
    Ok(())
}

/// The pip package that provides `phlite.<area>`.
fn package(area: &str) -> String {
    format!("phlite-{area}")
}

/// The pipeline called `name`, importing its package (`phlite.<area>`, from the name's first part) if need be.
pub(crate) fn find(py: Python<'_>, name: &str) -> PyResult<AlgorithmDefaults> {
    if let Some(defaults) = registered(name) {
        return Ok(defaults);
    }
    let area = name.split('.').next().unwrap_or_default();
    if !area.is_empty() && area != "core" {
        let module = format!("phlite.{area}");
        if let Err(err) = py.import(module.as_str()) {
            // Only if it's this package that's missing, not something it imports
            let missing: Option<String> = err
                .is_instance_of::<PyModuleNotFoundError>(py)
                .then(|| err.value(py).getattr("name").ok()?.extract().ok())
                .flatten();
            if missing.as_deref() == Some(module.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "Unknown pipeline '{name}': {module} is not installed (pip install {})",
                    package(area)
                )));
            }
            return Err(err);
        }
    }
    registered(name).ok_or_else(|| {
        let known: Vec<String> = all(py)
            .map(|all| all.iter().map(|d| format!("'{}'", d.pipeline)).collect())
            .unwrap_or_default();
        PyValueError::new_err(format!(
            "Unknown pipeline '{name}', expected one of {}",
            known.join(", ")
        ))
    })
}

/// Every pipeline of every installed phlite package: import them all (the subpackages of the namespace package
/// `phlite`, other than `phlite.core`), so that they register themselves.
fn all(py: Python<'_>) -> PyResult<Vec<AlgorithmDefaults>> {
    let path = py.import("phlite")?.getattr("__path__")?;
    let mut areas: Vec<String> = vec![];
    for module in py.import("pkgutil")?.call_method1("iter_modules", (path,))?.try_iter()? {
        let module = module?;
        if module.getattr("ispkg")?.extract()? {
            areas.push(module.getattr("name")?.extract()?);
        }
    }
    areas.sort();
    for area in areas.iter().filter(|area| *area != "core") {
        py.import(format!("phlite.{area}"))?;
    }
    let mut all = REGISTRY.lock().unwrap().clone();
    all.sort_by_key(|defaults| defaults.pipeline);
    Ok(all)
}

/// Every installed pipeline's defaults: a dict from the pipeline's name (`'complexes'` for
/// `phlite.complexes.FilteredComplex.persistence`, `'clouds.rips'` for `phlite.clouds.rips`, and so on) to its
/// presets, `{'serial': ..., 'parallel': ...}`, as they run there (`Algorithm.serial().resolve(name)` and
/// `Algorithm.parallel().resolve(name)`). Imports every installed phlite package.
#[gen_stub_pyfunction(module = "phlite.core")]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "builtins.dict[builtins.str, builtins.dict[builtins.str, Algorithm]]", imports = ("builtins")))]
pub(crate) fn defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for defaults in all(py)? {
        let presets = PyDict::new(py);
        for (name, kind) in [("serial", Kind::Clearing), ("parallel", Kind::ParallelClearing)] {
            let resolved = AlgorithmSpec::new(kind).resolve(&defaults)?;
            presets.set_item(name, AlgorithmConfig::from(&resolved))?;
        }
        out.set_item(defaults.pipeline, presets)?;
    }
    Ok(out)
}
