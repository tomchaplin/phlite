//! The registry of pipelines: every function that computes a barcode, with the [`AlgorithmDefaults`] it fills in
//! unset `Algorithm` options with.
//!
//! Each area declares its own pipelines, next to their functions (`cloud::PIPELINES`, ...); add a new pipeline's
//! defaults there, then use them in its function with [`algorithm_from_py`](super::shared::algorithm_from_py). Python
//! sees the registry through `phlite.defaults()` and `Algorithm.resolve(pipeline)`, and the book's table of
//! defaults is tested against it (`python-tests/tests/test_algorithm.py`), as is each function's docstring.

use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use pyo3_stub_gen::derive::*;

use super::{cloud, image, network, AlgorithmConfig, COMPLEX_DEFAULTS};
use crate::algorithm::{AlgorithmDefaults, AlgorithmSpec, Kind};

/// Every pipeline, in the order `phlite.defaults()` lists them.
pub(crate) fn all() -> impl Iterator<Item = &'static AlgorithmDefaults> {
    std::iter::once(&COMPLEX_DEFAULTS)
        .chain(cloud::PIPELINES)
        .chain(network::PIPELINES)
        .chain(image::PIPELINES)
}

/// The pipeline called `name`.
pub(crate) fn find(name: &str) -> PyResult<&'static AlgorithmDefaults> {
    all()
        .find(|defaults| defaults.pipeline == name)
        .ok_or_else(|| {
            let known: Vec<String> = all().map(|d| format!("'{}'", d.pipeline)).collect();
            PyValueError::new_err(format!(
                "Unknown pipeline '{name}', expected one of {}",
                known.join(", ")
            ))
        })
}

/// Every pipeline's defaults: a dict from the pipeline's name (`'complex'` for `FilteredComplex.persistence`,
/// `'cloud.rips'` for `phlite.cloud.rips`, and so on) to its presets, `{'serial': ..., 'parallel': ...}`, as they run
/// there (`Algorithm.serial().resolve(name)` and `Algorithm.parallel().resolve(name)`).
#[gen_stub_pyfunction(module = "phlite")]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "builtins.dict[builtins.str, builtins.dict[builtins.str, Algorithm]]", imports = ("builtins")))]
pub(crate) fn defaults(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for defaults in all() {
        let presets = PyDict::new(py);
        for (name, kind) in [
            ("serial", Kind::Clearing),
            ("parallel", Kind::ParallelClearing),
        ] {
            let resolved = AlgorithmSpec::new(kind).resolve(defaults)?;
            presets.set_item(name, AlgorithmConfig::from(&resolved))?;
        }
        out.set_item(defaults.pipeline, presets)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_names_are_unique() {
        let names: Vec<_> = all().map(|d| d.pipeline).collect();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(names.len(), unique.len(), "{names:?}");
    }
}
