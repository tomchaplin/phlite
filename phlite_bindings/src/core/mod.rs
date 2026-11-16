//! The `phlite.core` extension module (feature `extension`): the classes every phlite package shares,
//! [`AlgorithmConfig`] (`Algorithm`) and [`Barcode`], and the registry of pipelines ([`registry`]).

use pyo3::prelude::*;

mod algorithm;
mod barcode;
mod extended;
mod registry;

pub use algorithm::AlgorithmConfig;
pub use barcode::Barcode;
pub use extended::ExtendedBarcode;

const DOC: &str = "The core of phlite's Python packages: `Algorithm`, to choose how a barcode is computed, \
                   `Barcode`, what most pipelines return, `ExtendedBarcode`, extended persistence, and `defaults()`, every installed pipeline's defaults.";
pyo3_stub_gen::module_doc!("phlite.core", "{}", DOC);
pyo3_stub_gen::module_variable!("phlite.core", "__version__", String);

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__doc__", DOC)?;
    // Every phlite package is released on this version, and checks it on import (see `init_package`)
    m.add("__version__", crate::VERSION)?;
    m.add_class::<AlgorithmConfig>()?;
    m.add_class::<Barcode>()?;
    m.add_class::<ExtendedBarcode>()?;
    m.add_function(wrap_pyfunction!(registry::defaults, m)?)?;
    m.add_function(wrap_pyfunction!(registry::_register_pipeline, m)?)?;
    Ok(())
}

/// What `phlite_bindings_stub_gen` writes: `python/phlite/core/__init__.pyi`, the stubs of `phlite.core`, which
/// re-exports the extension module `phlite.core._core`.
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    crate::stub_info_for("phlite.core", env!("CARGO_MANIFEST_DIR"))
}
