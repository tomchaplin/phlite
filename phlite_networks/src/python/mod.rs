//! The `phlite.networks` extension module (feature `python`), `phlite.networks._networks`.
//!
//! Each pipeline is a file here, with its function, its `register` and its `DEFAULTS` (see `phlite_bindings` for
//! how to write one); add a new one to [`PIPELINES`] and [`_networks`].

use phlite_bindings::AlgorithmDefaults;
use pyo3::prelude::*;

mod grpph;

const DOC: &str = "Persistent homology of networks: `grpph` (grounded persistent path homology of weighted digraphs).";
pyo3_stub_gen::module_doc!("phlite.networks", "{}", DOC);

/// This package's pipelines, registered with `phlite.core` on import.
const PIPELINES: &[AlgorithmDefaults] = &[grpph::DEFAULTS];

#[pymodule(gil_used = false)]
fn _networks(m: &Bound<'_, PyModule>) -> PyResult<()> {
    phlite_bindings::init_package(m.py(), "phlite-networks", env!("CARGO_PKG_VERSION"), PIPELINES)?;
    pyo3_log::init();
    m.setattr("__doc__", DOC)?;
    phlite_bindings::add_functions(m, "phlite.networks", |m| {
        grpph::register(m)?;
        Ok(())
    })?;
    Ok(())
}
