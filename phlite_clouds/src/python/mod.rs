//! The `phlite.clouds` extension module (feature `python`), `phlite.clouds._clouds`.
//!
//! Each pipeline is a file here, with its function, its `register` and its `DEFAULTS` (see `phlite_bindings` for
//! how to write one); add a new one to [`PIPELINES`] and [`_clouds`].

use phlite_bindings::AlgorithmDefaults;
use pyo3::prelude::*;

mod alpha;
mod rips;

const DOC: &str = "Persistent homology of point clouds and distance matrices: `rips` (Vietoris–Rips), `alpha` (alpha \
                   complexes in the plane and in space) and `delaunay_cech` (Delaunay–Čech complexes).";
pyo3_stub_gen::module_doc!("phlite.clouds", "{}", DOC);

/// This package's pipelines, registered with `phlite.core` on import.
const PIPELINES: &[AlgorithmDefaults] =
    &[rips::DEFAULTS, rips::EXPLICIT_DEFAULTS, alpha::DEFAULTS, alpha::CECH_DEFAULTS];

#[pymodule(gil_used = false)]
fn _clouds(m: &Bound<'_, PyModule>) -> PyResult<()> {
    phlite_bindings::init_package(m.py(), "phlite-clouds", env!("CARGO_PKG_VERSION"), PIPELINES)?;
    pyo3_log::init();
    m.setattr("__doc__", DOC)?;
    phlite_bindings::add_functions(m, "phlite.clouds", |m| {
        rips::register(m)?;
        alpha::register(m)?;
        Ok(())
    })?;
    Ok(())
}
