//! The `phlite.cloud` submodule: persistent homology of point clouds and distance matrices, by [`phlite_cloud`].
//!
//! Each pipeline is a file here, with its function, its `register` and its `DEFAULTS`; add a new one to
//! [`PIPELINES`] and [`register`].

use pyo3::prelude::*;

use crate::algorithm::AlgorithmDefaults;

mod rips;

const DOC: &str =
    "Persistent homology of point clouds and distance matrices: `rips` (Vietoris–Rips).";
pyo3_stub_gen::module_doc!("phlite.cloud", "{}", DOC);

/// This submodule's pipelines (see [`super::pipelines`]).
pub(crate) const PIPELINES: &[AlgorithmDefaults] = &[rips::DEFAULTS];

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__doc__", DOC)?;
    rips::register(m)?;
    Ok(())
}
