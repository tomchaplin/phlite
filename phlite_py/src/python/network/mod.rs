//! The `phlite.network` submodule: persistent homology of networks, by [`phlite_network`].
//!
//! Each pipeline is a file here, with its function, its `register` and its `DEFAULTS`; add a new one to
//! [`PIPELINES`] and [`register`].

use pyo3::prelude::*;

use crate::algorithm::AlgorithmDefaults;

mod grpph;

const DOC: &str =
    "Persistent homology of networks: `grpph` (grounded persistent path homology of weighted digraphs).";
pyo3_stub_gen::module_doc!("phlite.network", "{}", DOC);

/// This submodule's pipelines (see [`super::pipelines`]).
pub(crate) const PIPELINES: &[AlgorithmDefaults] = &[grpph::DEFAULTS];

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__doc__", DOC)?;
    grpph::register(m)?;
    Ok(())
}
