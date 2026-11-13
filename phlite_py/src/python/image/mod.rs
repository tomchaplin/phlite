//! The `phlite.image` submodule: persistent homology of images, by [`phlite_image`]. Nothing yet.
//!
//! Each pipeline is a file here, with its function, its `register` and its `DEFAULTS`; add a new one to
//! [`PIPELINES`] and [`register`].

use pyo3::prelude::*;

use crate::algorithm::AlgorithmDefaults;

const DOC: &str = "Persistent homology of images (cubical complexes). Nothing here yet.";
pyo3_stub_gen::module_doc!("phlite.image", "{}", DOC);

/// This submodule's pipelines (see [`super::pipelines`]).
pub(crate) const PIPELINES: &[AlgorithmDefaults] = &[];

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__doc__", DOC)?;
    Ok(())
}
