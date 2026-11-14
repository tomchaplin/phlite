//! The `phlite.images` extension module (feature `python`), `phlite.images._images`.
//!
//! Each pipeline is a file here, with its function, its `register` and its `DEFAULTS` (see `phlite_bindings` for
//! how to write one); add a new one to [`PIPELINES`] and [`_images`].

use phlite_bindings::AlgorithmDefaults;
use pyo3::prelude::*;

const DOC: &str = "Persistent homology of images (cubical complexes). Nothing here yet.";
pyo3_stub_gen::module_doc!("phlite.images", "{}", DOC);

/// This package's pipelines, registered with `phlite.core` on import.
const PIPELINES: &[AlgorithmDefaults] = &[];

#[pymodule(gil_used = false)]
fn _images(m: &Bound<'_, PyModule>) -> PyResult<()> {
    phlite_bindings::init_package(m.py(), "phlite-images", env!("CARGO_PKG_VERSION"), PIPELINES)?;
    pyo3_log::init();
    m.setattr("__doc__", DOC)?;
    Ok(())
}
