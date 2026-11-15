//! Persistent homology of point clouds and distance matrices, implemented with [`phlite`].
//!
//! * [`rips`]: the Vietoris–Rips filtration of a distance matrix (dense or sparse) or points, à la Ripser.
//! * [`alpha`]: the alpha filtration of points in the plane or in space (as in GUDHI), from their Delaunay
//!   triangulation.
//!
//! With the feature `python`, this crate is also the Python package `phlite-clouds`, `phlite.clouds` (its bindings
//! are in `src/python`, one file per pipeline; see `phlite_bindings` for how to add one). The `phlite_clouds_rips`
//! binary computes Rips persistence of a distance matrix read from standard input.
//!
//! This crate was called `phlite_rips` (and its Python package the same) before it gathered every point-cloud
//! filtration.

pub mod alpha;
pub mod error;
pub mod rips;

pub use error::CloudError;
#[cfg(feature = "python")]
mod python;

/// What `phlite_clouds_stub_gen` writes: `python/phlite/clouds/__init__.pyi`.
#[cfg(feature = "python")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    phlite_bindings::stub_info_for("phlite.clouds", env!("CARGO_MANIFEST_DIR"))
}
