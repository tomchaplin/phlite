//! Persistent homology of point clouds and distance matrices, implemented with [`phlite`].
//!
//! * [`rips`]: the Vietoris–Rips filtration of a distance matrix, à la Ripser.
//!
//! In Python this crate is the `phlite.cloud` submodule of the `phlite` package (its bindings are in
//! `phlite_py/src/python/cloud`). The `phlite_cloud_rips` binary computes Rips persistence of a distance matrix read
//! from standard input.
//!
//! This crate was called `phlite_rips` (and its Python package the same) before it gathered every point-cloud
//! filtration.

pub mod rips;
