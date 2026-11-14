//! Persistent homology of networks, implemented with [`phlite`].
//!
//! * [`grpph`]: grounded persistent path homology (GrPPH) of weighted digraphs.
//!
//! With the feature `python`, this crate is also the Python package `phlite-networks`, `phlite.networks` (its
//! bindings are in `src/python`, one file per pipeline; see `phlite_bindings` for how to add one). The
//! `phlite_networks_grpph` binary times GrPPH on a complete digraph.
//!
//! This crate was called `phlite_grpph` (and its Python package the same) before it gathered every network
//! filtration.

pub mod grpph;
#[cfg(feature = "python")]
mod python;

/// What `phlite_networks_stub_gen` writes: `python/phlite/networks/__init__.pyi`.
#[cfg(feature = "python")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    phlite_bindings::stub_info_for("phlite.networks", env!("CARGO_MANIFEST_DIR"))
}
