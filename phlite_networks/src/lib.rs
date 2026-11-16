//! Persistent homology of networks, implemented with [`phlite`].
//!
//! * [`pph`]: persistent path homology (PPH, Chowdhury and Mémoli) of weighted digraphs, in any dimension.
//! * [`grpph`]: grounded persistent path homology (GrPPH) of weighted digraphs, and (without grounding) PPH of the
//!   shortest-path filtration, in dimensions 0 and 1.
//! * [`flag`]: directed flag complexes (as flagser), grounded or not, and ordered-tuple complexes (Turner's directed
//!   Rips), in any dimension.
//! * [`dowker`]: Dowker sink and source complexes.
//! * [`undirected`]: flag (clique) complexes of undirected graphs, filtered by edge weights, node values (lower-star)
//!   or both, by `phlite_clouds`' Rips machinery.
//!
//! [`flag`] and [`dowker`] are built on [`lazy`], which turns a complex described by its simplices' cofacets and facets
//! into coboundary and boundary oracles with Ripser's optimisations, on the digraphs of [`digraph`] and the packed
//! simplices of [`simplex`].
//!
//! Every constructor taking weighted edges checks them, and returns a [`NetworkError`] for an endpoint out of range, a
//! self-loop or a negative or `NaN` weight (an infinite weight is an edge that never appears).
//!
//! With the feature `python`, this crate is also the Python package `phlite-networks`, `phlite.networks` (its
//! bindings are in `src/python`, one file per pipeline; see `phlite_bindings` for how to add one). The
//! `phlite_networks_grpph` binary times GrPPH on a complete digraph, and `phlite_networks_pph` computes PPH of networks read
//! from stdin (see `matlab_compare/`).
//!
//! This crate was called `phlite_grpph` (and its Python package the same) before it gathered every network
//! filtration.

pub mod digraph;
pub mod dowker;
pub mod error;
pub mod flag;
pub mod grpph;
pub mod lazy;
pub mod pph;
pub mod simplex;
pub mod undirected;
#[cfg(feature = "python")]
mod python;

pub use error::NetworkError;

/// What `phlite_networks_stub_gen` writes: `python/phlite/networks/__init__.pyi`.
#[cfg(feature = "python")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    phlite_bindings::stub_info_for("phlite.networks", env!("CARGO_MANIFEST_DIR"))
}
