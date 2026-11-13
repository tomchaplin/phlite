//! Persistent homology of networks, implemented with [`phlite`].
//!
//! * [`grpph`]: grounded persistent path homology (GrPPH) of weighted digraphs.
//!
//! In Python this crate is the `phlite.network` submodule of the `phlite` package (its bindings are in
//! `phlite_py/src/python/network`). The `phlite_network_grpph` binary times GrPPH on a complete digraph.
//!
//! This crate was called `phlite_grpph` (and its Python package the same) before it gathered every network
//! filtration.

pub mod grpph;
