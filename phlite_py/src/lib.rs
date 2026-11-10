//! Python bindings for `phlite`: persistent homology of an explicit sparse filtered chain complex.
//!
//! The Rust side is split in two:
//! * [`complex`] stores the complex as a pair of sorted sparse matrices (boundary and coboundary) and implements
//!   `phlite`'s matrix traits for them. It has no Python dependency, so can be tested with `cargo test`.
//! * `python` (feature `python`) parses Python input into a [`complex::RawComplex`] and exposes the
//!   `FilteredComplex` and `Barcode` classes.
//!
//! Why not let users implement matrix oracles in Python? Every column access would need the GIL, a Python call and a
//! Python object per entry, which costs far more than the reduction itself, and serialises the parallel algorithms.
//! Since `phlite` recomputes columns on demand rather than storing them, this cost is paid many times per column.
//! Materialising the complex once and reducing it in Rust avoids all of that.

pub mod complex;
pub mod field;
#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
