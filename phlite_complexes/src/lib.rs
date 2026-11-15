//! Persistent homology of explicit sparse filtered chain complexes with `phlite`, and the `phlite.complexes` Python
//! package (feature `python`).
//!
//! * [`complex`] stores a chain complex as a sparse boundary matrix ([`complex::Chain`]) and, once a filtration is
//!   attached, as a pair of sorted sparse matrices (boundary and coboundary, [`complex::Complex`]) implementing
//!   `phlite`'s matrix traits. It has no Python dependency, so can be tested with `cargo test`.
//! * `python` (feature `python`) is the extension module `phlite.complexes._complexes`: it parses Python input into a
//!   [`complex::RawChain`] and exposes the `ChainComplex`, `FilteredComplex`, `ComplexBarcode` and `Bar` classes.
//!   `Algorithm` comes from `phlite.core` (see `phlite_bindings`).
//!
//! Why not let users implement matrix oracles in Python? Every column access would need the GIL, a Python call and a
//! Python object per entry, which costs far more than the reduction itself, and serialises the parallel algorithms.
//! Since `phlite` recomputes columns on demand rather than storing them, this cost is paid many times per column.
//! Materialising the complex once and reducing it in Rust avoids all of that.

pub mod complex;
pub mod field;
#[cfg(feature = "python")]
mod python;

/// What `phlite_complexes_stub_gen` writes: `python/phlite/complexes/__init__.pyi`.
#[cfg(feature = "python")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    phlite_bindings::stub_info_for("phlite.complexes", env!("CARGO_MANIFEST_DIR"))
}
