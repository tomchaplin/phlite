//! The `phlite` Python package: persistent homology of explicit sparse filtered chain complexes, and the bindings of
//! phlite's other crates.
//!
//! The Rust side is split in three:
//! * [`complex`] stores a chain complex as a sparse boundary matrix ([`complex::Chain`]) and, once a filtration is
//!   attached, as a pair of sorted sparse matrices (boundary and coboundary, [`complex::Complex`]) implementing
//!   `phlite`'s matrix traits. It has no Python dependency, so can be tested with `cargo test`.
//! * [`algorithm`] chooses a reduction algorithm ([`algorithm::AlgorithmSpec`], the Python `Algorithm`), filling in
//!   the options left unset with each pipeline's defaults. Also free of Python.
//! * `python` (feature `python`) is the extension module `phlite._phlite`: it parses Python input into a
//!   [`complex::RawChain`] and exposes the `ChainComplex`, `FilteredComplex`, `Barcode` and `Bar` classes, and holds
//!   the bindings of the other crates as submodules: `phlite.cloud` (the `phlite_cloud` crate: point clouds),
//!   `phlite.network` (`phlite_network`: networks) and `phlite.image` (`phlite_image`: images), one folder of
//!   `src/python` each. Those crates are plain Rust.
//!
//! Why not let users implement matrix oracles in Python? Every column access would need the GIL, a Python call and a
//! Python object per entry, which costs far more than the reduction itself, and serialises the parallel algorithms.
//! Since `phlite` recomputes columns on demand rather than storing them, this cost is paid many times per column.
//! Materialising the complex once and reducing it in Rust avoids all of that.

pub mod algorithm;
pub mod complex;
pub mod field;
#[cfg(feature = "python")]
mod python;

/// Gather what `phlite_py_stub_gen` writes to the stubs: one `__init__.pyi` per module of the `phlite` package
/// (`phlite`, `phlite.cloud`, ...), in `python/phlite`. The extension module itself is the private `phlite._phlite`,
/// which the Python files there re-export, so the stubs describe the public modules instead.
#[cfg(feature = "python")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    let pyproject = manifest_dir.join("pyproject.toml");
    let config = pyo3_stub_gen::StubInfo::from_pyproject_toml(&pyproject)?.config;
    pyo3_stub_gen::StubInfo::from_project_root(
        "phlite".to_string(),
        manifest_dir.join("python"),
        true,
        config,
    )
}
