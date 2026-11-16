//! The `phlite.networks` extension module (feature `python`), `phlite.networks._networks`.
//!
//! Each pipeline is a file here, with its function, its `register` and its `DEFAULTS` (see `phlite_bindings` for
//! how to write one); add a new one to [`PIPELINES`] and [`_networks`]. `graph.rs` reads digraphs, as every pipeline
//! takes them.

use phlite_bindings::AlgorithmDefaults;
use pyo3::prelude::*;

mod complexes;
mod flag;
mod graph;
mod grpph;
mod pph;

const DOC: &str = "Persistent homology of networks. Of weighted digraphs: `grpph` (grounded persistent path homology), \
                   `pph` (persistent path homology), `directed_flag` (directed flag complexes, as flagser, grounded or \
                   not), `ordered_tuples` (Turner's directed Rips) and `dowker` (Dowker sink and source complexes). Of \
                   undirected graphs: `flag` (flag complexes, by edge weights, node values or shortest paths).";
pyo3_stub_gen::module_doc!("phlite.networks", "{}", DOC);

/// This package's pipelines, registered with `phlite.core` on import.
const PIPELINES: &[AlgorithmDefaults] = &[
    grpph::DEFAULTS,
    pph::DEFAULTS,
    complexes::FLAG_DEFAULTS,
    complexes::TUPLES_DEFAULTS,
    complexes::DOWKER_DEFAULTS,
    flag::DEFAULTS,
    flag::IMPLICIT_DEFAULTS,
];

#[pymodule(gil_used = false)]
fn _networks(m: &Bound<'_, PyModule>) -> PyResult<()> {
    phlite_bindings::init_package(m.py(), "phlite-networks", env!("CARGO_PKG_VERSION"), PIPELINES)?;
    pyo3_log::init();
    m.setattr("__doc__", DOC)?;
    phlite_bindings::add_functions(m, "phlite.networks", |m| {
        grpph::register(m)?;
        pph::register(m)?;
        complexes::register(m)?;
        flag::register(m)?;
        Ok(())
    })?;
    Ok(())
}
