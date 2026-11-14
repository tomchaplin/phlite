//! Write the stubs of `phlite.networks`, `python/phlite/networks/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_networks::stub_info()?.generate()?;
    Ok(())
}
