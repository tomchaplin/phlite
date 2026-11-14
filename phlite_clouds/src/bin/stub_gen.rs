//! Write the stubs of `phlite.clouds`, `python/phlite/clouds/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_clouds::stub_info()?.generate()?;
    Ok(())
}
