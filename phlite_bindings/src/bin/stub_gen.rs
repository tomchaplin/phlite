//! Write the stubs of `phlite.core`, `python/phlite/core/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_bindings::stub_info()?.generate()?;
    Ok(())
}
