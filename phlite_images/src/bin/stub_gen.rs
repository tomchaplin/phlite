//! Write the stubs of `phlite.images`, `python/phlite/images/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_images::stub_info()?.generate()?;
    Ok(())
}
