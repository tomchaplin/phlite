//! Write the stubs of `phlite.complexes`, `python/phlite/complexes/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_complexes::stub_info()?.generate()?;
    Ok(())
}
