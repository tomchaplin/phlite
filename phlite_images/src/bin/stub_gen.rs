//! Write the stubs of `phlite.images`, `python/phlite/images/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_images::stub_info()?.generate()?;
    // The pure-Python filtrations of binary images, re-exported by __init__.py, aren't in the extension's stubs
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/python/phlite/images/__init__.pyi");
    let stubs = std::fs::read_to_string(path)?;
    if !stubs.contains("from . import filtrations") {
        let stubs = stubs.replacen("__all__ = [\n", "from . import filtrations as filtrations\n\n__all__ = [\n    \"filtrations\",\n", 1);
        std::fs::write(path, stubs)?;
    }
    Ok(())
}
