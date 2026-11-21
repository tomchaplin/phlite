//! Write the stubs of `phlite.core`, `python/phlite/core/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_bindings::stub_info()?.generate()?;
    // The pure-Python spectral sequence driver, re-exported by __init__.py, isn't in the extension module's stubs
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/python/phlite/core/__init__.pyi");
    let stubs = std::fs::read_to_string(path)?;
    let stubs = stubs.replacen("__all__ = [\n", "from . import spectral as spectral\nfrom .spectral import SpectralSequence as SpectralSequence\n\n__all__ = [\n    \"SpectralSequence\",\n", 1);
    std::fs::write(path, stubs)?;
    Ok(())
}
