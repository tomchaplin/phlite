//! Write the stubs of `phlite.networks`, `python/phlite/networks/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_networks::stub_info()?.generate()?;
    // The pure-Python magnitude homology results, re-exported by __init__.py, aren't in the extension module's stubs
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/python/phlite/networks/__init__.pyi");
    let stubs = std::fs::read_to_string(path)?;
    let stubs = stubs.replacen(
        "__all__ = [\n",
        concat!(
            "from . import magnitudes as magnitudes\n",
            "from .magnitudes import MagnitudeHomology as MagnitudeHomology\n\n",
            "__all__ = [\n    \"MagnitudeHomology\",\n"
        ),
        1,
    );
    std::fs::write(path, stubs)?;
    Ok(())
}
