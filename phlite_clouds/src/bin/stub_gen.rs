//! Write the stubs of `phlite.clouds`, `python/phlite/clouds/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_clouds::stub_info()?.generate()?;
    // The pure-Python zigzags, re-exported by __init__.py, aren't in the extension module's stubs
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/python/phlite/clouds/__init__.pyi");
    let stubs = std::fs::read_to_string(path)?;
    let stubs = stubs.replacen(
        "__all__ = [\n",
        concat!(
            "from . import zigzags as zigzags\n",
            "from .zigzags import cloud_zigzag as cloud_zigzag\n",
            "from .zigzags import sliding_windows as sliding_windows\n\n",
            "__all__ = [\n    \"cloud_zigzag\",\n    \"sliding_windows\",\n"
        ),
        1,
    );
    std::fs::write(path, stubs)?;
    Ok(())
}
