//! Write the stubs of `phlite.complexes`, `python/phlite/complexes/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_complexes::stub_info()?.generate()?;
    // The pure-Python zigzag layer, re-exported by __init__.py, isn't in the extension module's stubs
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/python/phlite/complexes/__init__.pyi");
    let stubs = std::fs::read_to_string(path)?;
    let stubs = stubs.replacen(
        "__all__ = [\n",
        concat!(
            "from . import zigzags as zigzags\n",
            "from .zigzags import Zigzag as Zigzag, ZigzagBar as ZigzagBar, ZigzagBarcode as ZigzagBarcode, zigzag as zigzag\n",
            "from .zigzags import TimedZigzagBar as TimedZigzagBar, TimedZigzagBarcode as TimedZigzagBarcode, timed_zigzag as timed_zigzag\n\n",
            "__all__ = [\n    \"zigzag\",\n    \"Zigzag\",\n    \"ZigzagBar\",\n    \"ZigzagBarcode\",\n    \"timed_zigzag\",\n    \"TimedZigzagBar\",\n    \"TimedZigzagBarcode\",\n"
        ),
        1,
    );
    std::fs::write(path, stubs)?;
    Ok(())
}
