//! Write the stubs of `phlite.core`, `python/phlite/core/__init__.pyi`. Run with `just stubs` at the root.

fn main() -> pyo3_stub_gen::Result<()> {
    phlite_bindings::stub_info()?.generate()?;
    // The pure-Python spectral sequence driver and six-packs, re-exported by __init__.py, aren't in the extension
    // module's stubs
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/python/phlite/core/__init__.pyi"
    );
    let stubs = std::fs::read_to_string(path)?;
    let stubs = stubs.replacen("__all__ = [\n", concat!(
            "from . import spectral as spectral\nfrom .spectral import SpectralSequence as SpectralSequence\n",
            "from . import six_pack as six_pack\n",
            "from .six_pack import Cone as Cone, PackBar as PackBar, PackBarcode as PackBarcode, SixPack as SixPack\n",
            "from . import zigzag as zigzag\n",
            "from .zigzag import ZigzagBar as ZigzagBar, ZigzagBarcode as ZigzagBarcode\n\n",
            "__all__ = [\n    \"SpectralSequence\",\n    \"SixPack\",\n    \"PackBarcode\",\n    \"PackBar\",\n    \"Cone\",\n    \"ZigzagBar\",\n    \"ZigzagBarcode\",\n"
        ), 1);
    std::fs::write(path, stubs)?;
    Ok(())
}
