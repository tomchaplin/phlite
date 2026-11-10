use std::any::TypeId;

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    // `stub_info` is a function defined by `define_stub_info_gatherer!` macro.
    let mut stub = phlite_py::stub_info()?;
    // The `Barcode` class of `phlite_pyutils` is linked in (via `AlgorithmArg`) but not part of this module,
    // which has its own `Barcode`
    for module in stub.modules.values_mut() {
        module
            .class
            .remove(&TypeId::of::<phlite_pyutils::Barcode>());
    }
    stub.generate()?;
    Ok(())
}
