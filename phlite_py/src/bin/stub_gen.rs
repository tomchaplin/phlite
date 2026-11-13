//! Write the type stubs of the `phlite` package: `python/phlite/__init__.pyi` and one `__init__.pyi` per submodule
//! (`python/phlite/cloud/__init__.pyi`, ...). Run with `just stubs`.

use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    phlite_py::stub_info()?.generate()?;
    Ok(())
}
