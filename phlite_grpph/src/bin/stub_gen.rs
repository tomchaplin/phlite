use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = phlite_grpph::stub_info()?;
    stub.generate()?;
    Ok(())
}
