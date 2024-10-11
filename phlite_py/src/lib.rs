use std::borrow::Borrow;

use log::info;
use phlite::{
    fields::{NonZeroCoefficient, Z2},
    matrices::{BasisElement, MatrixOracle},
};
use pyo3::intern;
use pyo3::{prelude::*, types::PyIterator};

struct PyMatrix(PyObject);
struct PyBasisElement(PyObject);

impl IntoPy<PyObject> for PyBasisElement {
    fn into_py(self, _py: Python<'_>) -> PyObject {
        self.0
    }
}

impl Clone for PyBasisElement {
    fn clone(&self) -> Self {
        PyBasisElement(Python::with_gil(|py| self.0.clone_ref(py)))
    }
}
impl PartialOrd for PyBasisElement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}
impl Ord for PyBasisElement {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}
impl PartialEq for PyBasisElement {
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}
impl Eq for PyBasisElement {}

impl BasisElement for PyBasisElement {}

impl MatrixOracle for PyMatrix {
    type CoefficientField = Z2;

    type ColT = PyBasisElement;

    type RowT = PyBasisElement;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, phlite::PhliteError>
    {
        Python::with_gil(|py| {
            // Collect entire column into a vec on the Rust side
            // This allows us to decouple the column iterator from the GIL
            Some(
                self.0
                    .call_method1(py, intern!(py, "column"), (col,))
                    .ok()?
                    .into_bound(py)
                    .iter()
                    .ok()?
                    .map(|i| (Z2::one(), PyBasisElement(i.unwrap().unbind())))
                    .collect::<Vec<_>>()
                    .into_iter(),
            )
        })
        .ok_or(phlite::PhliteError::NotInDomain)
    }
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[pymodule(name = "phlite")]
fn phlite_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
