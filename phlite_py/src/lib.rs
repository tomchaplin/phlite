use std::ops::Deref;

use phlite::{
    fields::{NonZeroCoefficient, Z2},
    matrices::{BasisElement, ColBasis, HasColBasis, MatrixOracle, SplitByDimension},
};
use pyo3::intern;
use pyo3::prelude::*;

struct PyMatrix(PyObject);
#[repr(transparent)]
struct PyBasis(PyObject);
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
        Some(self.cmp(other))
    }
}
impl Ord for PyBasisElement {
    fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}
impl PartialEq for PyBasisElement {
    fn eq(&self, _other: &Self) -> bool {
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
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        Python::with_gil(|py| {
            // Collect entire column into a vec on the Rust side
            // This allows us to decouple the column iterator from the GIL
            self.0
                .call_method1(py, intern!(py, "column"), (col,))
                .unwrap()
                .into_bound(py)
                .iter()
                .unwrap()
                .map(|i| (Z2::one(), PyBasisElement(i.unwrap().unbind())))
                .collect::<Vec<_>>()
                .into_iter()
        })
    }
}

impl ColBasis for PyBasis {
    type ElemT = PyBasisElement;

    fn element(&self, index: usize) -> Self::ElemT {
        PyBasisElement(
            Python::with_gil(|py| self.0.call_method1(py, intern!(py, "element"), (index,)))
                .unwrap(),
        )
    }

    fn size(&self) -> usize {
        Python::with_gil(|py| {
            self.0
                .call_method0(py, intern!(py, "size"))
                .unwrap()
                .extract(py)
                .unwrap()
        })
    }
}

struct BasisRef(PyObject);

impl Deref for BasisRef {
    type Target = PyBasis;

    fn deref(&self) -> &Self::Target {
        let ptr: *const _ = self.0.as_any();
        let basis_ptr = ptr.cast();
        unsafe { &*basis_ptr }
    }
}

impl HasColBasis for PyMatrix {
    type BasisT = PyBasis;

    type BasisRef<'a>
        = BasisRef
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisRef<'_> {
        BasisRef(Python::with_gil(|py| {
            self.0.call_method0(py, intern!(py, "basis")).unwrap()
        }))
    }
}

impl SplitByDimension for PyBasis {
    type SubBasisT = PyBasis;

    // Perhaps we need add a lookup table to PyBasis which we can populate with subbases
    fn in_dimension(&self, _dimension: usize) -> &Self::SubBasisT {
        todo!()
    }
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[pymodule(name = "phlite")]
fn phlite_py(_m: &Bound<'_, PyModule>) -> PyResult<()> {
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
