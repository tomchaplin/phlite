//! `ValueBarcode`: the barcode of the pipelines in the submodules (built by [`super::shared::SortedBars`]).

use pyo3::{prelude::*, types::PyList};

use super::AlgorithmConfig;
use pyo3_stub_gen::derive::*;

/// A persistence barcode in terms of filtration values, split by dimension, as returned by the pipelines in
/// `phlite`'s submodules (such as `phlite.cloud.rips` and `phlite.network.grpph`).
///
/// Bars of length zero are omitted.
/// Within each dimension, bars are sorted by birth (and then death).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite")]
pub struct ValueBarcode {
    pub(crate) essential: Py<PyList>,
    pub(crate) pairings: Py<PyList>,
    pub(crate) essential_representatives: Option<Py<PyList>>,
    pub(crate) pairing_representatives: Option<Py<PyList>>,
    pub(crate) algorithm: Py<AlgorithmConfig>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ValueBarcode {
    /// `essential[k]` lists the birth times of the infinite bars in dimension `k`.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[builtins.list[builtins.float]]", imports = ("builtins")))]
    fn essential(&self, py: Python<'_>) -> Py<PyList> {
        self.essential.clone_ref(py)
    }

    /// `pairings[k]` lists the `(birth, death)` times of the finite bars in dimension `k`.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[builtins.list[tuple[builtins.float, builtins.float]]]", imports = ("builtins")))]
    fn pairings(&self, py: Python<'_>) -> Py<PyList> {
        self.pairings.clone_ref(py)
    }

    /// `essential_representatives[k][i]` is a cycle representing the bar `essential[k][i]`,
    /// or `None` if representatives were not requested.
    ///
    /// A cycle is a list of cells, each given as a tuple of vertices (indices, or node labels for a networkx graph).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[builtins.list[builtins.list[builtins.list[tuple[typing.Any, ...]]]]]", imports = ("builtins", "typing")))]
    fn essential_representatives(&self, py: Python<'_>) -> Option<Py<PyList>> {
        self.essential_representatives
            .as_ref()
            .map(|reps| reps.clone_ref(py))
    }

    /// `pairing_representatives[k][i]` is a cycle representing the bar `pairings[k][i]`,
    /// or `None` if representatives were not requested.
    ///
    /// A cycle is a list of cells, each given as a tuple of vertices (indices, or node labels for a networkx graph).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Optional[builtins.list[builtins.list[builtins.list[tuple[typing.Any, ...]]]]]", imports = ("builtins", "typing")))]
    fn pairing_representatives(&self, py: Python<'_>) -> Option<Py<PyList>> {
        self.pairing_representatives
            .as_ref()
            .map(|reps| reps.clone_ref(py))
    }

    /// The algorithm that computed this barcode, with every option as it ran (see `Algorithm.resolve`).
    #[getter]
    fn algorithm(&self, py: Python<'_>) -> Py<AlgorithmConfig> {
        self.algorithm.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let counts = |list: &Py<PyList>| -> PyResult<Vec<usize>> {
            list.bind(py)
                .iter()
                .map(|dim| dim.len())
                .collect::<PyResult<_>>()
        };
        Ok(format!(
            "ValueBarcode(essential={:?}, pairings={:?}, representatives={}, algorithm={})",
            counts(&self.essential)?,
            counts(&self.pairings)?,
            if self.essential_representatives.is_some() {
                "True"
            } else {
                "False"
            },
            self.algorithm.get().__repr__(),
        ))
    }

    #[gen_stub(skip)]
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.essential)?;
        visit.call(&self.pairings)?;
        visit.call(&self.algorithm)?;
        if let Some(reps) = &self.essential_representatives {
            visit.call(reps)?;
        }
        if let Some(reps) = &self.pairing_representatives {
            visit.call(reps)?;
        }
        Ok(())
    }
}
