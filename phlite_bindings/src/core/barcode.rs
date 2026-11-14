//! `Barcode`: the barcode the pipelines of phlite's packages return (built by [`crate::SortedBars`]).

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
use pyo3_stub_gen::derive::*;

use super::AlgorithmConfig;
use crate::spec::spec_from_dict;

/// A persistence barcode in terms of filtration values, split by dimension, as returned by the pipelines of phlite's
/// packages (such as `phlite.clouds.rips` and `phlite.networks.grpph`).
/// (`phlite.complexes` gives a `ComplexBarcode` instead, in terms of your cells.)
///
/// Bars of length zero are omitted.
/// Within each dimension, bars are sorted by birth (and then death).
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite.core")]
pub struct Barcode {
    essential: Py<PyList>,
    pairings: Py<PyList>,
    essential_representatives: Option<Py<PyList>>,
    pairing_representatives: Option<Py<PyList>>,
    algorithm: Py<AlgorithmConfig>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Barcode {
    /// Build from the lists the getters return, taken as they are (not copied), and the spec of the algorithm that
    /// computed them (as `Algorithm._spec()` gives it). For phlite's packages, which call it through
    /// `phlite_bindings::SortedBars`; not a public API.
    #[staticmethod]
    #[gen_stub(skip)]
    fn _from_parts(
        py: Python<'_>,
        essential: Bound<'_, PyList>,
        pairings: Bound<'_, PyList>,
        essential_representatives: Option<Bound<'_, PyList>>,
        pairing_representatives: Option<Bound<'_, PyList>>,
        algorithm: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        if essential.len() != pairings.len() {
            return Err(PyValueError::new_err(
                "essential and pairings must have an entry for every dimension",
            ));
        }
        match (&essential_representatives, &pairing_representatives) {
            (None, None) => {}
            (Some(essential_reps), Some(pairing_reps)) => {
                for (bars, reps) in [(&essential, essential_reps), (&pairings, pairing_reps)] {
                    if reps.len() != bars.len() {
                        return Err(PyValueError::new_err(
                            "representatives must have an entry for every dimension",
                        ));
                    }
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "give representatives for both essential bars and pairings, or for neither",
                ))
            }
        }
        Ok(Self {
            essential: essential.unbind(),
            pairings: pairings.unbind(),
            essential_representatives: essential_representatives.map(Bound::unbind),
            pairing_representatives: pairing_representatives.map(Bound::unbind),
            algorithm: Py::new(
                py,
                AlgorithmConfig {
                    spec: spec_from_dict(algorithm)?,
                },
            )?,
        })
    }

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
            "Barcode(essential={:?}, pairings={:?}, representatives={}, algorithm={})",
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
