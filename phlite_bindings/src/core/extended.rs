//! `ExtendedBarcode`: extended persistence, as every package that computes it returns it (see `phlite::extended`).

use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use pyo3_stub_gen::derive::*;

use super::AlgorithmConfig;
use crate::spec::spec_from_dict;

/// The extended persistence of a filtration: up through the filtration, then back down through the pairs (complex,
/// superlevel subcomplex), so that every class dies. Four diagrams, each a list (by dimension) of sorted `(birth,
/// death)` pairs:
///
/// * `ordinary`: born and dying going up (as ordinary persistence);
/// * `relative`: born and dying going down; dimensions start at 1, as GUDHI's;
/// * `extended_plus`: born going up and dying going down, above its birth (a component, from its lowest point to its
///   highest);
/// * `extended_minus`: likewise, dying below its birth (a loop of a graph, from its highest point to its lowest).
///
/// `bars` lists every bar with the cells whose entry gives its birth and death (on the way down, the cell whose cone
/// enters). Bars of length zero are omitted, as is the cone vertex's component, the only essential class.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite.core")]
pub struct ExtendedBarcode {
    #[pyo3(get)]
    ordinary: Py<PyList>,
    #[pyo3(get)]
    relative: Py<PyList>,
    #[pyo3(get)]
    extended_plus: Py<PyList>,
    #[pyo3(get)]
    extended_minus: Py<PyList>,
    /// Every bar as `(kind, dimension, birth, death, birth_cell, death_cell)`, `kind` being `'ordinary'`,
    /// `'relative'`, `'extended+'` or `'extended-'`.
    #[pyo3(get)]
    bars: Py<PyList>,
    algorithm: Py<AlgorithmConfig>,
}

#[gen_stub_pymethods]
#[pymethods]
impl ExtendedBarcode {
    /// Build from every bar, as `bars` gives them (in any order), and the spec of the algorithm that computed them.
    /// For phlite's packages, which call it through `phlite_bindings::extended_into_python`; not a public API.
    #[staticmethod]
    #[gen_stub(skip)]
    fn _from_parts(py: Python<'_>, bars: Bound<'_, PyList>, algorithm: &Bound<'_, PyDict>) -> PyResult<Self> {
        let kinds = ["ordinary", "relative", "extended+", "extended-"];
        let mut items: Vec<(usize, usize, f64, f64, Bound<'_, PyAny>)> = vec![];
        for bar in bars.iter() {
            let (kind, dim, birth, death): (String, usize, f64, f64) =
                (bar.get_item(0)?.extract()?, bar.get_item(1)?.extract()?, bar.get_item(2)?.extract()?, bar.get_item(3)?.extract()?);
            let k = kinds.iter().position(|&k| k == kind).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("unknown kind of bar {kind:?}"))
            })?;
            items.push((k, dim, birth, death, bar));
        }
        items.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)).then(a.2.total_cmp(&b.2)).then(a.3.total_cmp(&b.3)));
        let top = items.iter().map(|i| i.1).max().unwrap_or(0);
        let mut diagrams = vec![vec![vec![]; top + 1]; 4];
        for (k, dim, birth, death, _) in &items {
            diagrams[*k][*dim].push((*birth, *death));
        }
        let list = |d: &Vec<Vec<(f64, f64)>>| -> PyResult<Py<PyList>> { Ok(PyList::new(py, d)?.unbind()) };
        let algorithm = AlgorithmConfig { spec: spec_from_dict(algorithm)? };
        Ok(Self {
            ordinary: list(&diagrams[0])?,
            relative: list(&diagrams[1])?,
            extended_plus: list(&diagrams[2])?,
            extended_minus: list(&diagrams[3])?,
            bars: PyList::new(py, items.into_iter().map(|i| i.4))?.unbind(),
            algorithm: Py::new(py, algorithm)?,
        })
    }

    /// The algorithm that computed it, with every option set.
    #[getter]
    fn algorithm(&self, py: Python<'_>) -> Py<AlgorithmConfig> {
        self.algorithm.clone_ref(py)
    }

    /// The four diagrams in GUDHI's format (as `SimplexTree.extended_persistence` returns them): a list of four lists
    /// of `(dimension, (birth, death))`.
    fn to_gudhi<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let diagrams = [&self.ordinary, &self.relative, &self.extended_plus, &self.extended_minus]
            .iter()
            .map(|diagram| {
                let mut out = vec![];
                for (dim, bars) in diagram.bind(py).iter().enumerate() {
                    for bar in bars.try_iter()? {
                        out.push(PyTuple::new(py, [dim.into_pyobject(py)?.into_any(), bar?])?);
                    }
                }
                PyList::new(py, out)
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, diagrams)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.bars.bind(py).len()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let count = |d: &Py<PyList>| -> PyResult<String> {
            let mut parts = vec![];
            for (i, bars) in d.bind(py).iter().enumerate() {
                let n = bars.len()?;
                if n > 0 {
                    parts.push(format!("H{i}: {n}"));
                }
            }
            Ok(parts.join(", "))
        };
        Ok(format!(
            "ExtendedBarcode(ordinary=[{}], relative=[{}], extended_plus=[{}], extended_minus=[{}])",
            count(&self.ordinary)?,
            count(&self.relative)?,
            count(&self.extended_plus)?,
            count(&self.extended_minus)?
        ))
    }

    #[gen_stub(skip)]
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        for list in [&self.ordinary, &self.relative, &self.extended_plus, &self.extended_minus, &self.bars] {
            visit.call(list)?;
        }
        visit.call(&self.algorithm)
    }
}
