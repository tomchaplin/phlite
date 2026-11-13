//! Helpers for every pipeline's bindings: reading the `method` and `algorithm` arguments, and building a
//! [`ValueBarcode`].
//!
//! A pipeline function in a submodule (see `cloud`, `network` and `image`) typically:
//! 1. takes `algorithm: Option<Bound<'_, AlgorithmConfig>>` and resolves it with [`algorithm_from_py`] against the
//!    pipeline's own [`AlgorithmDefaults`] (a `const` next to the function, listed in its area's `PIPELINES`, see
//!    [`super::pipelines`]), and states those defaults in its docstring;
//! 2. reads `method` with [`parse_method`];
//! 3. computes its bars without the GIL (`py.detach`), running an [`AlgorithmVisitor`](crate::algorithm::AlgorithmVisitor)
//!    with [`ResolvedAlgorithm::run`] and sorting the bars into [`SortedBars`];
//! 4. returns [`SortedBars::into_python`].

use ordered_float::NotNan;
use phlite::barcode::{Bar, Method};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyTuple},
};

use super::{AlgorithmConfig, ValueBarcode};
use crate::algorithm::{AlgorithmDefaults, AlgorithmError, ResolvedAlgorithm};

impl From<AlgorithmError> for PyErr {
    fn from(error: AlgorithmError) -> Self {
        PyValueError::new_err(error.0)
    }
}

/// Parse the `method` argument. In the stubs, type it as `typing.Literal['cohomology', 'homology']`.
pub(crate) fn parse_method(name: &str) -> PyResult<Method> {
    match name {
        "cohomology" => Ok(Method::Cohomology),
        "homology" => Ok(Method::Homology),
        other => Err(PyValueError::new_err(format!(
            "Unknown method {other:?}, expected 'cohomology' or 'homology'"
        ))),
    }
}

/// Read the `algorithm` argument (`None` for `Algorithm.serial()`), resolving the options it leaves unset with the
/// pipeline's `defaults`.
pub(crate) fn algorithm_from_py(
    algorithm: Option<&Bound<'_, AlgorithmConfig>>,
    defaults: &AlgorithmDefaults,
) -> PyResult<ResolvedAlgorithm> {
    let resolved = match algorithm {
        None => crate::algorithm::AlgorithmSpec::default().resolve(defaults),
        Some(algorithm) => algorithm.get().spec.resolve(defaults),
    };
    Ok(resolved?)
}

/// A representative cycle, as a list of cells each given by its vertices, if requested
type Rep = Option<Vec<Vec<u32>>>;

/// The bars of a barcode, sorted and split by dimension, still on the Rust side.
/// Build with [`new`](Self::new) (which can be done without the GIL), then convert with [`into_python`](Self::into_python).
pub(crate) struct SortedBars {
    essential: Vec<Vec<(f64, Rep)>>,
    pairings: Vec<Vec<((f64, f64), Rep)>>,
}

impl SortedBars {
    /// Split `bars` into dimensions `0..n_dims`, dropping zero-length bars, then sort.
    /// Cells with infinite filtration value are treated as never appearing.
    /// `cell_info` gives the dimension of a cell and the vertices it contains.
    pub(crate) fn new<C, CF>(
        bars: Vec<Bar<C, NotNan<f64>, CF>>,
        n_dims: usize,
        cell_info: impl Fn(&C) -> (usize, Vec<u32>),
    ) -> Self {
        let mut essential = vec![vec![]; n_dims];
        let mut pairings = vec![vec![]; n_dims];
        for bar in bars {
            if bar.is_zero_length() {
                continue;
            }
            let (dim, _) = cell_info(&bar.birth);
            let rep = bar.representative.map(|rep| {
                rep.iter()
                    .map(|(_coeff, cell)| cell_info(cell).1)
                    .collect::<Vec<_>>()
            });
            let birth = bar.birth_value.into_inner();
            // Nothing appears at time infinity, so classes born then never exist and those dying then are essential
            if birth == f64::INFINITY {
                continue;
            }
            match bar.death.map(|(_, death)| death.into_inner()) {
                None | Some(f64::INFINITY) => essential[dim].push((birth, rep)),
                Some(death) => pairings[dim].push(((birth, death), rep)),
            }
        }
        for dim in essential.iter_mut() {
            dim.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        }
        for dim in pairings.iter_mut() {
            dim.sort_by(|a, b| {
                (a.0 .0.total_cmp(&b.0 .0))
                    .then_with(|| a.0 .1.total_cmp(&b.0 .1))
                    .then_with(|| a.1.cmp(&b.1))
            });
        }
        Self {
            essential,
            pairings,
        }
    }

    /// Convert to a [`ValueBarcode`]; `representatives` must match whether the bars have representatives.
    ///
    /// If `labels` is given then vertex `i` appears in representatives as `labels[i]`, otherwise as the integer `i`.
    /// `algorithm` is what computed the bars, which the barcode records.
    pub(crate) fn into_python(
        self,
        py: Python<'_>,
        algorithm: &ResolvedAlgorithm,
        representatives: bool,
        labels: Option<&[Py<PyAny>]>,
    ) -> PyResult<ValueBarcode> {
        let cell_to_python = |cell: &Vec<u32>| -> PyResult<Bound<'_, PyTuple>> {
            match labels {
                None => PyTuple::new(py, cell),
                Some(labels) => PyTuple::new(py, cell.iter().map(|&v| labels[v as usize].bind(py))),
            }
        };
        let reps_to_python = |dims: Vec<Vec<&Rep>>| -> PyResult<Py<PyList>> {
            let dims = dims
                .into_iter()
                .map(|bars| {
                    let bars = bars
                        .into_iter()
                        .map(|rep| {
                            let cells = rep
                                .as_ref()
                                .expect("representative was requested")
                                .iter()
                                .map(cell_to_python)
                                .collect::<PyResult<Vec<_>>>()?;
                            PyList::new(py, cells)
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    PyList::new(py, bars)
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, dims)?.unbind())
        };

        let essential: Vec<Vec<f64>> = self
            .essential
            .iter()
            .map(|dim| dim.iter().map(|(birth, _)| *birth).collect())
            .collect();
        let pairings: Vec<Vec<(f64, f64)>> = self
            .pairings
            .iter()
            .map(|dim| dim.iter().map(|(bar, _)| *bar).collect())
            .collect();

        Ok(ValueBarcode {
            essential: PyList::new(py, essential)?.unbind(),
            pairings: PyList::new(py, pairings)?.unbind(),
            essential_representatives: representatives
                .then(|| reps_to_python(only_reps(&self.essential)))
                .transpose()?,
            pairing_representatives: representatives
                .then(|| reps_to_python(only_reps(&self.pairings)))
                .transpose()?,
            algorithm: Py::new(py, AlgorithmConfig::from(algorithm))?,
        })
    }
}

fn only_reps<T>(dims: &[Vec<(T, Rep)>]) -> Vec<Vec<&Rep>> {
    dims.iter()
        .map(|bars| bars.iter().map(|(_, rep)| rep).collect())
        .collect()
}
