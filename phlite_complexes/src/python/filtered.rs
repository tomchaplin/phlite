//! `FilteredComplex` (a chain complex with a filtration) and `ComplexBarcode` (its persistence, in terms of cells).

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use pyo3_stub_gen::derive::*;

use phlite_bindings::{algorithm_from_py, algorithm_to_py};

use super::{
    chain, AnyComplex, Bar, BarData, ChainComplex, Record, DEFAULTS,
};

/// A chain complex with a filtration attached, ready to compute persistent homology.
/// Made by `ChainComplex.filtration`.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite.complexes")]
pub struct FilteredComplex {
    pub(crate) complex: AnyComplex,
    pub(crate) chain: Py<ChainComplex>,
}

#[gen_stub_pymethods]
#[pymethods]
impl FilteredComplex {
    /// Compute persistent homology in dimensions `0, ..., max_dim`.
    ///
    /// Args:
    ///     max_dim: The maximum homology dimension to compute (default: the largest dimension of a cell).
    ///         Lower is faster: with `method='cohomology'` cells above dimension `max_dim + 1` are never looked at.
    ///     method: `'cohomology'` (the default) reduces the coboundary matrix, which is usually much faster.
    ///         `'homology'` reduces the boundary matrix instead.
    ///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
    ///         `Algorithm.serial()`).
    ///         For example `Algorithm.parallel(threads=8)`.
    ///         Pipeline defaults (`'complexes'`): columns='pivot', cache=False, store_r=True. Only serial clearing
    ///         uses pivot columns and stores R; the other algorithms merge columns.
    ///     representatives: Whether to also compute a representative cycle for every bar.
    ///         With `method='cohomology'` these are computed via involution.
    ///
    /// Returns:
    ///     The barcode. Bars of length zero (birth and death at the same filtration value) are omitted.
    #[pyo3(signature = (max_dim = None, *, method = "cohomology", algorithm = None, representatives = false))]
    fn persistence(
        slf: &Bound<'_, Self>,
        max_dim: Option<usize>,
        #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
        method: &str,
        #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
        algorithm: Option<Bound<'_, PyAny>>,
        representatives: bool,
    ) -> PyResult<ComplexBarcode> {
        let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
        let data = slf.get().complex.persistence(
            slf.py(),
            max_dim,
            method,
            &algorithm,
            representatives,
        )?;
        Ok(ComplexBarcode {
            data,
            complex: slf.clone().unbind(),
            algorithm: algorithm_to_py(slf.py(), &algorithm)?.unbind(),
        })
    }

    /// The spectral sequence of this filtration, over the complex's field: pages `E^1, ..., E^{r_max}`.
    ///
    /// The filtration's steps are its distinct values, in increasing order, and `E^r_p` is the page-`r` term at
    /// step `p`, whose differential `d^r` goes to step `p - r`. Over a field the spectral sequence is read off the
    /// barcode: a bar of degree `k` from step `b` to step `d` is a generator of `E^r_b` in degree `k` and of `E^r_d` in
    /// degree `k + 1` for every `r <= d - b`, and an essential bar one of `E^r_b` for every `r` (see `phlite::spectral`).
    /// Page `E^1` is the homology of the associated graded `F_p / F_{p-1}`, and the pages converge to the homology of
    /// the whole complex.
    ///
    /// Args:
    ///     r_max: The last page (default 3).
    ///     max_dim: The largest degree (default: one above the largest degree of a bar).
    ///     algorithm: The reduction algorithm, as for `persistence`.
    ///
    /// Returns:
    ///     `(values, pages)`: the distinct filtration values (the steps), and `pages[r - 1][p][k]`, the dimension of
    ///     `E^r_p` in degree `k`.
    #[pyo3(signature = (r_max = 3, max_dim = None, *, algorithm = None))]
    fn spectral_sequence(
        slf: &Bound<'_, Self>,
        r_max: usize,
        max_dim: Option<usize>,
        #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
        algorithm: Option<Bound<'_, PyAny>>,
    ) -> PyResult<(Vec<f64>, Vec<Vec<Vec<usize>>>)> {
        use phlite::spectral::{spectral_pages, step_of, StepBar};
        let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
        let complex = &slf.get().complex;
        let data = complex.persistence(slf.py(), max_dim, "cohomology", &algorithm, false)?;
        let mut steps: Vec<f64> = (0..complex.n_cells() as u32).map(|i| complex.value(i)).collect();
        steps.sort_by(f64::total_cmp);
        steps.dedup();
        let bars: Vec<StepBar> = data
            .bars
            .iter()
            .map(|bar| StepBar {
                degree: bar.dimension as usize,
                birth: step_of(&steps, &bar.birth_value),
                death: bar.death_value.is_finite().then(|| step_of(&steps, &bar.death_value)),
            })
            .collect();
        let n_degrees = max_dim.map_or_else(|| bars.iter().map(|b| b.degree + 2).max().unwrap_or(1), |d| d + 1);
        let pages = spectral_pages(&bars, steps.len(), n_degrees, r_max, None)
            .into_iter()
            .map(|page| page.into_iter().map(|row| row.into_iter().map(|x| x.unwrap_or(0)).collect()).collect())
            .collect();
        Ok((steps, pages))
    }

    /// Compute the extended persistence (Cohen-Steiner, Edelsbrunner and Harer) of this filtration: up through the
    /// filtration, then back down through the pairs (complex, superlevel subcomplex), so that every class dies, each
    /// bar being ordinary, relative, extended+ or extended- (see `ExtendedBarcode`). As GUDHI's
    /// `SimplexTree.extend_filtration` and `extended_persistence`, with exact values (GUDHI rescales them).
    ///
    /// Going down, each cell enters at the least value of it and its faces, so the superlevel subcomplexes are the
    /// largest subcomplexes of the superlevel sets. For the classical setting, values on the vertices of a complex,
    /// make the filtration with `chain.filtration(values, extend=True)` (values for the vertices, `None` elsewhere):
    /// each cell then enters at its highest vertex going up, and its lowest going down.
    ///
    /// Args:
    ///     method: `'cohomology'` (the default) or `'homology'`.
    ///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
    ///         `Algorithm.serial()`), filled in with the pipeline defaults (`'complexes'`).
    ///
    /// Returns:
    ///     A `phlite.core.ExtendedBarcode`, cells as this complex's. The cone vertex's component, the only essential class, and bars of length zero are
    ///     left out (as GUDHI's, with its default `min_persistence=0`).
    #[pyo3(signature = (*, method = "cohomology", algorithm = None))]
    #[gen_stub(override_return_type(type_repr = "core.ExtendedBarcode", imports = ("phlite.core")))]
    fn extended_persistence<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
        method: &str,
        #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
        algorithm: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
        let bars = self.complex.extended(py, method, &algorithm)?;
        // Cells as the chain complex's own (rebuilt from keys, if it stores keys)
        let chain = self.chain.get();
        let bars = bars
            .into_iter()
            .map(|bar| {
                Ok(phlite::extended::ExtendedBar {
                    kind: bar.kind,
                    dimension: bar.dimension,
                    birth: bar.birth,
                    death: bar.death,
                    birth_cell: chain.cell(py, bar.birth_cell.0)?,
                    death_cell: chain.cell(py, bar.death_cell.0)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        phlite_bindings::extended_into_python(py, bars, &algorithm)
    }

    /// The chain complex this filtration is on.
    #[getter]
    fn chain(&self, py: Python<'_>) -> Py<ChainComplex> {
        self.chain.clone_ref(py)
    }

    /// The filtration value of every cell, in cell order.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        (0..self.complex.n_cells() as u32)
            .map(|i| self.complex.value(i))
            .collect::<Vec<f64>>()
            .into_pyarray(py)
    }

    /// The filtration value of `cell`.
    fn value(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing")))] cell: &Bound<
            '_,
            PyAny,
        >,
    ) -> PyResult<f64> {
        let i = self.chain.get().index_or_err(cell)?;
        Ok(self.complex.value(i))
    }

    fn __len__(&self) -> usize {
        self.complex.n_cells()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("FilteredComplex({})", self.chain.bind(py).get().__repr__())
    }

    #[gen_stub(skip)]
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.chain)
    }
}

/// A persistence barcode, as computed by `FilteredComplex.persistence`.
///
/// Bars are identified by their birth cell (no cell gives birth to two bars). Cells are ints for a complex built
/// from columns or arrays, and your own objects for one built from cells.
/// Bars of length zero are omitted, and bars are sorted by the index of their birth cell.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite.complexes")]
pub struct ComplexBarcode {
    data: BarData,
    complex: Py<FilteredComplex>,
    algorithm: Py<PyAny>,
}

impl ComplexBarcode {
    fn chain(&self) -> &ChainComplex {
        self.complex.get().chain.get()
    }

    fn cell<'py>(&self, py: Python<'py>, i: u32) -> PyResult<Bound<'py, PyAny>> {
        self.chain().cell(py, i)
    }

    fn find(&self, birth: &Bound<'_, PyAny>) -> PyResult<&Record> {
        let index = self.chain().index_or_err(birth)?;
        self.data.find(index).ok_or_else(|| {
            PyValueError::new_err(format!(
                "No bar is born at {}",
                birth
                    .repr()
                    .map_or_else(|_| "that cell".into(), |r| r.to_string())
            ))
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl ComplexBarcode {
    /// Every bar, sorted by birth.
    #[getter]
    fn bars(&self, py: Python<'_>) -> PyResult<Vec<Bar>> {
        self.data.make_bars(py, |i| self.cell(py, i))
    }

    /// The finite bars, as `(birth, death)` pairs of cells, sorted by birth.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[tuple[typing.Any, typing.Any]]", imports = ("builtins", "typing")))]
    fn pairings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for bar in &self.data.bars {
            if let Some(death) = bar.death {
                list.append((self.cell(py, bar.birth)?, self.cell(py, death)?))?;
            }
        }
        Ok(list)
    }

    /// The birth cells of the essential (infinite) bars, sorted.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "builtins.list[typing.Any]", imports = ("builtins", "typing")))]
    fn essential<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for bar in &self.data.bars {
            if bar.death.is_none() {
                list.append(self.cell(py, bar.birth)?)?;
            }
        }
        Ok(list)
    }

    /// The bar born at the cell `birth`.
    fn bar(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing")))] birth: &Bound<
            '_,
            PyAny,
        >,
    ) -> PyResult<Bar> {
        let bar = self.find(birth)?;
        Bar::from_record(py, bar, self.data.representative(bar), |i| self.cell(py, i))
    }

    /// A representative cycle for the bar born at `birth`, as a list of `(cell, coefficient)` pairs in decreasing
    /// filtration order (so the birth cell is first), or `None` if representatives were not requested.
    #[gen_stub(override_return_type(type_repr = "typing.Optional[builtins.list[tuple[typing.Any, builtins.int]]]", imports = ("builtins", "typing")))]
    fn representative<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing")))] birth: &Bound<
            'py,
            PyAny,
        >,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        self.data
            .representative(self.find(birth)?)
            .map(|rep| chain(py, rep, |c| self.cell(py, c)))
            .transpose()
    }

    /// The persistence diagram in dimension `dim`, as an array of shape `(k, 2)` of `(birth, death)`
    /// filtration values, sorted. Essential bars have death `inf`; bars born at `inf` are left out.
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float64]", imports = ("numpy", "numpy.typing")))]
    fn diagram<'py>(&self, py: Python<'py>, dim: usize) -> Bound<'py, PyArray2<f64>> {
        self.data.diagram_array(py, dim)
    }

    /// The persistence diagrams in dimensions `0, ..., max_dim`; see `diagram`.
    #[gen_stub(override_return_type(type_repr = "builtins.list[numpy.typing.NDArray[numpy.float64]]", imports = ("builtins", "numpy", "numpy.typing")))]
    fn diagrams<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        (0..self.data.n_dims)
            .map(|dim| self.data.diagram_array(py, dim))
            .collect()
    }

    /// The filtered complex this barcode was computed from.
    #[getter]
    fn complex(&self, py: Python<'_>) -> Py<FilteredComplex> {
        self.complex.clone_ref(py)
    }

    /// The algorithm that computed this barcode, with every option as it ran (see `Algorithm.resolve`).
    #[getter]
    #[gen_stub(override_return_type(type_repr = "core.Algorithm", imports = ("phlite.core")))]
    fn algorithm(&self, py: Python<'_>) -> Py<PyAny> {
        self.algorithm.clone_ref(py)
    }

    fn __len__(&self) -> usize {
        self.data.bars.len()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self
            .data
            .repr("ComplexBarcode", &self.algorithm.bind(py).repr()?.to_string()))
    }

    #[gen_stub(skip)]
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.complex)?;
        visit.call(&self.algorithm)
    }
}
