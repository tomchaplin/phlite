//! `FilteredComplex` (a chain complex with a filtration) and `Barcode` (its persistence, in terms of cells).

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use pyo3_stub_gen::derive::*;

use phlite_pyutils::AlgorithmConfig;

use super::{chain, AnyComplex, Bar, BarData, ChainComplex, Record};

/// A chain complex with a filtration attached, ready to compute persistent homology.
/// Made by `ChainComplex.filtration`.
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite")]
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
    ///     algorithm: The reduction algorithm and its options, as an `Algorithm` (default: `Algorithm.serial()`).
    ///         For example `Algorithm.parallel(threads=8)`.
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
        algorithm: Option<Bound<'_, AlgorithmConfig>>,
        representatives: bool,
    ) -> PyResult<Barcode> {
        let algorithm = algorithm.map(|a| a.get().clone()).unwrap_or_default();
        let data = slf.get().complex.persistence(
            slf.py(),
            max_dim,
            method,
            &algorithm,
            representatives,
        )?;
        Ok(Barcode {
            data,
            complex: slf.clone().unbind(),
        })
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
#[pyclass(frozen, module = "phlite")]
pub struct Barcode {
    data: BarData,
    complex: Py<FilteredComplex>,
}

impl Barcode {
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
impl Barcode {
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

    fn __len__(&self) -> usize {
        self.data.bars.len()
    }

    fn __repr__(&self) -> String {
        self.data.repr("Barcode")
    }

    #[gen_stub(skip)]
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.complex)
    }
}
