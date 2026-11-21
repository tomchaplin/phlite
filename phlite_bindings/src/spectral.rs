//! The Rust side of `phlite.core.spectral.SpectralSequence`: [`SpectralBackend`] wraps any
//! [`phlite::spectral::SpectralBackend`] as a Python object computing one page at a time, which a package wraps in
//! the driver with [`spectral_sequence`].

use phlite::integral::{Group, Overflow};
use phlite::spectral::SpectralBackend as Backend;
use pyo3::{exceptions::PyOverflowError, prelude::*, types::PyDict};
use rayon::prelude::*;

/// A spectral sequence's entries, for `phlite.core.spectral.SpectralSequence` (which drives it page by page): each
/// package builds one around its own [`phlite::spectral::SpectralBackend`].
#[pyclass(frozen, module = "phlite.core.spectral")]
pub struct SpectralBackend {
    inner: Box<dyn Backend>,
    integral: bool,
    /// For each step, how many degrees its row has (by default all of them)
    row_lengths: Option<Vec<usize>>,
}

impl SpectralBackend {
    /// Wrap `inner`; `integral` says whether entries have torsion (over Z) or are ranks (over a field).
    pub fn new(inner: Box<dyn Backend>, integral: bool) -> Self {
        Self { inner, integral, row_lengths: None }
    }

    /// Give each step's row only its first `row_lengths[p]` degrees (as magnitude homology's `k <= l`).
    pub fn with_row_lengths(mut self, row_lengths: Vec<usize>) -> Self {
        self.row_lengths = Some(row_lengths);
        self
    }

    fn row_length(&self, p: usize) -> usize {
        self.row_lengths.as_ref().map_or(self.inner.n_degrees(), |rows| rows[p].min(self.inner.n_degrees()))
    }

    fn to_py<'py>(&self, py: Python<'py>, group: Option<Group>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match group {
            None => py.None().into_bound(py),
            Some(g) if self.integral => (g.rank, g.torsion).into_pyobject(py)?.into_any(),
            Some(g) => g.rank.into_pyobject(py)?.into_any(),
        })
    }
}

fn overflow(e: Overflow) -> PyErr {
    PyOverflowError::new_err(e.to_string())
}

#[pymethods]
impl SpectralBackend {
    /// The number of steps.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps()
    }

    /// The number of degrees.
    #[getter]
    fn n_degrees(&self) -> usize {
        self.inner.n_degrees()
    }

    /// Whether entries are groups over Z (`(rank, torsion)`) rather than ranks over a field.
    #[getter]
    fn integral(&self) -> bool {
        self.integral
    }

    /// A page from which every differential is zero, if known.
    #[getter]
    fn degenerates_at(&self) -> Option<usize> {
        self.inner.degenerates_at()
    }

    /// Page `r`: for each step `p`, the entries of its row (`None` where not determined).
    fn page<'py>(&self, py: Python<'py>, r: usize) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let cells: Vec<(usize, usize)> =
            (0..self.inner.n_steps()).flat_map(|p| (0..self.row_length(p)).map(move |k| (p, k))).collect();
        let entries = py
            .detach(|| cells.par_iter().map(|&(p, k)| self.inner.entry(r, p, k)).collect::<Result<Vec<_>, _>>())
            .map_err(overflow)?;
        let mut rows: Vec<Vec<Bound<'py, PyAny>>> = (0..self.inner.n_steps()).map(|_| vec![]).collect();
        for (&(p, _), entry) in cells.iter().zip(entries) {
            rows[p].push(self.to_py(py, entry)?);
        }
        Ok(rows)
    }

    /// The non-zero differentials of page `r`, as `(p, k, image)`: `d^r` from `(p, k)` to `(p - r, k - 1)` with this
    /// image (a rank, or `(rank, torsion)` over Z). Those not determined are left out.
    fn differentials<'py>(&self, py: Python<'py>, r: usize) -> PyResult<Vec<(usize, usize, Bound<'py, PyAny>)>> {
        let cells: Vec<(usize, usize)> = (r..self.inner.n_steps())
            .flat_map(|p| (1..self.row_length(p)).map(move |k| (p, k)))
            .filter(|&(p, k)| k - 1 < self.row_length(p - r))
            .collect();
        let images = py
            .detach(|| {
                cells.par_iter().map(|&(p, k)| self.inner.differential(r, p, k)).collect::<Result<Vec<_>, _>>()
            })
            .map_err(overflow)?;
        let mut out = vec![];
        for (&(p, k), image) in cells.iter().zip(images) {
            if let Some(image) = image.filter(|g| !g.is_zero()) {
                out.push((p, k, self.to_py(py, Some(image))?));
            }
        }
        Ok(out)
    }
}

/// A `phlite.core.spectral.SpectralSequence` driving `backend`, with these keyword arguments (`steps`, the filtration
/// values of the steps; `step_label` and `degree_label`, the names of the axes; `title`).
pub fn spectral_sequence<'py>(
    py: Python<'py>,
    backend: SpectralBackend,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("phlite.core.spectral")?
        .getattr("SpectralSequence")?
        .call((Bound::new(py, backend)?,), Some(kwargs))
}
