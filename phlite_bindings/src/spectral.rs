//! The Rust side of `phlite.core.spectral.SpectralSequence`: [`SpectralBackend`] wraps any
//! [`phlite::spectral::SpectralBackend`] as a Python object computing one page at a time, which a package wraps in
//! the driver with [`spectral_sequence`].

use phlite::integral::{Group, Overflow};
use phlite::spectral::SpectralBackend as Backend;
use pyo3::{
    exceptions::{PyOverflowError, PyValueError},
    prelude::*,
    types::{PyDict, PyTuple},
};

/// How a package names the cells of its complex in Python (the cells of generators' chains), and back.
pub trait CellNames: Send + Sync {
    /// The Python object for cell `cell`.
    fn name<'py>(&self, py: Python<'py>, cell: usize) -> PyResult<Bound<'py, PyAny>>;
    /// The cell a Python object names, or `None` if it isn't one.
    fn index(&self, py: Python<'_>, cell: &Bound<'_, PyAny>) -> PyResult<Option<usize>>;
}
use rayon::prelude::*;

/// A spectral sequence's entries, for `phlite.core.spectral.SpectralSequence` (which drives it page by page): each
/// package builds one around its own [`phlite::spectral::SpectralBackend`].
#[pyclass(frozen, module = "phlite.core.spectral")]
pub struct SpectralBackend {
    inner: Box<dyn Backend>,
    integral: bool,
    /// For each step, how many degrees its row has (by default all of them)
    row_lengths: Option<Vec<usize>>,
    /// How cells are named, for generators
    cells: Option<Box<dyn CellNames>>,
    /// Over `Z/p`, `p` (coefficients of generators are shown between `-p/2` and `p/2`)
    modulus: Option<u64>,
}

impl SpectralBackend {
    /// Wrap `inner`; `integral` says whether entries have torsion (over Z) or are ranks (over a field).
    pub fn new(inner: Box<dyn Backend>, integral: bool) -> Self {
        Self { inner, integral, row_lengths: None, cells: None, modulus: None }
    }

    /// Name the cells of generators' chains with `cells`, coefficients over `Z/modulus` if a modulus is given.
    pub fn with_cells(mut self, cells: Box<dyn CellNames>, modulus: Option<u64>) -> Self {
        self.cells = Some(cells);
        self.modulus = modulus;
        self
    }

    fn coefficient(&self, x: i128) -> i128 {
        match self.modulus {
            Some(p) if x > i128::from(p) / 2 => x - i128::from(p),
            _ => x,
        }
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

impl SpectralBackend {
    /// A chain from Python (a dict from cells to coefficients, or `(cell, coefficient)` pairs) as cell indices.
    fn read_chain(&self, py: Python<'_>, chain: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, i64)>> {
        let cells = self.cells.as_ref().ok_or_else(|| PyValueError::new_err("this spectral sequence has no generators"))?;
        let pairs: Vec<Bound<'_, PyAny>> = match chain.cast::<PyDict>() {
            Ok(dict) => dict.items().iter().collect(),
            Err(_) => chain.try_iter()?.collect::<PyResult<_>>()?,
        };
        let mut terms = Vec::with_capacity(pairs.len());
        for pair in pairs {
            let pair = pair.cast_into::<PyTuple>().map_err(|_| PyValueError::new_err("a chain's terms are (cell, coefficient)"))?;
            let cell = pair.get_item(0)?;
            let coefficient: i64 = pair.get_item(1)?.extract()?;
            let index = cells
                .index(py, &cell)?
                .ok_or_else(|| PyValueError::new_err(format!("{} is not a cell of the complex", cell.repr().map_or_else(|_| "?".into(), |r| r.to_string()))))?;
            terms.push((index, coefficient));
        }
        Ok(terms)
    }
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

    /// The characteristic of the coefficient field, `None` over the integers (or if not given).
    #[getter]
    fn modulus(&self) -> Option<u64> {
        if self.integral {
            None
        } else {
            self.modulus
        }
    }

    /// Whether generators (and classifying chains) are available.
    #[getter]
    fn has_generators(&self) -> bool {
        self.cells.is_some() && self.inner.has_generators()
    }

    /// `E^r_p` in degree `k` with generators: `(entry, [(order, chain)])`, `order` `None` for a free summand and
    /// `chain` a dict from cells to coefficients; `None` if not determined.
    #[allow(clippy::type_complexity)]
    fn generators<'py>(
        &self,
        py: Python<'py>,
        r: usize,
        p: usize,
        k: usize,
    ) -> PyResult<Option<(Bound<'py, PyAny>, Vec<(Option<u128>, Bound<'py, PyDict>)>)>> {
        let cells = self.cells.as_ref().ok_or_else(|| PyValueError::new_err("this spectral sequence has no generators"))?;
        let Some((group, generators)) = py.detach(|| self.inner.generators(r, p, k)).map_err(overflow)? else {
            return Ok(None);
        };
        let mut out = vec![];
        for g in generators {
            let chain = PyDict::new(py);
            for (cell, x) in g.chain {
                chain.set_item(cells.name(py, cell)?, self.coefficient(x))?;
            }
            out.push((g.order, chain));
        }
        Ok(Some((self.to_py(py, Some(group))?, out)))
    }

    /// The class of `chain` (a dict from cells to coefficients, or `(cell, coefficient)` pairs) in `E^r_p` (degree
    /// `k`), as coordinates on the generators, or `None` if it doesn't represent an element.
    fn classify(&self, py: Python<'_>, r: usize, p: usize, k: usize, chain: &Bound<'_, PyAny>) -> PyResult<Option<Vec<i128>>> {
        let terms = self.read_chain(py, chain)?;
        py.detach(|| self.inner.classify(r, p, k, &terms)).map_err(overflow)
    }

    /// The boundary of `chain` (as for `classify`), as a dict from cells to coefficients.
    fn boundary<'py>(&self, py: Python<'py>, chain: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        let cells = self.cells.as_ref().ok_or_else(|| PyValueError::new_err("this spectral sequence has no generators"))?;
        let terms = self.read_chain(py, chain)?;
        let boundary =
            self.inner.boundary(&terms).ok_or_else(|| PyValueError::new_err("this spectral sequence has no boundary"))?;
        let out = PyDict::new(py);
        for (cell, x) in boundary {
            out.set_item(cells.name(py, cell)?, self.coefficient(i128::from(x)))?;
        }
        Ok(out)
    }

    /// Page `r`: for each step `p`, the entries of its row (`None` where not determined).
    fn page<'py>(&self, py: Python<'py>, r: usize) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let cells: Vec<(usize, usize)> =
            (0..self.inner.n_steps()).flat_map(|p| (0..self.row_length(p)).map(move |k| (p, k))).collect();
        // On a pool of its own, so that a process forked afterwards doesn't hang (see `crate::detach`)
        let entries = crate::detach(py, || cells.par_iter().map(|&(p, k)| self.inner.entry(r, p, k)).collect::<Result<Vec<_>, _>>())
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
        let images = crate::detach(py, || {
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
