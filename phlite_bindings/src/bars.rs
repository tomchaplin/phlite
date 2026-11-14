//! [`SortedBars`]: a pipeline's bars, sorted by dimension, on their way to a `phlite.core.Barcode`.

use ordered_float::NotNan;
use phlite::barcode::Bar;
use pyo3::{
    prelude::*,
    types::{PyList, PyTuple},
};

use crate::{algorithm::ResolvedAlgorithm, core_attribute, spec::spec_to_dict, CoreAttribute};

/// A representative cycle, as a list of cells each given by its vertices, if requested
type Rep = Option<Vec<Vec<u32>>>;

/// The bars of a barcode, sorted and split by dimension, still on the Rust side.
/// Build with [`new`](Self::new) (which can be done without the GIL), then convert with [`into_python`](Self::into_python).
pub struct SortedBars {
    essential: Vec<Vec<(f64, Rep)>>,
    pairings: Vec<Vec<((f64, f64), Rep)>>,
}

impl SortedBars {
    /// Split `bars` into dimensions `0..n_dims`, dropping zero-length bars, then sort.
    /// Cells with infinite filtration value are treated as never appearing.
    /// `cell_info` gives the dimension of a cell and the vertices it contains.
    pub fn new<C, CF>(
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
        let mut bars = Self {
            essential,
            pairings,
        };
        bars.sort();
        bars
    }

    fn sort(&mut self) {
        for dim in self.essential.iter_mut() {
            dim.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        }
        for dim in self.pairings.iter_mut() {
            dim.sort_by(|a, b| {
                (a.0 .0.total_cmp(&b.0 .0))
                    .then_with(|| a.0 .1.total_cmp(&b.0 .1))
                    .then_with(|| a.1.cmp(&b.1))
            });
        }
    }

    /// Negate every birth and death, for a filtration by decreasing values computed as the sublevel filtration of
    /// the negated values. Build with [`new`](Self::new) from the negated values (so that a class born at `+inf`
    /// there, which never exists, is left out, but one born at `-inf`, which becomes `+inf`, is kept), then negate.
    pub fn negate(mut self) -> Self {
        for dim in self.essential.iter_mut() {
            for (birth, _) in dim.iter_mut() {
                *birth = -*birth;
            }
        }
        for dim in self.pairings.iter_mut() {
            for ((birth, death), _) in dim.iter_mut() {
                (*birth, *death) = (-*birth, -*death);
            }
        }
        self.sort();
        self
    }

    /// Convert to a `phlite.core.Barcode` (through its private constructor, `Barcode._from_parts`), recording
    /// `algorithm` as what computed it; `representatives` must match whether the bars have representatives.
    ///
    /// If `labels` is given then vertex `i` appears in representatives as `labels[i]`, otherwise as the integer `i`.
    pub fn into_python<'py>(
        self,
        py: Python<'py>,
        algorithm: &ResolvedAlgorithm,
        representatives: bool,
        labels: Option<&[Py<PyAny>]>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cell_to_python = |cell: &Vec<u32>| -> PyResult<Bound<'py, PyTuple>> {
            match labels {
                None => PyTuple::new(py, cell),
                Some(labels) => PyTuple::new(py, cell.iter().map(|&v| labels[v as usize].bind(py))),
            }
        };
        let reps_to_python = |dims: Vec<Vec<&Rep>>| -> PyResult<Bound<'py, PyList>> {
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
            PyList::new(py, dims)
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
        let essential_representatives = representatives
            .then(|| reps_to_python(only_reps(&self.essential)))
            .transpose()?;
        let pairing_representatives = representatives
            .then(|| reps_to_python(only_reps(&self.pairings)))
            .transpose()?;
        core_attribute(py, CoreAttribute::BarcodeFromParts)?.call1((
            PyList::new(py, essential)?,
            PyList::new(py, pairings)?,
            essential_representatives,
            pairing_representatives,
            spec_to_dict(py, &algorithm.to_spec())?,
        ))
    }
}

fn only_reps<T>(dims: &[Vec<(T, Rep)>]) -> Vec<Vec<&Rep>> {
    dims.iter()
        .map(|bars| bars.iter().map(|(_, rep)| rep).collect())
        .collect()
}
