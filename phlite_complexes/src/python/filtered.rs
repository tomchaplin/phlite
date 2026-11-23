//! `FilteredComplex` (a chain complex with a filtration) and `ComplexBarcode` (its persistence, in terms of cells).

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{
    exceptions::{PyOverflowError, PyValueError},
    prelude::*,
    types::{PyDict, PyList},
};
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

impl FilteredComplex {
    /// Which cells are in `subcomplex`: a sequence of cells, or of booleans (one per cell, in the cells' order).
    pub(crate) fn subcomplex_mask(&self, subcomplex: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
        let n = self.complex.n_cells();
        let chain = self.chain.get();
        let items: Vec<Bound<'_, PyAny>> = subcomplex.try_iter()?.collect::<PyResult<_>>()?;
        let mut mask = vec![false; n];
        // (Python's bools, or numpy's)
        let is_bool = |x: &Bound<'_, PyAny>| {
            x.is_instance_of::<pyo3::types::PyBool>()
                || x.get_type().name().is_ok_and(|name| {
                    name.to_cow()
                        .is_ok_and(|name| name == "bool" || name == "bool_")
                })
        };
        if items.len() == n && items.iter().all(is_bool) {
            for (m, x) in mask.iter_mut().zip(&items) {
                *m = x.extract()?;
            }
        } else {
            for x in &items {
                mask[chain.index_or_err(x)? as usize] = true;
            }
        }
        Ok(mask)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl FilteredComplex {
    /// A filtered simplicial complex in one step: from `(simplex, value)` pairs, as every exporter of a filtration
    /// returns them (`phlite.clouds.rips_complex`, `alpha_complex`, `cech_complex`, `witness_complex`, ...), or from
    /// simplices and their values. Shorthand for `ChainComplex.from_simplices(simplices, modulus=...)
    /// .filtration(values)`, so that any filtration goes on to `persistence`, `six_pack`, `spectral_sequence`, or (with
    /// `modulus=0`) `torsion`:
    ///
    /// ```python
    /// FilteredComplex.from_simplices(clouds.cech_complex(points, 2)).persistence(1)
    /// ```
    ///
    /// Args:
    ///     simplices: `(simplex, value)` pairs, or (with `values`) simplices, each a sequence of integer vertices.
    ///     values: The simplices' values, in the same order (if `simplices` are bare simplices).
    ///     modulus: The characteristic of the coefficient field, or 0 for the integers.
    #[staticmethod]
    #[pyo3(signature = (simplices, values = None, *, modulus = 2))]
    fn from_simplices<'py>(
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Iterable[typing.Any]", imports = ("typing")))] simplices: &Bound<'py, PyAny>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Any]", imports = ("typing")))] values: Option<&Bound<'py, PyAny>>,
        modulus: u64,
    ) -> PyResult<Py<FilteredComplex>> {
        let (simplices, values) = match values {
            Some(values) => (simplices.clone(), values.clone()),
            None => {
                let (mut cells, mut vals) = (vec![], vec![]);
                for pair in simplices.try_iter()? {
                    let (cell, value): (Bound<'py, PyAny>, f64) = pair?.extract().map_err(|_| {
                        PyValueError::new_err("give (simplex, value) pairs, or the simplices and their values")
                    })?;
                    cells.push(cell);
                    vals.push(value);
                }
                (PyList::new(py, cells)?.into_any(), PyList::new(py, vals)?.into_any())
            }
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item("modulus", modulus)?;
        let chain = py.get_type::<ChainComplex>().call_method("from_simplices", (simplices,), Some(&kwargs))?;
        chain.call_method1("filtration", (values,))?.extract().map_err(PyErr::from)
    }

    /// The six-pack of persistence barcodes of the inclusion of a subcomplex `L` into this filtered complex `K`: the
    /// barcodes of `L` and `K`, of the image, kernel and cokernel of `H(L) -> H(K)`, and of the pair `(K, L)`, with a
    /// representative for every bar (see `phlite.core.SixPack`).
    ///
    /// Args:
    ///     subcomplex: The cells of `L`: a sequence of cells, or of booleans (one per cell, in the cells' order). Every
    ///         face of a cell of `L` must be in `L`.
    ///     max_dim: The maximum dimension of the bars (default: all).
    ///     representatives: Whether to compute a representative for every bar.
    ///
    /// Returns:
    ///     A `phlite.core.SixPack`. Representatives are lists of `(cell, coefficient)`: cycles of `L` for the domain,
    ///     image and kernel, of `K` for the codomain and cokernel, and relative cycles (cells of `K` outside `L`) for the
    ///     relative barcode. Bars of length zero are left out.
    ///
    /// Raises:
    ///     ValueError: If the cells aren't a subcomplex, or the complex is over the integers.
    #[pyo3(signature = (subcomplex, *, max_dim = None, representatives = true))]
    #[gen_stub(override_return_type(type_repr = "phlite.core.SixPack", imports = ("phlite.core")))]
    fn six_pack<'py>(
        slf: &Bound<'py, Self>,
        subcomplex: &Bound<'py, PyAny>,
        max_dim: Option<usize>,
        representatives: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        let chain = this.chain.get();
        let mask = this.subcomplex_mask(subcomplex)?;
        let not_subcomplex = |(cell, face): (crate::complex::Cell, crate::complex::Cell)| {
            PyValueError::new_err(format!(
                "not a subcomplex: {} is in it but its face {} isn't",
                chain.name(py, cell.0),
                chain.name(py, face.0)
            ))
        };
        dispatch_field!(AnyComplex, &this.complex, c => {
            let pack = phlite_bindings::detach(py, || c.six_pack(&mask, representatives, max_dim)).map_err(not_subcomplex)?;
            super::maps::pack_to_python(py, pack, max_dim, chain, chain, representatives)
        }, integers => Err(PyValueError::new_err("six_pack needs a field (build the complex with a modulus other than 0)")))
    }

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
    /// Over the integers (a complex built with `modulus=0`) there is no barcode: this is the barcode over Q, computed
    /// exactly by a fraction-free reduction over Z (`method` and `algorithm` are ignored, and there are no
    /// representatives); `torsion()` gives the primes whose barcodes differ from it.
    ///
    /// Returns:
    ///     The barcode. Bars of length zero (birth and death at the same filtration value) are omitted.
    ///
    /// Raises:
    ///     ValueError: Over the integers, if `representatives` is set.
    ///     OverflowError: Over the integers, if an entry outgrows 128 bits.
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
        if let AnyComplex::Integers(integral) = &slf.get().complex {
            if representatives {
                return Err(PyValueError::new_err("no representatives over the integers"));
            }
            let _ = method;
            let result = integral_persistence(slf.py(), integral)?;
            let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
            return Ok(ComplexBarcode {
                data: super::BarData::from_rank_pairs(integral, &result.rational, max_dim),
                complex: slf.clone().unbind(),
                algorithm: algorithm_to_py(slf.py(), &algorithm)?.unbind(),
            });
        }
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

    /// The spectral sequence of this filtration, over the complex's coefficients, computed one page at a time: a
    /// `phlite.core.SpectralSequence` (`ss.next()` gives `E^1`, then `E^2`, ...; `ss.page(r)` any page; printing a page
    /// shows it as a table, with its differentials).
    ///
    /// The filtration's steps are its distinct values, in increasing order, and `E^r_p` is the page-`r` term at
    /// step `p`, whose differential `d^r` goes to step `p - r` (and down one degree). Page `E^1` is the homology of
    /// the associated graded `F_p / F_{p-1}`, and the pages converge to the homology of the whole complex.
    ///
    /// Over a field the spectral sequence is read off the barcode (computed once, here): a bar of degree `k` from step
    /// `b` to step `d` is a generator of `E^r_b` in degree `k` and of `E^r_d` in degree `k + 1` for every `r <= d - b`,
    /// cancelled by `d^{d-b}`. Over the integers (a complex built with `modulus=0`) there is no barcode, and each entry
    /// is computed exactly when its page is asked for, as a quotient of lattices,
    /// `E^r_p = Z^r_p / (Z^{r-1}_{p-1} + d Z^{r-1}_{p+r-1})` with `Z^r_p = {x in F_p : dx in F_{p-r}}`, from the cells
    /// with steps `p - r + 1, ..., p + r - 1` alone (by dense integer elimination: for up to a few thousand cells in
    /// such a window).
    ///
    /// Args:
    ///     max_dim: The largest degree (default: the top dimension).
    ///     integer_steps: Whether the filtration values are the steps themselves (they must then be non-negative
    ///         integers): `F_p` is the cells with value at most `p`, for every `p` up to the largest value, so empty
    ///         levels count (as in a Serre spectral sequence filtered by the dimension of the base). By default the
    ///         steps are the distinct values, in increasing order.
    ///     algorithm: The reduction algorithm, as for `persistence` (over a field only).
    ///
    /// Returns:
    ///     A `phlite.core.SpectralSequence`, whose steps are labelled by the filtration values.
    ///
    /// Raises:
    ///     ValueError: With `integer_steps`, if a value isn't a non-negative integer.
    #[pyo3(signature = (max_dim = None, *, integer_steps = false, algorithm = None))]
    #[gen_stub(override_return_type(type_repr = "core.SpectralSequence", imports = ("phlite.core")))]
    fn spectral_sequence<'py>(
        slf: &Bound<'py, Self>,
        max_dim: Option<usize>,
        integer_steps: bool,
        #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
        algorithm: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use phlite::integral::Ring;
        use phlite::spectral::{step_of, BarcodeSpectral, LocalSpectral, StepBar, WithGenerators};
        use phlite_bindings::spectral::{spectral_sequence, SpectralBackend};
        let py = slf.py();
        let complex = &slf.get().complex;
        let n_cells = complex.n_cells();
        let mut steps: Vec<f64> = (0..n_cells as u32).map(|i| complex.value(i)).collect();
        steps.sort_by(f64::total_cmp);
        steps.dedup();
        if integer_steps {
            if let Some(bad) = steps.iter().find(|v| **v < 0.0 || v.fract() != 0.0 || **v > 1e7) {
                return Err(PyValueError::new_err(format!(
                    "integer_steps needs the filtration values to be non-negative integers, got {bad}"
                )));
            }
            let last = steps.last().map_or(0.0, |&v| v);
            steps = (0..=last as usize).map(|p| p as f64).collect();
        }
        let step_of_value = |v: &f64| if integer_steps { *v as usize } else { step_of(&steps, v) };
        let top = complex.top_dimension().unwrap_or(0);
        let n_degrees = max_dim.unwrap_or(top) + 1;
        let filtered = if integer_steps { complex.integral_filtered_by_value() } else { complex.integral_filtered() };
        let modulus = complex.modulus();
        let (input_of_rank, rank_of_input) = complex.rank_maps();
        let cells = Box::new(ComplexCells { chain: slf.get().chain.clone_ref(py), input_of_rank, rank_of_input });
        let backend = if let AnyComplex::Integers(_) = complex {
            SpectralBackend::new(Box::new(LocalSpectral::new(filtered, n_degrees, None)), true)
        } else {
            let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
            let data = complex.persistence(py, Some(n_degrees - 1), "cohomology", &algorithm, false)?;
            let bars: Vec<StepBar> = data
                .bars
                .iter()
                .map(|bar| StepBar {
                    degree: bar.dimension as usize,
                    birth: step_of_value(&bar.birth_value),
                    death: bar.death_value.is_finite().then(|| step_of_value(&bar.death_value)),
                })
                .collect();
            let p = modulus.expect("a field");
            // Entries from the barcode; generators computed over Z/p, entry by entry, when asked for
            let backend = WithGenerators {
                entries: BarcodeSpectral::new(&bars, steps.len(), n_degrees, None),
                generators: LocalSpectral::new(filtered, n_degrees, None).over(Ring::Mod(p)),
            };
            SpectralBackend::new(Box::new(backend), false)
        };
        let backend = backend.with_cells(cells, modulus);
        let kwargs = PyDict::new(py);
        kwargs.set_item("steps", steps)?;
        spectral_sequence(py, backend, &kwargs)
    }

    /// The torsion of persistence over the integers (a complex built with `modulus=0`): every prime `p` whose barcode
    /// over `F_p` differs from that over Q (which `persistence()` gives), with that barcode.
    ///
    /// These are the primes of the torsion, whether in the homology of a subcomplex of the filtration or only in the
    /// maps between them (a loop that, later, is `p` times another). They are found exactly: a fraction-free reduction
    /// over Z, whose pivot pattern is that over Q, bounds them (a prime dividing none of its pivots and contents sees
    /// the same reduction mod p), and each candidate is checked by reducing mod p, as Perseus does.
    ///
    /// Args:
    ///     max_dim: The maximum dimension of the bars (default: the largest dimension of a cell).
    ///
    /// Returns:
    ///     A dict from each such prime (increasing) to its barcode over `F_p` (bars of length zero omitted).
    ///
    /// Raises:
    ///     ValueError: If the complex isn't over the integers.
    ///     OverflowError: If an entry outgrows 128 bits.
    #[pyo3(signature = (max_dim = None))]
    #[gen_stub(override_return_type(type_repr = "dict[builtins.int, ComplexBarcode]", imports = ("builtins")))]
    fn torsion<'py>(slf: &Bound<'py, Self>, max_dim: Option<usize>) -> PyResult<Bound<'py, PyDict>> {
        let AnyComplex::Integers(integral) = &slf.get().complex else {
            return Err(PyValueError::new_err("torsion() is for complexes over the integers (modulus=0)"));
        };
        let py = slf.py();
        let result = integral_persistence(py, integral)?;
        let algorithm = algorithm_to_py(py, &algorithm_from_py(None, &DEFAULTS)?)?;
        let out = PyDict::new(py);
        for (p, bars) in &result.torsion {
            let barcode = ComplexBarcode {
                data: super::BarData::from_rank_pairs(integral, bars, max_dim),
                complex: slf.clone().unbind(),
                algorithm: algorithm.clone().unbind(),
            };
            out.set_item(p, Bound::new(py, barcode)?)?;
        }
        Ok(out)
    }

    /// The persistent homology of the image, kernel and cokernel of the map `H(L) -> H(K)` induced by the inclusion of
    /// a subcomplex `L` (with the restricted filtration) into this filtered complex `K` (Cohen-Steiner, Edelsbrunner,
    /// Harer and Morozov), as phimaker computes them, and of the pair `(K, L)`.
    ///
    /// The image is what of `L`'s homology survives in `K`; the kernel what of it dies in `K` (a class of `L` that bounds
    /// in `K`); the cokernel what of `K`'s homology doesn't come from `L`. The barcodes of `K` and `L` come from the
    /// fast reductions (clearing, apparent pairs), which also tell which columns of the other matrices need reducing.
    ///
    /// Args:
    ///     subcomplex: The cells of `L`: a sequence of cells, or of booleans (one per cell, in the cells' order). Every
    ///         face of a cell of `L` must be in `L`.
    ///     max_dim: The maximum dimension of the bars (default: the largest dimension of a cell).
    ///
    /// Returns:
    ///     A dict of barcodes: `'image'`, `'kernel'`, `'cokernel'` and `'relative'` (of `(K, L)`), each bar in the
    ///     dimension of its homology (a kernel class of `H_k` is born at the `(k + 1)`-cell that makes a cycle of `L` a
    ///     boundary in `K`, and its cells are given as such). Bars of length zero are omitted.
    ///
    /// Raises:
    ///     ValueError: If the cells aren't a subcomplex, or the complex is over the integers.
    #[pyo3(signature = (subcomplex, max_dim = None))]
    #[gen_stub(override_return_type(type_repr = "dict[builtins.str, ComplexBarcode]", imports = ("builtins")))]
    fn image_kernel_cokernel<'py>(
        slf: &Bound<'py, Self>,
        subcomplex: &Bound<'py, PyAny>,
        max_dim: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let this = slf.get();
        let chain = this.chain.get();
        let mask = this.subcomplex_mask(subcomplex)?;
        let pairs = phlite_bindings::detach(py, || this.complex.image_kernel_cokernel(&mask))?
            .map_err(|(cell, face)| {
                PyValueError::new_err(format!(
                    "not a subcomplex: {} is in it but its face {} isn't",
                    chain.name(py, cell),
                    chain.name(py, face)
                ))
            })?;
        let algorithm = algorithm_to_py(py, &algorithm_from_py(None, &DEFAULTS)?)?;
        let out = PyDict::new(py);
        let AnyComplex::Integers(_) = &this.complex else {
            for (name, bars) in ["image", "kernel", "cokernel", "relative"].into_iter().zip(pairs) {
                // (A kernel class of H_k is born at the (k + 1)-cell that makes it a boundary in K)
                let shift = usize::from(name == "kernel");
                let data = this.complex.bar_data_from_rank_pairs(&bars, max_dim, shift);
                let barcode = ComplexBarcode { data, complex: slf.clone().unbind(), algorithm: algorithm.clone().unbind() };
                out.set_item(name, Bound::new(py, barcode)?)?;
            }
            return Ok(out);
        };
        unreachable!("refused over the integers")
    }

    /// The persistent homology group in dimension `dim` from `start` to `end`: the image of `H_dim` of the subcomplex
    /// of cells with value at most `start` in `H_dim` of the one at `end`.
    ///
    /// Over a field this is the number of bars of dimension `dim` born at or before `start` and alive past `end` (the
    /// persistent Betti number), with no torsion. Over the integers (a complex built with `modulus=0`) it is computed
    /// exactly, as `Z(F_start) / d{x in F_end : dx in F_start}`, with its torsion.
    ///
    /// Args:
    ///     dim: The homology dimension.
    ///     start: The filtration value of the subcomplex mapped from.
    ///     end: The filtration value of the subcomplex mapped into (at least `start`).
    ///     algorithm: The reduction algorithm, as for `persistence` (over a field only).
    ///
    /// Returns:
    ///     `(rank, torsion)`: the free rank, and the invariant factors of the torsion (each dividing the next).
    ///
    /// Raises:
    ///     OverflowError: If, over the integers, an entry outgrows 128 bits.
    #[pyo3(signature = (dim, start, end, *, algorithm = None))]
    fn persistent_group(
        slf: &Bound<'_, Self>,
        dim: usize,
        start: f64,
        end: f64,
        #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
        algorithm: Option<Bound<'_, PyAny>>,
    ) -> PyResult<(usize, Vec<u128>)> {
        if start.is_nan() || end.is_nan() || start > end {
            return Err(PyValueError::new_err(format!(
                "A persistent group needs start <= end, got {start} and {end}"
            )));
        }
        let py = slf.py();
        let complex = &slf.get().complex;
        if let AnyComplex::Integers(integral) = complex {
            let steps = integral.steps();
            // The last step at or before each value; before the first step the subcomplex is empty
            let up_to = |v: f64| steps.partition_point(|&s| s <= v).checked_sub(1);
            let Some(i) = up_to(start) else {
                return Ok((0, vec![]));
            };
            let j = up_to(end).expect("end >= start");
            let filtered = integral.integral();
            let group = py
                .detach(|| phlite::integral::persistent_group(&filtered, dim, i, j))
                .map_err(|e| PyOverflowError::new_err(e.to_string()))?;
            return Ok((group.rank, group.torsion));
        }
        let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
        let data = complex.persistence(py, Some(dim), "cohomology", &algorithm, false)?;
        let rank = data
            .bars
            .iter()
            .filter(|bar| bar.dimension as usize == dim && bar.birth_value <= start && bar.death_value > end)
            .count();
        Ok((rank, vec![]))
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

/// The cells of a filtered complex for generators' chains: `phlite::integral` numbers them by rank, Python by the
/// input's cells.
struct ComplexCells {
    chain: Py<ChainComplex>,
    input_of_rank: Vec<u32>,
    rank_of_input: Vec<u32>,
}

impl phlite_bindings::spectral::CellNames for ComplexCells {
    fn name<'py>(&self, py: Python<'py>, cell: usize) -> PyResult<Bound<'py, PyAny>> {
        self.chain.get().cell(py, self.input_of_rank[cell])
    }

    fn index(&self, _py: Python<'_>, cell: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        Ok(self.chain.get().lookup(cell)?.map(|i| self.rank_of_input[i as usize] as usize))
    }
}

/// The persistence over Z of a complex over the integers (see `phlite::integral_persistence`), without the GIL.
fn integral_persistence(
    py: Python<'_>,
    complex: &crate::complex::Complex<crate::field::Integer>,
) -> PyResult<phlite::integral_persistence::IntegralPersistence> {
    let filtered = complex.integral();
    py.detach(|| phlite::integral_persistence::integral_persistence(&filtered.boundary, &filtered.degree))
        .map_err(|e| PyOverflowError::new_err(e.to_string()))
}
