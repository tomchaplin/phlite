//! `FilteredMap` (a filtration-preserving chain map between filtered complexes) and six-packs into Python.

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use pyo3_stub_gen::derive::*;

use phlite::fields::NonZeroCoefficient;
use phlite::matrices::MatrixOracle;
use phlite::six_pack::{ConeCell, SixPackBar};
use phlite_bindings::{six_pack_into_python, PackBarParts};

use super::{AnyComplex, ChainComplex, FilteredComplex};
use crate::complex::Cell;
use crate::field::IntField;
use crate::maps::{ComplexMap, ComplexSixPack, MapError};

/// A six-pack into a `phlite.core.SixPack`, with cells of the domain from `domain` and of the codomain from
/// `codomain`, and bars of dimension above `max_dim` left out.
pub(crate) fn pack_to_python<'py, CF: IntField>(
    py: Python<'py>,
    pack: ComplexSixPack<CF>,
    max_dim: Option<usize>,
    domain: &ChainComplex,
    codomain: &ChainComplex,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    fn convert<'py, C, CF: IntField>(
        py: Python<'py>,
        bars: Vec<SixPackBar<f64, C, CF>>,
        max_dim: Option<usize>,
        entry: impl Fn(C, u64) -> PyResult<Bound<'py, PyTuple>>,
    ) -> PyResult<Vec<PackBarParts<'py>>> {
        bars.into_iter()
            .filter(|bar| max_dim.is_none_or(|m| bar.dimension <= m))
            .map(|bar| {
                let representative = bar
                    .representative
                    .map(|rep| {
                        let entries = rep
                            .into_iter()
                            .map(|(x, c)| entry(c, x.to_int()))
                            .collect::<PyResult<Vec<_>>>()?;
                        PyList::new(py, entries)
                    })
                    .transpose()?;
                Ok((
                    bar.dimension,
                    bar.birth,
                    bar.death.unwrap_or(f64::INFINITY),
                    representative,
                ))
            })
            .collect()
    }
    let in_domain = |c: Cell, x: u64| (domain.cell(py, c.0)?, x).into_pyobject(py);
    let in_codomain = |c: Cell, x: u64| (codomain.cell(py, c.0)?, x).into_pyobject(py);
    let in_cone = |c: ConeCell<Cell, Cell>, x: u64| match c {
        ConeCell::Codomain(k) => (codomain.cell(py, k.0)?, x).into_pyobject(py),
        ConeCell::Cone(l) => (domain.cell(py, l.0)?, x, true).into_pyobject(py),
    };
    let barcodes = [
        convert(py, pack.domain, max_dim, in_domain)?,
        convert(py, pack.codomain, max_dim, in_codomain)?,
        convert(py, pack.image, max_dim, in_domain)?,
        convert(py, pack.kernel, max_dim, in_domain)?,
        convert(py, pack.cokernel, max_dim, in_codomain)?,
        convert(py, pack.relative, max_dim, in_cone)?,
    ];
    six_pack_into_python(py, barcodes, representatives)
}

/// Both complexes over the same field, or an error.
macro_rules! dispatch_pair {
    ($a:expr, $b:expr, $x:ident, $y:ident => $body:expr) => {
        match ($a, $b) {
            (AnyComplex::Z2($x), AnyComplex::Z2($y)) => $body,
            (AnyComplex::Z3($x), AnyComplex::Z3($y)) => $body,
            (AnyComplex::Z5($x), AnyComplex::Z5($y)) => $body,
            (AnyComplex::Z7($x), AnyComplex::Z7($y)) => $body,
            (AnyComplex::Z11($x), AnyComplex::Z11($y)) => $body,
            (AnyComplex::Z13($x), AnyComplex::Z13($y)) => $body,
            _ => Err(PyValueError::new_err(
                "the domain and codomain must be over the same field (with a modulus other than 0)",
            )),
        }
    };
}

/// A filtration-preserving chain map `f: L -> K` between two filtered complexes over the same field, whose
/// `six_pack` gives the barcodes of `L` and `K`, of the image, kernel and cokernel of `H(L) -> H(K)`, and of the map's
/// mapping cone, each bar with a representative (computed through the map's mapping cylinder, lazily).
///
/// A map that is the inclusion of a subcomplex with the same values (each cell to one cell, with coefficient 1) is
/// recognised, and its six-pack computed directly, as `FilteredComplex.six_pack` does, without the cylinder.
///
/// ```python
/// L = ChainComplex.from_simplices(...).filtration(...)
/// K = ChainComplex.from_simplices(...).filtration(...)
/// f = FilteredMap.simplicial(L, K, {0: 0, 1: 0, 2: 1})   # from a map of vertices
/// pack = f.six_pack()
/// ```
#[gen_stub_pyclass]
#[pyclass(frozen, module = "phlite.complexes")]
pub struct FilteredMap {
    domain: Py<FilteredComplex>,
    codomain: Py<FilteredComplex>,
    /// By input cell of the domain: `(coefficient, input cell of the codomain)`
    images: Vec<Vec<(i64, u32)>>,
}

impl FilteredMap {
    /// If the map is the inclusion of a subcomplex with the restricted filtration (each cell to one cell, with
    /// coefficient 1, one to one, at the same value), which cells of `K` are its image, and the cell of `L` each comes
    /// from: its six-pack is then computed without the cylinder.
    fn as_inclusion(&self) -> Option<(Vec<bool>, Vec<u32>)> {
        let (l, k) = (&self.domain.get().complex, &self.codomain.get().complex);
        let mut mask = vec![false; k.n_cells()];
        let mut source = vec![u32::MAX; k.n_cells()];
        for (c, image) in self.images.iter().enumerate() {
            let &[(1, target)] = image.as_slice() else { return None };
            if mask[target as usize] || l.value(c as u32) != k.value(target) {
                return None;
            }
            mask[target as usize] = true;
            source[target as usize] = c as u32;
        }
        Some((mask, source))
    }

    /// The map over the complexes' field, checked.
    fn checked<'a, CF: IntField>(
        &self,
        py: Python<'_>,
        domain: &'a crate::complex::Complex<CF>,
        codomain: &'a crate::complex::Complex<CF>,
    ) -> PyResult<ComplexMap<'a, CF>> {
        let images = self
            .images
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .filter_map(|&(x, c)| CF::from_int(x).map(|x| (x, Cell(c))))
                    .collect()
            })
            .collect();
        let (l, k) = (
            self.domain.get().chain.get(),
            self.codomain.get().chain.get(),
        );
        ComplexMap::new(domain, codomain, images).map_err(|e| {
            PyValueError::new_err(match e {
                MapError::Dimension { cell, image } => {
                    format!(
                        "{} maps to {}, of another dimension",
                        l.name(py, cell.0),
                        k.name(py, image.0)
                    )
                }
                MapError::Later { cell, image } => format!(
                    "{} maps to {}, which enters later: the map must preserve the filtration",
                    l.name(py, cell.0),
                    k.name(py, image.0)
                ),
                MapError::NotChainMap { cell } => {
                    format!(
                        "not a chain map: it doesn't commute with the boundary on {}",
                        l.name(py, cell.0)
                    )
                }
            })
        })
    }
}

/// A chain, given as a dict `{cell: coefficient}` or an iterable of cells (each with coefficient 1), as `(coefficient,
/// index)` pairs of `complex`.
fn chain_from_py(complex: &ChainComplex, chain: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, u32)>> {
    if let Ok(dict) = chain.cast::<PyDict>() {
        dict.iter()
            .map(|(cell, x)| Ok((x.extract::<i64>()?, complex.index_or_err(&cell)?)))
            .collect()
    } else {
        chain
            .try_iter()?
            .map(|cell| Ok((1, complex.index_or_err(&cell?)?)))
            .collect()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl FilteredMap {
    /// The chain map from `domain` to `codomain` sending each cell to `images[cell]`.
    ///
    /// Args:
    ///     domain: The filtered complex `L`.
    ///     codomain: The filtered complex `K`, over the same field.
    ///     images: The image of each cell of `L`: a dict from cells of `L` (those left out map to zero), or a sequence
    ///         with an entry per cell (in the cells' order). An image is a chain of `K`: a dict `{cell: coefficient}`, or
    ///         an iterable of cells (coefficient 1 each).
    ///
    /// Raises:
    ///     ValueError: If a cell maps to cells of another dimension or that enter later than it, if the map doesn't
    ///         commute with the boundaries, or if the complexes aren't over the same field.
    #[new]
    fn new(
        py: Python<'_>,
        domain: Py<FilteredComplex>,
        codomain: Py<FilteredComplex>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing")))] images: &Bound<
            '_,
            PyAny,
        >,
    ) -> PyResult<Self> {
        let (l, k) = (domain.get().chain.get(), codomain.get().chain.get());
        let n = domain.get().complex.n_cells();
        let mut by_cell = vec![vec![]; n];
        if let Ok(dict) = images.cast::<PyDict>() {
            for (cell, chain) in dict.iter() {
                by_cell[l.index_or_err(&cell)? as usize] = chain_from_py(k, &chain)?;
            }
        } else {
            let items: Vec<Bound<'_, PyAny>> = images.try_iter()?.collect::<PyResult<_>>()?;
            if items.len() != n {
                return Err(PyValueError::new_err(format!(
                    "images has {} entries, but the domain has {n} cells (give a dict to leave some out)",
                    items.len()
                )));
            }
            for (slot, chain) in by_cell.iter_mut().zip(&items) {
                *slot = chain_from_py(k, chain)?;
            }
        }
        let map = Self {
            domain,
            codomain,
            images: by_cell,
        };
        let (dl, dk) = (&map.domain.get().complex, &map.codomain.get().complex);
        dispatch_pair!(dl, dk, a, b => map.checked(py, a, b).map(|_| ()))?;
        Ok(map)
    }

    /// The simplicial map from `domain` to `codomain` (both built by `ChainComplex.from_simplices`) induced by a map of
    /// vertices: a simplex goes to the simplex on its vertices' images (with the sign of the permutation sorting them),
    /// or to zero if two of its vertices go to the same one.
    ///
    /// Args:
    ///     domain: The filtered complex `L`.
    ///     codomain: The filtered complex `K`, with the image of every simplex of `L` no later than it.
    ///     vertex_map: The image of each vertex of `L`: a dict, or a sequence indexed by vertex.
    #[staticmethod]
    fn simplicial(
        py: Python<'_>,
        domain: Py<FilteredComplex>,
        codomain: Py<FilteredComplex>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing")))]
        vertex_map: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let l = domain.get().chain.get();
        let cells = (0..domain.get().complex.n_cells() as u32)
            .map(|i| l.cell(py, i))
            .collect::<PyResult<Vec<_>>>()?;
        let images = py
            .import("phlite.complexes")?
            .getattr("_simplicial_images")?
            .call1((cells, vertex_map))?;
        Self::new(py, domain, codomain, &images)
    }

    /// The six-pack of the map: the barcodes of `L` and `K`, of the image, kernel and cokernel of `H(L) -> H(K)`, and
    /// of the mapping cone (see `phlite.core.SixPack`).
    ///
    /// Args:
    ///     max_dim: The maximum dimension of the bars (default: all).
    ///     representatives: Whether to compute a representative for every bar.
    ///     method: `'auto'` (inclusions of subcomplexes directly, other maps through the mapping cylinder) or
    ///         `'cylinder'` (always through the cylinder).
    ///
    /// Returns:
    ///     A `phlite.core.SixPack`. Representatives are lists of `(cell, coefficient)`: cycles of `L` for the domain,
    ///     image and kernel, of `K` for the codomain and cokernel; those of the relative barcode are chains of the
    ///     mapping cone, with `phlite.core.Cone(cell)` the cone on a cell of `L`.
    #[pyo3(signature = (*, max_dim = None, representatives = true, method = "auto"))]
    #[gen_stub(override_return_type(type_repr = "phlite.core.SixPack", imports = ("phlite.core")))]
    fn six_pack<'py>(
        &self,
        py: Python<'py>,
        max_dim: Option<usize>,
        representatives: bool,
        #[gen_stub(override_type(type_repr = "typing.Literal['auto', 'cylinder']", imports = ("typing")))] method: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let cylinder = match method {
            "auto" => false,
            "cylinder" => true,
            other => return Err(PyValueError::new_err(format!("method must be 'auto' or 'cylinder', got {other:?}"))),
        };
        let (l, k) = (self.domain.get(), self.codomain.get());
        dispatch_pair!(&l.complex, &k.complex, a, b => {
            if let Some((mask, source)) = self.as_inclusion().filter(|_| !cylinder) {
                // (Checked as a chain map when made: so the image is a subcomplex)
                let pack = phlite_bindings::detach(py, || b.six_pack(&mask, representatives, max_dim))
                    .map_err(|_| PyValueError::new_err("the image of an inclusion isn't a subcomplex"))?;
                let mut pack = pack.map(|v| v, |c: Cell| Cell(source[c.idx()]), |c| c);
                // Relative cycles of (K, L) as cycles of the mapping cone, as the cylinder gives them: a chain `a` of
                // K outside L, with boundary in L, is the class of `a - Cone(f^-1 ∂a)`
                let boundary = b.boundary();
                for bar in &mut pack.relative {
                    if let Some(rep) = &mut bar.representative {
                        let mut cone = vec![];
                        for (x, cell) in rep.iter() {
                            if let &ConeCell::Codomain(k) = cell {
                                let x = *x;
                                for (y, face) in boundary.column(b.rank(k)) {
                                    cone.push((x * y, b.cell(face)));
                                }
                            }
                        }
                        for (x, face) in phlite::image_kernel::consolidated(cone.into_iter()) {
                            rep.push((x.additive_inverse(), ConeCell::Cone(Cell(source[face.idx()]))));
                        }
                    }
                }
                return pack_to_python(py, pack, max_dim, l.chain.get(), k.chain.get(), representatives);
            }
            let map = self.checked(py, a, b)?;
            let pack = phlite_bindings::detach(py, || map.six_pack(representatives, max_dim));
            pack_to_python(py, pack, max_dim, l.chain.get(), k.chain.get(), representatives)
        })
    }

    /// The domain `L`.
    #[getter]
    fn domain(&self, py: Python<'_>) -> Py<FilteredComplex> {
        self.domain.clone_ref(py)
    }

    /// The codomain `K`.
    #[getter]
    fn codomain(&self, py: Python<'_>) -> Py<FilteredComplex> {
        self.codomain.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "FilteredMap({} cells -> {} cells)",
            self.domain.get().complex.n_cells(),
            self.codomain.get().complex.n_cells()
        )
    }
}
