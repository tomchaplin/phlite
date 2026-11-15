//! `phlite.clouds.alpha` and `phlite.clouds.delaunay_cech`: persistence of filtrations of the Delaunay triangulation
//! of points in the plane or in space (and the filtered complexes themselves), by [`crate::alpha`].

use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use phlite_bindings::{
    algorithm_from_py, parse_method, AlgorithmDefaults, AlgorithmVisitor, Columns, SortedBars,
};

use super::rips::read_rows;
use crate::alpha::{delaunay_barcode, filtered_delaunay, vertices, DelaunayFiltration};

/// The pipeline `'clouds.alpha'`: reduced as an explicit complex, so with `phlite.complexes`' defaults.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "clouds.alpha",
    columns: Columns::Pivot,
    cache: false,
    store_r: true,
    // The complex knows the position of every cell in the filtration
    row_ranks: true,
    union_find: None,
};

/// The pipeline `'clouds.delaunay_cech'`: reduced as an explicit complex, as alpha complexes are.
pub(crate) const CECH_DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "clouds.delaunay_cech",
    ..DEFAULTS
};

/// Compute the alpha persistent homology of points in the plane or in space, in dimensions `0, ..., max_dim`.
///
/// The alpha complex at radius `r` is the nerve of the balls of radius `r` around the points, each cut down to its
/// point's Voronoi cell: it has the homotopy type of the union of the balls (so the same barcode as the Čech
/// filtration), and it is a subcomplex of the Delaunay triangulation, so it is small (in the plane, about 3 edges and
/// 2 triangles per point; in space, about 7 edges, 13 triangles and 6.5 tetrahedra per point), whatever the radius.
///
/// The Delaunay triangulation is exact in the plane (by spade) and in space (by axiolid-tetrahedralize, with exact
/// predicates and symbolic perturbation of cospherical points); repeated points are merged, and points spanning fewer
/// dimensions than they have coordinates (collinear, or coplanar in space) are triangulated in their own span.
///
/// Args:
///     points: The points, one per row, with 2 or 3 finite coordinates each: a 2D array or a sequence of sequences.
///     max_dim: The maximum homology dimension to compute (default: one less than the number of coordinates, the
///         highest that can be non-zero).
///     squared: Report squared radii, as GUDHI does, rather than radii (the default).
///     method: `'cohomology'` (the default) or `'homology'`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'clouds.alpha'`): columns='pivot', cache=False, store_r=True.
///     representatives: Whether to also compute a representative cycle for every bar.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`. Cells (births, deaths and representatives) are simplices of the
///     Delaunay triangulation, as tuples of point indices (a repeated point is the first of its copies).
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (points, max_dim = None, *, squared = false, method = "cohomology", algorithm = None, representatives = false))]
fn alpha<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    points: &Bound<'_, PyAny>,
    max_dim: Option<usize>,
    squared: bool,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
    method: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let points = read_rows(points, "points")?;
    let n_coordinates = points.first().map_or(2, Vec::len);
    let max_dim = max_dim.unwrap_or(n_coordinates.saturating_sub(1));
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let bars = py.detach(|| {
        algorithm.run(DelaunayRun {
            points,
            filtration: DelaunayFiltration::Alpha,
            max_dim,
            squared,
            method,
            representatives,
        })
    })?;
    bars.map_err(PyErr::from)?
        .into_python(py, &algorithm, representatives, None)
}

/// Compute the Delaunay–Čech persistent homology of points in the plane or in space, in dimensions
/// `0, ..., max_dim`.
///
/// The Delaunay–Čech filtration puts each simplex of the Delaunay triangulation at the radius of its minimal
/// enclosing ball, the radius at which the balls about its vertices first all meet (as GUDHI's `DelaunayCechComplex`):
/// as small as an alpha complex, and with the same barcode as the Čech filtration (Bauer and Edelsbrunner), but with
/// simplices entering at Čech times. Triangulated as for `alpha`.
///
/// Args:
///     points: The points, one per row, with 2 or 3 finite coordinates each: a 2D array or a sequence of sequences.
///     max_dim: The maximum homology dimension to compute (default: one less than the number of coordinates, the
///         highest that can be non-zero).
///     squared: Report squared radii, as GUDHI does, rather than radii (the default).
///     method: `'cohomology'` (the default) or `'homology'`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'clouds.delaunay_cech'`): columns='pivot', cache=False, store_r=True.
///     representatives: Whether to also compute a representative cycle for every bar.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`. Cells (births, deaths and representatives) are simplices of the
///     Delaunay triangulation, as tuples of point indices (a repeated point is the first of its copies).
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (points, max_dim = None, *, squared = false, method = "cohomology", algorithm = None, representatives = false))]
fn delaunay_cech<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    points: &Bound<'_, PyAny>,
    max_dim: Option<usize>,
    squared: bool,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
    method: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let points = read_rows(points, "points")?;
    let n_coordinates = points.first().map_or(2, Vec::len);
    let max_dim = max_dim.unwrap_or(n_coordinates.saturating_sub(1));
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &CECH_DEFAULTS)?;
    let bars = py.detach(|| {
        algorithm.run(DelaunayRun {
            points,
            filtration: DelaunayFiltration::DelaunayCech,
            max_dim,
            squared,
            method,
            representatives,
        })
    })?;
    bars.map_err(PyErr::from)?
        .into_python(py, &algorithm, representatives, None)
}

/// Compute the barcode of a filtration of the Delaunay triangulation with whichever algorithm is configured.
struct DelaunayRun {
    points: Vec<Vec<f64>>,
    filtration: DelaunayFiltration,
    max_dim: usize,
    squared: bool,
    method: Method,
    representatives: bool,
}

impl AlgorithmVisitor for DelaunayRun {
    type Output = Result<SortedBars, crate::CloudError>;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
        let (complex, bars) = delaunay_barcode(
            &self.points,
            self.filtration,
            Some(self.max_dim),
            self.squared,
            self.method,
            algorithm,
            self.representatives,
        )?;
        Ok(SortedBars::new(bars, self.max_dim + 1, |cell| {
            let (dim, simplex) = complex.cell(*cell);
            (dim, vertices(simplex).to_vec())
        }))
    }
}

/// The alpha complex of points in the plane or in space: every simplex of their Delaunay triangulation, with the
/// radius (or with `squared`, the squared radius) at which it appears, in filtration order (by value, then
/// dimension). The complex `alpha` computes the persistence of, for drawing or for your own computations.
///
/// Args:
///     points: The points, as for `alpha`.
///     squared: Squared radii, as GUDHI's, rather than radii (the default).
///
/// Returns:
///     A list of `(simplex, value)` pairs, each simplex a tuple of point indices in increasing order (a repeated point
///     is the first of its copies).
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[gen_stub(override_return_type(type_repr = "builtins.list[tuple[tuple[builtins.int, ...], builtins.float]]", imports = ("builtins")))]
#[pyfunction]
#[pyo3(signature = (points, *, squared = false))]
fn alpha_complex<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    points: &Bound<'_, PyAny>,
    squared: bool,
) -> PyResult<Vec<(Bound<'py, pyo3::types::PyTuple>, f64)>> {
    filtration_list(py, points, squared, DelaunayFiltration::Alpha)
}

/// The Delaunay–Čech complex of points in the plane or in space: every simplex of their Delaunay triangulation, with
/// the radius (or with `squared`, the squared radius) of its minimal enclosing ball, in filtration order (by value,
/// then dimension). The complex `delaunay_cech` computes the persistence of.
///
/// Args:
///     points: The points, as for `delaunay_cech`.
///     squared: Squared radii, as GUDHI's, rather than radii (the default).
///
/// Returns:
///     A list of `(simplex, value)` pairs, each simplex a tuple of point indices in increasing order (a repeated point
///     is the first of its copies).
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[gen_stub(override_return_type(type_repr = "builtins.list[tuple[tuple[builtins.int, ...], builtins.float]]", imports = ("builtins")))]
#[pyfunction]
#[pyo3(signature = (points, *, squared = false))]
fn delaunay_cech_complex<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    points: &Bound<'_, PyAny>,
    squared: bool,
) -> PyResult<Vec<(Bound<'py, pyo3::types::PyTuple>, f64)>> {
    filtration_list(py, points, squared, DelaunayFiltration::DelaunayCech)
}

/// Every simplex of the filtration, with its value, in filtration order.
fn filtration_list<'py>(
    py: Python<'py>,
    points: &Bound<'_, PyAny>,
    squared: bool,
    filtration: DelaunayFiltration,
) -> PyResult<Vec<(Bound<'py, pyo3::types::PyTuple>, f64)>> {
    let points = read_rows(points, "points")?;
    let complex = py
        .detach(|| filtered_delaunay(&points, filtration))?;
    let mut cells: Vec<(f64, usize, &crate::alpha::Simplex)> = complex
        .simplices
        .iter()
        .zip(&complex.values)
        .enumerate()
        .flat_map(|(dim, (simplices, values))| {
            simplices
                .iter()
                .zip(values)
                .map(move |(s, &v)| (if squared { v } else { v.sqrt() }, dim, s))
        })
        .collect();
    cells.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(b.2)));
    cells
        .into_iter()
        .map(|(value, _, s)| Ok((pyo3::types::PyTuple::new(py, vertices(s))?, value)))
        .collect()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(alpha, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_complex, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay_cech, m)?)?;
    m.add_function(wrap_pyfunction!(delaunay_cech_complex, m)?)
}
