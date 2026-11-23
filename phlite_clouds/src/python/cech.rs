//! `phlite.clouds.cech`: persistence of the Čech filtration of points in any dimension (and the complex itself), by
//! [`crate::cech`].

use phlite::{barcode::Algorithm, columns::ColumnRepresentation};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use phlite_bindings::{algorithm_from_py, AlgorithmDefaults, AlgorithmVisitor, Columns, SortedBars};

use super::rips::read_rows;
use crate::cech::{cech_barcode, cech_simplices};

/// The pipeline `'clouds.cech'`: reduced as an explicit complex, so with `phlite.complexes`' defaults.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "clouds.cech",
    columns: Columns::Pivot,
    cache: false,
    store_r: true,
    // The complex knows the position of every cell in the filtration
    row_ranks: true,
    union_find: None,
    collapse: None,
};

/// Compute the Čech persistent homology of points in any dimension, in dimensions `0, ..., max_dim`.
///
/// The Čech complex at radius `r` is the nerve of the balls of radius `r` about the points: a simplex enters at the
/// radius of the minimal enclosing ball of its points, when their balls first all meet (as GUDHI's `CechComplex`). By
/// the nerve theorem its barcode is that of the union of the balls. In the plane and in space `alpha` and
/// `delaunay_cech` give the same barcode from far smaller complexes; this is for points in more dimensions (or for the
/// Čech complex itself, `cech_complex`). Every simplex of up to `max_dim + 2` points within `max_radius` is listed
/// (candidates from the Rips complex at twice the radius) and reduced as an explicit complex, so give a `max_radius`
/// for more than a few hundred points.
///
/// Args:
///     points: The points, one per row, with any number of finite coordinates: a 2D array or a sequence of sequences.
///     max_dim: The maximum homology dimension to compute.
///     max_radius: The largest radius (default: all); bars still alive there are essential.
///     squared: Report squared radii, as GUDHI does, rather than radii (the default).
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'clouds.cech'`): columns='pivot', cache=False, store_r=True.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`; cells (births and deaths) are simplices, as tuples of point
///     indices.
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (points, max_dim = 1, *, max_radius = None, squared = false, algorithm = None))]
fn cech<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    points: &Bound<'_, PyAny>,
    max_dim: usize,
    max_radius: Option<f64>,
    squared: bool,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let points = read_rows(points, "points")?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let bars = py.detach(|| algorithm.run(CechRun { points, max_dim, max_radius, squared }))?;
    bars.into_python(py, &algorithm, false, None)
}

/// The Čech complex itself: every simplex of dimension at most `dimension` whose minimal enclosing ball has radius at
/// most `max_radius`, with that radius (or with `squared`, squared radius), faces before cofaces. For drawing, or your
/// own computations (over the integers: `phlite.complexes.ChainComplex.from_simplices(simplices, modulus=0)`).
///
/// Args:
///     points: The points, as for `cech`.
///     dimension: The largest dimension of a simplex.
///     max_radius: The largest radius (default: all).
///     squared: Squared radii rather than radii.
///
/// Returns:
///     A list of `(simplex, value)` pairs, each simplex a tuple of point indices in increasing order.
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[gen_stub(override_return_type(type_repr = "builtins.list[tuple[tuple[builtins.int, ...], builtins.float]]", imports = ("builtins")))]
#[pyfunction]
#[pyo3(signature = (points, dimension = 2, *, max_radius = None, squared = false))]
fn cech_complex<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    points: &Bound<'_, PyAny>,
    dimension: usize,
    max_radius: Option<f64>,
    squared: bool,
) -> PyResult<Vec<(Bound<'py, pyo3::types::PyTuple>, f64)>> {
    let points = read_rows(points, "points")?;
    let simplices = py.detach(|| cech_simplices(&points, dimension, max_radius));
    simplices
        .into_iter()
        .map(|(r2, s)| Ok((pyo3::types::PyTuple::new(py, s)?, if squared { r2 } else { r2.sqrt() })))
        .collect()
}

/// Compute the Čech barcode with whichever algorithm is configured.
struct CechRun {
    points: Vec<Vec<f64>>,
    max_dim: usize,
    max_radius: Option<f64>,
    squared: bool,
}

impl AlgorithmVisitor for CechRun {
    type Output = SortedBars;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> SortedBars {
        let bars = cech_barcode(&self.points, self.max_dim, self.max_radius, self.squared, algorithm);
        SortedBars::new(bars, self.max_dim + 1, |simplex: &Vec<u32>| (simplex.len() - 1, simplex.clone()))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cech, m)?)?;
    m.add_function(wrap_pyfunction!(cech_complex, m)?)
}
