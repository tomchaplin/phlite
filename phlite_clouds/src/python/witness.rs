//! `phlite.clouds.witness`: persistence of witness complexes (and the complexes themselves), by [`crate::witness`].

use phlite::{barcode::Algorithm, columns::ColumnRepresentation};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use phlite_bindings::{algorithm_from_py, AlgorithmDefaults, AlgorithmVisitor, SortedBars};

use super::rips::read_rows;
use crate::cech::simplicial_barcode;
use crate::witness::{witness_simplices, Witness};

/// The pipeline `'clouds.witness'`: reduced as an explicit complex, as Čech complexes are.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults { pipeline: "clouds.witness", ..super::cech::DEFAULTS };

fn kind(strong: bool) -> Witness {
    if strong {
        Witness::Strong
    } else {
        Witness::Relaxed
    }
}

/// Compute the persistent homology of a witness complex, in dimensions `0, ..., max_dim`, as GUDHI's
/// `EuclideanWitnessComplex` (relaxed) or, with `strong`, `EuclideanStrongWitnessComplex`.
///
/// A witness complex has simplices on a few *landmarks*, entering as the other points (*witnesses*) vouch for them: a
/// small complex approximating the topology of all the points. With `d(w, l)` the distance from a witness to a
/// landmark, values are `α²`:
///
/// * relaxed (the default; the witness complex of de Silva and Carlsson, relaxed by `α`): `w` witnesses `σ` at
///   `max_{l in σ} d(w, l)² - min_{l not in σ} d(w, l)²`, `σ` at the least of these over the witnesses, and enters once
///   all its faces have too;
/// * strong: `w` witnesses `σ` at `max_{l in σ} d(w, l)² - d(w, L)²` (beyond `w`'s nearest landmark), and `σ` enters at
///   the least of these.
///
/// Args:
///     landmarks: The landmarks, one per row (any number of coordinates).
///     witnesses: The witnesses, one per row (usually all the points, landmarks included).
///     max_dim: The maximum homology dimension to compute.
///     strong: The strong witness complex rather than the relaxed one.
///     max_alpha_square: The largest value (default: all); bars still alive there are essential.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'clouds.witness'`): columns='pivot', cache=False, store_r=True.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`, in values `α²`; cells are simplices, as tuples of landmark
///     indices.
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (landmarks, witnesses, max_dim = 1, *, strong = false, max_alpha_square = None, algorithm = None))]
fn witness<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    landmarks: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    witnesses: &Bound<'_, PyAny>,
    max_dim: usize,
    strong: bool,
    max_alpha_square: Option<f64>,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let landmarks = read_rows(landmarks, "landmarks")?;
    let witnesses = read_rows(witnesses, "witnesses")?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let bars = phlite_bindings::detach(py, || {
        algorithm.run(WitnessRun {
            landmarks,
            witnesses,
            kind: kind(strong),
            max_dim,
            max_alpha_square: max_alpha_square.unwrap_or(f64::INFINITY),
        })
    })?;
    bars.into_python(py, &algorithm, false, None)
}

/// The witness complex itself (see `witness`): every simplex of dimension at most `dimension` on the landmarks with
/// value (`α²`) at most `max_alpha_square`, with that value, faces before cofaces.
///
/// Args:
///     landmarks: The landmarks, as for `witness`.
///     witnesses: The witnesses, as for `witness`.
///     dimension: The largest dimension of a simplex.
///     strong: The strong witness complex rather than the relaxed one.
///     max_alpha_square: The largest value (default: all).
///
/// Returns:
///     A list of `(simplex, value)` pairs, each simplex a tuple of landmark indices in increasing order.
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[gen_stub(override_return_type(type_repr = "builtins.list[tuple[tuple[builtins.int, ...], builtins.float]]", imports = ("builtins")))]
#[pyfunction]
#[pyo3(signature = (landmarks, witnesses, dimension = 2, *, strong = false, max_alpha_square = None))]
fn witness_complex<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    landmarks: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    witnesses: &Bound<'_, PyAny>,
    dimension: usize,
    strong: bool,
    max_alpha_square: Option<f64>,
) -> PyResult<Vec<(Bound<'py, pyo3::types::PyTuple>, f64)>> {
    let landmarks = read_rows(landmarks, "landmarks")?;
    let witnesses = read_rows(witnesses, "witnesses")?;
    let bound = max_alpha_square.unwrap_or(f64::INFINITY);
    let simplices = phlite_bindings::detach(py, || witness_simplices(&landmarks, &witnesses, kind(strong), dimension, bound));
    simplices.into_iter().map(|(v, s)| Ok((pyo3::types::PyTuple::new(py, s)?, v))).collect()
}

struct WitnessRun {
    landmarks: Vec<Vec<f64>>,
    witnesses: Vec<Vec<f64>>,
    kind: Witness,
    max_dim: usize,
    max_alpha_square: f64,
}

impl AlgorithmVisitor for WitnessRun {
    type Output = SortedBars;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> SortedBars {
        let simplices =
            witness_simplices(&self.landmarks, &self.witnesses, self.kind, self.max_dim + 1, self.max_alpha_square);
        let bars = simplicial_barcode(&simplices, self.max_dim, algorithm);
        SortedBars::new(bars, self.max_dim + 1, |simplex: &Vec<u32>| (simplex.len() - 1, simplex.clone()))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(witness, m)?)?;
    m.add_function(wrap_pyfunction!(witness_complex, m)?)
}
