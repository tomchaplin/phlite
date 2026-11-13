//! `phlite.cloud.rips`: Vietoris–Rips persistence of a distance matrix, by [`phlite_cloud::rips`].

use numpy::{PyArray2, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
};
use phlite_cloud::rips::{rips_barcode, rips_index_fits};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use crate::{
    algorithm::{AlgorithmDefaults, AlgorithmVisitor, Columns},
    python::{
        shared::{algorithm_from_py, parse_method, SortedBars},
        AlgorithmConfig, ValueBarcode,
    },
};

/// The pipeline `'cloud.rips'`: Rips columns are cheap to recompute, and caching them dominates the memory used.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "cloud.rips",
    columns: Columns::Heap,
    cache: false,
    store_r: false,
    row_ranks: false,
};

// ====== The entry point ==================================================

/// Check the `n` by `n` matrix with entries `get(i, j)` is a valid distance matrix, and convert it.
fn validate(n: usize, get: impl Fn(usize, usize) -> f64) -> PyResult<Vec<Vec<NotNan<f64>>>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let d = get(i, j);
                    if d.is_nan() || d < 0.0 {
                        return Err(PyValueError::new_err(format!(
                            "distance_matrix[{i}][{j}] is {d}, distances must be non-negative (or infinite)"
                        )));
                    }
                    if d != get(j, i) {
                        return Err(PyValueError::new_err(format!(
                            "distance_matrix must be symmetric, but entries ({i}, {j}) and ({j}, {i}) differ"
                        )));
                    }
                    Ok(NotNan::new(d).unwrap())
                })
                .collect()
        })
        .collect()
}

fn check_square(rows: usize, cols: usize) -> PyResult<()> {
    if rows != cols {
        return Err(PyValueError::new_err(format!(
            "distance_matrix must be square, but has shape ({rows}, {cols})"
        )));
    }
    Ok(())
}

/// Read and validate the distance matrix.
/// A float64 or float32 numpy array is read directly; anything else (lists, other dtypes) is read as a sequence of sequences.
fn read_distances(distance_matrix: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<NotNan<f64>>>> {
    // Only try the numpy path for numpy's own types, so that numpy needn't be installed
    let from_numpy = distance_matrix
        .get_type()
        .module()
        .is_ok_and(|module| module.to_cow().is_ok_and(|m| m == "numpy"));
    if from_numpy {
        if let Ok(array) = distance_matrix.cast::<PyArray2<f64>>() {
            let array = array.readonly();
            let array = array.as_array();
            check_square(array.nrows(), array.ncols())?;
            return validate(array.nrows(), |i, j| array[[i, j]]);
        }
        if let Ok(array) = distance_matrix.cast::<PyArray2<f32>>() {
            let array = array.readonly();
            let array = array.as_array();
            check_square(array.nrows(), array.ncols())?;
            return validate(array.nrows(), |i, j| array[[i, j]] as f64);
        }
        if let Ok(array) = distance_matrix.cast::<PyUntypedArray>() {
            if array.ndim() != 2 {
                return Err(PyValueError::new_err(format!(
                    "distance_matrix must be 2-dimensional, but has {} dimensions",
                    array.ndim()
                )));
            }
        }
    }
    let rows: Vec<Vec<f64>> = distance_matrix.extract()?;
    for (i, row) in rows.iter().enumerate() {
        check_square(rows.len(), row.len()).map_err(|_| {
            PyValueError::new_err(format!(
                "distance_matrix must be square, but row {i} has length {} rather than {}",
                row.len(),
                rows.len()
            ))
        })?;
    }
    validate(rows.len(), |i, j| rows[i][j])
}

/// Compute the Vietoris-Rips persistent homology of a distance matrix, in dimensions `0, ..., max_dim`.
///
/// Args:
///     distance_matrix: A symmetric matrix of pairwise distances: a 2D numpy array (float64 and float32 arrays are read
///         directly, without converting each entry through Python), or a sequence of sequences of floats.
///         Distances must be non-negative. An infinite distance means the edge never appears. The diagonal is ignored.
///     max_dim: The maximum homology dimension to compute.
///     threshold: Only build the Rips complex up to this diameter (default: no limit).
///         Bars still alive at the threshold are reported as essential.
///         Simplices above the enclosing radius are always skipped, since by then the complex is contractible
///         and nothing changes, so a larger threshold gives the same result as none.
///     method: `'cohomology'` (the default) reduces the coboundary matrix, which is usually much faster.
///         `'homology'` reduces the boundary matrix instead, which needs simplices up to dimension `max_dim + 1`.
///     algorithm: The reduction algorithm and its options, as a `phlite.Algorithm` (default: `Algorithm.serial()`).
///         For example `Algorithm.parallel(threads=8)`.
///         Pipeline defaults (`'cloud.rips'`): columns='heap', cache=False, store_r=False.
///     representatives: Whether to also compute a representative cycle for every bar.
///         With `method='cohomology'` these are computed via involution.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`.
#[gen_stub_pyfunction(module = "phlite.cloud")]
#[pyfunction]
#[pyo3(signature = (distance_matrix, max_dim = 1, *, threshold = None, method = "cohomology", algorithm = None, representatives = false))]
#[allow(clippy::too_many_arguments)]
fn rips(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    distance_matrix: &Bound<'_, PyAny>,
    max_dim: usize,
    threshold: Option<f64>,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
    method: &str,
    algorithm: Option<Bound<'_, AlgorithmConfig>>,
    representatives: bool,
) -> PyResult<ValueBarcode> {
    let distances = read_distances(distance_matrix)?;
    let threshold = match threshold {
        None => NotNan::new(f64::INFINITY).unwrap(),
        Some(t) => {
            NotNan::new(t).map_err(|_| PyValueError::new_err("threshold must not be NaN"))?
        }
    };
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let n_points = distances.len();
    // Both methods need simplices of dimension max_dim + 1 (as cofacets, or the top of the boundary matrix)
    let top_dim = max_dim + 1;
    if !rips_index_fits(n_points, top_dim) {
        return Err(PyValueError::new_err(format!(
            "Too many points ({n_points}) to index simplices of dimension {top_dim}"
        )));
    }

    let bars = py.detach(|| {
        algorithm.run(RipsRun {
            distances,
            max_dim,
            threshold,
            method,
            representatives,
        })
    })?;
    bars.into_python(py, &algorithm, representatives, None)
}

/// Compute the Rips barcode with whichever algorithm is configured.
struct RipsRun {
    distances: Vec<Vec<NotNan<f64>>>,
    max_dim: usize,
    threshold: NotNan<f64>,
    method: Method,
    representatives: bool,
}

impl AlgorithmVisitor for RipsRun {
    type Output = SortedBars;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> SortedBars {
        let n_points = self.distances.len();
        let bars = rips_barcode::<_, usize>(
            self.distances,
            self.max_dim,
            self.threshold,
            self.method,
            algorithm,
            self.representatives,
        );
        SortedBars::new(bars, self.max_dim + 1, |simplex| {
            let vertices = simplex.to_vec(n_points);
            (
                vertices.len() - 1,
                vertices.into_iter().map(|v| v as u32).collect(),
            )
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rips, m)?)
}
