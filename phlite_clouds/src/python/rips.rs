//! `phlite.clouds.rips`: Vietoris–Rips persistence of a distance matrix (dense or sparse) or a point cloud, by
//! [`crate::rips`].

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods,
};
use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use phlite_bindings::{
    algorithm_from_py, parse_method, AlgorithmDefaults, AlgorithmVisitor, Columns, SortedBars,
};

use crate::rips::{rips_barcode, rips_index_fits, Distances, Metric, SparseDistances};

/// The pipeline `'clouds.rips'`: Rips columns are cheap to recompute, and caching them dominates the memory used.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "clouds.rips",
    columns: Columns::Heap,
    cache: false,
    store_r: false,
    row_ranks: false,
};

// ====== Reading the input ================================================

/// The distances to compute with: a dense matrix, or a sparse one (missing entries never appear).
enum Input {
    Dense(Vec<Vec<NotNan<f64>>>),
    Sparse(SparseDistances),
}

impl Input {
    fn n_points(&self) -> usize {
        match self {
            Input::Dense(d) => d.n_points(),
            Input::Sparse(d) => d.n_points(),
        }
    }
}

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

/// Whether `object` is an instance of one of numpy's own types (so that numpy needn't be installed otherwise).
fn is_numpy(object: &Bound<'_, PyAny>) -> bool {
    object
        .get_type()
        .module()
        .is_ok_and(|module| module.to_cow().is_ok_and(|m| m == "numpy"))
}

/// Read a 2D array of floats: a float64 or float32 numpy array directly, anything else (lists, other dtypes) as a
/// sequence of sequences. Returns its rows.
fn read_rows(object: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<Vec<f64>>> {
    if is_numpy(object) {
        if let Ok(array) = object.cast::<PyArray2<f64>>() {
            let array = array.readonly();
            return Ok(array.as_array().rows().into_iter().map(|row| row.to_vec()).collect());
        }
        if let Ok(array) = object.cast::<PyArray2<f32>>() {
            let array = array.readonly();
            return Ok(array
                .as_array()
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|&x| x as f64).collect())
                .collect());
        }
        if let Ok(array) = object.cast::<PyUntypedArray>() {
            if array.ndim() != 2 {
                return Err(PyValueError::new_err(format!(
                    "{name} must be 2-dimensional, but has {} dimensions",
                    array.ndim()
                )));
            }
        }
    }
    let rows: Vec<Vec<f64>> = object.extract()?;
    if let Some((i, row)) = rows.iter().enumerate().find(|(_, row)| row.len() != rows[0].len()) {
        return Err(PyValueError::new_err(format!(
            "The rows of {name} must have the same length, but row {i} has length {} rather than {}",
            row.len(),
            rows[0].len()
        )));
    }
    Ok(rows)
}

/// Read and validate a dense distance matrix.
fn read_dense(distance_matrix: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<NotNan<f64>>>> {
    // Read a numpy array without copying it first
    if is_numpy(distance_matrix) {
        if let Ok(array) = distance_matrix.cast::<PyArray2<f64>>() {
            let array = array.readonly();
            let array = array.as_array();
            check_square(array.nrows(), array.ncols())?;
            return validate(array.nrows(), |i, j| array[[i, j]]);
        }
    }
    let rows = read_rows(distance_matrix, "distance_matrix")?;
    check_square(rows.len(), rows.first().map_or(0, |row| row.len()))?;
    validate(rows.len(), |i, j| rows[i][j])
}

/// Whether `object` is a scipy sparse matrix or array.
fn is_scipy_sparse(object: &Bound<'_, PyAny>) -> bool {
    object
        .get_type()
        .module()
        .is_ok_and(|module| module.to_cow().is_ok_and(|m| m.starts_with("scipy.sparse")))
}

/// Read a scipy sparse distance matrix (any format), through its COO form.
fn read_sparse(matrix: &Bound<'_, PyAny>) -> PyResult<SparseDistances> {
    let (rows, cols): (usize, usize) = matrix.getattr("shape")?.extract()?;
    check_square(rows, cols)?;
    let coo = matrix.call_method0("tocoo")?;
    let as_array = |attribute: &str, dtype: &str| -> PyResult<Bound<'_, PyAny>> {
        coo.getattr(attribute)?.call_method1("astype", (dtype,))
    };
    let row = as_array("row", "int64")?;
    let col = as_array("col", "int64")?;
    let data = as_array("data", "float64")?;
    let row: PyReadonlyArray1<i64> = row.cast_into::<PyArray1<i64>>()?.readonly();
    let col: PyReadonlyArray1<i64> = col.cast_into::<PyArray1<i64>>()?.readonly();
    let data: PyReadonlyArray1<f64> = data.cast_into::<PyArray1<f64>>()?.readonly();
    let edges = row
        .as_array()
        .iter()
        .zip(col.as_array())
        .zip(data.as_array())
        .map(|((&i, &j), &d)| (i as usize, j as usize, d))
        .collect::<Vec<_>>();
    SparseDistances::from_edges(rows, edges).map_err(PyValueError::new_err)
}

fn parse_metric(metric: &str) -> PyResult<Metric> {
    match metric {
        "euclidean" => Ok(Metric::Euclidean),
        "manhattan" | "cityblock" => Ok(Metric::Manhattan),
        "chebyshev" => Ok(Metric::Chebyshev),
        other => Err(PyValueError::new_err(format!(
            "Unknown metric {other:?}, expected 'euclidean', 'manhattan' or 'chebyshev'"
        ))),
    }
}

/// Read the points (one per row) and turn them into distances: only those up to `threshold` (as a sparse matrix) if
/// it is finite, else all of them.
fn read_points(points: &Bound<'_, PyAny>, metric: Metric, threshold: NotNan<f64>) -> PyResult<Input> {
    let points = read_rows(points, "points")?;
    if points.iter().flatten().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("Every coordinate of points must be finite"));
    }
    Ok(if threshold.is_finite() {
        Input::Sparse(SparseDistances::from_points(&points, metric, threshold).map_err(PyValueError::new_err)?)
    } else {
        Input::Dense(metric.distance_matrix(&points))
    })
}

// ====== The entry point ==================================================
/// Compute the Vietoris-Rips persistent homology of a distance matrix or a point cloud, in dimensions
/// `0, ..., max_dim`.
///
/// Give either a distance matrix (dense or sparse), as the first argument, or `points`:
///
/// ```python
/// rips(distances)                                 # a dense distance matrix
/// rips(scipy.sparse.coo_matrix(...))              # a sparse one: missing entries never appear
/// rips(points=X, max_dim=2)                       # points (one per row), Euclidean distances
/// rips(points=X, metric="chebyshev", threshold=1.5)
/// ```
///
/// Args:
///     distance_matrix: A symmetric matrix of pairwise distances: a 2D numpy array (float64 and float32 arrays are read
///         directly, without converting each entry through Python), a sequence of sequences of floats, or a scipy
///         sparse matrix or array (any format), whose missing entries are edges that never appear, so that memory
///         and time grow with the number of edges rather than the square of the number of points. A sparse matrix
///         may give each edge once (either way round) or both ways round with the same distance.
///         Distances must be non-negative. An infinite distance means the edge never appears. The diagonal is ignored
///         (every point enters at 0).
///     max_dim: The maximum homology dimension to compute.
///     points: Instead of a distance matrix, the points (a 2D array or sequence of sequences, one point per row, with
///         finite coordinates), whose distances are measured with `metric`. With a `threshold` only the distances up
///         to it are kept (as a sparse matrix), else every distance is (as a dense matrix).
///     metric: How to measure the distance between `points`: `'euclidean'` (the default), `'manhattan'` (or
///         `'cityblock'`) or `'chebyshev'`.
///     threshold: Only build the Rips complex up to this diameter (default: no limit).
///         Bars still alive at the threshold are reported as essential.
///         Simplices above the enclosing radius are always skipped, since by then the complex is contractible
///         and nothing changes, so a larger threshold gives the same result as none.
///     method: `'cohomology'` (the default) reduces the coboundary matrix, which is usually much faster.
///         `'homology'` reduces the boundary matrix instead, which needs simplices up to dimension `max_dim + 1`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         For example `Algorithm.parallel(threads=8)`.
///         Pipeline defaults (`'clouds.rips'`): columns='heap', cache=False, store_r=False.
///     representatives: Whether to also compute a representative cycle for every bar.
///         With `method='cohomology'` these are computed via involution.
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`.
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (distance_matrix = None, max_dim = 1, *, points = None, metric = "euclidean", threshold = None, method = "cohomology", algorithm = None, representatives = false))]
#[allow(clippy::too_many_arguments)]
fn rips<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]], typing.Any]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    distance_matrix: Option<&Bound<'_, PyAny>>,
    max_dim: usize,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Union[numpy.typing.NDArray[numpy.floating], typing.Sequence[typing.Sequence[builtins.float]]]]", imports = ("builtins", "numpy", "numpy.typing", "typing")))]
    points: Option<&Bound<'_, PyAny>>,
    #[gen_stub(override_type(type_repr = "typing.Literal['euclidean', 'manhattan', 'cityblock', 'chebyshev']", imports = ("typing")))]
    metric: &str,
    threshold: Option<f64>,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
    method: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[core.Algorithm]", imports = ("typing", "phlite.core")))]
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let threshold = match threshold {
        None => NotNan::new(f64::INFINITY).unwrap(),
        Some(t) => {
            NotNan::new(t).map_err(|_| PyValueError::new_err("threshold must not be NaN"))?
        }
    };
    let metric = parse_metric(metric)?;
    let input = match (distance_matrix, points) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "Give either a distance matrix or points, not both",
            ))
        }
        (None, None) => {
            return Err(PyValueError::new_err(
                "Give a distance matrix (the first argument) or points",
            ))
        }
        (Some(matrix), None) if is_scipy_sparse(matrix) => Input::Sparse(read_sparse(matrix)?),
        (Some(matrix), None) => Input::Dense(read_dense(matrix)?),
        (None, Some(points)) => read_points(points, metric, threshold)?,
    };
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let n_points = input.n_points();
    // Both methods need simplices of dimension max_dim + 1 (as cofacets, or the top of the boundary matrix)
    let top_dim = max_dim + 1;
    if !rips_index_fits(n_points, top_dim) {
        return Err(PyValueError::new_err(format!(
            "Too many points ({n_points}) to index simplices of dimension {top_dim}"
        )));
    }

    let bars = py.detach(|| {
        algorithm.run(RipsRun {
            input,
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
    input: Input,
    max_dim: usize,
    threshold: NotNan<f64>,
    method: Method,
    representatives: bool,
}

impl AlgorithmVisitor for RipsRun {
    type Output = SortedBars;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> SortedBars {
        let n_points = self.input.n_points();
        let bars = match self.input {
            Input::Dense(distances) => rips_barcode::<_, usize, _>(
                distances,
                self.max_dim,
                self.threshold,
                self.method,
                algorithm,
                self.representatives,
            ),
            Input::Sparse(distances) => rips_barcode::<_, usize, _>(
                distances,
                self.max_dim,
                self.threshold,
                self.method,
                algorithm,
                self.representatives,
            ),
        };
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
