//! The `phlite_rips` Python module (feature `python`).
//!
//! The `Barcode` type and argument handling are shared with the other bindings, see [`phlite_pyutils`].

use ordered_float::NotNan;
use phlite_pyutils::{add_barcode_class, parse_method, AlgorithmArg, Barcode, SortedBars};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use crate::{rips_barcode, rips_index_fits};

// ====== The entry point ==================================================

fn validate(distance_matrix: &[Vec<f64>]) -> PyResult<Vec<Vec<NotNan<f64>>>> {
    let n = distance_matrix.len();
    for (i, row) in distance_matrix.iter().enumerate() {
        if row.len() != n {
            return Err(PyValueError::new_err(format!(
                "distance_matrix must be square, but row {i} has length {} rather than {n}",
                row.len()
            )));
        }
        for (j, &d) in row.iter().enumerate() {
            if d.is_nan() || d < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "distance_matrix[{i}][{j}] is {d}, distances must be non-negative (or infinite)"
                )));
            }
            if d != distance_matrix[j][i] {
                return Err(PyValueError::new_err(format!(
                    "distance_matrix must be symmetric, but entries ({i}, {j}) and ({j}, {i}) differ"
                )));
            }
        }
    }
    Ok(distance_matrix
        .iter()
        .map(|row| row.iter().map(|&d| NotNan::new(d).unwrap()).collect())
        .collect())
}

/// Compute the Vietoris-Rips persistent homology of a distance matrix, in dimensions `0, ..., max_dim`.
///
/// Args:
///     distance_matrix: A symmetric matrix of pairwise distances, e.g. a list of lists or a 2D numpy array.
///         Distances must be non-negative. An infinite distance means the edge never appears. The diagonal is ignored.
///     max_dim: The maximum homology dimension to compute.
///     method: `'cohomology'` (the default) reduces the coboundary matrix, which is usually much faster.
///         `'homology'` reduces the boundary matrix instead, which needs simplices up to dimension `max_dim + 1`.
///     algorithm: The reduction algorithm: `'standard'`, `'clearing'` (the default), `'parallel'`
///         (clearing, with each dimension reduced in parallel) or `'parallel-lock-free'`
///         (the same, using lock-free data structures).
///     representatives: Whether to also compute a representative cycle for every bar.
///         With `method='cohomology'` these are computed via involution.
///     threads: The number of threads for the parallel algorithms (default: all cores).
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (distance_matrix, max_dim = 1, *, method = "cohomology", algorithm = "clearing", representatives = false, threads = None))]
fn rips(
    py: Python<'_>,
    distance_matrix: Vec<Vec<f64>>,
    max_dim: usize,
    #[gen_stub(override_type(type_repr = "typing.Literal['cohomology', 'homology']", imports = ("typing")))]
    method: &str,
    #[gen_stub(override_type(type_repr = "typing.Literal['standard', 'clearing', 'parallel', 'parallel-lock-free']", imports = ("typing")))]
    algorithm: &str,
    representatives: bool,
    threads: Option<usize>,
) -> PyResult<Barcode> {
    let distances = validate(&distance_matrix)?;
    drop(distance_matrix);
    let method = parse_method(method)?;
    let algorithm = AlgorithmArg::parse(algorithm)?;
    let n_points = distances.len();
    // Both methods need simplices of dimension max_dim + 1 (as cofacets, or the top of the boundary matrix)
    let top_dim = max_dim + 1;
    if !rips_index_fits(n_points, top_dim) {
        return Err(PyValueError::new_err(format!(
            "Too many points ({n_points}) to index simplices of dimension {top_dim}"
        )));
    }

    let bars = py.detach(|| {
        algorithm.run(threads, |algorithm| {
            let bars = rips_barcode(distances, max_dim, method, algorithm, representatives);
            SortedBars::new(bars, max_dim + 1, |simplex| {
                let vertices = simplex.to_vec(n_points);
                (
                    vertices.len() - 1,
                    vertices.into_iter().map(|v| v as u32).collect(),
                )
            })
        })
    })?;
    bars.into_python(py, representatives, None)
}

/// Vietoris-Rips persistent homology, computed with `phlite`.
#[pymodule]
fn phlite_rips(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(rips, m)?)?;
    add_barcode_class(m, "phlite_rips")?;
    Ok(())
}
