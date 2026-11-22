//! The engine behind `phlite.images.filtrations` (whose Python functions check their arguments and call this): the
//! filtrations of binary images of [`crate::filtrations`].

use numpy::{IntoPyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use crate::filtrations::{self, Metric};

/// The filtration `kind` (`'height'`, `'radial'`, `'dilation'`, `'erosion'`, `'signed_distance'` or `'density'`) of a
/// boolean image, as an array of its shape (unreached pixels `inf`). See `phlite.images.filtrations`.
#[gen_stub_pyfunction(module = "phlite.images")]
#[pyfunction]
#[pyo3(signature = (kind, image, *, direction = None, center = None, radius = f64::INFINITY, metric = "euclidean", n_iterations = None))]
#[allow(clippy::too_many_arguments)]
fn _binary_filtration<'py>(
    py: Python<'py>,
    kind: &str,
    #[gen_stub(override_type(type_repr = "numpy.typing.NDArray[numpy.bool_]", imports = ("numpy", "numpy.typing")))]
    image: &Bound<'py, PyArrayDyn<bool>>,
    direction: Option<Vec<f64>>,
    center: Option<Vec<f64>>,
    radius: f64,
    metric: &str,
    n_iterations: Option<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let shape: Vec<usize> = image.shape().to_vec();
    // (In logical, row-major order, whatever the array's strides)
    let pixels: Vec<bool> = image.readonly().as_array().iter().copied().collect();
    let metric = match metric {
        "euclidean" => Metric::Euclidean,
        "manhattan" | "cityblock" => Metric::Manhattan,
        "chebyshev" => Metric::Chebyshev,
        other => return Err(PyValueError::new_err(format!("unknown metric {other:?}"))),
    };
    let error = |e: filtrations::FiltrationError| PyValueError::new_err(e.to_string());
    let values = py.detach(|| -> Result<Vec<f64>, filtrations::FiltrationError> {
        Ok(match kind {
            "height" => filtrations::height(&pixels, &shape, &direction.unwrap_or_else(|| vec![1.0; shape.len()]))?,
            "radial" => filtrations::radial(&pixels, &shape, &center.unwrap_or_else(|| vec![0.0; shape.len()]), radius, metric)?,
            "dilation" => filtrations::dilation(&pixels, &shape, n_iterations),
            "erosion" => filtrations::erosion(&pixels, &shape, n_iterations),
            "signed_distance" => filtrations::signed_distance(&pixels, &shape, n_iterations),
            "density" => filtrations::density(&pixels, &shape, radius, metric),
            _ => return Ok(vec![]),
        })
    });
    let values = values.map_err(error)?;
    if values.is_empty() && !pixels.is_empty() {
        return Err(PyValueError::new_err(format!("unknown filtration {kind:?}")));
    }
    values.into_pyarray(py).reshape(shape)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_binary_filtration, m)?)
}
