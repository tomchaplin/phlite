//! The engine behind `phlite.images.video_zigzag`: zigzag persistence of a video's frames, by [`crate::zigzag`].

use phlite::{barcode::Algorithm, columns::ColumnRepresentation};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use phlite_bindings::{algorithm_from_py, AlgorithmVisitor};

use super::cubical::{read_image, DEFAULTS};
use crate::cubical::{Construction, Filtration};
use crate::zigzag::{video_zigzag, VideoBar};

/// The zigzag barcode of a video (frames along the first axis) cut at `threshold`, as `(dimension, birth, death)` in
/// levels (frame `i` at `2 i`). See `phlite.images.video_zigzag`, which calls this.
#[gen_stub_pyfunction(module = "phlite.images")]
#[pyfunction]
#[pyo3(signature = (video, threshold, max_dim, construction, superlevel, periodic, algorithm = None))]
#[allow(clippy::too_many_arguments)]
fn _video_zigzag(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "numpy.typing.ArrayLike", imports = ("numpy.typing")))] video: &Bound<'_, PyAny>,
    threshold: f64,
    max_dim: usize,
    construction: &str,
    superlevel: bool,
    periodic: Vec<bool>,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Any]", imports = ("typing")))] algorithm: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<(usize, usize, usize)>> {
    let construction = match construction {
        "T" | "t" => Construction::T,
        "V" | "v" => Construction::V,
        other => return Err(PyValueError::new_err(format!("Unknown construction {other:?}: expected 'T' or 'V'"))),
    };
    let filtration = if superlevel { Filtration::Superlevel } else { Filtration::Sublevel };
    let (pixels, shape) = read_image(py, video)?;
    if shape.len() < 2 {
        return Err(PyValueError::new_err("a video is an array of frames: give at least two axes"));
    }
    let frame_shape = shape[1..].to_vec();
    if periodic.len() != frame_shape.len() {
        return Err(PyValueError::new_err("periodic needs one entry per axis of a frame"));
    }
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let run = Run { pixels, shape: frame_shape, periodic, threshold, construction, filtration, max_dim };
    let bars = py.detach(|| algorithm.run(run))?.map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(bars.into_iter().map(|b| (b.dimension, b.birth, b.death)).collect())
}

struct Run {
    pixels: Vec<f64>,
    shape: Vec<usize>,
    periodic: Vec<bool>,
    threshold: f64,
    construction: Construction,
    filtration: Filtration,
    max_dim: usize,
}

impl AlgorithmVisitor for Run {
    type Output = Result<Vec<VideoBar>, phlite_complexes::zigzag::ZigzagError>;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
        video_zigzag(
            &self.pixels,
            &self.shape,
            &self.periodic,
            self.threshold,
            self.construction,
            self.filtration,
            self.max_dim,
            algorithm,
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_video_zigzag, m)?)
}
