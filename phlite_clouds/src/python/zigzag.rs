//! The engine behind `phlite.clouds.cloud_zigzag`: zigzag persistence of a sequence of point clouds, by
//! [`crate::zigzag`].

use ordered_float::NotNan;
use phlite::{barcode::Algorithm, columns::ColumnRepresentation};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use phlite_bindings::{algorithm_from_py, AlgorithmDefaults, AlgorithmVisitor, Columns};
use phlite_complexes::zigzag::{Join, LevelBar};

use super::rips::read_rows;
use crate::rips::Metric;
use crate::witness::Witness;
use crate::zigzag::{cloud_zigzag, Alpha, Cech, DelaunayCech, Frame, Rips, RipsDistances, WitnessFiltration};
use crate::CloudError;

/// The pipeline `'clouds.cloud_zigzag'`: FastZigzag reduces an explicit complex, so with `phlite.complexes`' defaults.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "clouds.cloud_zigzag",
    columns: Columns::Pivot,
    cache: false,
    store_r: true,
    row_ranks: true,
    union_find: None,
    collapse: None,
};

/// The engine behind `phlite.clouds.cloud_zigzag` (which returns a `phlite.core.ZigzagBarcode`): the intervals, in
/// levels, as `(dimension, birth, death)` tuples.
///
/// Args:
///     frames: `(labels, points)` pairs: each frame's points and the label naming each in every frame.
///     threshold: The scale at which each frame's complex is cut.
///     max_dim: The largest homology dimension.
///     filtration: `'rips'`, `'cech'`, `'alpha'`, `'delaunay_cech'` or `'witness'`.
///     metric: For Rips: `'euclidean'`, `'manhattan'` or `'chebyshev'`.
///     join: `'union'` or `'intersection'`.
///     distances: For Rips: a distance matrix on the labels, used instead of the frames' points.
///     witnesses: For `'witness'`: the witnesses (each frame's points are the landmarks).
///     strong: For `'witness'`: strong rather than relaxed witnessing.
///     algorithm: The reduction algorithm (a `phlite.core.Algorithm`).
#[gen_stub_pyfunction(module = "phlite.clouds")]
#[pyfunction]
#[pyo3(signature = (frames, threshold, max_dim = 1, *, filtration = "rips", metric = "euclidean", join = "union", distances = None, witnesses = None, strong = false, algorithm = None))]
#[allow(clippy::too_many_arguments)]
fn _cloud_zigzag(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "typing.Sequence[tuple[typing.Sequence[builtins.int], typing.Any]]", imports = ("builtins", "typing")))]
    frames: &Bound<'_, PyAny>,
    threshold: f64,
    max_dim: usize,
    filtration: &str,
    metric: &str,
    join: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Any]", imports = ("typing")))] distances: Option<Bound<'_, PyAny>>,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Any]", imports = ("typing")))] witnesses: Option<Bound<'_, PyAny>>,
    strong: bool,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Any]", imports = ("typing")))] algorithm: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<(usize, usize, usize)>> {
    let mut parsed = vec![];
    for item in frames.try_iter()? {
        let item = item?;
        let labels: Vec<u32> = item.get_item(0)?.extract()?;
        let points = item.get_item(1)?;
        let points = if points.is_none() || points.len().is_ok_and(|n| n == 0) { vec![] } else { read_rows(&points, "points")? };
        if !points.is_empty() && points.len() != labels.len() {
            return Err(PyValueError::new_err("each frame needs one label per point"));
        }
        parsed.push(Frame { labels, points });
    }
    let join = match join {
        "union" => Join::Union,
        "intersection" => Join::Intersection,
        other => return Err(PyValueError::new_err(format!("Unknown join {other:?}: expected 'union' or 'intersection'"))),
    };
    let metric = match metric {
        "euclidean" => Metric::Euclidean,
        "manhattan" => Metric::Manhattan,
        "chebyshev" => Metric::Chebyshev,
        other => return Err(PyValueError::new_err(format!("Unknown metric {other:?}"))),
    };
    let kind = match filtration {
        "rips" => match distances {
            Some(d) => {
                let rows = read_rows(&d, "distances")?;
                let n = rows.len();
                if rows.iter().any(|r| r.len() != n) {
                    return Err(PyValueError::new_err("distances must be a square matrix"));
                }
                let matrix = rows
                    .into_iter()
                    .map(|r| r.into_iter().map(|x| NotNan::new(x).map_err(|_| PyValueError::new_err("a distance is NaN"))).collect())
                    .collect::<PyResult<Vec<Vec<NotNan<f64>>>>>()?;
                Kind::Distances(RipsDistances { distances: matrix })
            }
            None => Kind::Rips(Rips { metric }),
        },
        "cech" => Kind::Cech,
        "alpha" => Kind::Alpha,
        "delaunay_cech" => Kind::DelaunayCech,
        "witness" => {
            let witnesses = witnesses.ok_or_else(|| PyValueError::new_err("the witness filtration needs witnesses"))?;
            let witnesses = read_rows(&witnesses, "witnesses")?;
            Kind::Witness(WitnessFiltration { witnesses, kind: if strong { Witness::Strong } else { Witness::Relaxed } })
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown filtration {other:?}: expected 'rips', 'cech', 'alpha', 'delaunay_cech' or 'witness'"
            )))
        }
    };
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let run = Run { frames: parsed, kind, threshold, max_dim, join };
    let bars = py.detach(|| algorithm.run(run))?.map_err(PyErr::from)?;
    Ok(bars.into_iter().map(|b| (b.dimension, b.birth, b.death)).collect())
}

enum Kind {
    Rips(Rips),
    Distances(RipsDistances<Vec<Vec<NotNan<f64>>>>),
    Cech,
    Alpha,
    DelaunayCech,
    Witness(WitnessFiltration),
}

struct Run {
    frames: Vec<Frame>,
    kind: Kind,
    threshold: f64,
    max_dim: usize,
    join: Join,
}

impl AlgorithmVisitor for Run {
    type Output = Result<Vec<LevelBar>, CloudError>;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
        let Run { frames, kind, threshold, max_dim, join } = self;
        match &kind {
            Kind::Rips(f) => cloud_zigzag(&frames, f, threshold, max_dim, join, algorithm),
            Kind::Distances(f) => cloud_zigzag(&frames, f, threshold, max_dim, join, algorithm),
            Kind::Cech => cloud_zigzag(&frames, &Cech, threshold, max_dim, join, algorithm),
            Kind::Alpha => cloud_zigzag(&frames, &Alpha, threshold, max_dim, join, algorithm),
            Kind::DelaunayCech => cloud_zigzag(&frames, &DelaunayCech, threshold, max_dim, join, algorithm),
            Kind::Witness(f) => cloud_zigzag(&frames, f, threshold, max_dim, join, algorithm),
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_cloud_zigzag, m)?)
}
