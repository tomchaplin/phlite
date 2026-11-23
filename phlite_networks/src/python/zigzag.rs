//! The engine behind `phlite.networks.network_zigzag`: zigzag persistence of a sequence of network snapshots, by
//! [`crate::zigzag`].

use phlite::{barcode::Algorithm, columns::ColumnRepresentation};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::*;

use phlite_bindings::{algorithm_from_py, AlgorithmDefaults, AlgorithmVisitor, Columns};
use phlite_complexes::zigzag::{Join, LevelBar};

use crate::zigzag::{network_zigzag, NetworkComplex, Snapshot};

/// The pipeline `'networks.network_zigzag'`: FastZigzag reduces an explicit complex, so with `phlite.complexes`' defaults.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "networks.network_zigzag",
    columns: Columns::Pivot,
    cache: false,
    store_r: true,
    row_ranks: true,
    union_find: None,
    collapse: None,
};

/// A snapshot from Python: node labels and `(source, target)` edges.
type NodesAndEdges = (Vec<u32>, Vec<(u32, u32)>);

/// The engine behind `phlite.networks.network_zigzag` (which returns a `phlite.core.ZigzagBarcode`): the intervals,
/// in levels, as `(dimension, birth, death)` tuples.
///
/// Args:
///     snapshots: `(nodes, edges)` pairs: integer node labels and `(source, target)` edges.
///     max_dim: The largest homology dimension.
///     complex: `'clique'` (of the undirected graph) or `'directed_flag'` (directed cliques, ordered simplices).
///     join: `'union'` or `'intersection'`.
///     algorithm: The reduction algorithm (a `phlite.core.Algorithm`).
#[gen_stub_pyfunction(module = "phlite.networks")]
#[pyfunction]
#[pyo3(signature = (snapshots, max_dim = 1, *, complex = "clique", join = "union", algorithm = None))]
fn _network_zigzag(
    py: Python<'_>,
    snapshots: Vec<NodesAndEdges>,
    max_dim: usize,
    complex: &str,
    join: &str,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Any]", imports = ("typing")))] algorithm: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<(usize, usize, usize)>> {
    let complex = match complex {
        "clique" => NetworkComplex::Clique,
        "directed_flag" => NetworkComplex::DirectedFlag,
        other => return Err(PyValueError::new_err(format!("Unknown complex {other:?}: expected 'clique' or 'directed_flag'"))),
    };
    let join = match join {
        "union" => Join::Union,
        "intersection" => Join::Intersection,
        other => return Err(PyValueError::new_err(format!("Unknown join {other:?}: expected 'union' or 'intersection'"))),
    };
    let snapshots = snapshots.into_iter().map(|(nodes, edges)| Snapshot::new(nodes, edges)).collect();
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let bars = py.detach(|| algorithm.run(Run { snapshots, complex, max_dim, join }))?;
    Ok(bars.into_iter().map(|b| (b.dimension, b.birth, b.death)).collect())
}

struct Run {
    snapshots: Vec<Snapshot>,
    complex: NetworkComplex,
    max_dim: usize,
    join: Join,
}

impl AlgorithmVisitor for Run {
    type Output = Vec<LevelBar>;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> Self::Output {
        network_zigzag(&self.snapshots, self.complex, self.max_dim, self.join, algorithm)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_network_zigzag, m)?)
}
