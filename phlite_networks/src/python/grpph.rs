//! `phlite.networks.grpph`: grounded persistent path homology of a weighted digraph, by [`crate::grpph`] (whose
//! oracle, without the grounding, also computes `phlite.networks.pph` of the shortest-path filtration in dimensions 0
//! and 1).

use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::*;

use phlite_bindings::{
    algorithm_from_py, parse_method, AlgorithmDefaults, AlgorithmVisitor, Columns, SortedBars,
};

use super::graph::read_graph;
use crate::grpph::{
    build_edge_set, build_filtration,
    coboundary::{DigraphEdgeSet, PathHomCell, VecDigraphFiltration},
    grpph_barcode,
};

/// The pipeline `'networks.grpph'`: GrPPH columns are expensive to compute and many need reducing, so caching them
/// pays off (about 3× faster).
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "networks.grpph",
    columns: Columns::Merged,
    cache: true,
    store_r: false,
    row_ranks: false,
    union_find: None,
};

// ====== The entry points =================================================

/// Compute the grounded persistent path homology (GrPPH) of a weighted digraph, in dimensions 0 and 1.
///
/// Call either as `grpph(G)` with a networkx `DiGraph`, or as `grpph(n_vertices, edges)`.
///
/// Args:
///     graph: Either a networkx `DiGraph` (or `MultiDiGraph`), whose edge weights are read from the `weight` attribute
///         (default 1); or the number of nodes, which are then labelled `0, ..., n_vertices - 1`.
///     edges: Only when the first argument is the number of nodes: the weighted edges, as `(source, target, weight)` tuples.
///     weight: Only for a networkx graph: the edge attribute holding the weight (default `'weight'`).
///     method: `'cohomology'` (the default) reduces the coboundary matrix, which is usually much faster.
///         `'homology'` reduces the boundary matrix instead, which needs every 2-cell up front.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         For example `Algorithm.parallel(threads=8)`.
///         Pipeline defaults (`'networks.grpph'`): columns='merged', cache=True, store_r=False.
///     representatives: Whether to also compute a representative cycle for every bar.
///         With `method='cohomology'` these are computed via involution.
///         For a networkx graph, cells are tuples of node labels, otherwise of node indices.
///
/// Weights must be non-negative and may be infinite (the edge is then only used for grounding).
/// Self-loops are not supported.
///
/// Returns:
///     The barcode, with dimensions 0 and 1.
#[gen_stub_pyfunction(
    python_overload = r#"
    import builtins
    import typing
    import networkx
    import phlite.core

    @typing.overload
    def grpph(
        graph: networkx.DiGraph,
        /,
        *,
        weight: builtins.str = "weight",
        method: typing.Literal['cohomology', 'homology'] = "cohomology",
        algorithm: typing.Optional[core.Algorithm] = None,
        representatives: builtins.bool = False,
    ) -> core.Barcode: ...
    @typing.overload
    def grpph(
        n_vertices: builtins.int,
        edges: typing.Sequence[tuple[builtins.int, builtins.int, builtins.float]],
        /,
        *,
        method: typing.Literal['cohomology', 'homology'] = "cohomology",
        algorithm: typing.Optional[core.Algorithm] = None,
        representatives: builtins.bool = False,
    ) -> core.Barcode: ...
    "#,
    no_default_overload = true,
    module = "phlite.networks"
)]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, *, weight = "weight", method = "cohomology", algorithm = None, representatives = false))]
#[allow(clippy::too_many_arguments)]
fn grpph<'py>(
    py: Python<'py>,
    graph: &Bound<'_, PyAny>,
    edges: Option<Vec<(u32, u32, f64)>>,
    weight: &str,
    method: &str,
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    path_homology(py, graph, edges, weight, method, algorithm, representatives, true, &DEFAULTS)
}

/// GrPPH, or (not `grounded`) PPH of the shortest-path filtration in dimensions 0 and 1: the same oracle, grounded at
/// the digraph's edges or at none, with the pipeline `defaults`.
#[allow(clippy::too_many_arguments)]
pub(super) fn path_homology<'py>(
    py: Python<'py>,
    graph: &Bound<'_, PyAny>,
    edges: Option<Vec<(u32, u32, f64)>>,
    weight: &str,
    method: &str,
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
    grounded: bool,
    defaults: &AlgorithmDefaults,
) -> PyResult<Bound<'py, PyAny>> {
    let name = if grounded { "grpph" } else { "pph" };
    let (n_vertices, edges, labels) = read_graph(graph, edges, weight, name, u16::MAX as u32)?;
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), defaults)?;

    let bars = py.detach(|| {
        let filtration = build_filtration(n_vertices, &edges);
        let edge_set = if grounded {
            build_edge_set(&edges)
        } else {
            DigraphEdgeSet::default()
        };
        algorithm.run(GrPPHRun {
            filtration: &filtration,
            edge_set: &edge_set,
            n_vertices: n_vertices as u16,
            method,
            representatives,
        })
    })?;
    bars.into_python(py, &algorithm, representatives, labels.as_deref())
}

/// Compute the (Gr)PPH barcode with whichever algorithm is configured.
struct GrPPHRun<'a> {
    filtration: &'a VecDigraphFiltration,
    edge_set: &'a DigraphEdgeSet,
    n_vertices: u16,
    method: Method,
    representatives: bool,
}

impl AlgorithmVisitor for GrPPHRun<'_> {
    type Output = SortedBars;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> SortedBars {
        let bars = grpph_barcode(
            self.filtration,
            self.edge_set,
            self.n_vertices,
            self.method,
            algorithm,
            self.representatives,
        );
        SortedBars::new(bars, 2, |cell| match *cell {
            PathHomCell::Node(a) => (0, vec![a as u32]),
            PathHomCell::Edge(a, b) => (1, vec![a as u32, b as u32]),
            PathHomCell::TwoCell(_) => {
                unreachable!("(Gr)PPH bars and representatives live in dimensions 0 and 1")
            }
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(grpph, m)?)
}
