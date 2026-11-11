//! The `phlite_grpph` Python module.
//!
//! The `Barcode` type and argument handling are shared with the other bindings, see [`phlite_pyutils`].

use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
};
use phlite_pyutils::{
    add_algorithm_class, add_barcode_class, parse_method, AlgorithmConfig, AlgorithmVisitor,
    Barcode, SortedBars,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pyo3_stub_gen::derive::*;

use crate::{
    build_edge_set, build_filtration,
    coboundary::{DigraphEdgeSet, PathHomCell, VecDigraphFiltration},
    grpph_barcode,
};

// ====== The entry point ==================================================

fn validate(n_vertices: u32, edges: &[(u32, u32, f64)]) -> PyResult<()> {
    if n_vertices > u16::MAX as u32 {
        return Err(PyValueError::new_err(format!(
            "n_vertices must be at most {}",
            u16::MAX
        )));
    }
    for &(i, j, w) in edges {
        if i >= n_vertices || j >= n_vertices {
            return Err(PyValueError::new_err(format!(
                "Edge ({i}, {j}) refers to a node outside 0..{n_vertices}"
            )));
        }
        if i == j {
            return Err(PyValueError::new_err(format!(
                "Edge ({i}, {j}) is a self-loop, which is not supported"
            )));
        }
        if w.is_nan() || w < 0.0 {
            return Err(PyValueError::new_err(format!(
                "Edge ({i}, {j}) has weight {w}, weights must be non-negative (or infinite)"
            )));
        }
    }
    Ok(())
}

/// A digraph as `(n_vertices, edges, labels)`, where vertex `i` has node label `labels[i]`
type LabelledDigraph = (u32, Vec<(u32, u32, f64)>, Vec<Py<PyAny>>);

/// Read a networkx (multi)digraph into `(n_vertices, edges, labels)`, where node `labels[i]` becomes vertex `i`.
/// Edges without a `weight` attribute get weight 1, as in networkx's shortest path functions.
fn from_networkx(graph: &Bound<'_, PyAny>, weight: &str) -> PyResult<LabelledDigraph> {
    let not_a_graph = || {
        PyTypeError::new_err(
            "Expected a networkx DiGraph, or the number of nodes followed by a list of (source, target, weight) edges",
        )
    };
    let is_directed: bool = graph
        .call_method0("is_directed")
        .and_then(|d| d.extract())
        .map_err(|_| not_a_graph())?;
    if !is_directed {
        return Err(PyTypeError::new_err(
            "GrPPH needs a directed graph, try passing G.to_directed() (which has an edge each way for every edge of G)",
        ));
    }

    let labels: Vec<Bound<'_, PyAny>> = graph
        .getattr("nodes")?
        .try_iter()?
        .collect::<PyResult<_>>()?;
    let index = PyDict::new(graph.py());
    for (i, label) in labels.iter().enumerate() {
        index.set_item(label, i)?;
    }
    let vertex = |label: Bound<'_, PyAny>| -> PyResult<u32> {
        match index.get_item(&label)? {
            Some(i) => i.extract(),
            None => Err(PyValueError::new_err(format!(
                "Edge endpoint {label} is not a node of the graph"
            ))),
        }
    };

    let kwargs = PyDict::new(graph.py());
    kwargs.set_item("data", weight)?;
    kwargs.set_item("default", 1.0)?;
    let mut edges = vec![];
    for edge in graph.call_method("edges", (), Some(&kwargs))?.try_iter()? {
        let (source, target, w): (Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>) =
            edge?.extract()?;
        let w: f64 = w.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "Edge ({source}, {target}) has {weight}={w}, which is not a number"
            ))
        })?;
        if w.is_nan() || w < 0.0 {
            return Err(PyValueError::new_err(format!(
                "Edge ({source}, {target}) has {weight}={w}, weights must be non-negative (or infinite)"
            )));
        }
        if source.eq(&target)? {
            return Err(PyValueError::new_err(format!(
                "Edge ({source}, {target}) is a self-loop, which is not supported; \
                 remove them with G.remove_edges_from(nx.selfloop_edges(G))"
            )));
        }
        edges.push((vertex(source)?, vertex(target)?, w));
    }
    let labels = labels.into_iter().map(Bound::unbind).collect();
    Ok((index.len() as u32, edges, labels))
}

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
///     algorithm: The reduction algorithm and its options, as an `Algorithm` (default: `Algorithm.serial()`),
///         or one of the names `'standard'`, `'clearing'` (the same as `Algorithm.serial()`), `'parallel'`
///         (the same as `Algorithm.parallel()`) or `'parallel-lock-free'`.
///     representatives: Whether to also compute a representative cycle for every bar.
///         With `method='cohomology'` these are computed via involution.
///         For a networkx graph, cells are tuples of node labels, otherwise of node indices.
///     threads: The number of threads for a parallel algorithm (default: all cores);
///         short for `algorithm.with_threads(threads)`.
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

    @typing.overload
    def grpph(
        graph: networkx.DiGraph,
        /,
        *,
        weight: builtins.str = "weight",
        method: typing.Literal['cohomology', 'homology'] = "cohomology",
        algorithm: typing.Union[Algorithm, typing.Literal['standard', 'clearing', 'parallel', 'parallel-lock-free'], None] = None,
        representatives: builtins.bool = False,
        threads: typing.Optional[builtins.int] = None,
    ) -> Barcode: ...
    @typing.overload
    def grpph(
        n_vertices: builtins.int,
        edges: typing.Sequence[tuple[builtins.int, builtins.int, builtins.float]],
        /,
        *,
        method: typing.Literal['cohomology', 'homology'] = "cohomology",
        algorithm: typing.Union[Algorithm, typing.Literal['standard', 'clearing', 'parallel', 'parallel-lock-free'], None] = None,
        representatives: builtins.bool = False,
        threads: typing.Optional[builtins.int] = None,
    ) -> Barcode: ...
    "#,
    no_default_overload = true
)]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, *, weight = "weight", method = "cohomology", algorithm = None, representatives = false, threads = None))]
#[allow(clippy::too_many_arguments)]
fn grpph(
    py: Python<'_>,
    graph: &Bound<'_, PyAny>,
    edges: Option<Vec<(u32, u32, f64)>>,
    weight: &str,
    method: &str,
    algorithm: Option<&Bound<'_, PyAny>>,
    representatives: bool,
    threads: Option<usize>,
) -> PyResult<Barcode> {
    let (n_vertices, edges, labels) =
        match edges {
            Some(edges) => {
                let n_vertices: u32 = graph.extract().map_err(|_| {
                    PyTypeError::new_err(
                        "When edges are given, the first argument must be the number of nodes",
                    )
                })?;
                (n_vertices, edges, None)
            }
            None if graph.extract::<u32>().is_ok() => return Err(PyTypeError::new_err(
                "Missing edges: call grpph(n_vertices, edges) or grpph(G) for a networkx DiGraph G",
            )),
            None => {
                let (n_vertices, edges, labels) = from_networkx(graph, weight)?;
                (n_vertices, edges, Some(labels))
            }
        };
    validate(n_vertices, &edges)?;
    let method = parse_method(method)?;
    let algorithm = AlgorithmConfig::from_arg(algorithm, threads)?;

    let bars = py.detach(|| {
        let filtration = build_filtration(n_vertices, &edges);
        let edge_set = build_edge_set(&edges);
        algorithm.run(GrPPHRun {
            filtration: &filtration,
            edge_set: &edge_set,
            n_vertices: n_vertices as u16,
            method,
            representatives,
        })
    })?;
    bars.into_python(py, representatives, labels.as_deref())
}

/// Compute the GrPPH barcode with whichever algorithm is configured.
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
                unreachable!("GrPPH bars and representatives live in dimensions 0 and 1")
            }
        })
    }
}

/// Grounded persistent path homology (GrPPH) of weighted digraphs, computed with `phlite`.
#[pymodule]
fn phlite_grpph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(grpph, m)?)?;
    add_barcode_class(m, "phlite_grpph")?;
    add_algorithm_class(m, "phlite_grpph")?;
    Ok(())
}
