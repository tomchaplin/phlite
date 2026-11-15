//! `phlite.networks.flag`: the flag (clique) complex of an undirected graph, filtered by edge weights, node values or
//! both (see [`crate::undirected`]).

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
    values::INFINITY,
};
use phlite_bindings::{algorithm_from_py, parse_method, AlgorithmDefaults, AlgorithmVisitor, Columns, SortedBars};
use phlite_clouds::rips::rips_index_fits;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pyo3_stub_gen::derive::*;

use super::graph::network_error;
use crate::undirected::{flag_barcode, Graph};

/// The pipeline `'networks.flag'`: computed by Rips' machinery, so with Rips' defaults.
pub(crate) const DEFAULTS: AlgorithmDefaults = AlgorithmDefaults {
    pipeline: "networks.flag",
    columns: Columns::Heap,
    cache: false,
    store_r: false,
    row_ranks: false,
    union_find: Some(true),
};

/// An undirected graph as `(n_vertices, edges, node values, labels)`.
type Read = (u32, Vec<(u32, u32, f64)>, Option<Vec<f64>>, Option<Vec<Py<PyAny>>>);

/// Read `graph` (a networkx graph, or the number of nodes with `edges`), its edge weights (`weight`, or none) and its
/// node values (`node_weight`: an attribute name, or a value per node).
fn read(
    graph: &Bound<'_, PyAny>,
    edges: Option<Vec<Bound<'_, PyAny>>>,
    weight: Option<&str>,
    node_weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<Read> {
    let no_weight = f64::NEG_INFINITY;
    let number = |x: &Bound<'_, PyAny>, what: &dyn Fn() -> String| -> PyResult<f64> {
        x.extract::<f64>()
            .map_err(|_| PyTypeError::new_err(format!("{} is {x}, which is not a number", what())))
    };
    // A sequence of node values, if given as one
    let values_from_sequence = |n: u32| -> PyResult<Option<Vec<f64>>> {
        match node_weight {
            Some(values) if !values.is_instance_of::<pyo3::types::PyString>() => {
                let values: Vec<f64> = values.extract().map_err(|_| {
                    PyTypeError::new_err("node_weight must be a node attribute name, or a sequence of numbers")
                })?;
                if values.len() != n as usize {
                    return Err(PyValueError::new_err(format!(
                        "There are {} node values for {n} nodes",
                        values.len()
                    )));
                }
                Ok(Some(values))
            }
            _ => Ok(None),
        }
    };

    if let Some(edges) = edges {
        let n: u32 = graph.extract().map_err(|_| {
            PyTypeError::new_err("When edges are given, the first argument must be the number of nodes")
        })?;
        if node_weight.is_some_and(|v| v.is_instance_of::<pyo3::types::PyString>()) {
            return Err(PyTypeError::new_err(
                "With (n_vertices, edges), node_weight must be a sequence of values, not an attribute name",
            ));
        }
        let mut out = Vec::with_capacity(edges.len());
        for edge in edges {
            // (u, v) or (u, v, weight)
            let (u, v, w) = if let Ok((u, v, w)) = edge.extract::<(u32, u32, Bound<'_, PyAny>)>() {
                let w = match weight {
                    None => no_weight,
                    Some(_) => number(&w, &|| format!("The weight of edge ({u}, {v})"))?,
                };
                (u, v, w)
            } else if let Ok((u, v)) = edge.extract::<(u32, u32)>() {
                (u, v, if weight.is_none() { no_weight } else { 1.0 })
            } else {
                return Err(PyTypeError::new_err(format!(
                    "Edges must be (u, v) or (u, v, weight) tuples of node numbers, not {edge}"
                )));
            };
            out.push((u, v, w));
        }
        return Ok((n, out, values_from_sequence(n)?, None));
    }
    if graph.extract::<u32>().is_ok() {
        return Err(PyTypeError::new_err(
            "Missing edges: call flag(n_vertices, edges) or flag(G) for a networkx Graph G",
        ));
    }

    let not_a_graph = || {
        PyTypeError::new_err("Expected a networkx Graph, or the number of nodes followed by a list of edges")
    };
    let directed: bool = graph
        .call_method0("is_directed")
        .and_then(|d| d.extract())
        .map_err(|_| not_a_graph())?;
    if directed {
        return Err(PyTypeError::new_err(
            "Expected an undirected graph: for a directed one use directed_flag, or pass G.to_undirected()",
        ));
    }
    let labels: Vec<Bound<'_, PyAny>> = graph.getattr("nodes")?.try_iter()?.collect::<PyResult<_>>()?;
    let n = labels.len() as u32;
    let index = PyDict::new(graph.py());
    for (i, label) in labels.iter().enumerate() {
        index.set_item(label, i)?;
    }
    let vertex = |label: &Bound<'_, PyAny>| -> PyResult<u32> {
        match index.get_item(label)? {
            Some(i) => i.extract(),
            None => Err(PyValueError::new_err(format!("Edge endpoint {label} is not a node of the graph"))),
        }
    };
    let kwargs = PyDict::new(graph.py());
    kwargs.set_item("data", weight.unwrap_or("weight"))?;
    kwargs.set_item("default", 1.0)?;
    let mut edges = vec![];
    for edge in graph.call_method("edges", (), Some(&kwargs))?.try_iter()? {
        let (u, v, w): (Bound<'_, PyAny>, Bound<'_, PyAny>, Bound<'_, PyAny>) = edge?.extract()?;
        if u.eq(&v)? {
            return Err(PyValueError::new_err(format!(
                "Edge ({u}, {v}) is a self-loop, which is not supported; \
                 remove them with G.remove_edges_from(nx.selfloop_edges(G))"
            )));
        }
        let w = match weight {
            None => no_weight,
            Some(name) => number(&w, &|| format!("Edge ({u}, {v})'s {name}"))?,
        };
        edges.push((vertex(&u)?, vertex(&v)?, w));
    }
    let values = match node_weight {
        Some(name) if name.is_instance_of::<pyo3::types::PyString>() => {
            let name: String = name.extract()?;
            let data = graph.getattr("nodes")?;
            let values = labels
                .iter()
                .map(|label| {
                    let attributes = data.get_item(label)?;
                    let value = attributes.get_item(&name).map_err(|_| {
                        PyValueError::new_err(format!("Node {label} has no attribute {name:?}"))
                    })?;
                    number(&value, &|| format!("Node {label}'s {name}"))
                })
                .collect::<PyResult<Vec<f64>>>()?;
            Some(values)
        }
        _ => values_from_sequence(n)?,
    };
    Ok((n, edges, values, Some(labels.into_iter().map(Bound::unbind).collect())))
}

/// Compute the persistent homology of the flag (clique) complex of an undirected graph, in dimensions
/// `0, ..., max_dim`, filtered by its edge weights, its node values, or both.
///
/// The flag complex has a simplex for every clique of the graph (nodes pairwise joined by edges). A simplex enters at
/// the largest edge weight and node value among its edges and nodes, so node `v` at its value `f(v)` (0 by default),
/// and edge `uv` at `max(w(uv), f(u), f(v))`. In particular:
///
/// * `flag(G)`: the flag complex of the weighted graph, nodes at 0 (as `phlite.clouds.rips` of the graph's sparse
///   distance matrix, and GUDHI's `SimplexTree` from the edges, expanded);
/// * `flag(G, weight=None, node_weight='height')`: the **lower-star** filtration of the node values, each simplex
///   entering with its latest node (as GUDHI's `SimplexTree` with vertex values, made non-decreasing);
/// * both: each edge no earlier than its weight or its nodes.
///
/// With `filtration='shortest_path'`, every two nodes joined by a path are joined by an edge at the length of the
/// shortest one: the Vietoris–Rips filtration of the graph's shortest-path metric (as `phlite.clouds.rips` of the
/// matrix of path lengths; node values still apply). Edge weights are then lengths, so must be non-negative; with
/// `weight=None` every edge has length 1 (the hop distance). Pairs in different components are never joined.
///
/// It is computed by the same machinery as Vietoris–Rips (union-find in dimension 0, apparent pairs, and cofacets
/// enumerated through the neighbours of the sparsest vertex), so is fast on sparse graphs.
///
/// Call either as `flag(G)` with a networkx `Graph` (or `MultiGraph`, which keeps the lightest of parallel edges), or as
/// `flag(n_vertices, edges)`.
///
/// Args:
///     graph: Either a networkx `Graph`, whose edge weights are read from the `weight` attribute (default 1); or the
///         number of nodes, which are then labelled `0, ..., n_vertices - 1`.
///     edges: Only when the first argument is the number of nodes: the edges, as `(u, v)` or `(u, v, weight)` tuples.
///     max_dim: The maximum homology dimension to compute.
///     weight: The edge attribute holding the weight (default `'weight'`), or `None` for no edge weights (each edge
///         then enters with its later node). Weights may be negative, and an infinite weight is an edge that never
///         appears. An edge given twice keeps its smallest weight.
///     node_weight: The node values: a node attribute name (for a networkx graph), or a sequence with one value per
///         node, in the order of `G.nodes` (or `0, ..., n_vertices - 1`). Values must be finite. Default: all 0.
///     filtration: `'weight'` (the default: the graph's own edges, at their weights) or `'shortest_path'` (every pair
///         joined by a path, at its length).
///     threshold: Only build the complex up to this value (default: no limit). Bars still alive at the threshold are
///         reported as essential.
///     method: `'cohomology'` (the default) or `'homology'`.
///     algorithm: The reduction algorithm and its options, as a `phlite.core.Algorithm` (default:
///         `Algorithm.serial()`).
///         Pipeline defaults (`'networks.flag'`): columns='heap', cache=False, store_r=False, union_find=True.
///     representatives: Whether to also compute a representative cycle for every bar. Cells are tuples of nodes (node
///         labels for a networkx graph).
///
/// Returns:
///     The barcode, with dimensions `0, ..., max_dim`.
#[gen_stub_pyfunction(module = "phlite.networks")]
#[gen_stub(override_return_type(type_repr = "core.Barcode", imports = ("phlite.core")))]
#[pyfunction]
#[pyo3(signature = (graph, edges = None, /, max_dim = 1, *, weight = Some("weight"), node_weight = None, filtration = "weight", threshold = None, method = "cohomology", algorithm = None, representatives = false))]
#[allow(clippy::too_many_arguments)]
fn flag<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing")))] graph: &Bound<'_, PyAny>,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Sequence[typing.Union[typing.Tuple[builtins.int, builtins.int], typing.Tuple[builtins.int, builtins.int, builtins.float]]]]", imports = ("builtins", "typing")))]
    edges: Option<Vec<Bound<'_, PyAny>>>,
    max_dim: usize,
    weight: Option<&str>,
    #[gen_stub(override_type(type_repr = "typing.Optional[typing.Union[builtins.str, typing.Sequence[builtins.float]]]", imports = ("builtins", "typing")))]
    node_weight: Option<Bound<'_, PyAny>>,
    #[gen_stub(override_type(type_repr = "typing.Literal['weight', 'shortest_path']", imports = ("typing")))]
    filtration: &str,
    threshold: Option<f64>,
    method: &str,
    algorithm: Option<Bound<'_, PyAny>>,
    representatives: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (n_vertices, edges, values, labels) = read(graph, edges, weight, node_weight.as_ref())?;
    let graph = match filtration {
        "weight" => Graph::new(n_vertices, &edges, values.as_deref()),
        "shortest_path" => {
            // Without weights, every edge has length 1
            let lengths: Vec<_> = if weight.is_none() {
                edges.iter().map(|&(u, v, _)| (u, v, 1.0)).collect()
            } else {
                edges
            };
            py.detach(|| Graph::shortest_paths(n_vertices, &lengths, values.as_deref()))
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown filtration {other:?}, expected 'weight' or 'shortest_path'"
            )))
        }
    }
    .map_err(network_error)?;
    let threshold = match threshold {
        None => INFINITY,
        Some(t) => NotNan::new(t).map_err(|_| PyValueError::new_err("threshold must not be NaN"))?,
    };
    if !rips_index_fits(n_vertices as usize, max_dim + 1) {
        return Err(PyValueError::new_err(format!(
            "Too many nodes ({n_vertices}) to index simplices of dimension {}",
            max_dim + 1
        )));
    }
    let method = parse_method(method)?;
    let algorithm = algorithm_from_py(algorithm.as_ref(), &DEFAULTS)?;
    let bars = py.detach(|| {
        algorithm.run(FlagRun {
            graph: &graph,
            max_dim,
            threshold,
            method,
            representatives,
            union_find: algorithm.union_find(),
            parallel_setup: algorithm.parallel_setup(),
        })
    })?;
    bars.into_python(py, &algorithm, representatives, labels.as_deref())
}

/// Compute the flag complex's barcode with whichever algorithm is configured.
struct FlagRun<'a> {
    graph: &'a Graph,
    max_dim: usize,
    threshold: NotNan<f64>,
    method: Method,
    representatives: bool,
    union_find: bool,
    parallel_setup: bool,
}

impl AlgorithmVisitor for FlagRun<'_> {
    type Output = SortedBars;

    fn visit<R: ColumnRepresentation>(self, algorithm: Algorithm<'_, R>) -> SortedBars {
        use phlite_clouds::rips::Distances;
        let n = self.graph.n_points();
        let bars = flag_barcode::<_, usize>(
            self.graph,
            self.max_dim,
            self.threshold,
            self.method,
            algorithm,
            self.representatives,
            self.union_find,
            self.parallel_setup,
        );
        SortedBars::new(bars, self.max_dim + 1, |simplex| {
            let vertices = simplex.to_vec(n);
            (vertices.len() - 1, vertices.into_iter().map(|v| v as u32).collect())
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(flag, m)?)
}
