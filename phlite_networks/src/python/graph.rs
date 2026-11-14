//! Reading a weighted digraph from Python, as every pipeline here takes one: a networkx `DiGraph` (or
//! `MultiDiGraph`), or the number of nodes and a list of `(source, target, weight)` edges.

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};

use crate::digraph::Distances;

/// A digraph as `(n_vertices, edges, labels)`, where vertex `i` has node label `labels[i]` (if read from networkx).
pub(crate) type LabelledDigraph = (u32, Vec<(u32, u32, f64)>, Option<Vec<Py<PyAny>>>);

/// Check the edges: endpoints in range, no self-loops, weights non-negative (or infinite); and at most `max_nodes`
/// nodes.
pub(crate) fn validate(n_vertices: u32, edges: &[(u32, u32, f64)], max_nodes: u32) -> PyResult<()> {
    if n_vertices > max_nodes {
        return Err(PyValueError::new_err(format!(
            "n_vertices must be at most {max_nodes}"
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

/// Read a networkx (multi)digraph into `(n_vertices, edges, labels)`, where node `labels[i]` becomes vertex `i`.
/// Edges without a `weight` attribute get weight 1, as in networkx's shortest path functions.
fn from_networkx(graph: &Bound<'_, PyAny>, weight: &str) -> PyResult<(u32, Vec<(u32, u32, f64)>, Vec<Py<PyAny>>)> {
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
            "Expected a directed graph, try passing G.to_directed() (which has an edge each way for every edge of G)",
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

/// Read the digraph a pipeline was called with, as `f(G)` or `f(n_vertices, edges)` (`name` is the function's, for
/// the error messages), and validate it (with at most `max_nodes` nodes).
pub(crate) fn read_graph(
    graph: &Bound<'_, PyAny>,
    edges: Option<Vec<(u32, u32, f64)>>,
    weight: &str,
    name: &str,
    max_nodes: u32,
) -> PyResult<LabelledDigraph> {
    let (n_vertices, edges, labels) = match edges {
        Some(edges) => {
            let n_vertices: u32 = graph.extract().map_err(|_| {
                PyTypeError::new_err("When edges are given, the first argument must be the number of nodes")
            })?;
            (n_vertices, edges, None)
        }
        None if graph.extract::<u32>().is_ok() => {
            return Err(PyTypeError::new_err(format!(
                "Missing edges: call {name}(n_vertices, edges) or {name}(G) for a networkx DiGraph G"
            )))
        }
        None => {
            let (n_vertices, edges, labels) = from_networkx(graph, weight)?;
            (n_vertices, edges, Some(labels))
        }
    };
    validate(n_vertices, &edges, max_nodes)?;
    Ok((n_vertices, edges, labels))
}

/// Read the `filtration` argument: `'weight'` (each edge at its own weight) or `'shortest_path'` (each pair of nodes
/// at the length of the shortest directed path between them).
pub(crate) fn parse_distances(filtration: &str) -> PyResult<Distances> {
    match filtration {
        "weight" => Ok(Distances::Weight),
        "shortest_path" => Ok(Distances::ShortestPath),
        other => Err(PyValueError::new_err(format!(
            "Unknown filtration {other:?}, expected 'weight' or 'shortest_path'"
        ))),
    }
}
