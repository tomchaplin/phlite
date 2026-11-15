//! What can be wrong with a weighted digraph given to this crate.

use std::fmt;

/// Why a weighted digraph was rejected: every constructor that takes `(source, target, weight)` edges checks them
/// (see [`check_edges`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkError {
    /// An edge has an endpoint outside `0..n_vertices`.
    VertexOutOfRange {
        source: u32,
        target: u32,
        n_vertices: u32,
    },
    /// An edge from a node to itself, which none of the filtrations here support.
    SelfLoop { vertex: u32 },
    /// An edge with a negative or `NaN` weight (weights are times: non-negative, and infinite for an edge that
    /// never appears).
    BadWeight { source: u32, target: u32, weight: f64 },
    /// A weight matrix whose row `row` has `len` entries, not `n`.
    NotSquare { row: usize, len: usize, n: usize },
    /// More nodes than the structure can index.
    TooManyVertices { n_vertices: usize, max: usize },
    /// An edge of an undirected graph with a `NaN` weight (negative weights are fine there: they are only values).
    NotANumber { source: u32, target: u32 },
    /// A node value that is infinite or `NaN`.
    BadNodeValue { node: u32, value: f64 },
    /// One value per node is needed.
    WrongNumberOfNodeValues { values: usize, n_vertices: usize },
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            NetworkError::VertexOutOfRange {
                source,
                target,
                n_vertices,
            } => write!(f, "edge {source} -> {target} has a node outside 0..{n_vertices}"),
            NetworkError::SelfLoop { vertex } => {
                write!(f, "edge {vertex} -> {vertex} is a self-loop, which is not supported")
            }
            NetworkError::BadWeight {
                source,
                target,
                weight,
            } => write!(
                f,
                "edge {source} -> {target} has weight {weight}; weights must be non-negative (or infinite)"
            ),
            NetworkError::NotSquare { row, len, n } => {
                write!(f, "the weight matrix is not square: row {row} has {len} entries, not {n}")
            }
            NetworkError::TooManyVertices { n_vertices, max } => {
                write!(f, "{n_vertices} nodes is more than the {max} supported")
            }
            NetworkError::NotANumber { source, target } => write!(f, "edge {source} - {target} has a NaN weight"),
            NetworkError::BadNodeValue { node, value } => {
                write!(f, "node {node} has value {value}; node values must be finite")
            }
            NetworkError::WrongNumberOfNodeValues { values, n_vertices } => {
                write!(f, "there are {values} node values for {n_vertices} nodes")
            }
        }
    }
}

impl std::error::Error for NetworkError {}

/// Check edges on `n_vertices` nodes: endpoints in range, no self-loops, and weights non-negative (infinity allowed,
/// `NaN` not).
pub fn check_edges<'a>(
    n_vertices: u32,
    edges: impl IntoIterator<Item = &'a (u32, u32, f64)>,
) -> Result<(), NetworkError> {
    for &(source, target, weight) in edges {
        if source >= n_vertices || target >= n_vertices {
            return Err(NetworkError::VertexOutOfRange {
                source,
                target,
                n_vertices,
            });
        }
        if source == target {
            return Err(NetworkError::SelfLoop { vertex: source });
        }
        if weight.is_nan() || weight < 0.0 {
            return Err(NetworkError::BadWeight {
                source,
                target,
                weight,
            });
        }
    }
    Ok(())
}
