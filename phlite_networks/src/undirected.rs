//! Undirected weighted graphs, with values on their nodes, and the flag (clique) complex filtration they define.
//!
//! The flag complex of a graph has a simplex for every clique (every set of nodes pairwise joined by an edge). Here
//! each edge `uv` has a weight `w(uv)` and each node `v` a value `f(v)`, and a simplex enters at the largest weight
//! and value among its edges and nodes:
//! ```text
//! value(σ) = max( max over edges uv of σ of w(uv),  max over nodes v of σ of f(v) )
//! ```
//! So node `v` enters at `f(v)` (0 by default), and edge `uv` at `max(w(uv), f(u), f(v))`. Special cases:
//! * node values all 0: the flag complex of the weighted graph (as Rips is, for a distance matrix);
//! * edge weights all `-∞` (no weights): the **lower-star** filtration of the node values, each simplex entering
//!   with its latest node;
//! * both: GUDHI's flag filtration of a `SimplexTree` with vertex and edge values (made non-decreasing).
//!
//! [`Graph`] implements `phlite_clouds`' [`Distances`], so the filtration is computed by its Rips machinery (union-find
//! in dimension 0, apparent pairs, enumerating cofacets through the neighbours of the sparsest vertex): call
//! [`flag_barcode`].

use ordered_float::NotNan;
use phlite::{
    barcode::{Algorithm, Method},
    columns::ColumnRepresentation,
    values::{INFINITY, ZERO},
};
use phlite_clouds::rips::{rips_barcode_with, Distances, RipsBar, RipsInt};
// Listing a small flag complex and reducing it explicitly (shared with `phlite.clouds.rips` on sparse input)
pub use phlite_clouds::rips::explicit::{cliques_within, explicit_flag_barcode};

use crate::NetworkError;

/// An undirected graph, each edge with a weight and each node with a value (see the [module docs](self)), stored as
/// the symmetric sparse matrix of the values at which its edges enter.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Row `v` is `indices[indptr[v]..indptr[v + 1]]`, its neighbours in increasing order
    indptr: Vec<usize>,
    indices: Vec<u32>,
    /// When each edge enters: `max(w(uv), f(u), f(v))`
    values: Vec<NotNan<f64>>,
    /// When each node enters
    births: Vec<NotNan<f64>>,
}

impl Graph {
    /// The graph on `n_vertices` nodes with the given `(u, v, weight)` edges, either way round, and the given node
    /// values (`None` for all 0). An edge given more than once keeps its smallest weight. An edge of weight `+∞`
    /// never enters, so is left out; one of weight `-∞` has no weight of its own, entering with its later node.
    ///
    /// # Errors
    /// If an edge has an endpoint outside `0..n_vertices`, is a self-loop, or has a `NaN` weight; if there isn't one
    /// value per node; or if a node value is infinite or `NaN`.
    pub fn new(
        n_vertices: u32,
        edges: &[(u32, u32, f64)],
        node_values: Option<&[f64]>,
    ) -> Result<Self, NetworkError> {
        let n = n_vertices as usize;
        let births: Vec<NotNan<f64>> = match node_values {
            None => vec![ZERO; n],
            Some(values) => {
                if values.len() != n {
                    return Err(NetworkError::WrongNumberOfNodeValues {
                        values: values.len(),
                        n_vertices: n,
                    });
                }
                values
                    .iter()
                    .enumerate()
                    .map(|(node, &value)| {
                        NotNan::new(value)
                            .ok()
                            .filter(|v| v.is_finite())
                            .ok_or(NetworkError::BadNodeValue { node: node as u32, value })
                    })
                    .collect::<Result<_, _>>()?
            }
        };
        // Each edge both ways round, at max(weight, both values), then sorted and deduplicated (keeping the smallest)
        let mut entries: Vec<(u32, u32, NotNan<f64>)> = Vec::with_capacity(2 * edges.len());
        for &(u, v, w) in edges {
            if u >= n_vertices || v >= n_vertices {
                return Err(NetworkError::VertexOutOfRange { source: u, target: v, n_vertices });
            }
            if u == v {
                return Err(NetworkError::SelfLoop { vertex: u });
            }
            let Ok(w) = NotNan::new(w) else {
                return Err(NetworkError::NotANumber { source: u, target: v });
            };
            if w == INFINITY {
                continue;
            }
            let value = w.max(births[u as usize]).max(births[v as usize]);
            entries.push((u, v, value));
            entries.push((v, u, value));
        }
        entries.sort_unstable();
        entries.dedup_by_key(|&mut (u, v, _)| (u, v));
        let mut indptr = vec![0; n + 1];
        for &(u, _, _) in &entries {
            indptr[u as usize + 1] += 1;
        }
        for v in 0..n {
            indptr[v + 1] += indptr[v];
        }
        Ok(Self {
            indptr,
            indices: entries.iter().map(|&(_, v, _)| v).collect(),
            values: entries.iter().map(|&(_, _, d)| d).collect(),
            births,
        })
    }

    /// The graph of shortest paths: every pair of nodes `u`, `v` joined by a path, by an edge of weight the length of
    /// the shortest path between them (in the graph of `edges`, `(u, v, length)` either way round, ignoring infinite
    /// lengths), with the given node values. Its flag complex is the Vietoris–Rips filtration of the graph's
    /// shortest-path metric.
    ///
    /// # Errors
    /// As [`new`](Self::new), and if a length is negative (Dijkstra's algorithm needs them non-negative).
    pub fn shortest_paths(
        n_vertices: u32,
        edges: &[(u32, u32, f64)],
        node_values: Option<&[f64]>,
    ) -> Result<Self, NetworkError> {
        // Each edge both ways round, as a digraph (which checks the lengths), then each pair once
        let both: Vec<(u32, u32, f64)> = edges.iter().flat_map(|&(u, v, w)| [(u, v, w), (v, u, w)]).collect();
        let paths = crate::digraph::Digraph::shortest_paths(n_vertices, &both)?;
        let pairs: Vec<(u32, u32, f64)> = paths
            .edges()
            .filter(|&(u, v, _)| u < v)
            .map(|(u, v, length)| (u, v, length.into_inner()))
            .collect();
        Self::new(n_vertices, &pairs, node_values)
    }

    /// The number of edges.
    pub fn n_edges(&self) -> usize {
        self.indices.len() / 2
    }

    fn row(&self, v: usize) -> (&[u32], &[NotNan<f64>]) {
        let range = self.indptr[v]..self.indptr[v + 1];
        (&self.indices[range.clone()], &self.values[range])
    }
}

impl Distances for Graph {
    const HAS_BIRTHS: bool = true;

    fn n_points(&self) -> usize {
        self.births.len()
    }

    #[inline]
    fn distance(&self, i: usize, j: usize) -> NotNan<f64> {
        let (indices, values) = self.row(i);
        match indices.binary_search(&(j as u32)) {
            Ok(k) => values[k],
            Err(_) => INFINITY,
        }
    }

    #[inline]
    fn birth(&self, v: usize) -> NotNan<f64> {
        self.births[v]
    }

    #[inline]
    fn neighbours(&self, v: usize, from: usize) -> impl Iterator<Item = usize> + Clone + Send + '_ {
        let row = self.row(v).0;
        let start = row.partition_point(|&w| (w as usize) < from);
        row[start..].iter().map(|&w| w as usize)
    }

    #[inline]
    fn degree(&self, v: usize) -> usize {
        self.indptr[v + 1] - self.indptr[v]
    }

    fn enclosing_radius(&self) -> NotNan<f64> {
        // Only a node joined to every other has a finite eccentricity (which is at least every node's value, as each
        // edge's is at least its nodes')
        let n = self.n_points();
        (0..n)
            .filter(|&v| self.degree(v) == n - 1)
            .map(|v| {
                let row = self.row(v).1.iter().copied().max();
                row.map_or(self.births[v], |m| m.max(self.births[v]))
            })
            .min()
            .unwrap_or(INFINITY)
    }
}

/// The barcode of the flag complex of `graph` in dimensions `0..=max_dim`, as
/// [`rips_barcode_with`](phlite_clouds::rips::rips_barcode_with) (whose simplices are cliques of the graph): up to
/// `threshold` (bars still alive there are essential), by cohomology or homology, with union-find in dimension 0 if
/// `union_find`, building the bases in parallel if `parallel_setup`, collapsing edges first if `collapse` (see
/// [`collapse`](phlite_clouds::rips::collapse); not with node values), and optionally with representatives.
///
/// # Panics
/// If simplices of dimension `max_dim + 1` don't fit in a `RipsIndex<I>` (see
/// [`rips_index_fits`](phlite_clouds::rips::rips_index_fits)).
#[allow(clippy::too_many_arguments)]
pub fn flag_barcode<R: ColumnRepresentation, I: RipsInt>(
    graph: &Graph,
    max_dim: usize,
    threshold: NotNan<f64>,
    method: Method,
    algorithm: Algorithm<'_, R>,
    representatives: bool,
    union_find: bool,
    parallel_setup: bool,
    collapse: bool,
) -> Vec<RipsBar<I>> {
    rips_barcode_with(graph, max_dim, threshold, method, algorithm, representatives, union_find, parallel_setup, collapse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phlite::reduction::Clearing;

    fn bars(graph: &Graph, max_dim: usize) -> Vec<(usize, f64, Option<f64>)> {
        let n = graph.n_points();
        let mut out: Vec<_> = flag_barcode::<_, usize>(
            graph,
            max_dim,
            INFINITY,
            Method::Cohomology,
            Algorithm::Clearing(Clearing::new()),
            false,
            true,
            false,
            true,
        )
        .into_iter()
        .map(|bar| {
            let dim = bar.birth.dimension(n);
            (dim, bar.birth_value.into_inner(), bar.death.map(|(_, d)| d.into_inner()))
        })
        .collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out
    }

    #[test]
    fn a_square_with_a_diagonal() {
        // A 4-cycle 0-1-2-3 of weight 1, and the diagonal 0-2 at 2: a loop born at 1, filled at 2
        let edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 2.0)];
        let graph = Graph::new(4, &edges, None).unwrap();
        assert_eq!(graph.n_edges(), 5);
        assert_eq!(bars(&graph, 1), vec![(0, 0.0, None), (0, 0.0, Some(1.0)), (0, 0.0, Some(1.0)), (0, 0.0, Some(1.0)), (1, 1.0, Some(2.0))]);
    }

    #[test]
    fn lower_star_of_node_values() {
        // A path 0 - 1 - 2 with values 0, 5, -1 and no edge weights: two minima (0 and -1) joined at 5
        let edges = [(0, 1, f64::NEG_INFINITY), (1, 2, f64::NEG_INFINITY)];
        let graph = Graph::new(3, &edges, Some(&[0.0, 5.0, -1.0])).unwrap();
        assert_eq!(bars(&graph, 0), vec![(0, -1.0, None), (0, 0.0, Some(5.0))]);
    }

    #[test]
    fn shortest_paths_are_rips_of_the_graph_metric() {
        // A 6-cycle of unit edges: its geodesic metric is the circle's, so a loop from 1 until 3
        let edges: Vec<_> = (0..6u32).map(|i| (i, (i + 1) % 6, 1.0)).collect();
        let graph = Graph::shortest_paths(6, &edges, None).unwrap();
        assert_eq!(graph.n_edges(), 15);
        assert_eq!(graph.distance(0, 3), NotNan::new(3.0).unwrap());
        assert_eq!(bars(&graph, 1).last(), Some(&(1, 1.0, Some(2.0))));
        // Negative lengths are refused
        assert!(matches!(Graph::shortest_paths(2, &[(0, 1, -1.0)], None), Err(NetworkError::BadWeight { .. })));
        // Two components are never joined
        let graph = Graph::shortest_paths(4, &[(0, 1, 1.0), (2, 3, 1.0)], None).unwrap();
        assert_eq!(graph.n_edges(), 2);
    }

    #[test]
    fn bad_input() {
        assert_eq!(
            Graph::new(2, &[(0, 0, 1.0)], None).err(),
            Some(NetworkError::SelfLoop { vertex: 0 })
        );
        assert!(matches!(Graph::new(2, &[(0, 1, f64::NAN)], None), Err(NetworkError::NotANumber { .. })));
        assert!(matches!(
            Graph::new(2, &[], Some(&[0.0])),
            Err(NetworkError::WrongNumberOfNodeValues { values: 1, n_vertices: 2 })
        ));
        assert!(matches!(
            Graph::new(2, &[], Some(&[0.0, f64::INFINITY])),
            Err(NetworkError::BadNodeValue { node: 1, .. })
        ));
        // Negative weights are just values here; repeated edges keep the smallest
        let graph = Graph::new(2, &[(0, 1, 3.0), (1, 0, -2.0)], Some(&[-5.0, -4.0])).unwrap();
        assert_eq!(graph.distance(0, 1), NotNan::new(-2.0).unwrap());
    }
}
