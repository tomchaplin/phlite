//! Weighted digraphs, as the filtrations in this crate read them.
//!
//! Every filtration here starts from a list of weighted edges `(source, target, weight)` on the nodes
//! `0, ..., n - 1`, and turns it into a [`Digraph`] whose edge `a -> b` enters the filtration at a time `w(a, b)`,
//! chosen by [`Distances`]:
//!
//! * [`Distances::Weight`]: the weight of the edge itself (the lightest, if there are parallel edges). A pair of nodes
//!   with no edge between them is never joined.
//! * [`Distances::ShortestPath`]: the length of the shortest directed path from `a` to `b`, so `a -> b` enters as soon
//!   as there is any path from `a` to `b` (this is the filtration of [`grpph`](crate::grpph)).
//!
//! Nodes enter at time 0. Edges with infinite weight never enter (they still count for grounding, see
//! [`build_edge_set`](crate::grpph::build_edge_set)), and self-loops are ignored: the callers (`phlite.networks`)
//! reject them, as well as negative and `NaN` weights.

use ordered_float::NotNan;
use petgraph::{algo::dijkstra, graph::NodeIndex, Graph};

/// When the edge `a -> b` enters the filtration, see the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Distances {
    /// The edge's own weight (the lightest of parallel edges).
    #[default]
    Weight,
    /// The length of the shortest directed path from `a` to `b`.
    ShortestPath,
}

/// A weighted digraph on the nodes `0, ..., n - 1`, without self-loops or parallel edges, with finite non-negative
/// weights, stored as sorted adjacency lists in both directions (and also as a dense matrix when that is small),
/// so that the neighbours of a node can be listed in order and the weight of an edge looked up quickly.
#[derive(Debug, Clone)]
pub struct Digraph {
    n_vertices: u32,
    /// `out_edges[out_offsets[a]..out_offsets[a + 1]]` are the edges `a -> b` as `(b, w(a, b))`, sorted by `b`
    out_offsets: Vec<usize>,
    out_edges: Vec<(u32, NotNan<f64>)>,
    /// `in_edges[in_offsets[b]..in_offsets[b + 1]]` are the edges `a -> b` as `(a, w(a, b))`, sorted by `a`
    in_offsets: Vec<usize>,
    in_edges: Vec<(u32, NotNan<f64>)>,
    /// `dense[a * n + b] = w(a, b)`, infinite for no edge, if the graph is small enough (see [`Digraph::new`]).
    /// Looking weights up here is much faster than searching the adjacency lists.
    dense: Option<Vec<f64>>,
}

/// Build compressed sorted adjacency lists from `(from, to, weight)` triples, sorted by `(from, to)`.
fn compress(n: usize, mut triples: Vec<(u32, u32, NotNan<f64>)>) -> (Vec<usize>, Vec<(u32, NotNan<f64>)>) {
    triples.sort_unstable();
    let mut offsets = vec![0; n + 1];
    for &(from, _, _) in &triples {
        offsets[from as usize + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    (offsets, triples.into_iter().map(|(_, to, w)| (to, w)).collect())
}

impl Digraph {
    /// The digraph on `n_vertices` nodes with the given edges, each entering at its weight: parallel edges keep the
    /// lightest weight, and self-loops and edges with infinite (or `NaN`) weight are left out.
    ///
    /// # Panics
    /// If an edge refers to a node outside `0..n_vertices`.
    pub fn new(n_vertices: u32, edges: impl IntoIterator<Item = (u32, u32, f64)>) -> Self {
        let n = n_vertices as usize;
        let mut lightest: rustc_hash::FxHashMap<(u32, u32), NotNan<f64>> = Default::default();
        for (a, b, w) in edges {
            assert!(
                a < n_vertices && b < n_vertices,
                "edge ({a}, {b}) refers to a node outside 0..{n_vertices}"
            );
            if a == b || !w.is_finite() {
                continue;
            }
            let w = NotNan::new(w).unwrap();
            lightest
                .entry((a, b))
                .and_modify(|current| *current = (*current).min(w))
                .or_insert(w);
        }
        let n_edges = lightest.len();
        // Dense when that costs at most a few times the adjacency lists (or is tiny anyway)
        let dense = (n * n <= (16 * n_edges).max(1 << 16)).then(|| {
            let mut dense = vec![f64::INFINITY; n * n];
            for (&(a, b), w) in &lightest {
                dense[a as usize * n + b as usize] = w.into_inner();
            }
            dense
        });
        let (out_offsets, out_edges) =
            compress(n, lightest.iter().map(|(&(a, b), &w)| (a, b, w)).collect());
        let (in_offsets, in_edges) =
            compress(n, lightest.iter().map(|(&(a, b), &w)| (b, a, w)).collect());
        Self {
            n_vertices,
            out_offsets,
            out_edges,
            in_offsets,
            in_edges,
            dense,
        }
    }

    /// The digraph of `edges` filtered as `distances` says (see the [module docs](self)).
    pub fn with_distances(n_vertices: u32, edges: &[(u32, u32, f64)], distances: Distances) -> Self {
        match distances {
            Distances::Weight => Self::new(n_vertices, edges.iter().copied()),
            Distances::ShortestPath => Self::shortest_paths(n_vertices, edges),
        }
    }

    /// The complete digraph of shortest-path lengths: `a -> b` with weight the length of the shortest directed path
    /// from `a` to `b` in the digraph of `edges` (ignoring infinite weights), if there is one.
    pub fn shortest_paths(n_vertices: u32, edges: &[(u32, u32, f64)]) -> Self {
        let mut graph = Graph::<(), f64>::with_capacity(n_vertices as usize, edges.len());
        for _ in 0..n_vertices {
            graph.add_node(());
        }
        graph.extend_with_edges(
            edges
                .iter()
                .filter(|(a, b, w)| a != b && w.is_finite()),
        );
        let mut distances = vec![];
        for a in 0..n_vertices {
            let lengths = dijkstra(&graph, NodeIndex::new(a as usize), None, |e| *e.weight());
            distances.extend(
                lengths
                    .into_iter()
                    .map(|(b, length)| (a, b.index() as u32, length)),
            );
        }
        Self::new(n_vertices, distances)
    }

    /// The same digraph with every edge reversed.
    pub fn transpose(&self) -> Self {
        Self {
            n_vertices: self.n_vertices,
            out_offsets: self.in_offsets.clone(),
            out_edges: self.in_edges.clone(),
            in_offsets: self.out_offsets.clone(),
            in_edges: self.out_edges.clone(),
            dense: self.dense.as_ref().map(|dense| {
                let n = self.n_vertices as usize;
                let mut transposed = vec![f64::INFINITY; n * n];
                for a in 0..n {
                    for b in 0..n {
                        transposed[b * n + a] = dense[a * n + b];
                    }
                }
                transposed
            }),
        }
    }

    pub fn n_vertices(&self) -> u32 {
        self.n_vertices
    }

    /// The number of edges.
    pub fn n_edges(&self) -> usize {
        self.out_edges.len()
    }

    /// The edges `a -> b` out of `a`, as `(b, w(a, b))`, sorted by `b`.
    #[inline]
    pub fn out_edges(&self, a: u32) -> &[(u32, NotNan<f64>)] {
        &self.out_edges[self.out_offsets[a as usize]..self.out_offsets[a as usize + 1]]
    }

    /// The edges `a -> b` into `b`, as `(a, w(a, b))`, sorted by `a`.
    #[inline]
    pub fn in_edges(&self, b: u32) -> &[(u32, NotNan<f64>)] {
        &self.in_edges[self.in_offsets[b as usize]..self.in_offsets[b as usize + 1]]
    }

    /// The weight of the edge `a -> b`, if there is one.
    #[inline]
    pub fn weight(&self, a: u32, b: u32) -> Option<NotNan<f64>> {
        match &self.dense {
            Some(dense) => {
                let w = dense[a as usize * self.n_vertices as usize + b as usize];
                // Weights are finite, so infinity means there's no edge (and the value can't be NaN)
                (w != f64::INFINITY).then(|| unsafe { NotNan::new_unchecked(w) })
            }
            None => {
                let edges = self.out_edges(a);
                edges
                    .binary_search_by_key(&b, |&(target, _)| target)
                    .ok()
                    .map(|i| edges[i].1)
            }
        }
    }

    /// Every edge, as `(a, b, w(a, b))`, sorted by `(a, b)`.
    pub fn edges(&self) -> impl Iterator<Item = (u32, u32, NotNan<f64>)> + '_ {
        (0..self.n_vertices).flat_map(move |a| self.out_edges(a).iter().map(move |&(b, w)| (a, b, w)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(graph: &Digraph) -> Vec<(u32, u32, f64)> {
        graph.edges().map(|(a, b, w)| (a, b, w.into_inner())).collect()
    }

    #[test]
    fn parallel_edges_keep_the_lightest_and_infinite_edges_and_loops_are_dropped() {
        let graph = Digraph::new(
            3,
            [(0, 1, 2.0), (0, 1, 1.0), (1, 1, 0.0), (1, 2, f64::INFINITY), (2, 0, 3.0)],
        );
        assert_eq!(weights(&graph), vec![(0, 1, 1.0), (2, 0, 3.0)]);
        assert_eq!(graph.in_edges(0), &[(2, NotNan::new(3.0).unwrap())]);
        assert_eq!(graph.weight(1, 2), None);
    }

    #[test]
    fn shortest_paths_and_transpose() {
        let edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0), (2, 3, f64::INFINITY)];
        let graph = Digraph::shortest_paths(4, &edges);
        assert_eq!(weights(&graph), vec![(0, 1, 1.0), (0, 2, 3.0), (1, 2, 2.0)]);
        let transposed = graph.transpose();
        assert_eq!(weights(&transposed), vec![(1, 0, 1.0), (2, 0, 3.0), (2, 1, 2.0)]);
    }

    /// Weights looked up in the adjacency lists (large sparse graphs) and in the dense matrix (small ones) agree with
    /// the edges given
    #[test]
    fn lookups_match_the_edges() {
        for n in [20u32, 300] {
            let edges: Vec<_> = (0..n)
                .flat_map(|a| [(a, (a * 7 + 1) % n, a as f64), (a, (a * 13 + 5) % n, 1.0)])
                .collect();
            let graph = Digraph::new(n, edges.iter().copied());
            assert_eq!(graph.dense.is_some(), n == 20);
            for a in 0..n {
                for b in 0..n {
                    let expected = edges
                        .iter()
                        .filter(|e| (e.0, e.1) == (a, b) && a != b)
                        .map(|e| NotNan::new(e.2).unwrap())
                        .min();
                    assert_eq!(graph.weight(a, b), expected, "{a} -> {b}");
                }
            }
        }
    }
}
