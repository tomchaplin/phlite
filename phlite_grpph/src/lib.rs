pub mod boundary;
pub mod coboundary;
mod python;

use coboundary::{PathHomCell, VecDigraphFiltration};
use log::info;
use ordered_float::NotNan;
use petgraph::{algo::dijkstra, graph::NodeIndex, Graph};
use phlite::{
    barcode::{self, Algorithm, Bar},
    fields::Z2,
    matrices::{ColBasis, HasRowFiltration, HasSplitColBasis, MatrixOracle, SplitBasis},
};
use rustc_hash::FxHashSet;

use crate::{
    boundary::GrPPHBoundary,
    coboundary::{DigraphEdgeSet, GrPPHCoboundary},
};

/// The shortest-path filtration of a weighted digraph, as required by [`GrPPHCoboundary`].
///
/// Edges with infinite weight are ignored (they are still part of the graph for the purposes of grounding,
/// see [`build_edge_set`]).
pub fn build_filtration(n_vertices: u32, edges: &[(u32, u32, f64)]) -> VecDigraphFiltration {
    // Build graph
    let mut g = Graph::<(), f64>::new();
    // Add all nodes
    for i in 0..n_vertices {
        let new_index = g.add_node(());
        assert!(new_index.index() == i as usize)
    }
    // Add all edges with finite weight
    // We remove edges with infinite weights because we want finite shortest-path times
    // Note the infinite weighted edges are still added for the sake of grounding
    g.extend_with_edges(edges.iter().filter(|(_i, _j, t)| *t < f64::INFINITY));

    // Compute shortest path filtration
    let mut filtration: Vec<Vec<_>> = vec![];
    for i in 0..n_vertices {
        let sp_lengths = dijkstra(&g, NodeIndex::new(i as usize), None, |e| *e.weight());
        filtration.push(
            (0..n_vertices)
                .map(|j| {
                    if i == j {
                        return None;
                    };
                    sp_lengths
                        .get(&NodeIndex::new(j as usize))
                        .copied()
                        .map(|t| NotNan::new(t).unwrap())
                })
                .collect(),
        );
    }
    filtration
}

/// The edges of the digraph, all of which appear at time 0 in the grounded filtration.
pub fn build_edge_set(edges: &[(u32, u32, f64)]) -> DigraphEdgeSet {
    edges
        .iter()
        .map(|&(i, j, _t)| (i as u16, j as u16))
        .collect()
}

pub use phlite::barcode::Method;

/// A bar in the GrPPH barcode, see [`phlite::barcode::Bar`].
pub type GrPPHBar = Bar<PathHomCell, NotNan<f64>, Z2>;

/// Every 2-cell in the filtration, in filtration order, found by enumerating the coboundaries of all edges.
fn all_two_cells<M>(coboundary: M) -> Vec<PathHomCell>
where
    M: HasSplitColBasis
        + HasRowFiltration<FiltrationT = NotNan<f64>>
        + MatrixOracle<ColT = PathHomCell, RowT = PathHomCell>,
{
    let edges = coboundary.sub_basis(1);
    let mut two_cells: FxHashSet<(NotNan<f64>, PathHomCell)> = FxHashSet::default();
    for i in 0..edges.size() {
        two_cells.extend(
            coboundary
                .column_with_filtration(edges.element(i))
                .map(|entry| (entry.filtration_value, entry.row_index)),
        );
    }
    let mut two_cells: Vec<_> = two_cells.into_iter().collect();
    two_cells.sort_unstable();
    two_cells.into_iter().map(|(_f, cell)| cell).collect()
}

/// Compute the grounded persistent path homology (GrPPH) barcode in dimensions 0 and 1.
///
/// * `filtration` and `edge_set` describe the weighted digraph, see [`build_filtration`] and [`build_edge_set`].
/// * `method` chooses between reducing the coboundary or boundary matrix.
/// * `algorithm` is the reduction algorithm to use.
/// * If `representatives` is true then every bar comes with a representative cycle
///   (a sum of nodes in dimension 0 and of edges in dimension 1).
///
/// Bars are returned in no particular order, and zero-length bars are included.
pub fn grpph_barcode(
    filtration: &VecDigraphFiltration,
    edge_set: &DigraphEdgeSet,
    n_vertices: u16,
    method: Method,
    algorithm: Algorithm<'_>,
    representatives: bool,
) -> Vec<GrPPHBar> {
    let coboundary = GrPPHCoboundary::<Z2, _>::build(filtration, edge_set, n_vertices);
    info!("Built coboundary matrix");
    let boundary = GrPPHBoundary::<Z2, _>::new(filtration, edge_set);

    // Each matrix is given its transpose, so that the reduction can skip apparent pairs
    // (tests/transpose.rs checks that these really are transposes)
    let bars = match (method, representatives) {
        (Method::Cohomology, false) => {
            barcode::cohomology((&coboundary).with_transpose(&boundary), algorithm)
        }
        (Method::Cohomology, true) => {
            barcode::cohomology_with_involution((&coboundary).with_transpose(&boundary), algorithm)
        }
        (Method::Homology, _) => {
            let basis = SplitBasis::from_dimensions([
                to_vec(coboundary.sub_basis(0)),
                to_vec(coboundary.sub_basis(1)),
                all_two_cells(&coboundary),
            ]);
            info!("Enumerated 2-cells");
            let boundary = (&boundary).with_basis(basis);
            let boundary = (&boundary).with_transpose(&coboundary.matrix);
            let mut bars = barcode::homology(boundary, algorithm, representatives);
            // There are no 3-cells, so 2-cells look essential but we don't compute H_2
            bars.retain(|bar| bar.birth.dimension() <= 1);
            bars
        }
    };
    info!("Computed barcode");
    bars
}

fn to_vec<B: ColBasis>(basis: B) -> Vec<B::ElemT> {
    (0..basis.size()).map(|i| basis.element(i)).collect()
}

// A function that gathers stub information
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);

#[cfg(test)]
mod tests {
    use super::*;
    use phlite::reduction::{Clearing, Standard};

    /// Bars as (dimension, birth value, death value), sorted, with zero-length bars removed
    fn summary(bars: &[GrPPHBar]) -> Vec<(usize, f64, Option<f64>)> {
        let mut out: Vec<_> = bars
            .iter()
            .filter(|bar| !bar.is_zero_length())
            .map(|bar| {
                (
                    bar.birth.dimension(),
                    bar.birth_value.into_inner(),
                    bar.death.map(|(_, d)| d.into_inner()),
                )
            })
            .collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out
    }

    fn algorithms() -> Vec<Algorithm<'static>> {
        vec![
            Algorithm::Standard(Standard::new()),
            Algorithm::Clearing(Clearing::new()),
            Algorithm::ParallelClearing(Clearing::new().parallel()),
        ]
    }

    /// The example from `examples/reps.py`
    fn reps_example() -> (u32, Vec<(u32, u32, f64)>) {
        (
            6,
            vec![
                (0, 1, 10.0),
                (0, 2, 10.0),
                (1, 3, 10.0),
                (2, 3, 10.0),
                (0, 3, 20.0),
                (0, 4, 5.0),
                (4, 5, 10.0),
                (5, 3, 5.0),
            ],
        )
    }

    #[test]
    fn every_method_and_algorithm_agree() {
        let (n, edges) = reps_example();
        let filtration = build_filtration(n, &edges);
        let edge_set = build_edge_set(&edges);
        let expected = vec![
            (0, 0.0, None),
            (1, 0.0, Some(10.0)),
            (1, 0.0, Some(15.0)),
            (1, 0.0, Some(20.0)),
        ];
        for method in [Method::Cohomology, Method::Homology] {
            for algorithm in algorithms() {
                for representatives in [false, true] {
                    let bars = grpph_barcode(
                        &filtration,
                        &edge_set,
                        n as u16,
                        method,
                        algorithm.clone(),
                        representatives,
                    );
                    assert_eq!(summary(&bars), expected, "{method:?} {algorithm:?}");
                    for bar in bars.iter().filter(|bar| !bar.is_zero_length()) {
                        let rep = bar.representative.as_ref();
                        assert_eq!(rep.is_some(), representatives);
                        if let Some(rep) = rep {
                            assert!(rep
                                .iter()
                                .all(|(_, cell)| cell.dimension() == bar.birth.dimension()));
                        }
                    }
                }
            }
        }
    }

    /// Every H_1 representative must be a cycle: over Z2, each vertex meets an even number of its edges.
    #[test]
    fn representatives_are_cycles() {
        let (n, edges) = reps_example();
        let filtration = build_filtration(n, &edges);
        let edge_set = build_edge_set(&edges);
        for method in [Method::Cohomology, Method::Homology] {
            let bars = grpph_barcode(
                &filtration,
                &edge_set,
                n as u16,
                method,
                Algorithm::default(),
                true,
            );
            for bar in bars.iter().filter(|bar| bar.birth.dimension() == 1) {
                let mut degree = vec![0usize; n as usize];
                for (_, cell) in bar.representative.as_ref().unwrap() {
                    let PathHomCell::Edge(a, b) = cell else {
                        panic!("H_1 representative contains non-edge {cell:?}")
                    };
                    degree[*a as usize] += 1;
                    degree[*b as usize] += 1;
                }
                assert!(
                    degree.iter().all(|d| d.is_multiple_of(2)),
                    "{bar:?} is not a cycle"
                );
            }
        }
    }
}
