pub mod boundary;
pub mod coboundary;

use coboundary::PathHomCell;
use log::info;
use ordered_float::NotNan;
use petgraph::{adj::NodeIndex, algo::dijkstra, Graph};
use phlite::{
    fields::Z2,
    matrices::{combinators::product, HasRowFiltration, MatrixOracle},
    reduction::{standard_algo_with_diagram, ClearedReductionMatrix},
};
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

use crate::{
    boundary::GrPPHBoundary,
    coboundary::{GrPPHCoboundary, PathHomSingleBasis},
};

fn build_filtration(n_vertices: u32, edges: &[(u32, u32, f64)]) -> Vec<Vec<Option<NotNan<f64>>>> {
    // Build graph (don't add edges with infinite edge filtration)
    let mut g = Graph::<(), f64>::new();
    g.extend_with_edges(edges.iter().filter(|(_i, _j, t)| *t < f64::INFINITY));

    // Compute shortest path filtration
    let mut filtration: Vec<Vec<_>> = vec![];
    for i in 0..n_vertices {
        let sp_lengths = dijkstra(&g, NodeIndex::from(i), None, |e| *e.weight());
        filtration.push(
            (0..n_vertices)
                .map(|j| {
                    if i == j {
                        return None;
                    };
                    sp_lengths
                        .get(&NodeIndex::from(j))
                        .copied()
                        .map(|t| NotNan::new(t).unwrap())
                })
                .collect(),
        );
    }
    filtration
}

fn build_edge_set(edges: Vec<(u32, u32, f64)>) -> FxHashSet<(u16, u16)> {
    edges
        .into_iter()
        .map(|(i, j, _t)| (i as u16, j as u16))
        .collect()
}

type EssentialBars = Vec<Vec<f64>>;
type PairingBars = Vec<Vec<(f64, f64)>>;
// List of 1D persistent homology reps (each of which is a sum of edges)
type RepList = Vec<Vec<(u16, u16)>>;

#[pyfunction]
fn grpph(n_vertices: u32, edges: Vec<(u32, u32, f64)>) -> (EssentialBars, PairingBars) {
    let filtration = build_filtration(n_vertices, &edges);
    let edge_set = build_edge_set(edges);

    // Build coboundary
    // Absolute PcoH
    let d = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, n_vertices as u16);
    info!("Built coboundary matrix");
    // Relative PcoH
    let d_rev = (&d).reverse();

    // Reduce
    let (_v, diagram) = ClearedReductionMatrix::build_with_diagram(&d_rev, 0..=1);
    info!("Reduced coboundary matrix");

    // Translate diagram to Python compatible types
    let mut essential = vec![vec![], vec![]];

    for idx in diagram.essential.iter() {
        let idx = idx.0; //Pull out of reverse
        let f_val = d.filtration_value(idx).into_inner();
        let dimension = idx.dimension();
        essential[dimension].push(f_val);
    }

    let mut pairings = vec![vec![], vec![]];
    for (death_cell, birth_cell) in diagram.pairings.iter() {
        let birth_f = d.filtration_value(birth_cell.0);
        let death_f = d.filtration_value(death_cell.0);
        let dimension = birth_cell.0.dimension();

        if death_f == birth_f {
            continue;
        }
        pairings[dimension].push((birth_f.into_inner(), death_f.into_inner()));
    }
    info!("Translated persistence diagram");

    (essential, pairings)
}

// TODO: This only find the infinite reps, add in essential reps too!

#[pyfunction]
fn grpph_with_involution(
    n_vertices: u32,
    edges: Vec<(u32, u32, f64)>,
) -> (EssentialBars, PairingBars, RepList) {
    let filtration = build_filtration(n_vertices, &edges);
    let edge_set = build_edge_set(edges);

    // Build coboundary
    // Absolute PcoH
    let d = GrPPHCoboundary::<Z2, _>::build(&filtration, &edge_set, n_vertices as u16);
    info!("Built coboundary matrix");
    // Relative PcoH
    let d_rev = (&d).reverse();

    // Reduce
    let (_v, diagram) = ClearedReductionMatrix::build_with_diagram(&d_rev, 0..=1);
    info!("Reduced coboundary matrix");

    // Translate diagram to Python compatible types
    let mut essential = vec![vec![], vec![]];

    for idx in diagram.essential.iter() {
        let idx = idx.0; //Pull out of reverse
        let f_val = d.filtration_value(idx).into_inner();
        let dimension = idx.dimension();
        essential[dimension].push(f_val);
    }

    let mut involution_basis = vec![];
    let mut pairings = vec![vec![], vec![]];
    for (death_cell, birth_cell) in diagram.pairings.iter() {
        let birth_f = d.filtration_value(birth_cell.0);
        let death_f = d.filtration_value(death_cell.0);
        let dimension = birth_cell.0.dimension();
        if dimension == 1 {
            involution_basis.push((death_f, death_cell.0));
        }

        if death_f == birth_f {
            continue;
        }
        pairings[dimension].push((birth_f.into_inner(), death_f.into_inner()));
    }
    info!("Translated persistence diagram");

    // Sort involution basis
    involution_basis.sort_unstable();
    let involution_basis = PathHomSingleBasis(involution_basis);
    info!("Sorted involution basis");

    // Decompose boundary matrix restricted to involution basis
    let d_boundary = GrPPHBoundary::<Z2, _>::build(&filtration, &edge_set, involution_basis);
    let (v_boundary, boundary_diagram) = standard_algo_with_diagram(&d_boundary, false);
    info!("Reduced involution boundary matrix");
    let r_boundary = product(&d_boundary, &v_boundary);

    // Read off reps from pairings
    let mut reps = vec![];
    for pairing in boundary_diagram.pairings {
        let birth_cell = pairing.0;
        let death_cell = pairing.1;
        let birth_f = d.filtration_value(birth_cell);
        let death_f = d.filtration_value(death_cell);
        if death_f == birth_f {
            continue;
        }
        let mut rep = r_boundary.build_bhcol(death_cell);
        let new_rep: Vec<_> = rep
            .drain_sorted()
            .map(|entry| {
                let row = entry.row_index;
                match row {
                    PathHomCell::Edge(a, b) => (a, b),
                    _ => {
                        unreachable!("death_cell is a 2-cell so boundary should be a sum of edges")
                    }
                }
            })
            .collect();
        reps.push(new_rep)
    }
    info!("Built up representatives");

    (essential, pairings, reps)
}

/// A Python module implemented in Rust.
#[pymodule]
fn phlite_grpph(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(grpph, m)?)?;
    m.add_function(wrap_pyfunction!(grpph_with_involution, m)?)?;
    Ok(())
}
