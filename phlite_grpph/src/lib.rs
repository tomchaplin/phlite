pub mod coboundary;

use ordered_float::NotNan;
use petgraph::{adj::NodeIndex, algo::dijkstra, Graph};
use phlite::{
    fields::Z2,
    matrices::{combinators::product, HasRowFiltration, MatrixRef},
    reduction::ClearedReductionMatrix,
};
use pyo3::prelude::*;
use rustc_hash::FxHashSet;

use crate::coboundary::GrPPHCoboundary;

#[pyfunction]
fn grpph(n_vertices: u32, edges: Vec<(u32, u32, f64)>) -> (Vec<Vec<f64>>, Vec<Vec<(f64, f64)>>) {
    // Build graph
    let mut g = Graph::<(), f64>::new();
    g.extend_with_edges(edges.iter());
    println!("Built graph");

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
    println!("Computed filtration");

    // Compute edge set
    let edge_set: FxHashSet<(u16, u16)> = edges
        .into_iter()
        .map(|(i, j, _t)| (i as u16, j as u16))
        .collect();

    // Build coboundary
    // Absolute PcoH
    let d = GrPPHCoboundary::<Z2, _>::build(filtration, edge_set, n_vertices as u16);
    // Relative PcoH
    let d_rev = d.reverse();
    println!("Built coboundary matrix");

    // Reduce
    let (v, diagram) = ClearedReductionMatrix::build_with_diagram(&d_rev, 0..=1);
    println!("Reduced");

    let _r_rev = product(&d_rev, &v);
    let _r = product(&d, &v.unreverse());

    // Translate diagram to Python compatible types
    let mut essential = vec![vec![], vec![]];
    let mut pairings = vec![vec![], vec![]];

    for idx in diagram.essential.iter() {
        let idx = idx.0; //Pull out of reverse
        let f_val = d.filtration_value(idx).unwrap().into_inner();
        let dimension = idx.dimension();
        essential[dimension].push(f_val);
    }

    //let mut involution_basis = vec![];

    for (death_cell, birth_cell) in diagram.pairings.iter() {
        let birth_f = d.filtration_value(birth_cell.0).unwrap();
        let death_f = d.filtration_value(death_cell.0).unwrap();
        let dimension = birth_cell.0.dimension();
        //if dimension == 1 {
        //    involution_basis.push((death_f, tup.0));
        //}

        if death_f == birth_f {
            continue;
        }
        // println!("{:?},{:?}", tup.0, tup.1);
        // let mut cocycle = v.with_trivial_filtration().build_bhcol(tup.1).unwrap();
        // println!("Cycle start");
        // for entry in cocycle.drain_sorted() {
        //     println!("{:?} :: {:?}", entry.row_index, entry.filtration_value);
        // }

        // println!("Cycle end");
        pairings[dimension].push((birth_f.into_inner(), death_f.into_inner()));
    }

    //involution_basis.sort_unstable();
    //println!("Involution basis");
    //for cell in involution_basis.iter() {
    //    println!("{:?}", cell);
    //}
    //println!("{}", involution_basis.len());
    (essential, pairings)
}

/// A Python module implemented in Rust.
#[pymodule]
fn phlite_grpph(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(grpph, m)?)?;
    Ok(())
}
