use std::io;

use ordered_float::NotNan;
use phlite::{
    fields::Z2,
    filtrations::rips::cohomology::RipsCoboundaryAllDims,
    matrices::{combinators::product, ColBasis, HasColBasis, HasRowFiltration, SplitByDimension},
    reduction::ClearedReductionMatrix,
};

// TODO: Make a nice CLI using clap that accepts standard formats (akin to Ripser) and outputs diagram, optional plot?

pub fn main() {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(io::stdin());
    let distance_matrix: Vec<Vec<NotNan<f64>>> = rdr
        .records()
        .map(|row| {
            let row = row.unwrap();
            row.into_iter()
                .map(|entry| NotNan::new(entry.parse::<f64>().unwrap()).unwrap())
                .collect()
        })
        .collect();

    let n_points = distance_matrix.len();
    let max_dim = std::env::args()
        .nth(1)
        .map(|arg| {
            arg.parse()
                .expect("First argument should be the maximum homology dimension")
        })
        .unwrap_or(1);

    // Compute column basis
    let coboundary = RipsCoboundaryAllDims::<Z2>::build(distance_matrix, max_dim);
    println!("Built basis");
    println!("Basis sizes:");
    for dim in 0..=max_dim {
        let basis_size = coboundary.basis().in_dimension(dim).size();
        println!("Dimension {dim}: {basis_size}");
    }

    // Compute reduction matrix, in increasing dimension
    let (v, diagram) = ClearedReductionMatrix::build_with_diagram(&coboundary, 0..=max_dim);
    //let v = standard_algo(&coboundary);
    let _r = product(&coboundary, &v);

    // Report
    println!("Essential:");
    for idx in diagram.essential.iter() {
        let f_val = coboundary.filtration_value(*idx).unwrap().0;
        let dim = idx.dimension(n_points);
        println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
    }
    println!("\nPairings:");
    for tup in diagram.pairings.iter() {
        let dim = tup.1.dimension(n_points);
        let idx_tup = (tup.1, tup.0);
        let birth_f = coboundary.filtration_value(tup.1).unwrap().0.into_inner();
        let death_f = coboundary.filtration_value(tup.0).unwrap().0.into_inner();
        let difference = death_f - birth_f;
        if difference > 0.01 {
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }
    }
}
