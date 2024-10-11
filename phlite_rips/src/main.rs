use std::io;

use ordered_float::NotNan;
use phlite::{
    fields::Z2,
    matrices::{
        combinators::product, ColBasis, HasColBasis, HasRowFiltration, MatrixOracle,
        SplitByDimension,
    },
    reduction::ClearedReductionMatrix,
};
use phlite_rips::cohomology::build_rips_coboundary_matrix;

// TODO: Make a nice CLI using clap that accepts standard formats (akin to Ripser) and outputs diagram, optional plot?

pub fn main() {
    // TUTORIAL:
    // Welcome! First off we just do some IO to take in the distance matrix.

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

    // TUTORIAL:
    // Now we build the coboundary matrix (and a basis for the column up to max_dim)
    // Note we call `.reverse` to reverse the filtration order on the rows as well as reverse the column basis.
    // This yields the anti-transpose of the matrix which is what we will reduce.

    // Compute column basis
    let coboundary = build_rips_coboundary_matrix::<Z2>(distance_matrix, max_dim);
    let coboundary = coboundary.reverse();
    println!("Basis sizes:");
    for dim in 0..=max_dim {
        let basis_size = coboundary.basis().in_dimension(dim).size();
        println!("Dimension {dim}: {basis_size}");
    }

    // TUTORIAL:
    // Here's where we actually do the R=DV decomposition, with clearing.
    // We have to feed in the order in which to do the reduction.
    // Since the coboundary increases dimension we feed in `0..=max_dim`.
    //
    // We get back the reduction matrix `v` and the persistence diagram `diagram`.
    // If you want access to R, we can just take the product of the coboundary matrix and V.
    // Note all the indexing/filtration types are reversed.
    // Calling `phlite::matrices::MatrixOracle::unreverse` can resolve this.

    // Compute reduction matrix, in increasing dimension
    let (v, diagram) = ClearedReductionMatrix::build_with_diagram(&coboundary, 0..=max_dim);
    let _r = product(&coboundary, &v);

    // Report
    println!("\nEssential:");
    for idx in diagram.essential.iter() {
        let f_val = coboundary.filtration_value(*idx).unwrap().0;
        let dim = idx.0.dimension(n_points);
        println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
    }
    println!("\nPairings:");
    for tup in diagram.pairings.iter() {
        let dim = tup.1 .0.dimension(n_points);
        let idx_tup = (tup.1, tup.0);
        let birth_f = coboundary.filtration_value(tup.1).unwrap().0.into_inner();
        let death_f = coboundary.filtration_value(tup.0).unwrap().0.into_inner();
        let difference = death_f - birth_f;
        if difference > 0.01 {
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }
    }
}
