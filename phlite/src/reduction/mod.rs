//! R=DV reduction algorithms for `phlite` matrices.
//! Includes the standard algorithm as well as the clearing algorithm.
//!
//! Algorithms in this module should return an oracle to the V matrix (known as the reduction matrix).
//! If you held onto a reference to D then you can construct R (known as the reduced matrix) via [`product`](crate::matrices::combinators::product).

pub mod clearing;
pub mod standard;

pub use clearing::*;
pub use standard::*;

use rustc_hash::FxHashSet;
// TODO: Should this be just a vec to make construction quicker?
/// A persistence diagram, as obtained by one of the reduction algorithms.
#[derive(Debug, Clone)]
pub struct Diagram<T> {
    /// The unpaired or essential columns (infinite bars).
    pub essential: FxHashSet<T>,
    /// The paired columns (finite bars), stored as (birth, death) pairs.
    pub pairings: FxHashSet<(T, T)>,
}

#[cfg(test)]
mod tests {

    use crate::matrices::{combinators::product, implementors::SimpleZ2Matrix, MatrixOracle};

    use super::standard_algo;

    #[test]
    fn test_inefficient_reduction() {
        let matrix_d = SimpleZ2Matrix::new_all_one(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![3, 4, 5],
            vec![3, 4, 5],
        ]);
        let matrix_v = standard_algo((&matrix_d).with_trivial_filtration());
        let matrix_r = product(&matrix_d, &matrix_v);
        let true_matrix_r = SimpleZ2Matrix::new_all_one(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![],
            vec![3, 4, 5],
            vec![],
        ]);

        assert!((0..=7).all(|idx| matrix_r.eq_on_col(&true_matrix_r, idx)))
    }
}
