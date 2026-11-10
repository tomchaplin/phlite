//! R=DV reduction algorithms for `phlite` matrices.
//!
//! Every algorithm is a value implementing [`ReductionAlgorithm`], which you configure and then run on a matrix with
//! [`reduce`](ReductionAlgorithm::reduce). The result is always a [`Reduction`]:
//! an oracle for the matrix `V` (known as the reduction matrix) together with the persistence [`Diagram`].
//! If you held onto a reference to D then you can construct R (known as the reduced matrix) via [`product`](crate::matrices::combinators::product).
//!
//! | Algorithm | Notes |
//! |---|---|
//! | [`Standard`] | The standard algorithm, reducing columns in basis order. |
//! | [`Clearing`] | The standard algorithm with the clearing optimisation. Needs a [`HasSplitColBasis`](crate::matrices::HasSplitColBasis) matrix. |
//! | `ParallelClearing` | Clearing, with the columns of each dimension reduced in parallel (feature `parallel`). Obtained from `Clearing::parallel`. |
//!
//! # Doing less work
//!
//! In practice most columns need no reduction at all, and the clearing algorithms avoid as much work as they can on them:
//! * A column whose pivot no other column has claimed is recorded straight away, without building the column in full,
//!   using [`column_pivot`](crate::matrices::HasRowFiltration::column_pivot)
//!   (override it, or [`column_min`](crate::matrices::HasRowFiltration::column_min) for a [`reverse`](crate::matrices::MatrixOracle::reverse)d matrix,
//!   if you can find the pivot without visiting every entry). Its reduction column `V = e` is not stored.
//! * If the matrix knows the first column in each row
//!   ([`first_column_in_row`](crate::matrices::HasRowFiltration::first_column_in_row), e.g. by attaching its transpose with
//!   [`with_transpose`](crate::matrices::MatrixOracle::with_transpose)), then *apparent pairs* are recognised:
//!   a column `c` whose pivot `r` has `c` as the first column in its row is a persistence pair that needs no reduction.
//!   Apparent pairs are never stored in pivot tables, since whenever another column hits pivot `r` the owner `c` can be recomputed.
//!   So the column basis needn't even contain them; with
//!   [`skip_zero_apparent_pairs`](Clearing::skip_zero_apparent_pairs) those of zero persistence are also left out of the diagram.
//!
//! [`Standard`] makes none of these optimisations: it builds every column and needs the full column basis.
//!
//! ```
//! # use phlite::{fields::Z2, matrices::implementors::SimpleZ2Matrix, reduction::{Standard, ReductionAlgorithm}};
//! # use phlite::matrices::MatrixOracle;
//! # let boundary = SimpleZ2Matrix::new_all_one(vec![vec![], vec![], vec![0, 1]]).with_trivial_filtration();
//! let reduction = Standard::new().reduce(&boundary);
//! assert_eq!(reduction.diagram.pairings.len(), 1);
//! ```

pub mod clearing;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod standard;

pub use clearing::*;
#[cfg(feature = "parallel")]
pub use parallel::{ParallelClearing, Storage};
pub use standard::*;

use crate::matrices::MatrixOracle;
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

impl<T: std::hash::Hash + Eq + Clone> Diagram<T> {
    /// A column that is the pivot of another column is the birth of a finite bar, so is not essential.
    /// Algorithms that visit columns out of order call this once at the end.
    pub(crate) fn remove_births_from_essential(&mut self) {
        let births: FxHashSet<T> = self.pairings.iter().map(|(birth, _death)| birth.clone()).collect();
        self.essential.retain(|elem| !births.contains(elem));
    }
}

/// The output of a [`ReductionAlgorithm`]: the reduction matrix `V` and the persistence diagram.
pub struct Reduction<V, T> {
    /// An oracle for the reduction matrix `V`, such that `R = D V` is reduced.
    pub v: V,
    /// The persistence diagram, read off from the pivots of `R`.
    pub diagram: Diagram<T>,
}

/// A configured algorithm for reducing the matrix `M`, see the [module docs](self).
pub trait ReductionAlgorithm<M: MatrixOracle> {
    /// The oracle type used to represent `V`.
    type V;

    /// Reduce `boundary`, returning `V` and the persistence diagram.
    ///
    /// The matrix is consumed so you probably want to pass a reference to your matrix.
    fn reduce(self, boundary: M) -> Reduction<Self::V, M::ColT>;
}

#[cfg(test)]
mod tests {

    use crate::matrices::{combinators::product, implementors::SimpleZ2Matrix, MatrixOracle};

    use super::{ReductionAlgorithm, Standard};

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
        let matrix_v = Standard::new()
            .reduce((&matrix_d).with_trivial_filtration())
            .v;
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
