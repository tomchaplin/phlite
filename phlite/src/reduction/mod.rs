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
//!   So the column basis needn't even contain them; those of zero persistence are also left out of the diagram, unless
//!   you turn off [`skip_zero_apparent_pairs`](Clearing::skip_zero_apparent_pairs).
//!
//! [`Standard`] makes none of these optimisations: it builds every column and needs the full column basis.
//!
//! # Column representations
//!
//! Every algorithm is generic over how it represents the column `R_i = D V_i` that it is reducing, see
//! [`ColumnRepresentation`](crate::columns::ColumnRepresentation). The default,
//! [`MergedColumns`](crate::columns::MergedColumns), sorts each column of `D` once (when it is first needed in a dimension) and
//! adds it to `R_i` as a sorted run, merging the runs lazily, so that only one heap slot is needed per column added and terms
//! below the eventual pivot are never touched. [`HeapColumns`](crate::columns::HeapColumns) instead pushes every term into a heap,
//! which needs no extra memory but is much slower when reduced columns are long. Choose one with `with_columns`:
//!
//! ```
//! # use phlite::{fields::Z2, matrices::implementors::SimpleZ2Matrix, reduction::{Standard, ReductionAlgorithm}};
//! # use phlite::{columns::HeapColumns, matrices::MatrixOracle};
//! # let boundary = SimpleZ2Matrix::new_all_one(vec![vec![], vec![], vec![0, 1]]).with_trivial_filtration();
//! let reduction = Standard::new().with_columns(HeapColumns).reduce(&boundary);
//! assert_eq!(reduction.diagram.pairings.len(), 1);
//! ```
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

use crate::bookkeeping::ElementSet;
use crate::matrices::{BasisElement, MatrixOracle};

/// A persistence diagram, as obtained by one of the reduction algorithms.
///
/// Both lists are in no particular order, which may differ between algorithms; [`sorted`](Self::sorted) puts them in
/// order, for example to compare diagrams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagram<T> {
    /// The unpaired or essential columns (infinite bars).
    pub essential: Vec<T>,
    /// The paired columns (finite bars), stored as (birth, death) pairs.
    pub pairings: Vec<(T, T)>,
}

impl<T: Ord> Diagram<T> {
    /// The same diagram with both lists sorted, so that diagrams from different algorithms can be compared with `==`.
    #[must_use]
    pub fn sorted(mut self) -> Self {
        self.essential.sort_unstable();
        self.pairings.sort_unstable();
        self
    }
}

/// A [`Diagram`] being built by a reduction algorithm.
pub(crate) struct DiagramBuilder<T: BasisElement + std::hash::Hash> {
    /// The columns found to be zero so far. Some may turn out to be births, see [`finish`](Self::finish).
    pub(crate) essential: ElementSet<T>,
    pub(crate) pairings: Vec<(T, T)>,
}

impl<T: BasisElement + std::hash::Hash> Default for DiagramBuilder<T> {
    fn default() -> Self {
        Self {
            essential: ElementSet::default(),
            pairings: Vec::new(),
        }
    }
}

impl<T: BasisElement + std::hash::Hash> DiagramBuilder<T> {
    /// The diagram, leaving out of the essential columns those that are births of finite bars (algorithms that visit
    /// columns out of order may have marked them essential first), and those in `exclude`.
    pub(crate) fn finish(self, exclude: Option<&ElementSet<T>>) -> Diagram<T> {
        let mut births = ElementSet::default();
        for (birth, _death) in &self.pairings {
            births.insert(birth.clone());
        }
        let essential = self
            .essential
            .into_elements()
            .filter(|elem| !births.contains(elem) && !exclude.is_some_and(|set| set.contains(elem)))
            .collect();
        Diagram {
            essential,
            pairings: self.pairings,
        }
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
