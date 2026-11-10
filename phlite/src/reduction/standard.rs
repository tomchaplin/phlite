//! An implementation of the standard algorithm with no optimisations.
//!
//! Run it with `Standard::new().reduce(matrix)`, see [`ReductionAlgorithm`].

use std::hash::Hash;

use crate::matrices::combinators::product;
use crate::matrices::implementors::MapVecMatrix;
use crate::matrices::SquareMatrix;
use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{ColBasis, HasColBasis, HasRowFiltration, MatrixOracle},
};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::reduction::{Diagram, Reduction, ReductionAlgorithm};

// TODO: Convert to a MapVecMatrix so that we can use a common diagram read off

/// The type of `V` returned by [`Standard`] - reduction columns are stored behind a hash map.
pub type StandardReductionMatrix<CF, ColT> = MapVecMatrix<'static, CF, ColT, ColT>;

/// The standard reduction algorithm, with no optimisations.
///
/// Columns are reduced according to their order in the attached [`ColBasis`].
/// The reduction matrix V is returned as a [`StandardReductionMatrix`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Standard;

impl Standard {
    /// The standard algorithm. There are no options.
    pub fn new() -> Self {
        Self
    }
}

impl<M> ReductionAlgorithm<M> for Standard
where
    M: MatrixOracle + HasRowFiltration + HasColBasis + SquareMatrix,
    M::CoefficientField: Invertible + 'static,
    M::RowT: Hash + 'static,
{
    type V = StandardReductionMatrix<M::CoefficientField, M::ColT>;

    fn reduce(self, boundary: M) -> Reduction<Self::V, M::ColT> {
        let v = standard_algo(&boundary);
        let diagram = read_off_diagram(&boundary, &v);
        Reduction { v, diagram }
    }
}

fn standard_algo<M>(boundary: M) -> StandardReductionMatrix<M::CoefficientField, M::ColT>
where
    M: MatrixOracle + HasRowFiltration + HasColBasis + SquareMatrix,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let mut v = FxHashMap::default();

    // low_inverse[i]=(j, lambda) means R[j] has lowest non-zero in row i with coefficient lambda
    let mut low_inverse: FxHashMap<M::RowT, (M::ColT, M::CoefficientField)> = FxHashMap::default();

    for i in 0..boundary.basis().size() {
        // Reduce column i

        let basis_element = boundary.basis().element(i);

        let mut v_i = (&boundary).with_trivial_filtration().empty_bhcol();
        v_i.add_tuple((M::CoefficientField::one(), basis_element.clone(), ()));
        let mut r_i = boundary.build_bhcol(basis_element.clone());

        'reduction: loop {
            let Some(pivot_entry) = r_i.pop_pivot() else {
                // Column reduced to 0 -> found cycle -> move onto next column
                break 'reduction;
            };

            let pivot_j = pivot_entry.row_index.clone();
            let pivot_coeff = pivot_entry.coeff;

            // Push the pivot back in to keep r_col coorect
            r_i.add_entry(pivot_entry);

            // Check if there is a column with the same pivot
            let Some((j_basis_element, j_coeff)) = low_inverse.get(&pivot_j) else {
                // Cannot reduce further -> found boundary -> break and save pivot
                break 'reduction;
            };

            // If so then we add a multiple of that column to cancel out the pivot in r_col
            let col_multiple = pivot_coeff.additive_inverse() * (j_coeff.mult_inverse());

            let v_matrix = MapVecMatrix::from(&v);
            let v_matrix = v_matrix.with_trivial_filtration();
            let r_matrix = product(&boundary, &v_matrix);

            // Add the multiple of that column
            r_i.add_entries(
                r_matrix
                    .column_with_filtration(j_basis_element.clone())
                    .map(|entry| entry * col_multiple),
            );

            // Update V
            v_i.add_entries(
                v_matrix
                    .column_with_filtration(j_basis_element.clone())
                    .map(|entry| entry * col_multiple),
            );
        }

        // Save pivot if we have one
        if let Some(pivot_entry) = r_i.pop_pivot() {
            low_inverse.insert(
                pivot_entry.row_index,
                (basis_element.clone(), pivot_entry.coeff),
            );
        };

        // Save V
        v.insert(
            basis_element,
            v_i.drain_sorted()
                .map(|entry| (entry.coeff, entry.row_index))
                .collect(),
        );
    }

    MapVecMatrix::from(v)
}

/// Reads off the pairings from the reduced matrix to construct the persistence diagram.
///
/// The result doesn't depend on the order in which columns are visited:
/// the essential classes are the columns that reduce to zero and are not the pivot of any other column.
fn read_off_diagram<M>(
    boundary: M,
    reduction_matrix: &StandardReductionMatrix<M::CoefficientField, M::ColT>,
) -> Diagram<M::ColT>
where
    M: SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let r = product(&boundary, &reduction_matrix);

    let mut essential = FxHashSet::default();
    let mut pairings = FxHashSet::default();
    let mut pivot_rows = FxHashSet::default();

    let basis = boundary.basis();
    for i in 0..basis.size() {
        let basis_element = basis.element(i);
        match r.build_bhcol(basis_element.clone()).pop_pivot() {
            None => {
                essential.insert(basis_element);
            }
            Some(piv) => {
                pivot_rows.insert(piv.row_index.clone());
                pairings.insert((piv.row_index, basis_element));
            }
        }
    }
    // A column that is the pivot of another is the birth of a finite bar
    essential.retain(|elem| !pivot_rows.contains(elem));

    Diagram {
        essential,
        pairings,
    }
}
