//! An implementation of the standard algorithm with no optimisations.
//!
//! To run this algorithm you should attempt to construct the V matrix by calling [`standard_algo`] or [`standard_algo_with_diagram`].
//!
//! Some of the functions in this module require you to specify a boolean parameter `reverse_order`.
//! If your chain complex increases the dimension (e.g. a coboundary matrix) then set `reverse_order=true`, otherwise set to `false`.

use std::hash::Hash;

use crate::matrices::combinators::product;
use crate::matrices::implementors::MapVecMatrix;
use crate::matrices::SquareMatrix;
use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matrices::{ColBasis, HasColBasis, HasRowFiltration, MatrixOracle},
};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::reduction::Diagram;

// TODO: Convert to a MapVecMatrix so that we can use a common diagram read off

/// The return type of [`standard_algo`] - reduction columns are stored behind a hash map.
pub type StandardReductionMatrix<CF, ColT> = MapVecMatrix<'static, CF, ColT, ColT>;

/// Run the standard algorithm to reduce the provided boundary matrix.
/// The reduction matrix V is returned as a [`StandardReductionMatrix`].
///
/// Columns will be reduced according to their order in the attached [`ColBasis`].
pub fn standard_algo<M>(boundary: M) -> StandardReductionMatrix<M::CoefficientField, M::ColT>
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
pub fn read_off_diagram<M>(
    boundary: M,
    reduction_matrix: &StandardReductionMatrix<M::CoefficientField, M::ColT>,
    reverse_order: bool,
) -> Diagram<M::ColT>
where
    M: SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let r = product(&boundary, &reduction_matrix);

    let mut essential = FxHashSet::default();
    let mut pairings = FxHashSet::default();

    let col_iter: Box<dyn Iterator<Item = usize>> = if reverse_order {
        Box::new((0..boundary.basis().size()).rev())
    } else {
        Box::new(0..boundary.basis().size())
    };

    for i in col_iter {
        let basis_element = boundary.basis().element(i);
        let mut r_i = r.build_bhcol(basis_element.clone());
        match r_i.pop_pivot() {
            None => {
                essential.insert(basis_element);
            }
            Some(piv) => {
                pairings.insert((piv.row_index.clone(), basis_element));
                essential.remove(&piv.row_index);
            }
        }
        if r_i.pop_pivot().is_none() {}
    }

    Diagram {
        essential,
        pairings,
    }
}

#[allow(clippy::type_complexity)]
/// Calls [`standard_algo`] and then [`read_off_diagram`].
pub fn standard_algo_with_diagram<M>(
    boundary: M,
    reverse_order: bool,
) -> (
    StandardReductionMatrix<M::CoefficientField, M::ColT>,
    Diagram<M::ColT>,
)
where
    M: SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let v = standard_algo(&boundary);
    let diagram = read_off_diagram(&boundary, &v, reverse_order);
    (v, diagram)
}
