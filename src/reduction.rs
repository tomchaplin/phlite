use std::collections::HashMap;
use std::hash::Hash;

use crate::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    matrices::{implementors::VecVecMatrix, ColBasis, HasColBasis, HasRowFiltration, MatrixRef},
};

// TODO:
// 1. Convert to oracle
// 2. Implement clearing

pub fn standard_algo<M>(boundary: M) -> VecVecMatrix<'static, M::CoefficientField, M::ColT>
where
    M: MatrixRef + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let mut v = vec![];
    for i in 0..boundary.basis().size() {
        v.push(vec![(
            M::CoefficientField::one(),
            boundary.basis().element(i),
        )]);
    }

    // low_inverse[i]=(j, lambda) means R[j] has lowest non-zero in row i with coefficient lambda
    let mut low_inverse: HashMap<M::RowT, (M::ColT, M::CoefficientField)> = HashMap::new();

    'column_loop: for i in 0..boundary.basis().size() {
        // Reduce column i

        let basis_element = boundary.basis().element(i);
        let mut r_col = boundary.build_bhcol(basis_element).unwrap();

        'reduction: loop {
            let Some(pivot_entry) = r_col.pop_pivot() else {
                // Column reduced to 0 -> found cycle -> move onto next column
                continue 'column_loop;
            };

            let pivot_row_index = pivot_entry.row_index;
            let pivot_coeff = pivot_entry.coeff;

            // Push the pivot back in to keep r_col coorect
            r_col.push(pivot_entry);

            // Check if there is a column with the same pivot
            let Some((other_col_basis_element, other_col_coeff)) =
                low_inverse.get(&pivot_row_index)
            else {
                // Cannot reduce further -> found boundary -> break and save pivot
                break 'reduction;
            };

            // If so then we add a multiple of that column to cancel out the pivot in r_col
            let col_multiple = pivot_coeff.additive_inverse() * (other_col_coeff.inverse());

            //let other_col_basis_element = boundary.basis().element(*other_col_idx);

            // Add the multiple of that column
            r_col.add_entries(
                boundary
                    .column_with_filtration(*other_col_basis_element)
                    .unwrap()
                    .map(|e| e.unwrap())
                    .map(|entry| {
                        ColumnEntry::from((
                            entry.coeff * col_multiple,
                            entry.row_index,
                            entry.filtration_value,
                        ))
                    }),
            );

            // Update V
            v[i].push((col_multiple, *other_col_basis_element))
        }

        // Save pivot if we have one
        if let Some(pivot_entry) = r_col.pop_pivot() {
            low_inverse.insert(pivot_entry.row_index, (basis_element, pivot_entry.coeff));
        };
    }

    VecVecMatrix::from(v)
}

#[cfg(test)]
mod tests {

    use crate::matrices::{
        combinators::product, implementors::simple_Z2_matrix, MatrixOracle, MatrixRef,
    };

    use super::standard_algo;

    #[test]
    fn test_inefficient_reduction() {
        let matrix_d = simple_Z2_matrix(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![3, 4, 5],
            vec![3, 4, 5],
        ]);
        let matrix_v = standard_algo(&matrix_d.with_trivial_filtration());
        let matrix_r = product(matrix_d.using_col_basis_index(), &matrix_v);
        let true_matrix_r = simple_Z2_matrix(vec![
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
