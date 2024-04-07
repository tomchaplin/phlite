use std::collections::HashMap;
use std::hash::Hash;

use crate::{
    fields::{Invertible, NonZeroCoefficient},
    matricies::{product, FiniteOrderedColBasis, HasRowFiltration, MatrixRef, VecVecMatrix},
};

// TODO:
// 1. Convert to oracle
// 2. Implement clearing
// 3. Don't keep dropping and rebuilding r_col

pub fn inefficient_reduction<M: MatrixRef + FiniteOrderedColBasis>(
    boundary: M,
) -> VecVecMatrix<'static, M::CoefficientField, usize>
where
    M: MatrixRef + HasRowFiltration,
    M::CoefficientField: Invertible,
    M::RowT: Hash,
{
    let mut v = vec![];
    for i in 0..boundary.n_cols() {
        v.push(vec![(M::CoefficientField::one(), i)]);
    }

    let mut low_inverse: HashMap<M::RowT, usize> = HashMap::new();

    for i in 0..boundary.n_cols() {
        // Reduce column i
        loop {
            // Build up temporary oracle for product
            let v_oracle = VecVecMatrix::from(&v);
            // Build up r_col from the product
            let mut r_col = product(boundary, &v_oracle).build_bhcol(i).unwrap();

            // Check if we have a pivot to reduce
            let Some(pivot_entry) = r_col.pop_pivot() else {
                break;
            };

            // Check if there is a column with the same pivot
            let Some(col_idx_to_add) = low_inverse.get(&pivot_entry.row_index) else {
                break;
            };

            // If so then we add -1/coeff with the corresponding index to V
            let col_multiple = pivot_entry.coeff.inverse().additive_inverse();

            v[i].push((col_multiple, *col_idx_to_add))
        }

        // Keep low_inverse up to date

        // Build up temporary oracle for product
        let v_oracle = VecVecMatrix::from(&v);
        // Build up r_col from the product
        let mut r_col = product(boundary, &v_oracle).build_bhcol(i).unwrap();

        // Check if we have a pivot to reduce
        if let Some(pivot_entry) = r_col.pop_pivot() {
            low_inverse.insert(pivot_entry.row_index, i);
        };
    }

    VecVecMatrix::from(v)
}

#[cfg(test)]
mod tests {
    use crate::matricies::{product, simple_Z2_matrix, MatrixOracle, MatrixRef};

    use super::inefficient_reduction;

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
        let matrix_d_with_basis = matrix_d.with_trivial_filtration();

        let matrix_v = inefficient_reduction(&matrix_d_with_basis);

        let matrix_r = product(&matrix_d_with_basis, &matrix_v);

        assert!((0..=6).all(|idx| matrix_r.eq_on_col(&true_matrix_r, idx)))
    }
}
