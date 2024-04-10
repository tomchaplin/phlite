use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::matrices::combinators::product;
use crate::matrices::SquareMatrix;
use crate::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    matrices::{implementors::VecVecMatrix, ColBasis, HasColBasis, HasRowFiltration, MatrixOracle},
};

enum ReductionColumn<CF, ColT> {
    Cleared(ColT),
    Reduced(Vec<(CF, ColT)>),
}

// TODO:
// 1. Convert to oracle
// 2. Implement clearing

#[derive(Debug, Clone)]
pub struct Diagram<T> {
    pub essential: HashSet<T>,
    pub pairings: HashSet<(T, T)>,
}

pub type StandardReductionMatrix<CF, ColT> = VecVecMatrix<'static, CF, ColT>;

/// If your operator goes up in column order (e.g. coboundary) then you will need to set `reverse_order=True`.
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

/// If your operator goes up in column order (e.g. coboundary) then you will need to set `reverse_order=True`.
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

    let mut essential = HashSet::new();
    let mut pairings = HashSet::new();

    let col_iter: Box<dyn Iterator<Item = usize>> = if reverse_order {
        Box::new((0..r.basis().size()).rev())
    } else {
        Box::new(0..r.basis().size())
    };

    for i in col_iter {
        let mut r_i = r.build_bhcol(i).unwrap();
        match r_i.pop_pivot() {
            None => {
                essential.insert(boundary.basis().element(i));
            }
            Some(piv) => {
                let death_idx = boundary.basis().element(i);
                pairings.insert((piv.row_index, death_idx));
                essential.remove(&piv.row_index);
            }
        }
        if r_i.pop_pivot().is_none() {}
    }

    let diagram = Diagram {
        essential,
        pairings,
    };

    diagram
}

pub fn standard_algo<M>(boundary: M) -> StandardReductionMatrix<M::CoefficientField, M::ColT>
where
    M: MatrixOracle + HasRowFiltration + HasColBasis,
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
