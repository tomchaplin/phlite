use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;

use crate::matrices::combinators::product;
use crate::matrices::{MatrixRef, SplitByDimension, SquareMatrix};
use crate::{
    columns::ColumnEntry,
    fields::{Invertible, NonZeroCoefficient},
    matrices::{implementors::VecVecMatrix, ColBasis, HasColBasis, HasRowFiltration, MatrixOracle},
};
use crate::{matrix_col_product, PhliteError};

pub enum ReductionColumn<CF, ColT> {
    Cleared(ColT), // This gets set when (i, j) is found as a pair in which case column i can be reduced by R_j, we store j here
    Reduced(Vec<(CF, ColT)>), // The sum of columns required to reduce (minus the +1 with self index)
}

pub struct ClearedReductionMatrix<M>
where
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    boundary: M,
    reduction_columns: HashMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>>,
}

impl<M> MatrixOracle for ClearedReductionMatrix<M>
where
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash + Debug,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::ColT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, crate::PhliteError>
    {
        let reduction_col = self
            .reduction_columns
            .get(&col)
            .ok_or(PhliteError::NotInDomain)?;

        // TODO: Is there a way to do this without Box?

        let output_iter: Box<dyn Iterator<Item = (Self::CoefficientField, Self::RowT)>> =
            match reduction_col {
                ReductionColumn::Cleared(death_idx) => {
                    // This returns the death_idx column of R = D V
                    let v_j = self.column(*death_idx)?;
                    Box::new(matrix_col_product!(self.boundary, v_j))
                }
                ReductionColumn::Reduced(vec) => Box::new(
                    // We don't store the diagonal so we have to chain +1 on the diagonal to the output
                    vec.iter()
                        .copied()
                        .chain(iter::once((M::CoefficientField::one(), col))),
                ),
            };

        Ok(output_iter)
    }
}

impl<M> HasColBasis for ClearedReductionMatrix<M>
where
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    M::CoefficientField: Invertible,
    M::ColT: Hash + Debug,
{
    type BasisT = M::BasisT;

    fn basis(&self) -> &Self::BasisT {
        self.boundary.basis()
    }
}

impl<M> ClearedReductionMatrix<M>
where
    M: MatrixRef + SquareMatrix + HasRowFiltration + HasColBasis,
    <M as HasColBasis>::BasisT: SplitByDimension,
    M::CoefficientField: Invertible,
    M::ColT: Hash,
{
    pub fn build(boundary: M, dimension_order: impl Iterator<Item = usize>) -> Self {
        let mut reduction_columns: HashMap<M::ColT, ReductionColumn<M::CoefficientField, M::ColT>> =
            HashMap::new();

        for dim in dimension_order {
            let sub_matrix = boundary.sub_matrix_in_dimension(dim);
            // Reduce submatrix

            // low_inverse[i]=(j, lambda) means R[j] has lowest non-zero in row i with coefficient lambda
            let mut low_inverse: HashMap<M::RowT, (M::ColT, M::CoefficientField)> = HashMap::new();

            'column_loop: for i in 0..sub_matrix.basis().size() {
                // Reduce column i
                let basis_element = sub_matrix.basis().element(i);

                // First check whether already cleared
                if reduction_columns.contains_key(&basis_element) {
                    continue 'column_loop;
                }

                let mut v_i = vec![];
                let mut r_col = sub_matrix.build_bhcol(basis_element).unwrap();

                'reduction: loop {
                    let Some(pivot_entry) = r_col.pop_pivot() else {
                        // Column reduced to 0 -> found cycle -> move onto next column
                        break 'reduction;
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
                    v_i.push((col_multiple, *other_col_basis_element))
                }

                // If we have a pivot
                if let Some(pivot_entry) = r_col.pop_pivot() {
                    // Save it to low inverse
                    low_inverse.insert(pivot_entry.row_index, (basis_element, pivot_entry.coeff));
                    // and clear out the birth column
                    reduction_columns.insert(
                        pivot_entry.row_index,
                        ReductionColumn::Cleared(basis_element),
                    );
                };

                // Then save v_i to reduction matrix
                reduction_columns.insert(basis_element, ReductionColumn::Reduced(v_i));
            }
        }

        Self {
            reduction_columns,
            boundary,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Diagram<T> {
    pub essential: HashSet<T>,
    pub pairings: HashSet<(T, T)>,
}

// TODO: Convert to a MapVecMatrix so that we can use a common diagram read off

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

    use std::collections::HashSet;

    use ordered_float::NotNan;

    use crate::{
        fields::Z2,
        filtrations::rips::cohomology::RipsCoboundaryAllDims,
        matrices::{
            combinators::product, implementors::simple_Z2_matrix, ColBasis, HasColBasis,
            HasRowFiltration, MatrixOracle, MatrixRef,
        },
        reduction::ClearedReductionMatrix,
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
        let matrix_v = standard_algo(matrix_d.with_trivial_filtration());
        let matrix_r = product(&matrix_d, &matrix_v);
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

    fn distance_matrix() -> Vec<Vec<NotNan<f64>>> {
        let zero = NotNan::new(0.0).unwrap();
        let one = NotNan::new(1.0).unwrap();
        let sqrt2 = NotNan::new(2.0_f64.sqrt()).unwrap();
        let distance_matrix: Vec<Vec<NotNan<f64>>> = vec![
            vec![zero, one, sqrt2, one],
            vec![one, zero, one, sqrt2],
            vec![sqrt2, one, zero, one],
            vec![one, sqrt2, one, zero],
        ];
        distance_matrix
    }

    #[test]
    fn test_clearing() {
        let distance_matrix = distance_matrix();
        let n_points = distance_matrix.len();
        let max_dim = 1;

        // Compute column basis
        let coboundary = RipsCoboundaryAllDims::<Z2>::build(distance_matrix, max_dim);
        // Compute reduction matrix, in increasing dimension
        let v = ClearedReductionMatrix::build(&coboundary, 0..=max_dim);
        let r = product(&coboundary, &v);

        let mut essential = HashSet::new();
        let mut pairings = HashSet::new();

        for i in (0..r.basis().size()).rev() {
            let mut r_i = r.build_bhcol(r.basis().element(i)).unwrap();
            match r_i.pop_pivot() {
                None => {
                    essential.insert(r.basis().element(i));
                }
                Some(piv) => {
                    let death_idx = r.basis().element(i);
                    pairings.insert((piv.row_index, death_idx));
                    essential.remove(&piv.row_index);
                }
            }
            if r_i.pop_pivot().is_none() {}
        }

        // Report
        println!("Essential:");
        for idx in essential.iter() {
            let f_val = coboundary.filtration_value(*idx).unwrap().0;
            let dim = idx.dimension(n_points);
            println!(" dim={dim}, birth={idx:?}, f=({f_val}, ∞)");
        }
        println!("\nPairings:");
        for tup in pairings.iter() {
            let dim = tup.1.dimension(n_points);
            let idx_tup = (tup.1, tup.0);
            let birth_f = coboundary.filtration_value(tup.1).unwrap().0;
            let death_f = coboundary.filtration_value(tup.0).unwrap().0;
            println!(" dim={dim}, pair={idx_tup:?}, f=({birth_f}, {death_f})");
        }

        // Ignored 2-dimensional void
        assert_eq!(pairings.len(), 6);
        assert_eq!(essential.len(), 1);
    }
}
