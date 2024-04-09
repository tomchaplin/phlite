// ======== Combinators ========================================

// ====== Product ==============================

use crate::PhliteError;

use super::{FiniteOrderedColBasis, HasRowFiltration, MatrixOracle, MatrixRef};

pub fn product<M1: MatrixRef, M2: MatrixRef>(left: M1, right: M2) -> Product<M1, M2>
where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT>,
{
    Product { left, right }
}

#[derive(Clone, Copy)]
pub struct Product<M1: MatrixRef, M2: MatrixRef> {
    left: M1,
    right: M2,
}

impl<M1: MatrixRef, M2: MatrixRef> MatrixOracle for Product<M1, M2>
where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT>,
{
    type CoefficientField = M1::CoefficientField;

    type ColT = M2::ColT;

    type RowT = M1::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        // Pull out right col
        let right_col_entries = self.right.column(col)?;
        // This tells us what linear combination of columns in the left matrix
        // should be formed to yield the product column
        Ok(
            right_col_entries.flat_map(|(right_coeff, right_row_index)| {
                let left_col = self.left.column(right_row_index).unwrap();
                left_col.map(move |(left_coeff, left_row_index)| {
                    (left_coeff * right_coeff, left_row_index)
                })
            }),
        )
    }
}

impl<M1: MatrixRef, M2: MatrixRef> MatrixRef for Product<M1, M2> where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT>
{
}

// In product there is an obvious row filtration if the LHS has a row filtration
impl<M1: MatrixRef, M2: MatrixRef> HasRowFiltration for Product<M1, M2>
where
    M1: HasRowFiltration,
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT>,
{
    type FiltrationT = M1::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.left.filtration_value(row)
    }
}

impl<M1: MatrixRef, M2: MatrixRef> FiniteOrderedColBasis for Product<M1, M2>
where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT>
        + FiniteOrderedColBasis,
{
    fn n_cols(&self) -> usize {
        self.right.n_cols()
    }
}

// ====== Sum ==================================

pub fn sum<M1: MatrixRef, M2: MatrixRef>(left: M1, right: M2) -> Sum<M1, M2>
where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, ColT = M1::ColT, RowT = M1::RowT>,
{
    Sum { left, right }
}

// Note: We don't implement HasRowFiltration in case the filtrations disagree
// Note: We don't implement FiniteOrderedColBasis in case the number of cols disagrees
#[derive(Clone, Copy)]
pub struct Sum<M1: MatrixRef, M2: MatrixRef> {
    left: M1,
    right: M2,
}

impl<M1: MatrixRef, M2: MatrixRef> MatrixOracle for Sum<M1, M2>
where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, ColT = M1::ColT, RowT = M1::RowT>,
{
    type CoefficientField = M1::CoefficientField;
    type ColT = M1::ColT;
    type RowT = M1::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        Ok(self.left.column(col)?.chain(self.right.column(col)?))
    }
}

impl<M1: MatrixRef, M2: MatrixRef> MatrixRef for Sum<M1, M2> where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, ColT = M1::ColT, RowT = M1::RowT>
{
}
