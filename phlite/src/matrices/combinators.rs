// ======== Combinators ========================================

// ====== Product ==============================

use crate::{columns::ColumnEntry, PhliteError};

use super::{HasColBasis, HasRowFiltration, MatrixOracle};

pub fn product<M1, M2>(left: M1, right: M2) -> Product<M1, M2>
where
    M1: MatrixOracle,
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT> + MatrixOracle,
{
    Product { left, right }
}

#[derive(Clone, Copy)]
pub struct Product<M1: MatrixOracle, M2: MatrixOracle> {
    left: M1,
    right: M2,
}

/// Given a matrix `M` and a column `v`, returns an iterator over the entries in the column vector `Mv`.
///
/// Inputs:
/// * `$matrix` - Should implement [`MatrixOracle`].
/// * `$col` - Should implement
/// `Iterator<Item=($matrix::CoefficientField, $matrix::ColT)>`
/// so that types align and matrix multiplication can be performed.
/// Usually obtained by calling [`column`](MatrixOracle::column) on another matrix.
#[macro_export]
macro_rules! matrix_col_product {
    (  $matrix: expr, $col: expr ) => {{
        $col.flat_map(|(right_coeff, right_row_index)| {
            let left_col = $matrix.column(right_row_index);
            left_col
                .map(move |(left_coeff, left_row_index)| (left_coeff * right_coeff, left_row_index))
        })
    }};
}

impl<M1: MatrixOracle, M2: MatrixOracle> MatrixOracle for Product<M1, M2>
where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT>,
{
    type CoefficientField = M1::CoefficientField;

    type ColT = M2::ColT;

    type RowT = M1::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        // Pull out right col
        let right_col = self.right.column(col);
        // This tells us what linear combination of columns in the left matrix
        // should be formed to yield the product column
        matrix_col_product!(self.left, right_col)
    }
}

// In product there is an obvious row filtration if the LHS has a row filtration
impl<M1: MatrixOracle, M2: MatrixOracle> HasRowFiltration for Product<M1, M2>
where
    M1: HasRowFiltration,
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT>,
{
    type FiltrationT = M1::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.left.filtration_value(row)
    }

    // In case left has a more efficient column with filtration, we try to use it
    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<M1::FiltrationT, M1::RowT, M1::CoefficientField>> {
        let right_col = self.right.column(col);
        right_col.flat_map(|(right_coeff, right_row_index)| {
            let left_col = self.left.column_with_filtration(right_row_index);
            left_col.map(move |left_entry| ColumnEntry {
                coeff: left_entry.coeff * right_coeff,
                row_index: left_entry.row_index,
                filtration_value: left_entry.filtration_value,
            })
        })
    }
}

impl<M1: MatrixOracle, M2: MatrixOracle> HasColBasis for Product<M1, M2>
where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT> + HasColBasis,
{
    type BasisT = M2::BasisT;
    type BasisRef<'a>
        = M2::BasisRef<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisRef<'_> {
        self.right.basis()
    }
}

// ====== Sum ==================================

pub fn sum<M1, M2>(left: M1, right: M2) -> Sum<M1, M2>
where
    M1: MatrixOracle,
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, ColT = M1::ColT, RowT = M1::RowT>
        + MatrixOracle,
{
    Sum { left, right }
}

// Note: We don't implement HasRowFiltration in case the filtrations disagree
// Note: We don't implement HasColBasis in case the number of cols disagrees
#[derive(Clone, Copy)]
pub struct Sum<M1: MatrixOracle, M2: MatrixOracle> {
    left: M1,
    right: M2,
}

impl<M1: MatrixOracle, M2: MatrixOracle> MatrixOracle for Sum<M1, M2>
where
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, ColT = M1::ColT, RowT = M1::RowT>,
{
    type CoefficientField = M1::CoefficientField;
    type ColT = M1::ColT;
    type RowT = M1::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.left.column(col.clone()).chain(self.right.column(col))
    }
}
