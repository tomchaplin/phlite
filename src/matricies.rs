use std::marker::PhantomData;

use ordered_float::NotNan;

use crate::{fields::NonZeroCoefficient, PhliteError};

pub trait BasisElement: Ord {}
pub trait FiltrationT: Ord {}

impl BasisElement for usize {}
impl BasisElement for isize {}
impl FiltrationT for NotNan<f32> {}
impl FiltrationT for NotNan<f64> {}
impl FiltrationT for () {}

pub trait MatrixOracle {
    type CoefficientField: NonZeroCoefficient;
    type ColT: BasisElement;
    type RowT: BasisElement + Copy;
    type FiltrationT: FiltrationT + Copy;

    /// Implement your oracle on the widest range of [`ColT`](Self::ColT) possible.
    /// To specify a given matrix, you will later provide an oracle, alongside a basis for the column space.
    /// If you are unable to produce a column, please return [`PhliteError::NotInDomain`].
    ///
    /// It is dis-advantageous to produce the rows in ascending order (see [`BinaryHeap::push`](std::collections::BinaryHeap::push)).
    /// Since checking and sorting by filtration values is typically slow, prefer to produce in descending order with respect to the ordering on [`RowT`](Self::RowT).
    /// Lower bounds on iterator size provided via [`Iterator::size_hint`] will be used to preallocate.
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<
        impl Iterator<Item = (Self::FiltrationT, Self::RowT, Self::CoefficientField)>,
        PhliteError,
    >;
}

pub struct VecVecMatrix<CF> {
    columns: Vec<Vec<usize>>,
    phantom: PhantomData<CF>,
}

impl<CF> From<Vec<Vec<usize>>> for VecVecMatrix<CF> {
    fn from(value: Vec<Vec<usize>>) -> Self {
        Self {
            columns: value,
            phantom: PhantomData,
        }
    }
}

impl<CF: NonZeroCoefficient> MatrixOracle for VecVecMatrix<CF> {
    type CoefficientField = CF;

    type ColT = usize;

    type RowT = usize;

    type FiltrationT = ();

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<
        impl Iterator<Item = (Self::FiltrationT, Self::RowT, Self::CoefficientField)>,
        PhliteError,
    > {
        Ok(self
            .columns
            .get(col)
            .ok_or(PhliteError::NotInDomain)?
            .iter()
            .map(|&row_idx| ((), row_idx, CF::one())))
    }
}

#[cfg(test)]
mod tests {
    use crate::columns::ColumnEntry;
    use crate::{
        fields::{NonZeroCoefficient, Z2},
        matricies::VecVecMatrix,
    };

    use super::MatrixOracle;
    use crate::columns::BHCol;

    #[test]
    fn test_add() {
        let matrix: VecVecMatrix<Z2> = vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![3, 4, 5],
        ]
        .into();

        let mut column: BHCol<VecVecMatrix<Z2>> = BHCol::default();
        column.add_entries(matrix.column(5).unwrap());
        column.add_entries(matrix.column(4).unwrap());
        println!("{:?}", column);
        assert_eq!(
            column.pop_pivot(),
            Some(ColumnEntry::from(((), 1, Z2::one())))
        );
        assert_eq!(
            column.pop_pivot(),
            Some(ColumnEntry::from(((), 0, Z2::one())))
        );
        assert_eq!(column.pop_pivot(), None);

        let mut column: BHCol<VecVecMatrix<Z2>> = BHCol::default();
        column.add_entries(matrix.column(5).unwrap());
        column.add_entries(matrix.column(4).unwrap());
        column.add_entries(matrix.column(3).unwrap());
        assert_eq!(column.pop_pivot(), None);

        let mut column: BHCol<VecVecMatrix<Z2>> = BHCol::default();
        column.add_entries(matrix.column(6).unwrap());
        column.add_entries(matrix.column(6).unwrap());
        println!("{:?}", column);
        assert_eq!(column.to_sorted_vec().len(), 0);
    }
}
