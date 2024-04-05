use std::{borrow::Borrow, marker::PhantomData};

use ordered_float::NotNan;

use crate::{
    columns::{BHCol, ColumnEntry},
    fields::{NonZeroCoefficient, Z2},
    PhliteError,
};

// ========= Traits for matrix indices and filtrations =========

pub trait BasisElement: Ord + Copy {}
pub trait FiltrationT: Ord + Copy {}

// Default implementors

impl BasisElement for usize {}
impl BasisElement for isize {}
impl FiltrationT for NotNan<f32> {}
impl FiltrationT for NotNan<f64> {}
impl FiltrationT for usize {}
impl FiltrationT for isize {}
impl FiltrationT for () {}

// ======== Abstract matrix oracle trait =======================

pub trait MatrixOracle {
    type CoefficientField: NonZeroCoefficient;
    type ColT: BasisElement;
    type RowT: BasisElement;

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
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError>;

    // TODO: Offer consuming versions to allow method chaining

    fn with_trivial_filtration(self) -> WithTrivialFiltration<Self, Self>
    where
        Self: Sized,
    {
        WithTrivialFiltration {
            matrix_ref: self,
            phantom: PhantomData,
        }
    }

    fn borrow_trivial_filtration(&self) -> WithTrivialFiltration<&Self, Self>
    where
        Self: Sized,
    {
        WithTrivialFiltration {
            matrix_ref: &self,
            phantom: PhantomData,
        }
    }
}

pub trait HasRowFiltration: MatrixOracle + Sized {
    type FiltrationT: FiltrationT;
    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError>;

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = Result<ColumnEntry<Self>, PhliteError>>, PhliteError> {
        let column = self.column(col)?;
        Ok(column.map(|(coeff, row_index)| {
            let f_val = self.filtration_value(row_index)?;
            let entry = (coeff, row_index, f_val).into();
            Ok(entry)
        }))
    }

    fn empty_bhcol(&self) -> BHCol<Self> {
        BHCol::<Self>::default()
    }
}

// ======== Matrix oracle adaptors ============================

// ====== WithTrivialFiltration ================

pub struct WithTrivialFiltration<Ref: Borrow<M>, M: MatrixOracle> {
    matrix_ref: Ref,
    phantom: PhantomData<M>,
}

impl<Ref: Borrow<M>, M: MatrixOracle> MatrixOracle for WithTrivialFiltration<Ref, M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        self.matrix_ref.borrow().column(col)
    }
}

impl<Ref: Borrow<M>, M: MatrixOracle> HasRowFiltration for WithTrivialFiltration<Ref, M> {
    type FiltrationT = ();

    fn filtration_value(&self, _row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        Ok(())
    }
}

// ====== WithOrderedBasis =====================

pub struct WithOrderedColBasis<Ref: Borrow<M>, M: MatrixOracle> {
    oracle: Ref,
    col_basis: Vec<M::ColT>,
}

impl<Ref: Borrow<M>, M: MatrixOracle> MatrixOracle for WithOrderedColBasis<Ref, M> {
    type CoefficientField = M::CoefficientField;

    type ColT = usize;

    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (M::CoefficientField, M::RowT)>, PhliteError> {
        let col_idx = self.col_basis.get(col).ok_or(PhliteError::NotInDomain)?;
        self.oracle.borrow().column(*col_idx)
    }
}

impl<Ref: Borrow<M>, M: MatrixOracle> HasRowFiltration for WithOrderedColBasis<Ref, M>
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;
    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.oracle.borrow().filtration_value(row)
    }
}

// ======== Combinators ========================================

// ====== Product ==============================

pub struct Product<Ref1: Borrow<M1>, Ref2: Borrow<M2>, M1, M2> {
    left: Ref1,
    right: Ref2,
    phantom_left: PhantomData<M1>,
    phantom_right: PhantomData<M2>,
}

impl<Ref1: Borrow<M1>, Ref2: Borrow<M2>, M1, M2> MatrixOracle for Product<Ref1, Ref2, M1, M2>
where
    M1: MatrixOracle,
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
        let right_col_entries = self.right.borrow().column(col)?;
        // This tells us what linear combination of columns in the left matrix
        // should be formed to yield the product column
        Ok(
            right_col_entries.flat_map(|(right_coeff, right_row_index)| {
                let left_col = self.left.borrow().column(right_row_index).unwrap();
                left_col.map(move |(left_coeff, left_row_index)| {
                    (left_coeff * right_coeff, left_row_index)
                })
            }),
        )
    }
}

// In product there is an obvious row filtration if the LHS has a row filtration
impl<Ref1: Borrow<M1>, Ref2: Borrow<M2>, M1, M2> HasRowFiltration for Product<Ref1, Ref2, M1, M2>
where
    M1: HasRowFiltration,
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT>,
{
    type FiltrationT = M1::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.left.borrow().filtration_value(row)
    }
}

// ====== Add ==================================

// pub struct Add<'a, M1, M2> {
//     left: &'a M1,
//     right: &'a M2,
// }
//
// impl<'a, M1, M2> MatrixOracle for Add<'a, M1, M2> {
//     type CoefficientField;
//
//     type ColT;
//
//     type RowT;
//
//     type FiltrationT;
//
//     fn column(
//         &self,
//         col: Self::ColT,
//     ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
//         todo!()
//     }
//
//     fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
//         todo!()
//     }
// }

// ======== Default matrix oracles =============================

// ====== VecVecMatrix =========================

pub struct VecVecMatrix<CF: NonZeroCoefficient, RowT: BasisElement> {
    columns: Vec<Vec<(CF, RowT)>>,
    phantom: PhantomData<CF>,
}

impl<CF, RowT> From<Vec<Vec<(CF, RowT)>>> for VecVecMatrix<CF, RowT>
where
    CF: NonZeroCoefficient,
    RowT: BasisElement,
{
    fn from(value: Vec<Vec<(CF, RowT)>>) -> Self {
        Self {
            columns: value,
            phantom: PhantomData,
        }
    }
}

impl<RowT, CF> MatrixOracle for VecVecMatrix<CF, RowT>
where
    RowT: BasisElement,
    CF: NonZeroCoefficient,
{
    type CoefficientField = CF;

    type ColT = usize;

    type RowT = RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        Ok(self
            .columns
            .get(col)
            .ok_or(PhliteError::NotInDomain)?
            .iter()
            .copied())
    }
}

type SimpleZ2Matrix = VecVecMatrix<Z2, usize>;

#[allow(non_snake_case)]
pub fn simple_Z2_matrix(cols: Vec<Vec<usize>>) -> SimpleZ2Matrix {
    cols.into_iter()
        .map(|col| {
            col.into_iter()
                .map(|col_idx| (Z2::one(), col_idx))
                .collect()
        })
        .collect::<Vec<Vec<(Z2, usize)>>>()
        .into()
}

// ======== Tests ==============================================

#[cfg(test)]
mod tests {

    use crate::fields::{NonZeroCoefficient, Z2};
    use crate::matricies::{simple_Z2_matrix, HasRowFiltration};

    use super::MatrixOracle;
    use crate::columns::{BHCol, ColumnEntry};

    #[test]
    fn test_add() {
        let matrix = simple_Z2_matrix(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![3, 4, 5],
        ])
        .with_trivial_filtration();
        let add = |column: &mut BHCol<_>, index| {
            column.add_entries(
                matrix
                    .column(index)
                    .unwrap()
                    .map(|(coef, idx)| (coef, idx, ()))
                    .map(ColumnEntry::from),
            );
        };

        let mut column = matrix.empty_bhcol();
        add(&mut column, 5);
        add(&mut column, 4);
        println!("{:?}", column);
        assert_eq!(
            column.pop_pivot().map(|e| e.into()),
            Some((Z2::one(), 1, ()))
        );
        assert_eq!(
            column.pop_pivot().map(|e| e.into()),
            Some((Z2::one(), 0, ()))
        );
        assert_eq!(column.pop_pivot(), None);

        // Forget filtration

        let mut column = matrix.empty_bhcol();
        add(&mut column, 5);
        add(&mut column, 4);
        println!("{:?}", column);
        assert_eq!(
            column.pop_pivot().map(|e| e.into()),
            Some((Z2::one(), 1, ()))
        );
        assert_eq!(
            column.pop_pivot().map(|e| e.into()),
            Some((Z2::one(), 0, ()))
        );
        assert_eq!(column.pop_pivot(), None);

        // Extra tests

        let mut column = matrix.empty_bhcol();
        add(&mut column, 5);
        add(&mut column, 4);
        add(&mut column, 3);
        assert_eq!(column.pop_pivot(), None);

        let mut column = matrix.empty_bhcol();
        add(&mut column, 6);
        add(&mut column, 6);
        println!("{:?}", column);
        assert_eq!(column.to_sorted_vec().len(), 0);
    }
}
