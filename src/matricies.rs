use std::{borrow::Borrow, marker::PhantomData};

use itertools::equal;
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
    /// It is dis-advantageous to produce the rows in ascending order because inserting into binary heaps would require traversing the full height (see [`BinaryHeap::push`](std::collections::BinaryHeap::push)).
    /// Since checking and sorting by filtration values is typically slow, prefer to produce in descending order with respect to the ordering on [`RowT`](Self::RowT).
    /// Lower bounds on iterator size provided via [`Iterator::size_hint`] will be used to preallocate.
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError>;

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

    fn with_filtration<FT: FiltrationT, F: Fn(Self::RowT) -> Result<FT, PhliteError>>(
        self,
        filtration_function: F,
    ) -> WithFuncFiltration<Self, Self, FT, F>
    where
        Self: Sized,
    {
        WithFuncFiltration {
            oracle: self,
            filtration: filtration_function,
            phantom: PhantomData,
        }
    }

    fn borrow_with_filtration<FT: FiltrationT, F: Fn(Self::RowT) -> Result<FT, PhliteError>>(
        &self,
        filtration_function: F,
    ) -> WithFuncFiltration<&Self, Self, FT, F>
    where
        Self: Sized,
    {
        WithFuncFiltration {
            oracle: self,
            filtration: filtration_function,
            phantom: PhantomData,
        }
    }

    /// Checks that the matricies are equal on the specified col, ignoring ordering due to filtration values
    fn eq_on_col<M2>(&self, other: &M2, col: Self::ColT) -> bool
    where
        Self: Sized,
        M2: MatrixOracle<
                CoefficientField = Self::CoefficientField,
                ColT = Self::ColT,
                RowT = Self::RowT,
            > + Sized,
    {
        let self_trivial = self.borrow_trivial_filtration();
        let other_trivial = other.borrow_trivial_filtration();

        let mut self_col = self_trivial.build_bhcol(col).unwrap();
        let self_col_sorted = self_col
            .drain_sorted()
            .map(|e| Into::<(Self::CoefficientField, Self::RowT, ())>::into(e));
        let mut other_col = other_trivial.build_bhcol(col).unwrap();
        let other_col_sorted = other_col
            .drain_sorted()
            .map(|e| Into::<(Self::CoefficientField, Self::RowT, ())>::into(e));

        equal(self_col_sorted, other_col_sorted)
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

    fn build_bhcol(&self, col: Self::ColT) -> Result<BHCol<Self>, PhliteError> {
        let mut output = self.empty_bhcol();
        output.add_entries(self.column_with_filtration(col)?.map(|e| e.unwrap()));
        Ok(output)
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

// ====== WithFuncFiltration ===================

pub struct WithFuncFiltration<
    Ref: Borrow<M>,
    M: MatrixOracle,
    FT: FiltrationT,
    F: Fn(M::RowT) -> Result<FT, PhliteError>,
> {
    oracle: Ref,
    filtration: F,
    phantom: PhantomData<M>,
}

impl<
        Ref: Borrow<M>,
        M: MatrixOracle,
        FT: FiltrationT,
        F: Fn(M::RowT) -> Result<FT, PhliteError>,
    > WithFuncFiltration<Ref, M, FT, F>
{
    pub fn discard_filtration(self) -> Ref {
        self.oracle
    }
}

impl<
        Ref: Borrow<M>,
        M: MatrixOracle,
        FT: FiltrationT,
        F: Fn(M::RowT) -> Result<FT, PhliteError>,
    > MatrixOracle for WithFuncFiltration<Ref, M, FT, F>
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        self.oracle.borrow().column(col)
    }
}

impl<
        Ref: Borrow<M>,
        M: MatrixOracle,
        FT: FiltrationT,
        F: Fn(M::RowT) -> Result<FT, PhliteError>,
    > HasRowFiltration for WithFuncFiltration<Ref, M, FT, F>
{
    type FiltrationT = FT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        (self.filtration)(row)
    }
}

// ====== WithOrderedBasis =====================

pub struct WithOrderedColBasis<Ref: Borrow<M>, M: MatrixOracle> {
    oracle: Ref,
    pub col_basis: Vec<M::ColT>,
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

// ====== Consolidator =========================

fn consolidate<Ref: Borrow<M>, M>(oracle: Ref) -> Consolidator<Ref, M> {
    Consolidator {
        oracle,
        phantom: PhantomData,
    }
}

pub struct Consolidator<Ref: Borrow<M>, M> {
    oracle: Ref,
    phantom: PhantomData<M>,
}

pub struct ConsolidatorColumn<'a, M: MatrixOracle> {
    bh_col: BHCol<WithTrivialFiltration<&'a M, M>>,
}

impl<'a, M: MatrixOracle> Iterator for ConsolidatorColumn<'a, M> {
    type Item = (M::CoefficientField, M::RowT);

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.bh_col.pop_pivot()?;
        let (coef, index, _) = next.into();
        Some((coef, index))
    }
}

impl<Ref: Borrow<M>, M> MatrixOracle for Consolidator<Ref, M>
where
    M: MatrixOracle,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        // Take all the entries in the column and store them in a binary heap
        let bh_col = self
            .oracle
            .borrow()
            .borrow_trivial_filtration()
            .build_bhcol(col)?;
        // This iterator will consolidate all entries with the same row index into a new iterator
        Ok(ConsolidatorColumn { bh_col })
    }
}

impl<Ref: Borrow<M>, M> HasRowFiltration for Consolidator<Ref, M>
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

pub fn product<Ref1: Borrow<M1>, Ref2: Borrow<M2>, M1, M2>(
    left: Ref1,
    right: Ref2,
) -> Product<Ref1, Ref2, M1, M2>
where
    M1: MatrixOracle,
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, RowT = M1::ColT>,
{
    Product {
        left,
        right,
        phantom_left: PhantomData,
        phantom_right: PhantomData,
    }
}

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

// ====== Sum ==================================

pub fn sum<Ref1: Borrow<M1>, Ref2: Borrow<M2>, M1, M2>(
    left: Ref1,
    right: Ref2,
) -> Sum<Ref1, Ref2, M1, M2>
where
    M1: MatrixOracle,
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, ColT = M1::ColT, RowT = M1::RowT>,
{
    Sum {
        left,
        right,
        phantom_left: PhantomData,
        phantom_right: PhantomData,
    }
}

// Note: We don't implement has filtration in case the filtrations disagree
pub struct Sum<Ref1: Borrow<M1>, Ref2: Borrow<M2>, M1, M2> {
    left: Ref1,
    right: Ref2,
    phantom_left: PhantomData<M1>,
    phantom_right: PhantomData<M2>,
}

impl<Ref1: Borrow<M1>, Ref2: Borrow<M2>, M1, M2> MatrixOracle for Sum<Ref1, Ref2, M1, M2>
where
    M1: MatrixOracle,
    M2: MatrixOracle<CoefficientField = M1::CoefficientField, ColT = M1::ColT, RowT = M1::RowT>,
{
    type CoefficientField = M1::CoefficientField;
    type ColT = M1::ColT;
    type RowT = M1::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        Ok(self
            .left
            .borrow()
            .column(col)?
            .chain(self.right.borrow().column(col)?))
    }
}

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

    use super::{consolidate, product, MatrixOracle};
    use crate::columns::BHCol;

    #[test]
    fn test_matrix_product() {
        let matrix_d = simple_Z2_matrix(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![3, 4, 5],
        ]);
        let matrix_v = simple_Z2_matrix(vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![3, 4, 5],
            vec![6],
        ]);

        let true_matrix_r = simple_Z2_matrix(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![],
            vec![3, 4, 5],
        ]);

        let matrix_r = product(matrix_d, matrix_v);

        assert!((0..=6).all(|i| matrix_r.eq_on_col(&true_matrix_r, i)))
    }

    #[test]
    fn test_matrix_bhcol_interface() {
        let matrix = simple_Z2_matrix(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![3, 4, 5],
        ])
        .with_filtration(|idx| Ok(idx * 10));
        let add = |column: &mut BHCol<_>, index| {
            column.add_entries(
                matrix
                    .column_with_filtration(index)
                    .unwrap()
                    .map(|e| e.unwrap()),
            );
        };

        let mut column = matrix.empty_bhcol();
        add(&mut column, 5);
        add(&mut column, 4);
        assert_eq!(
            column.pop_pivot().map(|e| e.into()),
            Some((Z2::one(), 1, 10))
        );
        assert_eq!(
            column.pop_pivot().map(|e| e.into()),
            Some((Z2::one(), 0, 0))
        );
        assert_eq!(column.pop_pivot(), None);

        // Opposite filtration
        let opp_matrix = matrix
            .borrow_with_filtration(|idx| Ok(-(matrix.filtration_value(idx).unwrap() as isize)));
        let opp_add = |column: &mut BHCol<_>, index| {
            column.add_entries(
                opp_matrix
                    .column_with_filtration(index)
                    .unwrap()
                    .map(|e| e.unwrap()),
            );
        };
        let mut column = opp_matrix.empty_bhcol();
        opp_add(&mut column, 5);
        opp_add(&mut column, 4);
        assert_eq!(
            column.pop_pivot().map(|e| e.into()),
            Some((Z2::one(), 0, 0))
        );
        assert_eq!(
            column.pop_pivot().map(|e| e.into()),
            Some((Z2::one(), 1, -10))
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
        assert_eq!(column.to_sorted_vec().len(), 0);
    }

    #[test]
    fn test_consolidate() {
        let build_mat = || simple_Z2_matrix(vec![vec![0], vec![0, 1]]);
        // Working over Z^2 so M^2 = Id

        let build_mat4 = product(
            product(build_mat(), build_mat()),
            product(build_mat(), build_mat()),
        );

        let col1: Vec<_> = build_mat4.column(1).unwrap().collect();

        // TODO: Why can't types be inferred when I pass in a reference?

        let col2: Vec<_> = consolidate(build_mat4).column(1).unwrap().collect();

        // Lots of entries adding up
        assert_eq!(col1.len(), 5);
        // Consolidated down to single entry
        assert_eq!(col2.len(), 1);
    }
}
