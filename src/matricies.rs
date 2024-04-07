use std::{borrow::Cow, marker::PhantomData};

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

// TODO: Try and get rid of Sized bounds, is there a better way to summarise ColumnEntry?

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
        let self_trivial = self.with_trivial_filtration();
        let other_trivial = other.with_trivial_filtration();

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

pub trait FiniteOrderedColBasis: MatrixOracle<ColT = usize> {
    fn n_cols(&self) -> usize;
}

pub trait MatrixRef: MatrixOracle + Copy {
    fn with_trivial_filtration(self) -> WithTrivialFiltration<Self>
    where
        Self: Sized,
    {
        WithTrivialFiltration { matrix_ref: self }
    }

    fn with_filtration<FT: FiltrationT, F: Fn(Self::RowT) -> Result<FT, PhliteError>>(
        self,
        filtration_function: F,
    ) -> WithFuncFiltration<Self, FT, F>
    where
        Self: Sized,
    {
        WithFuncFiltration {
            oracle: self,
            filtration: filtration_function,
        }
    }
}

impl<'a, M> MatrixOracle for &'a M
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
        (*self).column(col)
    }
}

impl<'a, M> HasRowFiltration for &'a M
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        (*self).filtration_value(row)
    }
}

impl<'a, M> FiniteOrderedColBasis for &'a M
where
    M: FiniteOrderedColBasis,
{
    fn n_cols(&self) -> usize {
        (*self).n_cols()
    }
}

impl<'a, M> MatrixRef for &'a M where M: MatrixOracle {}

// ======== Matrix oracle adaptors ============================

// ====== WithTrivialFiltration ================

#[derive(Clone, Copy)]
pub struct WithTrivialFiltration<M: MatrixRef> {
    matrix_ref: M,
}

impl<M: MatrixRef> MatrixOracle for WithTrivialFiltration<M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        self.matrix_ref.column(col)
    }
}

impl<M: MatrixRef> MatrixRef for WithTrivialFiltration<M> {}

impl<M: MatrixRef> HasRowFiltration for WithTrivialFiltration<M> {
    type FiltrationT = ();

    fn filtration_value(&self, _row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        Ok(())
    }
}

impl<M: MatrixRef> FiniteOrderedColBasis for WithTrivialFiltration<M>
where
    M: FiniteOrderedColBasis,
{
    fn n_cols(&self) -> usize {
        self.matrix_ref.n_cols()
    }
}

// ====== WithFuncFiltration ===================

#[derive(Clone)]
pub struct WithFuncFiltration<
    M: MatrixRef,
    FT: FiltrationT,
    F: Fn(M::RowT) -> Result<FT, PhliteError>,
> {
    oracle: M,
    filtration: F,
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> Copy
    for WithFuncFiltration<M, FT, F>
where
    F: Copy,
{
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>>
    WithFuncFiltration<M, FT, F>
{
    pub fn discard_filtration(self) -> M {
        self.oracle
    }
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> MatrixOracle
    for WithFuncFiltration<M, FT, F>
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        self.oracle.column(col)
    }
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> MatrixRef
    for WithFuncFiltration<M, FT, F>
where
    F: Copy,
{
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> HasRowFiltration
    for WithFuncFiltration<M, FT, F>
{
    type FiltrationT = FT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        (self.filtration)(row)
    }
}

impl<M: MatrixRef, FT: FiltrationT, F: Fn(M::RowT) -> Result<FT, PhliteError>> FiniteOrderedColBasis
    for WithFuncFiltration<M, FT, F>
where
    M: FiniteOrderedColBasis,
{
    fn n_cols(&self) -> usize {
        self.oracle.n_cols()
    }
}

// ====== WithOrderedBasis =====================

// TODO: Change so it only stores a ref?
pub struct WithOrderedColBasis<M: MatrixRef> {
    oracle: M,
    pub col_basis: Vec<M::ColT>,
}

impl<M: MatrixRef> WithOrderedColBasis<M> {
    pub fn new(oracle: M, col_basis: Vec<M::ColT>) -> Self {
        Self { oracle, col_basis }
    }
}

impl<M: MatrixRef> MatrixOracle for WithOrderedColBasis<M> {
    type CoefficientField = M::CoefficientField;

    type ColT = usize;

    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (M::CoefficientField, M::RowT)>, PhliteError> {
        let col_idx = self.col_basis.get(col).ok_or(PhliteError::NotInDomain)?;
        self.oracle.column(*col_idx)
    }
}

impl<M: MatrixRef> HasRowFiltration for WithOrderedColBasis<M>
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;
    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<M: MatrixRef> FiniteOrderedColBasis for WithOrderedColBasis<M> {
    fn n_cols(&self) -> usize {
        self.col_basis.len()
    }
}

// ====== Consolidator =========================

pub fn consolidate<M: MatrixRef>(oracle: M) -> Consolidator<M> {
    Consolidator { oracle }
}

#[derive(Clone, Copy)]
pub struct Consolidator<M: MatrixRef> {
    oracle: M,
}

pub struct ConsolidatorColumn<M: MatrixRef> {
    bh_col: BHCol<WithTrivialFiltration<M>>,
}

impl<M: MatrixRef> Iterator for ConsolidatorColumn<M> {
    type Item = (M::CoefficientField, M::RowT);

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.bh_col.pop_pivot()?;
        let (coef, index, _) = next.into();
        Some((coef, index))
    }
}

impl<M: MatrixRef> MatrixOracle for Consolidator<M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> Result<impl Iterator<Item = (Self::CoefficientField, Self::RowT)>, PhliteError> {
        // Take all the entries in the column and store them in a binary heap
        let bh_col = self.oracle.with_trivial_filtration().build_bhcol(col)?;
        // This iterator will consolidate all entries with the same row index into a new iterator
        Ok(ConsolidatorColumn { bh_col })
    }
}

impl<M: MatrixRef> MatrixRef for Consolidator<M> {}

impl<M: MatrixRef> HasRowFiltration for Consolidator<M>
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError> {
        self.oracle.filtration_value(row)
    }
}

impl<M: MatrixRef> FiniteOrderedColBasis for Consolidator<M>
where
    M: FiniteOrderedColBasis,
{
    fn n_cols(&self) -> usize {
        self.oracle.n_cols()
    }
}

// ======== Combinators ========================================

// ====== Product ==============================

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

// ======== Default matrix oracles =============================

// ====== VecVecMatrix =========================

pub struct VecVecMatrix<'a, CF: NonZeroCoefficient, RowT: BasisElement> {
    columns: Cow<'a, Vec<Vec<(CF, RowT)>>>,
    phantom: PhantomData<CF>,
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> From<Cow<'a, Vec<Vec<(CF, RowT)>>>>
    for VecVecMatrix<'a, CF, RowT>
{
    fn from(value: Cow<'a, Vec<Vec<(CF, RowT)>>>) -> Self {
        Self {
            columns: value,
            phantom: PhantomData,
        }
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> From<&'a Vec<Vec<(CF, RowT)>>>
    for VecVecMatrix<'a, CF, RowT>
{
    fn from(value: &'a Vec<Vec<(CF, RowT)>>) -> Self {
        Self {
            columns: Cow::Borrowed(value),
            phantom: PhantomData,
        }
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> From<Vec<Vec<(CF, RowT)>>>
    for VecVecMatrix<'static, CF, RowT>
{
    fn from(value: Vec<Vec<(CF, RowT)>>) -> Self {
        Self {
            columns: Cow::Owned(value),
            phantom: PhantomData,
        }
    }
}

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> MatrixOracle for VecVecMatrix<'a, CF, RowT> {
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

impl<'a, CF: NonZeroCoefficient, RowT: BasisElement> FiniteOrderedColBasis
    for VecVecMatrix<'a, CF, RowT>
{
    fn n_cols(&self) -> usize {
        self.columns.len()
    }
}

#[allow(non_snake_case)]
pub fn simple_Z2_matrix(cols: Vec<Vec<usize>>) -> VecVecMatrix<'static, Z2, usize> {
    let cols_with_coeffs = cols
        .into_iter()
        .map(|col| {
            col.into_iter()
                .map(|col_idx| (Z2::one(), col_idx))
                .collect()
        })
        .collect::<Vec<Vec<(Z2, usize)>>>();

    <VecVecMatrix<'_, Z2, usize>>::from(Cow::Owned(cols_with_coeffs))
}

// ======== Tests ==============================================

#[cfg(test)]
mod tests {

    use crate::fields::{NonZeroCoefficient, Z2};
    use crate::matricies::{simple_Z2_matrix, HasRowFiltration, MatrixRef};

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

        let matrix_r = product(&matrix_d, &matrix_v);

        assert!((0..=6).all(|i| matrix_r.eq_on_col(&true_matrix_r, i)))
    }

    #[test]
    fn test_matrix_bhcol_interface() {
        let base_matrix = simple_Z2_matrix(vec![
            vec![],
            vec![],
            vec![],
            vec![0, 1],
            vec![1, 2],
            vec![0, 2],
            vec![3, 4, 5],
        ]);
        let matrix = base_matrix.with_filtration(|idx| Ok(idx * 10));

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
        let opp_matrix =
            matrix.with_filtration(|idx| Ok(-(matrix.filtration_value(idx).unwrap() as isize)));
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
        let mat = simple_Z2_matrix(vec![vec![0], vec![0, 1]]);
        // Working over Z^2 so M^2 = Id

        let mat4 = product(product(&mat, &mat), product(&mat, &mat));

        let col1: Vec<_> = mat4.column(1).unwrap().collect();

        let col2: Vec<_> = consolidate(&mat4).column(1).unwrap().collect();

        // Lots of entries adding up
        assert_eq!(col1.len(), 5);
        // Consolidated down to single entry
        assert_eq!(col2.len(), 1);
    }

    #[test]
    fn test_projection() {
        let base_matrix = simple_Z2_matrix(vec![
            vec![4, 3, 12],
            vec![5, 9, 4],
            vec![0, 1, 0],
            vec![1, 2, 4, 4],
        ]);

        // Dumb way to build - better to make custom oracle
        let mut projection_cols = vec![vec![]; 13];
        projection_cols[4] = vec![4];
        let projection_matrix = simple_Z2_matrix(projection_cols);

        let projected = product(&projection_matrix, &base_matrix);

        let true_matrix = simple_Z2_matrix(vec![vec![4], vec![4], vec![], vec![]]);

        assert!((0..4).all(|i| projected.eq_on_col(&true_matrix, i)))
    }
}
