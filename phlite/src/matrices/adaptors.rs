//! Contains matrix adaptors which alter the behaviour of existing matrices, e.g. reversing indices or adding a filtration/basis.
//!
//! All structs in this submodule should be constructed by calling the relevant provided methods on [`MatrixOracle`] or [`HasColBasis`].
//! Their utility is explained in the documentation for these constructors.

// ======== Matrix oracle adaptors ============================

// ====== WithTrivialFiltration ================

use std::cmp::Reverse;

use crate::columns::{BHCol, ColumnEntry};

use super::{
    BasisElement, ColBasis, FiltrationValue, HasColBasis, HasRowFiltration, HasSplitColBasis,
    HasTranspose, MatrixOracle, SplitByDimension, SquareMatrix,
};

/// Return type of [`MatrixOracle::with_trivial_filtration`].
#[derive(Clone, Copy, Debug)]
pub struct WithTrivialFiltration<M: MatrixOracle> {
    pub(crate) oracle: M,
}

impl<M: MatrixOracle> MatrixOracle for WithTrivialFiltration<M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.oracle.column(col)
    }
}

impl<M: MatrixOracle> HasRowFiltration for WithTrivialFiltration<M> {
    type FiltrationT = ();

    fn filtration_value(&self, _row: Self::RowT) -> Self::FiltrationT {}
}

impl<M: MatrixOracle> HasColBasis for WithTrivialFiltration<M>
where
    M: HasColBasis,
{
    type BasisT<'a>
        = M::BasisT<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisT<'_> {
        self.oracle.basis()
    }
}

impl<M: MatrixOracle> HasSplitColBasis for WithTrivialFiltration<M>
where
    M: HasSplitColBasis,
{
    type SubBasisT<'a>
        = M::SubBasisT<'a>
    where
        Self: 'a;

    fn n_dimensions(&self) -> usize {
        self.oracle.n_dimensions()
    }

    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_> {
        self.oracle.sub_basis(dimension)
    }
}

// ====== WithFuncFiltration ===================

/// Return type of [`MatrixOracle::with_filtration`].
#[derive(Clone, Copy, Debug)]
pub struct WithFuncFiltration<M: MatrixOracle, FT: FiltrationValue, F: Fn(M::RowT) -> FT> {
    pub(crate) oracle: M,
    pub(crate) filtration: F,
}

impl<M: MatrixOracle, FT: FiltrationValue, F: Fn(M::RowT) -> FT> WithFuncFiltration<M, FT, F> {
    /// Discard the filtration, returning the matrix from which this one was originally constructed.
    pub fn discard_filtration(self) -> M {
        self.oracle
    }
}

impl<M: MatrixOracle, FT: FiltrationValue, F: Fn(M::RowT) -> FT> MatrixOracle
    for WithFuncFiltration<M, FT, F>
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.oracle.column(col)
    }
}

impl<M: MatrixOracle, FT: FiltrationValue, F: Fn(M::RowT) -> FT> HasRowFiltration
    for WithFuncFiltration<M, FT, F>
{
    type FiltrationT = FT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        (self.filtration)(row)
    }
}

impl<M: MatrixOracle, FT: FiltrationValue, F: Fn(M::RowT) -> FT> HasColBasis
    for WithFuncFiltration<M, FT, F>
where
    M: HasColBasis,
{
    type BasisT<'a>
        = M::BasisT<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisT<'_> {
        self.oracle.basis()
    }
}

impl<M: MatrixOracle, FT: FiltrationValue, F: Fn(M::RowT) -> FT> HasSplitColBasis for WithFuncFiltration<M, FT, F>
where
    M: HasSplitColBasis,
{
    type SubBasisT<'a>
        = M::SubBasisT<'a>
    where
        Self: 'a;

    fn n_dimensions(&self) -> usize {
        self.oracle.n_dimensions()
    }

    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_> {
        self.oracle.sub_basis(dimension)
    }
}

// ====== Consolidator =========================

/// Return type of [`MatrixOracle::consolidate`].
#[derive(Clone, Copy, Debug)]
pub struct Consolidator<M: MatrixOracle> {
    pub(crate) oracle: M,
}

/// Return type of [`Consolidator::column`].
#[derive(Debug, Clone)]
pub struct ConsolidatorColumn<M: MatrixOracle> {
    bh_col: BHCol<(), M::RowT, M::CoefficientField>,
}

impl<M: MatrixOracle> Iterator for ConsolidatorColumn<M> {
    type Item = (M::CoefficientField, M::RowT);

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.bh_col.pop_pivot()?;
        let (coef, index, ()) = next.into();
        Some((coef, index))
    }
}

impl<M: MatrixOracle> MatrixOracle for Consolidator<M> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        // Take all the entries in the column and store them in a binary heap
        let bh_col = (&self.oracle).with_trivial_filtration().build_bhcol(col);
        // This iterator will consolidate all entries with the same row index into a new iterator
        ConsolidatorColumn::<Self> { bh_col }
    }
}

impl<M: MatrixOracle> HasRowFiltration for Consolidator<M>
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.oracle.filtration_value(row)
    }
}

impl<M: MatrixOracle> HasColBasis for Consolidator<M>
where
    M: HasColBasis,
{
    type BasisT<'a>
        = M::BasisT<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisT<'_> {
        self.oracle.basis()
    }
}

impl<M: MatrixOracle> HasSplitColBasis for Consolidator<M>
where
    M: HasSplitColBasis,
{
    type SubBasisT<'a>
        = M::SubBasisT<'a>
    where
        Self: 'a;

    fn n_dimensions(&self) -> usize {
        self.oracle.n_dimensions()
    }

    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_> {
        self.oracle.sub_basis(dimension)
    }
}

// ====== MatrixWithBasis ======================

/// Return type of [`MatrixOracle::with_basis`], can also be constructed manually.
///
/// Implements [`HasColBasis`] by endowing `matrix` with the basis `basis`, inherits any filtration
#[derive(Debug, Clone, Copy)]
pub struct MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
{
    /// The underlying matrix (potentially with a filtration)
    pub matrix: M,
    /// The new basis, with which we implement [`HasColBasis`]
    pub basis: B,
}

impl<M, B> MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
{
    /// The obvious constructor.
    pub fn new(matrix: M, basis: B) -> Self {
        Self { matrix, basis }
    }
}

impl<M, B> MatrixOracle for MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;
    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.matrix.column(col)
    }
}

impl<M, B> HasRowFiltration for MatrixWithBasis<M, B>
where
    M: MatrixOracle + HasRowFiltration,
    B: ColBasis<ElemT = M::ColT>,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.matrix.filtration_value(row)
    }

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>
    {
        self.matrix.column_with_filtration(col)
    }

    fn column_pivot(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        self.matrix.column_pivot(col)
    }

    fn column_min(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        self.matrix.column_min(col)
    }

    fn first_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.matrix.first_column_in_row(row)
    }

    fn last_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.matrix.last_column_in_row(row)
    }

    fn sorted_column(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        self.matrix.sorted_column(col)
    }

    fn sorted_column_ascending(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        self.matrix.sorted_column_ascending(col)
    }
}

impl<M, B> HasColBasis for MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
{
    type BasisT<'a>
        = &'a B
    where
        Self: 'a;

    fn basis(&self) -> &B {
        &self.basis
    }
}

impl<M, B> HasSplitColBasis for MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
    for<'a> &'a B: SplitByDimension + ColBasis<ElemT = M::ColT>,
{
    type SubBasisT<'a>
        = <&'a B as SplitByDimension>::SubBasisT
    where
        Self: 'a;

    fn n_dimensions(&self) -> usize {
        (&self.basis).n_dimensions()
    }

    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_> {
        (&self.basis).in_dimension(dimension)
    }
}

// ====== UsingColBasisIndex ==================

/// Return type of [`MatrixOracle::using_col_basis_index`].
#[derive(Clone, Copy, Debug)]
pub struct UsingColBasisIndex<M: MatrixOracle + HasColBasis> {
    pub(crate) oracle: M,
}

impl<M> MatrixOracle for UsingColBasisIndex<M>
where
    M: MatrixOracle + HasColBasis,
{
    type CoefficientField = M::CoefficientField;

    type ColT = usize;

    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.oracle.column(self.oracle.basis().element(col))
    }
}

impl<M> HasRowFiltration for UsingColBasisIndex<M>
where
    M: MatrixOracle + HasColBasis + HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.oracle.filtration_value(row)
    }
}

// TODO: Should we give it the standard column basis?

// ====== WithSubBasis =========================

/// Return type of [`HasSplitColBasis::sub_matrix_in_dimension`].
#[derive(Clone, Copy, Debug)]
pub struct WithSubBasis<M: HasSplitColBasis> {
    pub(crate) oracle: M,
    pub(crate) dimension: usize,
}

impl<M> MatrixOracle for WithSubBasis<M>
where
    M: HasSplitColBasis,
{
    type CoefficientField = M::CoefficientField;

    type ColT = M::ColT;

    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.oracle.column(col)
    }
}

impl<M> HasRowFiltration for WithSubBasis<M>
where
    M: HasSplitColBasis + HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.oracle.filtration_value(row)
    }

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>
    {
        self.oracle.column_with_filtration(col)
    }

    fn column_pivot(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        self.oracle.column_pivot(col)
    }

    fn column_min(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        self.oracle.column_min(col)
    }

    fn first_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.oracle.first_column_in_row(row)
    }

    fn last_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.oracle.last_column_in_row(row)
    }

    fn sorted_column(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        self.oracle.sorted_column(col)
    }

    fn sorted_column_ascending(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        self.oracle.sorted_column_ascending(col)
    }
}

impl<M> HasColBasis for WithSubBasis<M>
where
    M: HasSplitColBasis,
{
    type BasisT<'a>
        = M::SubBasisT<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisT<'_> {
        self.oracle.sub_basis(self.dimension)
    }
}

// ====== ReverseMatrix ========================

/// Return type of [`MatrixOracle::reverse`].
#[derive(Clone, Copy, Debug)]
pub struct ReverseMatrix<M> {
    pub(crate) oracle: M,
}

impl<M> ReverseMatrix<M> {
    /// Override the usual [`MatrixOracle::unreverse`].
    /// This is more efficient because it just takes the inner oracle out of its [`ReverseMatrix`] wrapper.
    pub fn unreverse(self) -> M {
        self.oracle
    }
}

impl<M> MatrixOracle for ReverseMatrix<M>
where
    M: MatrixOracle,
{
    type CoefficientField = M::CoefficientField;

    type ColT = Reverse<M::ColT>;

    type RowT = Reverse<M::RowT>;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        let normal_col = self.oracle.column(col.0);
        normal_col.map(|(coeff, row)| (coeff, Reverse(row)))
    }
}

impl<M> HasRowFiltration for ReverseMatrix<M>
where
    M: MatrixOracle + HasRowFiltration,
{
    type FiltrationT = Reverse<M::FiltrationT>;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        Reverse(self.oracle.filtration_value(row.0))
    }

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>
    {
        let normal_col = self.oracle.column_with_filtration(col.0);
        normal_col.map(|entry| ColumnEntry {
            filtration_value: Reverse(entry.filtration_value),
            row_index: Reverse(entry.row_index),
            coeff: entry.coeff,
        })
    }

    // Reversing the order swaps the first and last columns in each row
    fn first_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.oracle.last_column_in_row(row.0).map(Reverse)
    }

    fn last_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.oracle.first_column_in_row(row.0).map(Reverse)
    }

    // Reversing the order reverses each column
    fn sorted_column(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        let column = self.oracle.sorted_column_ascending(col.0);
        column.map(|column| {
            column.map(|entry| ColumnEntry {
                filtration_value: Reverse(entry.filtration_value),
                row_index: Reverse(entry.row_index),
                coeff: entry.coeff,
            })
        })
    }

    fn sorted_column_ascending(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        let column = self.oracle.sorted_column(col.0);
        column.map(|column| {
            column.map(|entry| ColumnEntry {
                filtration_value: Reverse(entry.filtration_value),
                row_index: Reverse(entry.row_index),
                coeff: entry.coeff,
            })
        })
    }

    // Reversing the order swaps the largest and smallest entries of each column
    fn column_pivot(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        self.oracle.column_min(col.0).map(|entry| ColumnEntry {
            filtration_value: Reverse(entry.filtration_value),
            row_index: Reverse(entry.row_index),
            coeff: entry.coeff,
        })
    }

    fn column_min(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        self.oracle.column_pivot(col.0).map(|entry| ColumnEntry {
            filtration_value: Reverse(entry.filtration_value),
            row_index: Reverse(entry.row_index),
            coeff: entry.coeff,
        })
    }
}

impl<M> HasColBasis for ReverseMatrix<M>
where
    M: MatrixOracle + HasColBasis,
{
    type BasisT<'a>
        = ReverseBasis<M::BasisT<'a>>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisT<'_> {
        ReverseBasis(self.oracle.basis())
    }
}

impl<M> HasSplitColBasis for ReverseMatrix<M>
where
    M: MatrixOracle + HasSplitColBasis,
{
    type SubBasisT<'a>
        = ReverseBasis<M::SubBasisT<'a>>
    where
        Self: 'a;

    fn n_dimensions(&self) -> usize {
        self.oracle.n_dimensions()
    }

    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_> {
        ReverseBasis(self.oracle.sub_basis(dimension))
    }
}

/// Return type of [`ReverseMatrix::basis`].
///
/// This is just a by-value wrapper around another basis view, so nothing here needs `unsafe`.
#[derive(Clone, Copy, Debug)]
pub struct ReverseBasis<B>(pub B);

impl<B> ReverseBasis<B> {
    /// Override the usual [`ColBasis::unreverse`].
    /// This is more efficient because it just takes the inner basis out of its [`ReverseBasis`] wrapper.
    pub fn unreverse(self) -> B {
        self.0
    }
}

impl<B> ColBasis for ReverseBasis<B>
where
    B: ColBasis,
{
    type ElemT = Reverse<B::ElemT>;

    fn element(&self, index: usize) -> Self::ElemT {
        Reverse(self.0.element(self.0.size() - 1 - index))
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<B> SplitByDimension for ReverseBasis<B>
where
    B: SplitByDimension,
{
    type SubBasisT = ReverseBasis<B::SubBasisT>;

    fn n_dimensions(&self) -> usize {
        self.0.n_dimensions()
    }

    fn in_dimension(self, dimension: usize) -> Self::SubBasisT {
        ReverseBasis(self.0.in_dimension(dimension))
    }
}

// ====== UnreverseMatrix ======================
// Used to un-reverse a matrix/basis that is indexed + filtered by Reverse<T>
// Returns a matrix/basis that is index + filtered by T
// If you naively call reverse again, you will not be able to multiply matrices because
// Reverse<Reverse<T>> != T
// TODO: Could we fix this by just requiring Into<ColT> on traits? Or maybe we only need it on multiply?

/// Return type of [`MatrixOracle::unreverse`].
#[derive(Clone, Copy, Debug)]
pub struct UnreverseMatrix<M> {
    pub(crate) oracle: M,
}

impl<M, ColT, RowT> MatrixOracle for UnreverseMatrix<M>
where
    M: MatrixOracle<ColT = Reverse<ColT>, RowT = Reverse<RowT>>,
    RowT: BasisElement,
    ColT: BasisElement,
{
    type CoefficientField = M::CoefficientField;

    type ColT = ColT;

    type RowT = RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        let reversed = self.oracle.column(Reverse(col));
        reversed.map(|(coeff, rev_row)| (coeff, rev_row.0))
    }
}

impl<M, ColT, RowT, FilT> HasRowFiltration for UnreverseMatrix<M>
where
    M: MatrixOracle<ColT = Reverse<ColT>, RowT = Reverse<RowT>>,
    RowT: BasisElement,
    ColT: BasisElement,
    M: HasRowFiltration<FiltrationT = Reverse<FilT>>,
    FilT: FiltrationValue,
{
    type FiltrationT = FilT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.oracle.filtration_value(Reverse(row)).0
    }

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<FilT, RowT, Self::CoefficientField>> {
        let rev_col = self.oracle.column_with_filtration(Reverse(col));
        rev_col.map(|entry| ColumnEntry {
            filtration_value: entry.filtration_value.0,
            row_index: entry.row_index.0,
            coeff: entry.coeff,
        })
    }

    // Reversing the order swaps the first and last columns in each row
    fn first_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.oracle
            .last_column_in_row(Reverse(row))
            .map(|col| col.0)
    }

    fn last_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.oracle
            .first_column_in_row(Reverse(row))
            .map(|col| col.0)
    }

    // Reversing the order reverses each column
    fn sorted_column(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        let column = self.oracle.sorted_column_ascending(Reverse(col));
        column.map(|column| {
            column.map(|entry| ColumnEntry {
                filtration_value: entry.filtration_value.0,
                row_index: entry.row_index.0,
                coeff: entry.coeff,
            })
        })
    }

    fn sorted_column_ascending(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        let column = self.oracle.sorted_column(Reverse(col));
        column.map(|column| {
            column.map(|entry| ColumnEntry {
                filtration_value: entry.filtration_value.0,
                row_index: entry.row_index.0,
                coeff: entry.coeff,
            })
        })
    }

    // Reversing the order swaps the largest and smallest entries of each column
    fn column_pivot(&self, col: Self::ColT) -> Option<ColumnEntry<FilT, RowT, Self::CoefficientField>> {
        self.oracle.column_min(Reverse(col)).map(|entry| ColumnEntry {
            filtration_value: entry.filtration_value.0,
            row_index: entry.row_index.0,
            coeff: entry.coeff,
        })
    }

    fn column_min(&self, col: Self::ColT) -> Option<ColumnEntry<FilT, RowT, Self::CoefficientField>> {
        self.oracle.column_pivot(Reverse(col)).map(|entry| ColumnEntry {
            filtration_value: entry.filtration_value.0,
            row_index: entry.row_index.0,
            coeff: entry.coeff,
        })
    }
}

impl<M, ColT, RowT> HasColBasis for UnreverseMatrix<M>
where
    M: MatrixOracle<ColT = Reverse<ColT>, RowT = Reverse<RowT>>,
    RowT: BasisElement,
    ColT: BasisElement,
    M: HasColBasis,
{
    type BasisT<'a>
        = UnreverseBasis<M::BasisT<'a>>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisT<'_> {
        UnreverseBasis(self.oracle.basis())
    }
}

impl<M, ColT, RowT> HasSplitColBasis for UnreverseMatrix<M>
where
    M: MatrixOracle<ColT = Reverse<ColT>, RowT = Reverse<RowT>>,
    RowT: BasisElement,
    ColT: BasisElement,
    M: HasSplitColBasis,
{
    type SubBasisT<'a>
        = UnreverseBasis<M::SubBasisT<'a>>
    where
        Self: 'a;

    fn n_dimensions(&self) -> usize {
        self.oracle.n_dimensions()
    }

    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_> {
        UnreverseBasis(self.oracle.sub_basis(dimension))
    }
}

/// Return type of [`UnreverseMatrix::basis`].
#[derive(Clone, Copy, Debug)]
pub struct UnreverseBasis<B>(pub B);

impl<B, T> ColBasis for UnreverseBasis<B>
where
    B: ColBasis<ElemT = Reverse<T>>,
    T: BasisElement,
{
    type ElemT = T;

    fn element(&self, index: usize) -> Self::ElemT {
        let reversed = self.0.element(self.0.size() - 1 - index);
        reversed.0
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

impl<B, T> SplitByDimension for UnreverseBasis<B>
where
    B: ColBasis<ElemT = Reverse<T>> + SplitByDimension,
    B::SubBasisT: ColBasis<ElemT = Reverse<T>>,
    T: BasisElement,
{
    type SubBasisT = UnreverseBasis<B::SubBasisT>;

    fn n_dimensions(&self) -> usize {
        self.0.n_dimensions()
    }

    fn in_dimension(self, dimension: usize) -> Self::SubBasisT {
        UnreverseBasis(self.0.in_dimension(dimension))
    }
}

// ========= Attaching a transpose ==============================

/// Return type of [`MatrixOracle::with_transpose`]: a matrix together with an oracle for its transpose,
/// which is used to find the first and last columns in each row (and hence apparent pairs).
/// Everything else is forwarded to the matrix.
#[derive(Debug, Clone, Copy)]
pub struct WithTranspose<M, T> {
    /// The matrix.
    pub matrix: M,
    /// An oracle for the transpose: column `r` of `transpose` lists the non-zero entries in row `r` of `matrix`.
    pub transpose: T,
}

impl<M: MatrixOracle, T> MatrixOracle for WithTranspose<M, T> {
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        self.matrix.column(col)
    }
}

impl<M, T> WithTranspose<M, T>
where
    M: HasRowFiltration + SquareMatrix,
    T: MatrixOracle<CoefficientField = M::CoefficientField, ColT = M::RowT, RowT = M::ColT>,
{
    /// The least (or greatest) column with a non-zero entry in `row`, in the order on (filtration value, column).
    fn extreme_column_in_row(&self, row: M::RowT, greatest: bool) -> Option<M::ColT> {
        let key = |col: &M::ColT| (self.matrix.filtration_value(col.clone()), col.clone());
        let better = |candidate: &(M::FiltrationT, M::ColT),
                      current: &(M::FiltrationT, M::ColT)| {
            if greatest {
                candidate > current
            } else {
                candidate < current
            }
        };
        // The best column seen so far, and the summed coefficient of every entry in it (None if they cancel)
        let mut best: Option<(M::FiltrationT, M::ColT)> = None;
        let mut sum: Option<M::CoefficientField> = None;
        for (coeff, col) in self.transpose.column(row.clone()) {
            let k = key(&col);
            match &best {
                Some(current) if *current == k => sum = coeff + sum,
                Some(current) if !better(&k, current) => {}
                _ => {
                    best = Some(k);
                    sum = Some(coeff);
                }
            }
        }
        match (best, sum) {
            (None, _) => None,
            (Some((_f, col)), Some(_)) => Some(col),
            // The best column cancelled out, so sum everything up properly
            (Some(_), None) => {
                let mut entries: Vec<_> = self
                    .transpose
                    .column(row)
                    .map(|(coeff, col)| (key(&col), coeff))
                    .collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                if greatest {
                    entries.reverse();
                }
                let mut i = 0;
                while i < entries.len() {
                    let mut total = Some(entries[i].1);
                    let mut j = i + 1;
                    while j < entries.len() && entries[j].0 == entries[i].0 {
                        total = entries[j].1 + total;
                        j += 1;
                    }
                    if total.is_some() {
                        return Some(entries[i].0 .1.clone());
                    }
                    i = j;
                }
                None
            }
        }
    }
}

impl<M, T> HasRowFiltration for WithTranspose<M, T>
where
    M: HasRowFiltration + SquareMatrix,
    T: MatrixOracle<CoefficientField = M::CoefficientField, ColT = M::RowT, RowT = M::ColT>,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.matrix.filtration_value(row)
    }

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>
    {
        self.matrix.column_with_filtration(col)
    }

    fn column_pivot(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        self.matrix.column_pivot(col)
    }

    fn column_min(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        self.matrix.column_min(col)
    }

    // Use the matrix's own (presumably faster) implementation if it has one, otherwise scan the transpose
    fn first_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.matrix
            .first_column_in_row(row.clone())
            .or_else(|| self.extreme_column_in_row(row, false))
    }

    fn last_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        self.matrix
            .last_column_in_row(row.clone())
            .or_else(|| self.extreme_column_in_row(row, true))
    }

    fn sorted_column(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        self.matrix.sorted_column(col)
    }

    fn sorted_column_ascending(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        self.matrix.sorted_column_ascending(col)
    }
}

impl<M: HasColBasis, T> HasColBasis for WithTranspose<M, T> {
    type BasisT<'a>
        = M::BasisT<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisT<'_> {
        self.matrix.basis()
    }
}

impl<M: HasSplitColBasis, T> HasSplitColBasis for WithTranspose<M, T> {
    type SubBasisT<'a>
        = M::SubBasisT<'a>
    where
        Self: 'a;

    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_> {
        self.matrix.sub_basis(dimension)
    }

    fn n_dimensions(&self) -> usize {
        self.matrix.n_dimensions()
    }
}

impl<M, T> HasTranspose for WithTranspose<M, T>
where
    M: MatrixOracle,
    T: MatrixOracle<CoefficientField = M::CoefficientField, ColT = M::RowT, RowT = M::ColT>,
{
    type TransposeT<'a>
        = &'a T
    where
        Self: 'a;

    fn transpose(&self) -> Self::TransposeT<'_> {
        &self.transpose
    }
}
