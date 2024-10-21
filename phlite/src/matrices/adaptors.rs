//! Contains matrix adaptors which alter the behaviour of existing matrices, e.g. reversing indices or adding a filtration/basis.
//!
//! All structs in this submodule should be constructed by calling the relevant provided methods on [`MatrixOracle`] or [`HasColBasis`].
//! Their utility is explained in the documentation for these constructors.

// ======== Matrix oracle adaptors ============================

// ====== WithTrivialFiltration ================

use std::{cmp::Reverse, marker::PhantomData, ops::Deref};

use crate::columns::{BHCol, ColumnEntry};

use super::{
    BasisElement, ColBasis, FiltrationValue, HasColBasis, HasRowFiltration, MatrixOracle,
    SplitByDimension,
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
    type BasisT = M::BasisT;
    type BasisRef<'a>
        = M::BasisRef<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisRef<'_> {
        self.oracle.basis()
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
    type BasisT = M::BasisT;
    type BasisRef<'a>
        = M::BasisRef<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisRef<'_> {
        self.oracle.basis()
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
    type BasisT = M::BasisT;
    type BasisRef<'a>
        = M::BasisRef<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisRef<'_> {
        self.oracle.basis()
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
}

impl<M, B> HasColBasis for MatrixWithBasis<M, B>
where
    M: MatrixOracle,
    B: ColBasis<ElemT = M::ColT>,
{
    type BasisT = B;
    type BasisRef<'a>
        = &'a B
    where
        Self: 'a;

    fn basis(&self) -> &B {
        &self.basis
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

/// Return type of [`HasColBasis::sub_matrix_in_dimension`].
#[derive(Clone, Copy, Debug)]
pub struct WithSubBasis<M: MatrixOracle + HasColBasis<BasisT: SplitByDimension>> {
    pub(crate) oracle: M,
    pub(crate) dimension: usize,
}

impl<M> MatrixOracle for WithSubBasis<M>
where
    M: MatrixOracle + HasColBasis<BasisT: SplitByDimension>,
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
    M: MatrixOracle + HasColBasis<BasisT: SplitByDimension> + HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.oracle.filtration_value(row)
    }
}

struct SubBasisRefInner<'a, B, Ref> {
    total_basis: Ref,
    dimension: usize,
    phantom: PhantomData<&'a B>,
}

impl<B, Ref> Deref for SubBasisRefInner<'_, B, Ref>
where
    Ref: Deref<Target = B>,
    B: SplitByDimension,
{
    type Target = B::SubBasisT;
    fn deref(&self) -> &B::SubBasisT {
        self.total_basis.deref().in_dimension(self.dimension)
    }
}

/// Return type of [`WithSubBasis::basis`].
pub struct SubBasisRef<'a, M>(SubBasisRefInner<'a, M::BasisT, M::BasisRef<'a>>)
where
    M: HasColBasis + 'a;

impl<'a, M> Deref for SubBasisRef<'a, M>
where
    M: HasColBasis<BasisT: SplitByDimension> + 'a,
{
    type Target = <M::BasisT as SplitByDimension>::SubBasisT;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<M> HasColBasis for WithSubBasis<M>
where
    M: MatrixOracle + HasColBasis<BasisT: SplitByDimension>,
{
    type BasisT = <M::BasisT as SplitByDimension>::SubBasisT;
    type BasisRef<'a>
        = SubBasisRef<'a, M>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisRef<'_> {
        SubBasisRef(SubBasisRefInner {
            total_basis: self.oracle.basis(),
            dimension: self.dimension,
            phantom: PhantomData,
        })
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
}

impl<M> HasColBasis for ReverseMatrix<M>
where
    M: MatrixOracle + HasColBasis,
{
    type BasisT = ReverseBasis<M::BasisT>;
    type BasisRef<'a>
        = ReverseBasis<M::BasisRef<'a>>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisRef<'_> {
        ReverseBasis(self.oracle.basis())
    }
}

/// Return type of [`ReverseMatrix::basis`].
///
/// Note that layout is `repr(transparent)` so can transmute references.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct ReverseBasis<B>(pub B);

impl<B, Ref> Deref for ReverseBasis<Ref>
where
    Ref: Deref<Target = B>,
{
    type Target = ReverseBasis<B>;

    fn deref(&self) -> &Self::Target {
        let original_basis: *const _ = self.0.deref();
        let rev_basis = original_basis.cast();
        // Is this sound if Ref is not &B ?
        // Should be because deref gives us a &B
        unsafe { &*rev_basis }
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

    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT {
        let original_basis: *const _ = self.0.in_dimension(dimension);
        let rev_basis = original_basis.cast();
        unsafe { &*rev_basis }
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
}

impl<M, ColT, RowT, BasisT> HasColBasis for UnreverseMatrix<M>
where
    M: MatrixOracle<ColT = Reverse<ColT>, RowT = Reverse<RowT>>,
    RowT: BasisElement,
    ColT: BasisElement,
    M: HasColBasis<BasisT = BasisT>,
    BasisT: ColBasis<ElemT = Reverse<ColT>>,
{
    type BasisT = UnreverseBasis<BasisT>;
    type BasisRef<'a>
        = UnreverseBasis<M::BasisRef<'a>>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisRef<'_> {
        UnreverseBasis(self.oracle.basis())
        //let rev_basis: *const _ = self.oracle.basis();
        //let unrev_basis = rev_basis.cast();
        //unsafe { &*unrev_basis }
    }
}

impl<B, Ref> Deref for UnreverseBasis<Ref>
where
    Ref: Deref<Target = B>,
{
    type Target = UnreverseBasis<B>;

    fn deref(&self) -> &Self::Target {
        let rev_basis: *const _ = self.0.deref();
        let unrev_basis = rev_basis.cast();
        unsafe { &*unrev_basis }
    }
}

/// Return type of [`UnreverseMatrix::basis`].
///
/// Note that layout is `repr(transparent)` so can transmute references.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
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

impl<B, T, SbT> SplitByDimension for UnreverseBasis<B>
where
    B: ColBasis<ElemT = Reverse<T>>,
    B: SplitByDimension<SubBasisT = SbT>,
    SbT: ColBasis<ElemT = Reverse<T>>,
    T: BasisElement,
{
    type SubBasisT = UnreverseBasis<SbT>;
    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT {
        let rev_basis: *const _ = self.0.in_dimension(dimension);
        let unrev_basis = rev_basis.cast();
        unsafe { &*unrev_basis }
    }
}
