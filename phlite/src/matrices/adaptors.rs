//! Contains matrix adaptors which alter the behaviour of existing matrices, e.g. reversing indices or adding a filtration/basis.
//!
//! All structs in this submodule should be constructed by calling the relevant provided methods on [`MatrixOracle`] or [`HasColBasis`].
//! Their utility is explained in the documentation for these constructors.

// ======== Matrix oracle adaptors ============================

// ====== WithTrivialFiltration ================

use std::{
    cell::{Ref, RefCell},
    cmp::Reverse,
    collections::HashMap,
    hash::Hash,
};

use crate::columns::{BHCol, ColumnEntry};

use super::{
    BasisElement, ColBasis, FiltrationValue, HasColBasis, HasRowFiltration, HasSplitColBasis,
    MatrixOracle, SplitByDimension,
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

// ====== WithCache ============================

/// Return type of [`MatrixOracle::cache_cols`], can also be constructed manually.
///
/// Caches any calls to [`MatrixOracle::column`] in an internal [`HashMap`].
#[derive(Clone, Debug)]
pub struct WithCachedCols<M: MatrixOracle> {
    /// The underlying matrix being wrapped.
    /// Non-cached columns fall back to this, as well as implementations of [`HasColBasis`] and [`HasRowFiltration`].
    pub oracle: M,
    /// The internal [`column`](MatrixOracle::column) cache, stored behind a [`RefCell`] to allow interior mutability.
    pub cache: RefCell<HashMap<M::ColT, Vec<(M::CoefficientField, M::RowT)>>>,
}

// Essentialy https://stackoverflow.com/questions/33541492/returning-iterator-of-a-vec-in-a-refcell
struct CachedColIter<'a, T> {
    inner: Option<Ref<'a, [T]>>,
}

impl<'a, T> Iterator for CachedColIter<'a, T> {
    type Item = Ref<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.take() {
            Some(borrow) => match *borrow {
                [] => None,
                [_, ..] => {
                    let (head, tail) = Ref::map_split(borrow, |slice| (&slice[0], &slice[1..]));
                    self.inner.replace(tail);
                    Some(head)
                }
            },
            None => None,
        }
    }
}

impl<M> MatrixOracle for WithCachedCols<M>
where
    M: MatrixOracle<ColT: Hash>,
{
    type CoefficientField = M::CoefficientField;

    type ColT = M::ColT;

    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        // Compute column if not yet cached
        if !self.cache.borrow().contains_key(&col) {
            let col_as_vec: Vec<_> = self.oracle.column(col.clone()).collect();
            self.cache
                .borrow_mut()
                .insert(col.clone(), col_as_vec.clone());
        }
        // Get a borrow to the column
        let cache_borrow = self.cache.borrow();
        let col_borrow = Ref::map(cache_borrow, |c| {
            &c.get(&col).expect("Just populated key")[..]
        });
        // Return an iterator
        CachedColIter {
            inner: Some(col_borrow),
        }
        .map(|entry| entry.clone())
    }
}

impl<M> HasRowFiltration for WithCachedCols<M>
where
    M: HasRowFiltration,
    M::ColT: Hash,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
        self.oracle.filtration_value(row)
    }
}

impl<M> HasColBasis for WithCachedCols<M>
where
    M: HasColBasis,
    M::ColT: Hash,
{
    type BasisT<'a>

        = M::BasisT<'a>
    where
        Self: 'a;

    fn basis(&self) -> Self::BasisT<'_> {
        self.oracle.basis()
    }
}

impl<M> HasSplitColBasis for WithCachedCols<M>
where
    M: HasSplitColBasis,
    M::ColT: Hash,
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
