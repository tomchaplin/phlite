//! The core framework for implementing lazy oracles for sparse matrices.
//! Provides matrix traits that should be implemented by users.
//! Also provides various wrappers to attach additional data to matrices, change their indexing types or multiply two matrices.

// TODO: Add reasonable constraints to reverse and unreverse methods on bases and matrices

use std::hash::Hash;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;
use std::{cmp::Reverse, collections::HashMap};

use itertools::equal;
use ordered_float::NotNan;

use crate::{
    columns::{BHCol, ColumnEntry},
    fields::NonZeroCoefficient,
    PhliteError,
};

use self::adaptors::{
    MatrixWithBasis, ReverseBasis, ReverseMatrix, UnreverseBasis, UnreverseMatrix,
    UsingColBasisIndex, WithFuncFiltration, WithSubBasis, WithTrivialFiltration,
};

pub mod adaptors;
pub mod combinators;
pub mod implementors;

// ========= Traits for matrix indices and filtrations =========

pub trait BasisElement: Ord + Clone {}
pub trait FiltrationT: Ord + Clone {}

// Default implementors

// NOTE: We do not blanket impl because we want to catch errors
//       where we index with the wrong type.

impl BasisElement for usize {}
impl BasisElement for isize {}
impl FiltrationT for NotNan<f32> {}
impl FiltrationT for NotNan<f64> {}
impl FiltrationT for usize {}
impl FiltrationT for isize {}
impl FiltrationT for () {}

impl<T> BasisElement for Reverse<T> where T: BasisElement {}
impl<T> FiltrationT for Reverse<T> where T: FiltrationT {}

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
    fn column(&self, col: Self::ColT)
        -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)>;

    /// Checks that the matricies are equal on the specified col, ignoring ordering due to filtration values
    fn eq_on_col<M2>(&self, other: &M2, col: Self::ColT) -> bool
    where
        Self: Sized,
        M2: MatrixOracle<
            CoefficientField = Self::CoefficientField,
            ColT = Self::ColT,
            RowT = Self::RowT,
        >,
    {
        let self_trivial = self.with_trivial_filtration();
        let other_trivial = other.with_trivial_filtration();

        let mut self_col = self_trivial.build_bhcol(col.clone());
        let self_col_sorted = self_col
            .drain_sorted()
            .map(Into::<(Self::CoefficientField, Self::RowT, ())>::into);
        let mut other_col = other_trivial.build_bhcol(col);
        let other_col_sorted = other_col
            .drain_sorted()
            .map(Into::<(Self::CoefficientField, Self::RowT, ())>::into);

        equal(self_col_sorted, other_col_sorted)
    }

    fn with_trivial_filtration(self) -> WithTrivialFiltration<Self>
    where
        Self: Sized,
    {
        WithTrivialFiltration { oracle: self }
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

    fn with_basis<B>(self, basis: B) -> MatrixWithBasis<Self, B>
    where
        Self: Sized,
        B: ColBasis<ElemT = Self::ColT>,
    {
        MatrixWithBasis {
            matrix: self,
            basis,
        }
    }

    fn using_col_basis_index(self) -> UsingColBasisIndex<Self>
    where
        Self: Sized + HasColBasis,
    {
        UsingColBasisIndex { oracle: self }
    }

    fn reverse(self) -> ReverseMatrix<Self>
    where
        Self: Sized,
    {
        ReverseMatrix { oracle: self }
    }

    fn unreverse(self) -> UnreverseMatrix<Self>
    where
        Self: Sized,
    {
        UnreverseMatrix { oracle: self }
    }
}

// ======== Square matrices ====================================

pub trait SquareMatrix: MatrixOracle<ColT = <Self as MatrixOracle>::RowT> {}

impl<M> SquareMatrix for M where M: MatrixOracle<ColT = <Self as MatrixOracle>::RowT> {}

// ======== Default implementors ===============================

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
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        (*self).column(col)
    }
}

impl<M> MatrixOracle for Rc<M>
where
    M: MatrixOracle,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        (**self).column(col)
    }
}

impl<M> MatrixOracle for Arc<M>
where
    M: MatrixOracle,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        (**self).column(col)
    }
}

impl<M> MatrixOracle for Box<M>
where
    M: MatrixOracle,
{
    type CoefficientField = M::CoefficientField;
    type ColT = M::ColT;
    type RowT = M::RowT;

    fn column(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = (Self::CoefficientField, Self::RowT)> {
        (**self).column(col)
    }
}

// ======== Filtration on rows to order them ===================

pub trait HasRowFiltration: MatrixOracle {
    type FiltrationT: FiltrationT;
    fn filtration_value(&self, row: Self::RowT) -> Result<Self::FiltrationT, PhliteError>;

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>
    {
        let column = self.column(col);
        column.map(|(coeff, row_index)| {
            let f_val = self
                .filtration_value(row_index.clone())
                .expect("Rows should all have filtration values");
            (coeff, row_index, f_val).into()
        })
    }

    fn empty_bhcol(&self) -> BHCol<Self::FiltrationT, Self::RowT, Self::CoefficientField> {
        BHCol::default()
    }

    fn build_bhcol(
        &self,
        col: Self::ColT,
    ) -> BHCol<Self::FiltrationT, Self::RowT, Self::CoefficientField> {
        let mut output = self.empty_bhcol();
        output.add_entries(self.column_with_filtration(col));
        output
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

// ======== Ordered basis on columns + arbitrary access ========

pub trait ColBasis {
    type ElemT: BasisElement;
    fn element(&self, index: usize) -> Self::ElemT;
    fn size(&self) -> usize;

    fn construct_reverse_lookup(&self) -> HashMap<Self::ElemT, usize>
    where
        Self::ElemT: Hash,
    {
        (0..(self.size()))
            .map(|idx| (self.element(idx), idx))
            .collect()
    }

    fn reverse(self) -> ReverseBasis<Self>
    where
        Self: Copy,
    {
        ReverseBasis(self)
    }

    fn unreverse(self) -> UnreverseBasis<Self>
    where
        Self: Copy,
    {
        UnreverseBasis(self)
    }
}

impl<T> ColBasis for Vec<T>
where
    T: BasisElement,
{
    type ElemT = T;

    fn element(&self, index: usize) -> Self::ElemT {
        self[index].clone()
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<'a, T> ColBasis for &'a T
where
    T: ColBasis,
{
    type ElemT = T::ElemT;

    fn element(&self, index: usize) -> Self::ElemT {
        (*self).element(index)
    }

    fn size(&self) -> usize {
        (*self).size()
    }
}

pub trait HasColBasis: MatrixOracle {
    type BasisT: ColBasis<ElemT = Self::ColT>;
    type BasisRef<'a>: Deref<Target = Self::BasisT>
    where
        Self: 'a;

    fn basis<'a>(&'a self) -> Self::BasisRef<'a>;

    fn sub_matrix_in_dimension(self, dimension: usize) -> WithSubBasis<Self>
    where
        Self: MatrixOracle + HasColBasis<BasisT: SplitByDimension> + Sized,
    {
        WithSubBasis {
            oracle: self,
            dimension,
        }
    }
}

impl<'a, T> HasColBasis for &'a T
where
    T: HasColBasis,
{
    type BasisT = T::BasisT;
    type BasisRef<'b>
        = T::BasisRef<'b>
    where
        Self: 'b;

    fn basis<'b>(&'b self) -> Self::BasisRef<'b> {
        (*self).basis()
    }
}

// TODO: This puts a pretty strong constraint on the ColBasis
// In particular at must own each of the SubBasisT pre-constructed and then piece them together in order to extract its elements.
// Is there a better way to do this?
// TODO: Repeat what we did with HasColBasis, allow arbitrary ref type
pub trait SplitByDimension: ColBasis {
    type SubBasisT: ColBasis<ElemT = Self::ElemT>;

    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT;
}

impl<'a, T> SplitByDimension for &'a T
where
    T: SplitByDimension,
{
    type SubBasisT = T::SubBasisT;

    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT {
        (*self).in_dimension(dimension)
    }
}
