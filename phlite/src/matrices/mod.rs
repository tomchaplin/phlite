//! The core framework for implementing lazy oracles for sparse matrices.
//!
//! This module provides matrix traits that should be implemented by users.
//! Also provides various wrappers to attach additional data to matrices, change their indexing types or multiply two matrices.

// TODO: Add reasonable constraints to reverse and unreverse methods on bases and matrices

use std::hash::Hash;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;
use std::{cmp::Reverse, collections::HashMap};

use adaptors::Consolidator;
use itertools::equal;
use ordered_float::NotNan;

use crate::{
    columns::{BHCol, ColumnEntry},
    fields::NonZeroCoefficient,
};

use self::adaptors::{
    MatrixWithBasis, ReverseBasis, ReverseMatrix, UnreverseBasis, UnreverseMatrix,
    UsingColBasisIndex, WithFuncFiltration, WithSubBasis, WithTrivialFiltration,
};

pub mod adaptors;
pub mod combinators;
pub mod implementors;
#[cfg(test)]
mod tests;

// ========= Traits for matrix indices and filtrations =========

/// Row and column indices must explicity implement this (see [`MatrixOracle`](MatrixOracle::ColT)).
///
/// Ideally [`clone`](Clone::clone) should be *very* cheap as it is called regularly, [`Copy`] would be ideal.
pub trait BasisElement: Ord + Clone {}
/// Filtration types must explicity implement this (see [`HasRowFiltration`](HasRowFiltration::FiltrationT)).
///
/// Ideally [`clone`](Clone::clone) should be *very* cheap as it is called regularly, [`Copy`] would be ideal.
pub trait FiltrationValue: Ord + Clone {}

// Default implementors

// NOTE: We do not blanket impl because we want to catch errors
//       where we index with the wrong type.

impl BasisElement for usize {}
impl BasisElement for isize {}
impl FiltrationValue for NotNan<f32> {}
impl FiltrationValue for NotNan<f64> {}
impl FiltrationValue for usize {}
impl FiltrationValue for isize {}
impl FiltrationValue for () {}

impl<T> BasisElement for Reverse<T> where T: BasisElement {}
impl<T> FiltrationValue for Reverse<T> where T: FiltrationValue {}

// ======== Abstract matrix oracle trait =======================

/// A type implementing [`MatrixOracle`] represents a linear transformation, together with a *choice* of basis for both the target and domain.
///
/// First, there are a number of associated types:
/// * [`ColT`](MatrixOracle::ColT) represents elements in the domain basis;
/// * [`RowT`](MatrixOracle::RowT) represents elements in the target basis;
/// * [`CoefficientField`](MatrixOracle::CoefficientField) represents the non-zero elements in the field over which we are doing linear algebra.
///
/// In order to implement [`MatrixOracle`] for your matrix, you must decide on these types and also provide an implementation of [`column`](MatrixOracle::column).
/// Some important things to note:
/// * While you must have enough information in `T` to have *chosen* the basis, you do not necessarily need the basis to hand in order to implement [`MatrixOracle`]. Indeed, it is probably more memory-efficient *not* to store the row basis, whilst the column basis will be provided via the separate [`HasColBasis`] trait.
/// * It is up to you to ensure that we never construct a [`ColT`](MatrixOracle::ColT) or [`RowT`](MatrixOracle::RowT) that doesn't correspond to an element of the chosen bases.
/// * An object of type [`CoefficientField`](MatrixOracle::CoefficientField) should represent a *non-zero* coefficient; ideally `0` is un-representable in this type. Since we essentially represent our columns as linear combinations of the row basis, `0` is represented by the *absence* of that basis element in the combination, i.e. `None` rather than `Some`. A good choice is [`Z2`](crate::fields::Z2).
pub trait MatrixOracle {
    /// Represents the non-zero elements in the field over which we are doing linear algebra.   
    type CoefficientField: NonZeroCoefficient;
    /// Represents elements in the domain basis.
    type ColT: BasisElement;
    /// Represents elements in the target basis.
    type RowT: BasisElement;

    /// Given an element `col` in the domain basis, express the image of `col` under the linear transformation as a linear combination of elements in the target basis.
    /// You should provide this combination is an iterator of `(coeff, row)` tuples where each `row` is an element of the target basis.
    /// If `coeff` is `0` then **omit this term** from the linear combination.
    ///
    /// # Performance Notes
    /// It is dis-advantageous to produce the rows in ascending order because inserting into binary heaps would require traversing the full height (see [`BinaryHeap::push`](std::collections::BinaryHeap::push)).
    /// Since checking and sorting by filtration values is typically slow, prefer to produce in descending order with respect to the ordering on [`RowT`](Self::RowT).
    /// Lower bounds on iterator size provided via [`Iterator::size_hint`] will be used to preallocate.
    ///
    /// In principle, you can repeat the same [`RowT`](Self::RowT) multiple times, but fewer terms is better for memory and performance.
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

    // TODO: Change all these to accept &self
    // Then implement constructors for each of the structs so we can consume self if we want to.

    /// Endows `self` with a filtration in which all rows have the same filtration value: `()`.
    fn with_trivial_filtration(self) -> WithTrivialFiltration<Self>
    where
        Self: Sized,
    {
        WithTrivialFiltration { oracle: self }
    }

    /// Endows `self` with the filtration given by the provided `filtration_function`.
    fn with_filtration<FT: FiltrationValue, F: Fn(Self::RowT) -> FT>(
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

    /// Endows `self` with the basis `basis`.
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

    /// Turns `self` into a matrix indexed by `usize`, using the attached basis.
    fn using_col_basis_index(self) -> UsingColBasisIndex<Self>
    where
        Self: Sized + HasColBasis,
    {
        UsingColBasisIndex { oracle: self }
    }

    /// Takes the anti-transpose matrix.
    /// Both the indices and the filtration values are now the [`Reverse`] of their previous type.
    fn reverse(self) -> ReverseMatrix<Self>
    where
        Self: Sized,
    {
        ReverseMatrix { oracle: self }
    }

    /// Takes the anti-transpose matrix
    /// Can only be called when the indices are of the form `Reverse<ColT>` and `Reverse<RowT>`
    /// Returns a matrix indexed by `ColT` and `RowT`.
    /// Additionally, the filtration value can be 'unreversed'.
    /// Can be useful when `self` was obtained by reducing a [`reverse`](Self::reverse)d matrix.
    fn unreverse<ColT, RowT>(self) -> UnreverseMatrix<Self>
    where
        Self: Sized,
        Self: MatrixOracle<RowT = Reverse<RowT>, ColT = Reverse<ColT>>,
    {
        UnreverseMatrix { oracle: self }
    }

    /// When accessing a [`column`](Self::column) of the [`consolidate`](Self::consolidate)d matrix, the corresponding column of `self` will be requested and stored into a binary heap.
    /// An iterator through this binary heap is then returned.
    ///
    /// Essentailly, if we view the output of [`column`](Self::column) as a [`CF`](Self::CoefficientField)-weighted sum of row basis elements, this simplifies this sum by combining all of the summand with the same basis element.
    /// This can reduce memory usage when computing products/sums of large matrices (see Flagser's dynamic heap for an alternative approach).
    fn consolidate(self) -> Consolidator<Self>
    where
        Self: Sized,
    {
        Consolidator { oracle: self }
    }
}

// ======== Square matrices ====================================

/// Alias for a matrix whose column and row types match.
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
    type FiltrationT: FiltrationValue;
    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT;

    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>
    {
        let column = self.column(col);
        column.map(|(coeff, row_index)| {
            let f_val = self.filtration_value(row_index.clone());
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

impl<M> HasRowFiltration for &M
where
    M: HasRowFiltration,
{
    type FiltrationT = M::FiltrationT;

    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT {
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

impl<T> ColBasis for &T
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

    fn basis(&self) -> Self::BasisRef<'_>;

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

impl<T> HasColBasis for &T
where
    T: HasColBasis,
{
    type BasisT = T::BasisT;
    type BasisRef<'b>
        = T::BasisRef<'b>
    where
        Self: 'b;

    fn basis(&self) -> Self::BasisRef<'_> {
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

impl<T> SplitByDimension for &T
where
    T: SplitByDimension,
{
    type SubBasisT = T::SubBasisT;

    fn in_dimension(&self, dimension: usize) -> &Self::SubBasisT {
        (*self).in_dimension(dimension)
    }
}
