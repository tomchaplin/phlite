//! The core framework for implementing lazy oracles for sparse matrices.
//!
//! This module provides matrix traits that should be implemented by users.
//! Also provides various wrappers to attach additional data to matrices, change their indexing types or multiply two matrices.

// TODO: Add reasonable constraints to reverse and unreverse methods on bases and matrices

use std::cell::RefCell;
use std::hash::Hash;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;
use std::{cmp::Reverse, collections::HashMap};

use adaptors::{Consolidator, WithCachedCols};
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
/// * It is up to you to ensure that we never construct a [`ColT`](MatrixOracle::ColT) or [`RowT`](MatrixOracle::RowT) that doesn't correspond to an element of the chosen bases (this may change in future versions to allow graceful failure).
/// * An object of type [`CoefficientField`](MatrixOracle::CoefficientField) should represent a *non-zero* coefficient; ideally `0` is un-representable in this type. Since we essentially represent our columns as linear combinations of the row basis, `0` is represented by the *absence* of that basis element in the combination, i.e. `None` rather than `Some`. A good choice is [`Z2`](crate::fields::Z2).
///
/// Note that an implementation of [`MatrixOracle`] for `M` automatically derives an implementation for `&'a M`, `Box<M>`, `Rc<M>` and `Arc<M>`.
pub trait MatrixOracle {
    /// Represents the non-zero elements in the field over which we are doing linear algebra.   
    type CoefficientField: NonZeroCoefficient;
    /// Represents elements in the domain basis.
    type ColT: BasisElement;
    /// Represents elements in the target basis.
    type RowT: BasisElement;

    // TODO: Should we allow column to be fallible and return a Result type.
    //       When reducing we assume that we can read the column corresponding to each lowest 1.
    //       This might not be able to be guaranteed at the type level.

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

    /// Reverse the order on both the rows and columns.
    /// Both the indices and the filtration values are now the [`Reverse`] of their previous type.
    fn reverse(self) -> ReverseMatrix<Self>
    where
        Self: Sized,
    {
        ReverseMatrix { oracle: self }
    }

    /// Reverse the order on both the rows and columns.
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

    /// Returns a wrapper around `self` which caches any calls to [`column`](MatrixOracle::column) in an internal [`HashMap`].
    ///
    /// This may be useful if your matrix is relatively small but expensive to re-compute on the fly.
    /// This way your matrix can still be lazily evaluated, but each column is only computed once.
    /// Implementations of [`HasColBasis`] and [`HasRowFiltration`] are passed through transparently.
    fn cache_cols(self) -> WithCachedCols<Self>
    where
        Self: Sized,
    {
        WithCachedCols {
            oracle: self,
            cache: RefCell::new(Default::default()),
        }
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

/// Represents a matrix that has a filtration on its row basis (this basis and filtration need not be pre-computed).
/// Either implemented explicitily or via [`with_filtration`](MatrixOracle::with_filtration).
pub trait HasRowFiltration: MatrixOracle {
    /// The type that the filtration function is valued in.
    type FiltrationT: FiltrationValue;
    /// Implementing the filtration function on the row basis.
    fn filtration_value(&self, row: Self::RowT) -> Self::FiltrationT;

    /// Uses [`MatrixOracle::column`] and [`HasRowFiltration::filtration_value`] to provide an iterator over non-zero column entries, enriches with the row filtration value.
    ///
    /// **Note:** You may wish to override the default implementation for efficiency's sake.
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

    /// Initialise an empty binary heap that can accept entries from this matrix.
    /// Mostly an implementation detail of [`build_bhcol`](HasRowFiltration::build_bhcol).
    fn empty_bhcol(&self) -> BHCol<Self::FiltrationT, Self::RowT, Self::CoefficientField> {
        BHCol::default()
    }

    /// Builds a binary heap out of the non-zero entries in this column, sorted according to the filtration value and then the default ordering on [`RowT`](MatrixOracle::RowT).
    /// Popping top entries off this binary heap will allow access to the column pivot.
    ///
    /// **Warning:** A given row may appear multiple times in the binary heap!
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

/// Represents an indexable column basis (of a matrix), typically pre-computed and stored in memory.
pub trait ColBasis {
    /// The type of elements within the column basis.
    type ElemT: BasisElement;
    /// Return the element at position `index` within the basis.
    fn element(&self, index: usize) -> Self::ElemT;
    /// Return the size of the basis.
    fn size(&self) -> usize;

    /// Construct a hashmap that stores the index of each element in the basis.
    /// Note [`ElemT`](Self::ElemT) must implement [`Hash`].
    fn construct_reverse_lookup(&self) -> HashMap<Self::ElemT, usize>
    where
        Self::ElemT: Hash,
    {
        (0..(self.size()))
            .map(|idx| (self.element(idx), idx))
            .collect()
    }

    /// Construct a basis with the opposite ordering so the element at position `index` in the new basis is in position `len - index` in `self`.
    fn reverse(self) -> ReverseBasis<Self>
    where
        Self: Copy,
    {
        ReverseBasis(self)
    }

    /// Essentially the inverse operation to [`reverse`](Self::reverse).
    /// You should call this if [`ElemT`](Self::ElemT) is a [`Reverse<T>`] and you would like a basis in which [`ElemT`](Self::ElemT) is `T`, rather than `Reverse<Reverse<T>>`.
    fn unreverse<T>(self) -> UnreverseBasis<Self>
    where
        Self: Copy,
        Self: ColBasis<ElemT = Reverse<T>>,
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

/// Represents a matrix that has an indexable column basis, typically this is pre-computed to save time.
/// Usually constructed via [`with_basis`](MatrixOracle::with_basis).
pub trait HasColBasis: MatrixOracle {
    /// The type of the column basis that this matrix is equipped with.
    type BasisT: ColBasis<ElemT = Self::ColT>;
    /// The type of the reference to the column basis returned by [`basis`](Self::basis), this is often `&'a Self::BasisT`.
    /// However, if the basis is cheap to clone (or wrapped in a smart pointer), you might prefer to construct on the fly and hand over ownership.
    type BasisRef<'a>: Deref<Target = Self::BasisT>
    where
        Self: 'a;

    /// Return a reference to the basis attached to this matrix.
    fn basis(&self) -> Self::BasisRef<'_>;

    /// Yields the same matrix but equipped with the sub-basis in the provided `dimension`.
    /// Basis indices are now according to this sub-basis.
    /// This is essentially a sub-matrix view.
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
// TODO: Maybe provided a default implementation, storing a split basis as a Vec<B> and maybe a dimension offset?

/// Represents a [`ColBasis`] which is split into multiple sub-bases, according to an integer "dimension".
///
/// For example, the Rips (co)boundary matrix whose basis is split according to simplex dimension.
/// A basis implementing this type is typically stored as a [`Vec<B>`] where [`B: ColBasis`](ColBasis).
pub trait SplitByDimension: ColBasis {
    /// The type of each sub-basis.
    type SubBasisT: ColBasis<ElemT = Self::ElemT>;

    /// Should return a reference to the sub-basis of columns in the provided `dimension`.
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
