//! The core framework for implementing lazy oracles for sparse matrices.
//!
//! This module provides matrix traits that should be implemented by users.
//! Also provides various wrappers to attach additional data to matrices, change their indexing types or multiply two matrices.
//!
//! # The traits
//!
//! * [`MatrixOracle`] (required): the non-zero entries of each column, produced lazily.
//! * [`HasRowFiltration`] (required for reduction): a filtration value for each row.
//!   It also has optional methods which the reduction algorithms use to do less work, see below.
//! * [`HasColBasis`] / [`HasSplitColBasis`] (required for reduction): the ordered basis of columns to reduce,
//!   split by dimension for the clearing algorithms. Usually attached with [`with_basis`](MatrixOracle::with_basis),
//!   for example as a [`SplitBasis`].
//! * [`HasTranspose`] (optional): an oracle for the transpose, attached with [`with_transpose`](MatrixOracle::with_transpose).
//!   For a coboundary matrix the transpose is the boundary matrix, and vice versa.
//!
//! # Optional performance hooks
//!
//! Default implementations are provided for these, but overriding them can make a big difference
//! (see the [`reduction`](crate::reduction#doing-less-work) module docs for how they are used):
//! * [`HasRowFiltration::column_with_filtration`]: the column together with its rows' filtration values,
//!   if you can compute these together more cheaply than one row at a time.
//! * [`HasRowFiltration::column_pivot`] and [`HasRowFiltration::column_min`]: the largest and smallest entries of a column.
//!   Most columns need no reduction beyond finding their pivot, so if you can find it without visiting every entry
//!   (say, by enumerating the column in order and knowing a bound on its filtration values) then override these.
//!   A [`reverse`](MatrixOracle::reverse)d matrix's pivot is the original matrix's [`column_min`](HasRowFiltration::column_min).
//! * [`HasRowFiltration::first_column_in_row`] and [`HasRowFiltration::last_column_in_row`]:
//!   these let the reduction algorithms recognise *apparent pairs*.
//!   [`with_transpose`](MatrixOracle::with_transpose) provides them by scanning the transpose,
//!   but implement them on your matrix if you can find them faster (the transpose's version is then used only as a fallback).
//!
//! The wrappers in [`adaptors`] forward these methods to the matrix they wrap where that makes sense,
//! and [`ReverseMatrix`] swaps the first/last and largest/smallest versions.

// TODO: Add reasonable constraints to reverse and unreverse methods on bases and matrices

// Bases are handed out as by-value views, so nothing in this module (including its adaptors) needs `unsafe`.
#![forbid(unsafe_code)]

use std::hash::Hash;
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
    UsingColBasisIndex, WithFuncFiltration, WithSubBasis, WithTranspose, WithTrivialFiltration,
};
pub use self::bases::{BasisSlice, SplitBasis};

pub mod adaptors;
pub mod bases;
pub mod combinators;
pub mod implementors;
#[cfg(test)]
mod tests;

// ========= Traits for matrix indices and filtrations =========

/// Row and column indices must explicity implement this (see [`MatrixOracle`](MatrixOracle::ColT)).
///
/// Ideally [`clone`](Clone::clone) should be *very* cheap as it is called regularly, [`Copy`] would be ideal.
///
/// # Dense indices
///
/// If basis elements are numbered by small integers (as when the cells of a complex are `0, ..., n - 1`), set
/// [`DENSE_INDEX`](Self::DENSE_INDEX) and implement [`dense_index`](Self::dense_index) and
/// [`from_dense_index`](Self::from_dense_index). The reduction algorithms then keep their per-column state (pivot
/// tables, pairs, reduction columns) in vectors indexed by these numbers instead of hash maps, see
/// [`ElementMap`](crate::bookkeeping::ElementMap). Vectors are as long as the largest number used, so only do this
/// when the numbers are dense.
pub trait BasisElement: Ord + Clone {
    /// Whether basis elements are densely numbered, see the [trait docs](Self). Off by default.
    const DENSE_INDEX: bool = false;

    /// The number of this basis element: injective, and small. Only called if [`DENSE_INDEX`](Self::DENSE_INDEX).
    fn dense_index(&self) -> usize {
        unimplemented!("dense_index is only called on types with DENSE_INDEX set")
    }

    /// The basis element numbered `index`, inverting [`dense_index`](Self::dense_index).
    /// Only called if [`DENSE_INDEX`](Self::DENSE_INDEX).
    fn from_dense_index(index: usize) -> Self {
        let _ = index;
        unimplemented!("from_dense_index is only called on types with DENSE_INDEX set")
    }
}
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
impl FiltrationValue for u32 {}
impl FiltrationValue for u64 {}
impl FiltrationValue for isize {}
impl FiltrationValue for () {}

// Reversing the order doesn't change the numbering
impl<T> BasisElement for Reverse<T>
where
    T: BasisElement,
{
    const DENSE_INDEX: bool = T::DENSE_INDEX;

    #[inline]
    fn dense_index(&self) -> usize {
        self.0.dense_index()
    }

    #[inline]
    fn from_dense_index(index: usize) -> Self {
        Reverse(T::from_dense_index(index))
    }
}
impl<T> FiltrationValue for Reverse<T> where T: FiltrationValue {}

/// The position of every row in the order the reductions use (filtration value, then row index), as a dense
/// number `0 <= rank < bound`, for rows whose [`dense_index`](BasisElement::dense_index) is known.
///
/// A matrix that can give this (see [`row_ranks`](HasRowFiltration::row_ranks)) lets
/// [`PivotColumns`](crate::columns::PivotColumns) keep the column being reduced as a bitset over ranks, as PHAT does,
/// rather than as a heap. The ranks are a shared table, cheap to clone, from the dense index of each row to its rank.
#[derive(Clone, Debug)]
pub struct RowRanks {
    /// `table[row.dense_index()]` is the rank of `row` in increasing order, or `None` if that is the dense index itself
    table: Option<std::sync::Arc<[u32]>>,
    /// One more than the largest rank
    bound: usize,
    /// Whether the order is reversed, so that the rank is `bound - 1 - table[..]`
    descending: bool,
}

impl RowRanks {
    /// Ranks from a table: `table[i]` is the rank of the row with dense index `i`, where ranks must be distinct, less
    /// than `table.len()`, and increase with the order on [`ColumnEntry`] (filtration value, then row index).
    pub fn new(table: std::sync::Arc<[u32]>) -> Self {
        Self {
            bound: table.len(),
            table: Some(table),
            descending: false,
        }
    }

    /// Ranks that are the dense indices `0..bound` themselves, for rows numbered in order (no table to look up).
    pub fn identity(bound: usize) -> Self {
        Self {
            table: None,
            bound,
            descending: false,
        }
    }

    /// The ranks for the same rows in the reverse order (as for a [`reverse`](MatrixOracle::reverse)d matrix).
    #[must_use]
    pub fn reversed(self) -> Self {
        Self {
            descending: !self.descending,
            ..self
        }
    }

    /// One more than the largest rank.
    #[inline]
    pub fn bound(&self) -> usize {
        self.bound
    }

    /// The table from dense index to rank (`None` for the identity), and whether ranks are read descending
    /// (`bound() - 1 - table[i]`), for loops that look up many ranks and want to branch on these once.
    #[inline]
    pub(crate) fn parts(&self) -> (Option<&[u32]>, bool) {
        (self.table.as_deref(), self.descending)
    }

    /// The rank of `row`.
    #[inline]
    pub fn rank<R: BasisElement>(&self, row: &R) -> usize {
        let index = row.dense_index();
        let rank = match &self.table {
            Some(table) => table[index] as usize,
            None => index,
        };
        if self.descending {
            self.bound - 1 - rank
        } else {
            rank
        }
    }
}

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

    /// Endows `self` with an oracle for its transpose, so that the reduction algorithms can use apparent pairs
    /// (see [`first_column_in_row`](HasRowFiltration::first_column_in_row)).
    ///
    /// Column `r` of `transpose` must contain exactly the non-zero entries of row `r` of `self`.
    /// For a coboundary matrix, the transpose is the boundary matrix, and vice versa.
    fn with_transpose<T>(self, transpose: T) -> WithTranspose<Self, T>
    where
        Self: Sized,
        T: MatrixOracle<
            CoefficientField = Self::CoefficientField,
            ColT = Self::RowT,
            RowT = Self::ColT,
        >,
    {
        WithTranspose {
            matrix: self,
            transpose,
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

    /// Whether rows `a` and `b` have the same filtration value, so that a pair between them is a bar of length zero
    /// (which the reductions can skip, see [`Clearing::skip_zero_apparent_pairs`](crate::reduction::Clearing)).
    ///
    /// By default this compares [`filtration_value`](Self::filtration_value)s. A matrix whose filtration values are
    /// finer than its filtration (for example positions in it, which are cheaper to look up and compare) overrides
    /// this to compare the real values.
    fn same_filtration_value(&self, a: Self::RowT, b: Self::RowT) -> bool {
        self.filtration_value(a) == self.filtration_value(b)
    }

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

    /// The pivot of the column `col`: its largest non-zero entry, ordered by filtration value and then [`RowT`](MatrixOracle::RowT),
    /// or `None` if the column is zero.
    ///
    /// The default implementation makes a single pass over [`column_with_filtration`](HasRowFiltration::column_with_filtration)
    /// without allocating, summing the coefficients of any repeated rows.
    /// Only if the largest row cancels out does it fall back to building the whole column with [`build_bhcol`](HasRowFiltration::build_bhcol).
    ///
    /// The reduction algorithms call this for every column, and most columns need no further work
    /// (their pivot is not yet claimed), so you may wish to override it if you can find the pivot without visiting every entry.
    fn column_pivot(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        // The largest entry seen so far, and the summed coefficient of every entry in its row (None if they cancel)
        let mut best: Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> =
            None;
        let mut sum: Option<Self::CoefficientField> = None;
        // Internal iteration, see `BHCol::add_entries`
        self.column_with_filtration(col.clone()).for_each(|entry| {
            let ordering = best.as_ref().map(|current| entry.cmp(current));
            match ordering {
                Some(std::cmp::Ordering::Less) => {}
                Some(std::cmp::Ordering::Equal) => sum = entry.coeff + sum,
                Some(std::cmp::Ordering::Greater) | None => {
                    sum = Some(entry.coeff);
                    best = Some(entry);
                }
            }
        });
        match (best, sum) {
            (None, _) => None,
            (Some(mut entry), Some(coeff)) => {
                entry.coeff = coeff;
                Some(entry)
            }
            // The largest row cancelled, so the pivot is further down
            (Some(_), None) => self.build_bhcol(col).pop_pivot(),
        }
    }

    /// The smallest non-zero entry of the column `col`, ordered by filtration value and then [`RowT`](MatrixOracle::RowT),
    /// or `None` if the column is zero.
    ///
    /// This is the pivot of the [`reverse`](MatrixOracle::reverse)d matrix, which is how cohomology is computed,
    /// so override it if you can find the smallest entry without visiting every entry
    /// (for example, if you can enumerate the column in order and know a lower bound on its filtration values).
    /// The default implementation is a single pass like [`column_pivot`](HasRowFiltration::column_pivot).
    fn column_min(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        // The smallest entry seen so far, and the summed coefficient of every entry in its row (None if they cancel)
        let mut best: Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> =
            None;
        let mut sum: Option<Self::CoefficientField> = None;
        // Internal iteration, see `BHCol::add_entries`
        self.column_with_filtration(col.clone()).for_each(|entry| {
            let ordering = best.as_ref().map(|current| entry.cmp(current));
            match ordering {
                Some(std::cmp::Ordering::Greater) => {}
                Some(std::cmp::Ordering::Equal) => sum = entry.coeff + sum,
                Some(std::cmp::Ordering::Less) | None => {
                    sum = Some(entry.coeff);
                    best = Some(entry);
                }
            }
        });
        match (best, sum) {
            (None, _) => None,
            (Some(mut entry), Some(coeff)) => {
                entry.coeff = coeff;
                Some(entry)
            }
            // The smallest row cancelled, so sum up the whole column and take the smallest survivor
            (Some(_), None) => {
                let mut entries: Vec<_> = self.column_with_filtration(col).collect();
                entries.sort();
                let mut i = 0;
                while i < entries.len() {
                    let mut total = Some(entries[i].coeff);
                    let mut j = i + 1;
                    while j < entries.len() && entries[j] == entries[i] {
                        total = entries[j].coeff + total;
                        j += 1;
                    }
                    if let Some(coeff) = total {
                        let mut entry = entries.swap_remove(i);
                        entry.coeff = coeff;
                        return Some(entry);
                    }
                    i = j;
                }
                None
            }
        }
    }

    /// The entries of column `col` largest first (by filtration value, then [`RowT`](MatrixOracle::RowT)), with each row
    /// at most once and no zero coefficients, if the matrix can give them in this order without sorting; otherwise
    /// `None` (the default).
    ///
    /// The reduction algorithms repeatedly add columns of the matrix to the column being reduced, and with
    /// [`MergedColumns`](crate::columns::MergedColumns) each must be sorted first. If your matrix stores its columns
    /// sorted, implement this (and [`sorted_column_ascending`](HasRowFiltration::sorted_column_ascending)) to skip that.
    /// If you implement it, the order must be right: entries out of order give wrong results.
    fn sorted_column(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        let _ = col;
        None::<std::iter::Empty<_>>
    }

    /// As [`sorted_column`](HasRowFiltration::sorted_column), but smallest first. Used for [`reverse`](MatrixOracle::reverse)d
    /// matrices (and so for cohomology).
    fn sorted_column_ascending(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        let _ = col;
        None::<std::iter::Empty<_>>
    }

    /// Of the columns with a non-zero entry in `row`, the first in the order on filtration value and then [`ColT`](MatrixOracle::ColT),
    /// or `None` if this is not known (the default).
    ///
    /// Only meaningful for square matrices (so that columns have filtration values), whose column bases are sorted in this order.
    /// The reduction algorithms use it to detect *apparent pairs*: if the pivot of column `c` is `r`, and `c` is the first column in row `r`,
    /// then `(r, c)` is a persistence pair that needs no reduction (nor storage, since it can be recomputed).
    /// Implement it if you can find this column cheaply, typically by computing the pivot of the corresponding column of the transpose
    /// (see [`with_transpose`](MatrixOracle::with_transpose)).
    /// If you implement it, it must be correct: returning a column that is not the first in its row gives wrong results.
    fn first_column_in_row(&self, _row: Self::RowT) -> Option<Self::ColT> {
        None
    }

    /// As [`first_column_in_row`](HasRowFiltration::first_column_in_row), but the last column. Used for [`reverse`](MatrixOracle::reverse)d matrices.
    fn last_column_in_row(&self, _row: Self::RowT) -> Option<Self::ColT> {
        None
    }

    /// The rank of every row in the order on filtration value and then row index, if known (by default it is not).
    /// Only meaningful when [`RowT`](MatrixOracle::RowT) has a [`dense_index`](BasisElement::dense_index).
    ///
    /// [`PivotColumns`](crate::columns::PivotColumns) uses it to keep the column being reduced as a bitset over ranks.
    /// If you implement it, it must agree with the order: wrong ranks give wrong results.
    fn row_ranks(&self) -> Option<RowRanks> {
        None
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

    fn same_filtration_value(&self, a: Self::RowT, b: Self::RowT) -> bool {
        (*self).same_filtration_value(a, b)
    }

    fn column_pivot(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        (*self).column_pivot(col)
    }

    fn column_min(
        &self,
        col: Self::ColT,
    ) -> Option<ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>> {
        (*self).column_min(col)
    }

    fn first_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        (*self).first_column_in_row(row)
    }

    fn last_column_in_row(&self, row: Self::RowT) -> Option<Self::ColT> {
        (*self).last_column_in_row(row)
    }

    fn sorted_column(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        (*self).sorted_column(col)
    }

    fn sorted_column_ascending(
        &self,
        col: Self::ColT,
    ) -> Option<
        impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>,
    > {
        (*self).sorted_column_ascending(col)
    }

    fn row_ranks(&self) -> Option<RowRanks> {
        (*self).row_ranks()
    }
    fn column_with_filtration(
        &self,
        col: Self::ColT,
    ) -> impl Iterator<Item = ColumnEntry<Self::FiltrationT, Self::RowT, Self::CoefficientField>>
    {
        (*self).column_with_filtration(col)
    }
}

// ======== Access to the transpose ============================

/// A matrix with an oracle for its transpose, typically attached with [`with_transpose`](MatrixOracle::with_transpose).
///
/// For a coboundary matrix the transpose is the boundary matrix, and vice versa.
/// This is used to find apparent pairs during reduction, and to compute representatives by involution
/// (see [`cohomology_with_involution`](crate::barcode::cohomology_with_involution)).
pub trait HasTranspose: MatrixOracle {
    /// The type of the oracle for the transpose.
    type TransposeT<'a>: MatrixOracle<
            CoefficientField = Self::CoefficientField,
            ColT = Self::RowT,
            RowT = Self::ColT,
        > + Copy
    where
        Self: 'a;

    /// The transpose: column `r` lists the non-zero entries in row `r` of `self`.
    fn transpose(&self) -> Self::TransposeT<'_>;
}

impl<M: HasTranspose> HasTranspose for &M {
    type TransposeT<'a>
        = M::TransposeT<'a>
    where
        Self: 'a;

    fn transpose(&self) -> Self::TransposeT<'_> {
        (*self).transpose()
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
///
/// The basis is handed out as a cheap, [`Copy`] *view* (often `&'a B`, or a wrapper around such a reference).
/// Adaptors such as [`ReverseBasis`] and [`WithSubBasis`] simply wrap or transform these views by value,
/// so no references into temporaries (and hence no `unsafe` pointer casts) are ever required.
pub trait HasColBasis: MatrixOracle {
    /// The type of the view of the column basis that this matrix is equipped with.
    /// This borrows from `self` for `'a` and should be cheap to copy.
    type BasisT<'a>: ColBasis<ElemT = Self::ColT> + Copy
    where
        Self: 'a;

    /// Return a view of the basis attached to this matrix.
    fn basis(&self) -> Self::BasisT<'_>;
}

impl<T> HasColBasis for &T
where
    T: HasColBasis,
{
    type BasisT<'b>
        = T::BasisT<'b>
    where
        Self: 'b;

    fn basis(&self) -> Self::BasisT<'_> {
        (*self).basis()
    }
}

/// Represents a matrix whose column basis is split into multiple sub-bases, according to an integer "dimension".
///
/// For example, the Rips (co)boundary matrix whose basis is split according to simplex dimension.
/// This is what the clearing algorithm needs in order to reduce one dimension at a time.
///
/// The lifetime of the sub-basis view appears only in the signature of [`sub_basis`](Self::sub_basis),
/// never in a `for<'a>` bound, which keeps generic code (and borrowed matrices) free of `'static` requirements.
/// The sub-basis is returned by value, so it never needs to exist as an owned object inside the matrix.
/// For matrices built with [`with_basis`](MatrixOracle::with_basis) this is implemented automatically
/// whenever `&B` implements [`SplitByDimension`] (e.g. [`SplitBasis`]).
pub trait HasSplitColBasis: HasColBasis {
    /// The type of the view of the sub-basis in a given dimension.
    type SubBasisT<'a>: ColBasis<ElemT = Self::ColT> + Copy
    where
        Self: 'a;

    /// The number of dimensions, i.e. [`sub_basis`](Self::sub_basis) is valid for `0..n_dimensions()`.
    fn n_dimensions(&self) -> usize;

    /// Return a view of the sub-basis of columns in the provided `dimension`.
    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_>;

    /// Yields the same matrix but equipped with the sub-basis in the provided `dimension`.
    /// Basis indices are now according to this sub-basis.
    /// This is essentially a sub-matrix view.
    fn sub_matrix_in_dimension(self, dimension: usize) -> WithSubBasis<Self>
    where
        Self: Sized,
    {
        WithSubBasis {
            oracle: self,
            dimension,
        }
    }
}

impl<T> HasSplitColBasis for &T
where
    T: HasSplitColBasis,
{
    type SubBasisT<'b>
        = T::SubBasisT<'b>
    where
        Self: 'b;

    fn n_dimensions(&self) -> usize {
        (*self).n_dimensions()
    }

    fn sub_basis(&self, dimension: usize) -> Self::SubBasisT<'_> {
        (*self).sub_basis(dimension)
    }
}

/// Represents a [`ColBasis`] which is split into multiple sub-bases, according to an integer "dimension".
///
/// Implementors are expected to be cheap [`Copy`] *views* (e.g. `&'a SplitBasis<T>`), and
/// [`in_dimension`](Self::in_dimension) consumes the view, returning another view.
/// Since the returned sub-basis is not tied to a borrow of `self`, wrappers such as [`ReverseBasis`] can build
/// it on the fly without owning pre-constructed sub-bases (and without `unsafe`).
/// See [`SplitBasis`] for a ready-made implementation backed by a single flat allocation.
pub trait SplitByDimension: ColBasis + Sized {
    /// The type of each sub-basis.
    type SubBasisT: ColBasis<ElemT = Self::ElemT> + Copy;

    /// The number of dimensions, i.e. [`in_dimension`](Self::in_dimension) is valid for `0..n_dimensions()`.
    fn n_dimensions(&self) -> usize;

    /// Return the sub-basis of columns in the provided `dimension`.
    fn in_dimension(self, dimension: usize) -> Self::SubBasisT;
}
